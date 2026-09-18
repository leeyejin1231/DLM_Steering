"""The single diffusion sampler, driven by an optional Defender.

Two decoders share the defender protocol (before_step / forward / after_block):

`block` (LLaDA): semi-autoregressive block decoding. Per step the defender may
reopen committed tokens (before_step), one forward runs detection + steering
(forward), then the sampler commits the top-confidence masked tokens of the
current block. After a repair the transfer schedule for the rest of the block
is recomputed as an even split of everything masked before block_end; the
last step of every block fills whatever is still masked there. Between
blocks the defender gets a boundary hook (after_block) that may audit the
finished block or remask and regenerate it.

`dream` (Dream): a verbatim port of Dream's own `diffusion_generate` loop
(generation_utils._sample): no blocks, the whole masked sequence is denoised
over `steps` timesteps linspace(1, eps); each step transfers
n_masked * (1 - s/t) tokens chosen by `alg` (origin = uniformly at random,
maskgit_plus / topk_margin / entropy = by confidence) and sampled from
softmax(logits / temperature) after top_p / top_k. The defender's boundary
hook fires every `block_length` committed answer tokens (a "block" is the
set of tokens committed since the previous audit, wherever they sit) and
once more when the sequence is complete, so v3's first-block audit and
block recovery keep their meaning without a positional block structure.

CFG is not supported. Answer slots are every mask in the sequence, so masks
planted inside the prompt (DIJA) are filled, steered, and eligible for repair
like the rest.
"""

import torch
import torch.nn.functional as F
from torch import distributions as dists

from common import MASK_ID, step_scale
from llada import add_gumbel_noise, get_num_transfer_tokens

DECODERS = ("block", "dream")
DREAM_ALGS = ("origin", "maskgit_plus", "topk_margin", "entropy")


# ------------------------------------------------------------------ block (LLaDA)
def commit_sample(x, logits, eligible, count, temperature, remasking, final=False):
    """Commit the `count` highest-confidence eligible positions in place.

    eligible: index tensor of still-masked candidate positions. Returns the
    eligible indices left unfilled.
    """
    n = eligible.numel()
    k = n if final else min(int(count), n)
    if k <= 0:
        return eligible
    sub = logits[0, eligible]
    predicted = add_gumbel_noise(sub, temperature).argmax(-1)
    if remasking == "low_confidence":
        sub64 = sub.to(torch.float64)
        confidence = (sub64 - sub64.logsumexp(dim=-1, keepdim=True)).exp()
        confidence = confidence.gather(1, predicted[:, None]).squeeze(-1)
    elif remasking == "random":
        confidence = torch.rand(n, dtype=torch.float64, device=x.device)
    else:
        raise NotImplementedError(remasking)
    selected = confidence.topk(k).indices
    x[0, eligible[selected]] = predicted[selected]
    keep = torch.ones(n, dtype=torch.bool, device=x.device)
    keep[selected] = False
    return eligible[keep]


# ------------------------------------------------------------------ dream (native)
def _top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def _top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    return logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)


def dream_sample_tokens(logits, temperature=0.0, top_p=None, top_k=None,
                        margin_confidence=False, neg_entropy=False):
    """Dream's sample_tokens (generation_utils.py), unchanged: returns
    (confidence, token) per row."""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except Exception:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        confidence = sorted_probs[:, 0] - sorted_probs[:, 1]
    if neg_entropy:
        log_probs = torch.log(probs + 1e-10)
        confidence = torch.sum(probs * log_probs, dim=-1)
    return confidence, x0


def dream_timesteps(steps, eps, device):
    return torch.linspace(1, eps, steps + 1, device=device)


def dream_transfer(x, logits, mask_index, t, s, last, *, alg="origin", alg_temp=None,
                   temperature=0.0, top_p=None, top_k=None, mask_id=MASK_ID):
    """One Dream denoising step over the masked positions `mask_index`
    (bool [1, seq]), in place: the body of Dream's _sample loop for one i.
    `last` fills every remaining mask (Dream's final step)."""
    if alg not in DREAM_ALGS:
        raise ValueError(f"alg must be one of {DREAM_ALGS}, not {alg!r}")
    mask_logits = logits[mask_index]
    if alg == "origin":
        p_transfer = 1 - s / t if not last else 1
        x0 = torch.zeros_like(x[mask_index], device=x.device, dtype=torch.long) + mask_id
        transfer_index_t_s = torch.rand(*x0.shape, device=x.device) < p_transfer
        _, x0[transfer_index_t_s] = dream_sample_tokens(
            mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
        x[mask_index] = x0.clone()
        return
    if alg == "maskgit_plus":
        confidence, x0 = dream_sample_tokens(mask_logits, temperature=temperature,
                                             top_p=top_p, top_k=top_k)
    elif alg == "topk_margin":
        confidence, x0 = dream_sample_tokens(mask_logits, temperature=temperature,
                                             top_p=top_p, top_k=top_k, margin_confidence=True)
    else:  # entropy
        confidence, x0 = dream_sample_tokens(mask_logits, temperature, top_p=top_p,
                                             top_k=top_k, neg_entropy=True)
    num_mask_token = mask_index.sum() / mask_index.shape[0]
    number_transfer_tokens = (int(num_mask_token * (1 - s / t)) if not last
                              else int(num_mask_token))
    full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
    full_confidence[mask_index] = confidence
    if number_transfer_tokens > 0:
        if alg_temp is None or alg_temp == 0:
            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
        else:
            full_confidence = F.softmax(full_confidence / alg_temp, dim=-1)
            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)
        x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mask_id
        x_[mask_index] = x0.clone()
        row_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(transfer_index)
        x[row_indices, transfer_index] = x_[row_indices, transfer_index]


def dream_denoise(x, targets, n_steps, forward, *, eps=1e-3, **sampling):
    """Denoise the masked positions among `targets` (bool [1, seq]) over
    n_steps Dream timesteps; `forward(x)` returns the (realigned) logits.
    Used by v3 block recovery so regenerated tokens follow the same rule as
    the base decoding. Returns the number of forwards run."""
    timesteps = dream_timesteps(n_steps, eps, x.device)
    forwards = 0
    for i in range(n_steps):
        mask_index = targets & (x == sampling.get("mask_id", MASK_ID))
        if not bool(mask_index.any()):
            break
        logits = forward(x)
        forwards += 1
        dream_transfer(x, logits, mask_index, timesteps[i], timesteps[i + 1],
                       i == n_steps - 1, **sampling)
    return forwards


# ------------------------------------------------------------------ entry point
@torch.no_grad()
def generate(model, prompt_ids, defender=None, *, steps=128, gen_length=128,
             block_length=32, temperature=0.0, remasking="low_confidence",
             schedule="const", decoder="block", alg="origin", alg_temp=None,
             top_p=None, top_k=None, eps=1e-3):
    """Diffusion sampling driven by a Defender; None runs undefended.

    gen_length=0 runs pure infilling: the prompt's own mask slots (e.g. DIJA
    spans) are denoised over `steps` steps with no assistant suffix appended
    (one block under the block decoder).

    decoder="dream" uses Dream's native rule (see module docstring):
    remasking/schedule are ignored, alg/alg_temp/top_p/top_k/eps are Dream's
    diffusion_generate arguments, and block_length is the audit granularity."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("generate supports one prompt at a time")
    if gen_length < 0 or block_length <= 0 or steps <= 0:
        raise ValueError("gen_length must be >= 0; block length and steps must be positive")
    if decoder not in DECODERS:
        raise ValueError(f"decoder must be one of {DECODERS}, not {decoder!r}")
    prompt_length = prompt_ids.shape[1]
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long,
                   device=model.device)
    x[:, :prompt_length] = prompt_ids.clone()
    region = x == MASK_ID
    if not region.any():
        return x
    if defender is not None:
        defender.reset()
    if decoder == "dream":
        return _generate_dream(model, x, region, defender, prompt_length, steps=steps,
                               block_length=block_length, temperature=temperature,
                               alg=alg, alg_temp=alg_temp, top_p=top_p, top_k=top_k, eps=eps)
    if gen_length and (gen_length % block_length or steps % (gen_length // block_length)):
        raise ValueError("gen_length must be a multiple of block_length and steps of num_blocks")
    return _generate_block(model, x, region, defender, prompt_length, steps=steps,
                           gen_length=gen_length, block_length=block_length,
                           temperature=temperature, remasking=remasking, schedule=schedule)


def _generate_block(model, x, region, defender, prompt_length, *, steps, gen_length,
                    block_length, temperature, remasking, schedule):
    sampling = {"decoder": "block", "temperature": temperature, "remasking": remasking}
    num_blocks = gen_length // block_length if gen_length else 1
    steps_per_block = steps // num_blocks
    for num_block in range(num_blocks):
        block_end = min(prompt_length + (num_block + 1) * block_length, x.shape[1])
        block_start = 0 if gen_length == 0 else prompt_length + num_block * block_length
        scope = region.clone()
        scope[:, block_end:] = False
        block_positions = scope.clone()
        block_positions[:, :block_start] = False
        schedule_counts = get_num_transfer_tokens(
            (x == MASK_ID) & scope, steps_per_block)[0].tolist()
        eligible = ((x == MASK_ID) & scope)[0].nonzero().flatten()

        for i in range(steps_per_block):
            commit_count = None
            if defender is not None:
                commit_count = defender.before_step(
                    x, region, scope=scope, steps_remaining=steps_per_block - i)
            if commit_count is not None:
                # Reopened slots: spread everything still masked before block_end
                # evenly over the remaining steps of this block.
                schedule_counts = schedule_counts[:i] + get_num_transfer_tokens(
                    (x == MASK_ID) & scope, steps_per_block - i)[0].tolist()
                eligible = ((x == MASK_ID) & scope)[0].nonzero().flatten()
            if eligible.numel() == 0:
                break

            if defender is None:
                logits = model(x).logits
            else:
                logits = defender.forward(
                    x, region, schedule_scale=step_scale(schedule, i, steps_per_block)).logits

            eligible = commit_sample(x, logits, eligible, schedule_counts[i],
                                     temperature, remasking,
                                     final=(i == steps_per_block - 1))

        if defender is not None:
            defender.after_block(
                x, region, block_index=num_block, block_positions=block_positions,
                prompt_length=prompt_length,
                temperature=temperature, remasking=remasking,
                last_block=num_block == num_blocks - 1, sampling=sampling)

    return x


def _generate_dream(model, x, region, defender, prompt_length, *, steps, block_length,
                    temperature, alg, alg_temp, top_p, top_k, eps):
    sampling = {"decoder": "dream", "temperature": temperature, "alg": alg,
                "alg_temp": alg_temp, "top_p": top_p, "top_k": top_k, "eps": eps}
    rule = {k: sampling[k] for k in ("alg", "alg_temp", "temperature", "top_p", "top_k")}
    timesteps = dream_timesteps(steps, eps, x.device)
    audited = torch.zeros_like(region)     # answer slots already handed to after_block
    block_index = 0
    for i in range(steps):
        if defender is not None:
            # Whole sequence in scope; steps_remaining counts Dream timesteps.
            defender.before_step(x, region, scope=region, steps_remaining=steps - i)
        mask_index = x == MASK_ID
        if not bool(mask_index.any()):
            break
        if defender is None:
            logits = model(x).logits
        else:
            logits = defender.forward(x, region, schedule_scale=1.0).logits
        dream_transfer(x, logits, mask_index, timesteps[i], timesteps[i + 1],
                       i == steps - 1, mask_id=MASK_ID, **rule)

        if defender is None:
            continue
        committed = region & (x != MASK_ID)
        done = not bool((region & (x == MASK_ID)).any())
        fresh = committed & ~audited
        n_fresh = int(fresh.sum())
        if n_fresh and (n_fresh >= block_length or done):
            audited |= fresh
            defender.after_block(
                x, region, block_index=block_index, block_positions=fresh,
                prompt_length=prompt_length, temperature=temperature,
                remasking=alg, last_block=done, sampling=sampling)
            block_index += 1
            # Recovery may have reopened slots; they are re-committed by the
            # remaining timesteps (transfer counts follow the live mask count).
            audited &= x != MASK_ID
        if done and not bool((region & (x == MASK_ID)).any()):
            break
    if bool((region & (x == MASK_ID)).any()):
        # Recovery at the final boundary can leave masks with no timestep
        # left: fill them with one more forward, as Dream's last step does.
        mask_index = x == MASK_ID
        logits = model(x).logits if defender is None else \
            defender.forward(x, region, schedule_scale=1.0).logits
        dream_transfer(x, logits, mask_index, timesteps[-2], timesteps[-1], True,
                       mask_id=MASK_ID, **rule)
    return x
