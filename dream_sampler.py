import torch
import torch.nn.functional as F
from dlm_steering.runtime.constants import MASK_ID

DREAM_ALGS = ("origin", "maskgit_plus", "topk_margin", "entropy")

def _top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    if probs.is_cuda and torch.are_deterministic_algorithms_enabled():
        cumulative_probs = probs.to(torch.float64)
        offset = 1
        while offset < probs.shape[-1]:
            cumulative_probs = torch.cat([
                cumulative_probs[..., :offset],
                cumulative_probs[..., offset:] + cumulative_probs[..., :-offset]], dim=-1)
            offset *= 2
        cumulative_probs = cumulative_probs.to(probs.dtype)
    else:
        cumulative_probs = torch.cumsum(probs, dim=-1)
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


def _categorical_would_raise(normalised):
    """torch.distributions.Categorical's argument validation (validate_args is
    on by default): every renormalised row must be a simplex within 1e-6."""
    if normalised.dim() < 1 or normalised.numel() == 0:
        return normalised.dim() < 1
    ok = (torch.all(normalised >= 0, dim=-1) & ((normalised.sum(dim=-1) - 1).abs() < 1e-6))
    return not bool(ok.all())


def dream_sample_tokens(logits, temperature=0.0, top_p=None, top_k=None,
                        margin_confidence=False, neg_entropy=False, rng=None):
    """Dream's sample_tokens rule with an explicit row RNG: returns
    (confidence, token) per row."""
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = _top_p_logits(logits, top_p)
    if top_k is not None:
        logits = _top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)
    if temperature > 0:
        normalised = probs / probs.sum(dim=-1, keepdim=True)
        if _categorical_would_raise(normalised):
            confidence, x0 = probs.max(dim=-1)
        else:
            x0 = torch.multinomial(normalised, 1, generator=rng).squeeze(-1)
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
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


def dream_transfer(x, logits, mask_index, t, s, last, *, alg="origin", alg_temp=None, temperature=0.0, top_p=None, top_k=None, mask_id=MASK_ID, rng=None):
    if alg not in DREAM_ALGS:
        raise ValueError(f"alg must be one of {DREAM_ALGS}, not {alg!r}")
    mask_logits = logits[mask_index]
    if alg == "origin":
        p_transfer = 1 - s / t if not last else 1
        x0 = torch.zeros_like(x[mask_index], device=x.device, dtype=torch.long) + mask_id
        transfer_index_t_s = torch.rand(*x0.shape, device=x.device, generator=rng) < p_transfer
        _, x0[transfer_index_t_s] = dream_sample_tokens(
            mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k, rng=rng)
        x[mask_index] = x0.clone()
        return
    if alg == "maskgit_plus":
        confidence, x0 = dream_sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, rng=rng)
    elif alg == "topk_margin":
        confidence, x0 = dream_sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True, rng=rng)
    else:  # entropy
        confidence, x0 = dream_sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True, rng=rng)
    num_mask_token = mask_index.sum() / mask_index.shape[0]
    number_transfer_tokens = (int(num_mask_token * (1 - s / t)) if not last else int(num_mask_token))
    full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
    full_confidence[mask_index] = confidence
    if number_transfer_tokens > 0:
        if alg_temp is None or alg_temp == 0:
            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
        else:
            full_confidence = F.softmax(full_confidence / alg_temp, dim=-1)
            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens, generator=rng)
        x_ = torch.zeros_like(x, device=x.device, dtype=torch.long) + mask_id
        x_[mask_index] = x0.clone()
        row_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(transfer_index)
        x[row_indices, transfer_index] = x_[row_indices, transfer_index]


def dream_denoise(x, targets, n_steps, forward, *, eps=1e-3, **sampling):
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


def _generate_dream(model, x, region, defender, prompt_length, *, steps, block_length, temperature, alg, alg_temp, top_p, top_k, eps, rng=None):
    sampling = {"decoder": "dream", "temperature": temperature, "alg": alg, "alg_temp": alg_temp, "top_p": top_p, "top_k": top_k, "eps": eps}
    rule = {k: sampling[k] for k in ("alg", "alg_temp", "temperature", "top_p", "top_k")}
    rule["rng"] = rng
    timesteps = dream_timesteps(steps, eps, x.device)
    audited = torch.zeros_like(region)     # answer slots already handed to after_block
    block_index = 0
    for i in range(steps):
        mask_index = x == MASK_ID
        if not bool(mask_index.any()):
            break
        if defender is None:
            logits = model(x).logits
        else:
            logits = defender.forward(x, region, schedule_scale=1.0).logits
        dream_transfer(x, logits, mask_index, timesteps[i], timesteps[i + 1], i == steps - 1, mask_id=MASK_ID, **rule)

        if defender is None:
            continue
        committed = region & (x != MASK_ID)
        done = not bool((region & (x == MASK_ID)).any())
        fresh = committed & ~audited
        n_fresh = int(fresh.sum())
        if n_fresh and (n_fresh >= block_length or done):
            audited |= fresh
            defender.after_block(x, region, block_number=block_index, block_positions=fresh, prompt_length=prompt_length, temperature=temperature, remasking=alg, last_block=done, sampling=sampling, rng=rng)
            block_index += 1
            audited &= x != MASK_ID
        if done and not bool((region & (x == MASK_ID)).any()):
            break
    if bool((region & (x == MASK_ID)).any()):
        mask_index = x == MASK_ID
        logits = model(x).logits if defender is None else defender.forward(x, region, schedule_scale=1.0).logits
        dream_transfer(x, logits, mask_index, timesteps[-2], timesteps[-1], True, mask_id=MASK_ID, **rule)
    return x


@torch.no_grad()
def generate(model, prompt_ids, defender=None, *, steps=128, gen_length=128, block_length=32, temperature=0.0, alg="origin", alg_temp=None, top_p=0.95, top_k=50, eps=1e-3, rng=None):
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("Dream generate supports one prompt at a time")
    if gen_length < 0 or steps <= 0 or block_length <= 0:
        raise ValueError("Invalid generation length, steps or audit interval")
    if alg not in DREAM_ALGS or not 0 < eps < 1:
        raise ValueError("Invalid Dream sampling rule")
    prompt_length = prompt_ids.shape[1]
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long, device=prompt_ids.device)
    x[:, :prompt_length] = prompt_ids
    region = x == MASK_ID
    if defender is not None:
        defender.reset()
    return _generate_dream(model, x, region, defender, prompt_length, steps=steps, block_length=block_length, temperature=temperature, alg=alg, 
                            alg_temp=alg_temp, top_p=top_p, top_k=top_k, eps=eps, rng=rng)
