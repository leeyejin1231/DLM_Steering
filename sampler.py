"""The single diffusion sampler, driven by an optional Defender.

Semi-autoregressive block decoding: per step the defender may reopen committed
tokens (before_step), one forward runs detection + steering (forward), then
the sampler commits the top-confidence masked tokens. After a repair the
transfer schedule for the rest of the block is recomputed as an even split of
everything masked before block_end; the last step of every block fills
whatever is still masked there, so reopened slots from earlier blocks can
never be left open. Between blocks the defender gets a boundary hook
(after_block) that may audit the finished block or remask and regenerate it.

CFG is not supported. Answer slots are every mask in the sequence, so masks
planted inside the prompt (DIJA) are filled, steered, and eligible for repair
like the rest.
"""

import torch

from common import MASK_ID, MODEL_LOCK, step_scale
from llada import PAD_ID, add_gumbel_noise, get_num_transfer_tokens


def commit_sample(x, logits, eligible, count, temperature, remasking,
                  final=False, rng=None):
    """Commit the `count` highest-confidence eligible positions in place.

    eligible: index tensor of still-masked candidate positions. Returns the
    eligible indices left unfilled.
    """
    n = eligible.numel()
    k = n if final else min(int(count), n)
    if k <= 0:
        return eligible
    # logits may be full [1, seq, vocab] or pre-sliced to the eligible rows
    # [1, n, vocab] (ln_f hook); identical when every position is eligible.
    sub = logits[0, eligible] if logits.shape[1] == x.shape[1] else logits[0]
    predicted = add_gumbel_noise(sub, temperature, rng).argmax(-1)
    if remasking == "low_confidence":
        sub64 = sub.to(torch.float64)
        confidence = (sub64 - sub64.logsumexp(dim=-1, keepdim=True)).exp()
        confidence = confidence.gather(1, predicted[:, None]).squeeze(-1)
    elif remasking == "random":
        confidence = torch.rand(n, dtype=torch.float64, device=x.device,
                                generator=rng)
    else:
        raise NotImplementedError(remasking)
    selected = confidence.topk(k).indices
    x[0, eligible[selected]] = predicted[selected]
    keep = torch.ones(n, dtype=torch.bool, device=x.device)
    keep[selected] = False
    return eligible[keep]


@torch.no_grad()
def generate(model, prompt_ids, defender=None, *, steps=128, gen_length=128,
             block_length=32, temperature=0.0, remasking="low_confidence",
             schedule="const", rng=None):
    """Diffusion sampling driven by a Defender; None runs undefended.

    gen_length=0 runs pure infilling: the prompt's own mask slots (e.g. DIJA
    spans) form a single block denoised over `steps` steps, with no assistant
    suffix appended. rng: per-row torch.Generator; all sampling draws come
    from it so concurrent row workers stay bit-identical to serial runs."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("generate supports one prompt at a time")
    if gen_length < 0 or block_length <= 0 or steps <= 0:
        raise ValueError("gen_length must be >= 0; block length and steps must be positive")
    if gen_length and (gen_length % block_length or steps % (gen_length // block_length)):
        raise ValueError("gen_length must be a multiple of block_length and steps of num_blocks")
    prompt_length = prompt_ids.shape[1]
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long,
                   device=model.device)
    x[:, :prompt_length] = prompt_ids.clone()
    region = x == MASK_ID
    if not region.any():
        return x
    if defender is not None:
        defender.reset()

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
        # Infilling checkpoints. With gen_length 0 the prompt's mask slots are
        # one block, exactly like DIJA's own loop, so there is no block boundary
        # to audit part-way through. A defender that wants one sets
        # infill_checkpoint = N and gets an audit after every N committed slots
        # (checkpoint k after N*(k+1)). The denoising order is untouched: only
        # when the audit happens changes.
        every = (defender.infill_checkpoint
                 if defender is not None and gen_length == 0 else 0)
        total = eligible.numel()
        checkpoints = 0

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
                ln_f = model.model.transformer.ln_f
                with MODEL_LOCK:
                    handle = ln_f.register_forward_hook(
                        lambda m, i, o: o[:, eligible])
                    try:
                        logits = model(x).logits
                    finally:
                        handle.remove()
            else:
                logits = defender.forward(
                    x, region, schedule_scale=step_scale(schedule, i, steps_per_block),
                    logit_positions=eligible).logits

            eligible = commit_sample(x, logits, eligible, schedule_counts[i],
                                     temperature, remasking,
                                     final=(i == steps_per_block - 1), rng=rng)

            # A checkpoint reached with slots still open audits what has been
            # committed so far; the audit rides the next forward, so it costs
            # nothing unless it triggers a recovery.
            while every and eligible.numel() and total - eligible.numel() >= every * (checkpoints + 1):
                defender.after_block(
                    x, region, block_number=checkpoints,
                    block_positions=region & (x != MASK_ID),
                    prompt_length=prompt_length, temperature=temperature,
                    remasking=remasking, last_block=False, rng=rng)
                checkpoints += 1

        if defender is not None:
            number = num_block
            if every:
                # The end of infilling is checkpoint k only when it lands exactly
                # on one; otherwise it is an extra audit that no --audit-boundary
                # can select (-1), so a row that never reaches checkpoint k gets
                # no recovery at k.
                number = total // every - 1 if total and total % every == 0 else -1
            defender.after_block(
                x, region, block_number=number, block_positions=block_positions,
                prompt_length=prompt_length,
                temperature=temperature, remasking=remasking,
                last_block=num_block == num_blocks - 1, rng=rng)

    return x


@torch.no_grad()
def generate_batch(model, prompts, *, steps=128, gen_length=128,
                   block_length=32, temperature=0.0, remasking="low_confidence",
                   schedule="const", pad_id=PAD_ID, rng=None):
    """Batched undefended diffusion sampling over left-padded prompts.

    prompts: list of (1, L_i) token-id tensors. Rows are left-padded with
    pad_id so every prompt ends at the same column and the appended answer
    region (hence block boundaries) aligns across the batch; pad columns are
    excluded via attention_mask, so each row denoises as in generate() (RoPE
    keeps real-token relative positions unchanged). Defense policies hold
    per-sequence state with divergent control flow (remask/recovery), so
    they cannot run here -- this is the --defense none path. `schedule` is
    accepted for signature parity with generate(); it only feeds defender
    hooks, which do not run. Returns one (1, L_i + gen_length) sequence per
    input prompt, padding stripped.

    Batched kernels reduce in a different order than the per-prompt path, so
    outputs are numerically equivalent but not guaranteed bit-identical.
    """
    if gen_length < 0 or block_length <= 0 or steps <= 0:
        raise ValueError("gen_length must be >= 0; block length and steps must be positive")
    if gen_length and (gen_length % block_length or steps % (gen_length // block_length)):
        raise ValueError("gen_length must be a multiple of block_length and steps of num_blocks")
    for p in prompts:
        if p.ndim != 2 or p.shape[0] != 1:
            raise ValueError("generate_batch takes a list of (1, L) tensors")
    if not prompts:
        return []
    if len(prompts) == 1:
        # rng and schedule must be forwarded: this is the same generation the
        # caller would get from defend(), and dropping rng would silently move
        # sampling back to the global RNG, breaking per-row determinism.
        return [generate(model, prompts[0], defender=None, steps=steps,
                         gen_length=gen_length, block_length=block_length,
                         temperature=temperature, remasking=remasking,
                         schedule=schedule, rng=rng)]

    batch = len(prompts)
    prompt_len = max(p.shape[1] for p in prompts)
    width = prompt_len + gen_length
    x = torch.full((batch, width), MASK_ID, dtype=torch.long,
                   device=model.device)
    attention_mask = torch.zeros(batch, width, dtype=torch.long,
                                 device=model.device)
    for r, p in enumerate(prompts):
        pad = prompt_len - p.shape[1]
        x[r, :pad] = pad_id
        x[r, pad:pad + p.shape[1]] = p[0]
        attention_mask[r, pad:] = 1
    region = x == MASK_ID
    if not region.any():
        return [x[r:r + 1, prompt_len - p.shape[1]:]
                for r, p in enumerate(prompts)]

    num_blocks = gen_length // block_length if gen_length else 1
    steps_per_block = steps // num_blocks
    ln_f = model.model.transformer.ln_f
    for num_block in range(num_blocks):
        block_end = (min(prompt_len + (num_block + 1) * block_length, width)
                     if gen_length else width)
        scope = region.clone()
        scope[:, block_end:] = False
        schedule_counts = get_num_transfer_tokens(
            (x == MASK_ID) & scope, steps_per_block)
        eligible = [((x[r] == MASK_ID) & scope[r]).nonzero().flatten()
                    for r in range(batch)]

        for i in range(steps_per_block):
            if not any(e.numel() for e in eligible):
                break
            # Per-row eligible positions differ, so flatten them into
            # (row * width + pos) and slice ln_f's [B*seq, hidden] once:
            # the vocab projection runs on total eligible rows only.
            flat = torch.cat([e + r * width
                              for r, e in enumerate(eligible) if e.numel()])
            with MODEL_LOCK:
                handle = ln_f.register_forward_hook(
                    lambda m, _i, o: o.reshape(-1, o.shape[-1])[flat].unsqueeze(0))
                try:
                    logits = model(x, attention_mask=attention_mask).logits
                finally:
                    handle.remove()
            offset = 0
            for r in range(batch):
                n = eligible[r].numel()
                if not n:
                    continue
                eligible[r] = commit_sample(
                    x[r:r + 1], logits[:, offset:offset + n], eligible[r],
                    int(schedule_counts[r, i]), temperature, remasking,
                    final=(i == steps_per_block - 1), rng=rng)
                offset += n

    return [x[r:r + 1, prompt_len - p.shape[1]:]
            for r, p in enumerate(prompts)]
