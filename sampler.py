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

from common import MASK_ID, step_scale
from llada import add_gumbel_noise, get_num_transfer_tokens


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


@torch.no_grad()
def generate(model, prompt_ids, defender=None, *, steps=128, gen_length=128,
             block_length=32, temperature=0.0, remasking="low_confidence",
             schedule="const"):
    """Diffusion sampling driven by a Defender; None runs undefended.

    gen_length=0 runs pure infilling: the prompt's own mask slots (e.g. DIJA
    spans) form a single block denoised over `steps` steps, with no assistant
    suffix appended."""
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
                last_block=num_block == num_blocks - 1)

    return x
