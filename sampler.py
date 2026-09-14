"""The single diffusion sampler, driven by an optional Defender.

Semi-autoregressive block decoding: per step the defender may reopen committed
tokens (before_step), one forward runs detection + steering (forward), then
the sampler commits the top-confidence masked tokens. After a repair the
transfer schedule for the rest of the block is recomputed as an even split of
everything masked before block_end; the last step of every block fills
whatever is still masked there, so reopened slots from earlier blocks can
never be left open.

CFG is not supported. Answer slots are every mask in the sequence, so masks
planted inside the prompt (DIJA) are filled, steered, and eligible for repair
like the rest.
"""

import torch
import torch.nn.functional as F

from common import MASK_ID, step_scale
from llada import add_gumbel_noise, get_num_transfer_tokens


@torch.no_grad()
def generate(model, prompt_ids, defender=None, *, steps=128, gen_length=128,
             block_length=32, temperature=0.0, remasking="low_confidence",
             schedule="const"):
    """Diffusion sampling driven by a Defender; None runs undefended."""
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("generate supports one prompt at a time")
    if gen_length <= 0 or block_length <= 0 or steps <= 0:
        raise ValueError("generation length, block length, and steps must be positive")
    if gen_length % block_length or steps % (gen_length // block_length):
        raise ValueError("gen_length must be a multiple of block_length and steps of num_blocks")
    prompt_length = prompt_ids.shape[1]
    x = torch.full((1, prompt_length + gen_length), MASK_ID, dtype=torch.long,
                   device=model.device)
    x[:, :prompt_length] = prompt_ids.clone()
    region = x == MASK_ID
    if defender is not None:
        defender.reset()

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks
    for num_block in range(num_blocks):
        block_end = prompt_length + (num_block + 1) * block_length
        scope = region.clone()
        scope[:, block_end:] = False
        schedule_counts = get_num_transfer_tokens((x == MASK_ID) & scope, steps_per_block)

        for i in range(steps_per_block):
            commit_count = None
            if defender is not None:
                commit_count = defender.before_step(
                    x, region, scope=scope, steps_remaining=steps_per_block - i)
            if commit_count is not None:
                # Reopened slots: spread everything still masked before block_end
                # evenly over the remaining steps of this block.
                schedule_counts = torch.cat(
                    [schedule_counts[:, :i],
                     get_num_transfer_tokens((x == MASK_ID) & scope, steps_per_block - i)], dim=1)
            mask_index = x == MASK_ID
            if not (mask_index & scope).any():
                break

            if defender is None:
                logits = model(x).logits
            else:
                logits = defender.forward(
                    x, region, schedule_scale=step_scale(schedule, i, steps_per_block)).logits

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float64)
            else:
                raise NotImplementedError(remasking)

            eligible = mask_index & scope
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(eligible, x0_p, torch.tensor(-float("inf"), device=x0.device))
            n_eligible = int(eligible.sum())
            k = int(schedule_counts[0, i])
            if i == steps_per_block - 1:
                k = n_eligible
            k = min(k, n_eligible)
            if k > 0:
                _, select_index = torch.topk(confidence[0], k=k)
                x[0, select_index] = x0[0, select_index]

    return x
