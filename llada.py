import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
MASK_ID = 126336  # [MASK] token id used by LLaDA
PAD_ID = 126081   # <|endoftext|>; also the eos/pad id in the model config


def add_gumbel_noise(logits, temperature, rng=None):
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float64)
    noise = (torch.rand_like(logits) if rng is None else torch.rand(logits.shape, dtype=logits.dtype, device=logits.device, generator=rng))
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    offsets = torch.arange(steps, device=mask_index.device)
    return base + (offsets < remainder).to(torch.int64)


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
):
    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long, device=model.device)
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_index = x != MASK_ID

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == MASK_ID
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == MASK_ID

            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = MASK_ID
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == "random":
                x0_p = torch.rand_like(x0, dtype=torch.float64)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, block_end:] = -float("inf")

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, torch.tensor(-float("inf"), device=x0.device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = (AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).to(DEVICE).eval())

    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(formatted)["input_ids"]
    input_ids = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    out = generate(
        model,
        input_ids,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        remasking="low_confidence",
    )
    answer = tokenizer.batch_decode(
        out[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0]
    print(f"\nPrompt: {prompt}")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
