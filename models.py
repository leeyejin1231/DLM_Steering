"""Target-model registry: which masked diffusion LM the pipeline runs on.

The model is chosen once per process, before common.py is imported, because
common.py binds MODEL_NAME / MASK_ID / MASK_TOKEN / EOT_ID as module
constants that every other module imports by value. Selection order:

    --model <key> on the command line   (exp.py and the steering/*.py fitters)
    DLM_MODEL=<key> in the environment  (what --gpus shards inherit)
    "llada"                              (default; all existing outputs)

Only keys listed in MODELS are honoured from argv, so scripts whose own
--model means something else (run_sr_eval.py's ollama judge) are unaffected.

Per-model fields:
    family          compatible decoder/defense family (llada or dream)
    name            HF repo id
    mask_id         id of the mask token the sampler denoises
    mask_token      its text form, for building DIJA prompts
    eot_id          token closing the user turn in the chat template
    n_layers        transformer blocks; hidden_states[1..n_layers-1] are the
                    block outputs (hidden_states[n_layers] is post-final-norm)
    chat_control    tokens the tokenizer must treat as special so
                    skip_special_tokens removes them from decoded text
    out_dir         where fitted vectors / detectors live by default
    detector_layer  default gate layer (None = detector bundle's best_layer)
    steer_layers    default steering layers (None = vector bundle's best_layer)
    shift_logits    lm_head predicts the NEXT position (AR-style); realign
                    logits so logits[:, p] scores slot p (Dream)
    user_header_id  control token after which the user turn's role name and
                    newline(s) precede the user text; with eot_id it bounds
                    the user content (v3 --remask-prompt)
    newline_id      newline token id ending the header
    turn_breakers   ids that cannot occur inside a user turn (end-of-text,
                    turn/header control); banned at rewritten prompt slots
"""

import os
import sys

MODELS = {
    "llada": dict(
        family="llada",
        name="GSAI-ML/LLaDA-8B-Instruct",
        mask_id=126336, mask_token="<|mdm_mask|>", eot_id=126348,
        n_layers=32, chat_control=(),
        out_dir="outputs", detector_layer=18, steer_layers="25",
        shift_logits=False, user_header_id=126347, newline_id=198,   # <|end_header_id|>
        turn_breakers=(126080, 126081, 126346, 126347, 126348),
    ),
    "llada1.5": dict(
        family="llada", name="GSAI-ML/LLaDA-1.5",
        mask_id=126336, mask_token="<|mdm_mask|>", eot_id=126348,
        n_layers=32, chat_control=(),
        out_dir="outputs/llada1.5", detector_layer=18, steer_layers="25",
        shift_logits=False, user_header_id=126347, newline_id=198,
        turn_breakers=(126080, 126081, 126346, 126347, 126348),
    ),
    "dream": dict(
        family="dream",
        name="Dream-org/Dream-v0-Instruct-7B",
        mask_id=151666, mask_token="<|mask|>", eot_id=151645,   # <|im_end|>
        n_layers=28, chat_control=("<|im_start|>", "<|im_end|>"),
        out_dir="outputs/dream", detector_layer=None, steer_layers=None,
        shift_logits=True, user_header_id=151644, newline_id=198,    # <|im_start|>
        turn_breakers=(151643, 151644, 151645),   # <|endoftext|>, <|im_start|>, <|im_end|>
    ),
}


def key_from_argv(argv=None):
    """--model KEY / --model=KEY from argv, if KEY names a registered model."""
    argv = sys.argv[1:] if argv is None else argv
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            key = argv[i + 1]
        elif a.startswith("--model="):
            key = a.split("=", 1)[1]
        else:
            continue
        if key in MODELS:
            return key
    return None


def select():
    """Resolve the model key for this process and pin it in the environment."""
    key = key_from_argv() or os.environ.get("DLM_MODEL", "llada")
    if key not in MODELS:
        raise SystemExit(f"DLM_MODEL={key!r} is not a registered model: {sorted(MODELS)}")
    os.environ["DLM_MODEL"] = key
    return key


MODEL_KEY = select()
MODEL = MODELS[MODEL_KEY]


def add_model_arg(parser):
    """Register --model on an entry-point parser (the value is read at import
    time by common.py; argparse only validates and records it)."""
    parser.add_argument("--model", choices=sorted(MODELS), default=MODEL_KEY,
                        help="Target diffusion LM (default llada). Must be given "
                             "on the command line, not via config, because it is "
                             "resolved before common.py is imported.")
