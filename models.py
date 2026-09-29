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
    """Register --model on an entry-point parser (the value is read from
    sys.argv when this module is imported; argparse only validates and records it)."""
    parser.add_argument("--model", choices=sorted(MODELS), default=MODEL_KEY, 
                        help="Target diffusion LM (default llada). Must be given on the command line, not via config, because it is resolved when models.py is imported.")
