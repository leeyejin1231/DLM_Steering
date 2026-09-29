import threading
from models import MODEL


MODEL_NAME = MODEL["name"]
MASK_ID = MODEL["mask_id"]
MASK_TOKEN = MODEL["mask_token"]     # text form, expanded into DIJA prompts
EOT_ID = MODEL["eot_id"]             # closes the user turn in the chat template
USER_HEADER_ID = MODEL["user_header_id"]
NEWLINE_ID = MODEL["newline_id"]
TURN_BREAKERS = tuple(MODEL["turn_breakers"])
N_LAYERS = MODEL["n_layers"]
OUT_DIR = MODEL["out_dir"]           # default home of fitted vectors/detectors
DETECTOR_LAYER = MODEL["detector_layer"]   # None: detector bundle best_layer
STEER_LAYERS = MODEL["steer_layers"]       # None: vector bundle best_layer
MODEL_LOCK = threading.Lock()


def block_index(layer):
    if layer < 1:
        raise ValueError(f"layer numbering is 1-based (layer 1 = blocks[0]); got {layer}")
    return layer - 1


FIT_LAYERS = list(range(1, N_LAYERS))
ERROR_SENTINEL = "[STEERING_ERROR]"
