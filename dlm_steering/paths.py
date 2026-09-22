"""Repository paths, independent of the importing module location."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = REPO / ".venv/bin/python"
DATA_DIR = REPO / "data"
