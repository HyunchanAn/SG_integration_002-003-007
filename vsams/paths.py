from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PACKAGE_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def checkpoint_path(filename):
    return CHECKPOINTS_DIR / filename
