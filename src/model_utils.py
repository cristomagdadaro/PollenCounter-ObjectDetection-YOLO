"""YOLO model discovery and image collection utilities.

Shared by inference.py and compare_val.py.

Usage:
    from src.model_utils import get_latest_weights, collect_images
"""

from pathlib import Path
from src.paths import RUNS_DETECT, IMAGE_EXTS


def get_latest_weights(detect_dir: Path = RUNS_DETECT) -> Path | None:
    """Find the most recently modified best.pt in runs/detect/.

    Returns None if no trained weights exist yet.
    """
    if not detect_dir.exists():
        return None
    weight_files = list(detect_dir.rglob("weights/best.pt"))
    if not weight_files:
        return None
    return max(weight_files, key=lambda p: p.stat().st_mtime)


def collect_images(folder: Path) -> list[Path]:
    """Recursively find all supported image files in a folder, sorted alphabetically."""
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
