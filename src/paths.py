"""Centralized path constants and file extensions for the project.

Every script imports from here instead of redefining PROJECT_ROOT.

Usage:
    from src.paths import PROJECT_ROOT, IMAGE_EXTS, DATASET_ROOT
"""

from pathlib import Path

import sys
import os

# Root of the workspace
if "POLLEN_WORKSPACE" in os.environ:
    PROJECT_ROOT = Path(os.environ["POLLEN_WORKSPACE"]).resolve()
elif getattr(sys, 'frozen', False):
    # In PyInstaller, fallback to the folder containing the .exe
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
else:
    # Source mode
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset paths ────────────────────────────────────────────────────
DATASET_ROOT    = PROJECT_ROOT / "datasets_sliced"
DATASET_YAML    = PROJECT_ROOT / "config" / "pollen_dataset.yaml"
TRAINING_YAML   = PROJECT_ROOT / "config" / "training.yaml"
SETTINGS_JSON   = PROJECT_ROOT / "config" / "inference_settings.json"

TRAIN_IMAGES    = DATASET_ROOT / "images" / "train"
TRAIN_LABELS    = DATASET_ROOT / "labels" / "train"
VAL_IMAGES      = DATASET_ROOT / "images" / "val"
VAL_LABELS      = DATASET_ROOT / "labels" / "val"
EXCLUDED_IMAGES = DATASET_ROOT / "images" / "excluded"
EXCLUDED_LABELS = DATASET_ROOT / "labels" / "excluded"
RAW_IMAGES      = DATASET_ROOT / "raw"

# ── Runs / output ────────────────────────────────────────────────────
RUNS_DETECT     = PROJECT_ROOT / "runs" / "detect"
DEFAULT_OUTPUT  = RUNS_DETECT / "results"

# ── Pretrained models ────────────────────────────────────────────────
DEFAULT_MODEL   = "pretrained_models/yolo11n.pt"

# ── Supported image extensions ───────────────────────────────────────
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
