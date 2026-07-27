"""Read/write config/inference_settings.json.

Single source of truth for persisting UI state across annotate.py and inference.py.

Usage:
    from src.settings import load_settings, save_settings
    config = load_settings()            # returns dict (empty if file missing)
    save_settings(config)               # merges and writes back
"""

import json
from pathlib import Path
from src.paths import SETTINGS_JSON


def load_settings(path: Path = SETTINGS_JSON) -> dict:
    """Load settings dict from JSON. Returns {} if file is missing or corrupt."""
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_settings(updates: dict, path: Path = SETTINGS_JSON) -> None:
    """Merge `updates` into the existing settings file and write back.

    Creates parent directories if they don't exist.
    Existing keys not in `updates` are preserved.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = load_settings(path)
    existing.update(updates)

    try:
        with open(path, "w") as f:
            json.dump(existing, f)
    except OSError as e:
        print(f"[WARNING] Failed to save settings: {e}")
