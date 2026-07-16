#!/usr/bin/env python3
"""Convert TIFF training images to JPG in the train folder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def convert_tiff_folder(folder: Path, overwrite: bool = True, remove_source: bool = False) -> int:
    """Convert all TIFF images in the folder to JPEG format."""
    folder = folder.expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    converted = 0
    for ext in ("*.tif", "*.tiff"):
        for tiff_path in sorted(folder.glob(ext)):
            jpg_path = tiff_path.with_suffix(".jpg")
            if jpg_path.exists() and not overwrite:
                print(f"Skipping existing JPG: {jpg_path.name}")
                continue

            image = cv2.imread(str(tiff_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"[ERROR] Failed to read TIFF: {tiff_path}")
                continue

            success = cv2.imwrite(str(jpg_path), image)
            if not success:
                print(f"[ERROR] Failed to write JPG: {jpg_path}")
                continue

            if remove_source:
                try:
                    tiff_path.unlink()
                    print(f"Converted and removed: {tiff_path.name} -> {jpg_path.name}")
                except OSError as exc:
                    print(f"[ERROR] Converted {tiff_path.name}, but failed to delete source: {exc}")
            else:
                print(f"Converted: {tiff_path.name} -> {jpg_path.name}")

            converted += 1

    return converted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TIFF images to JPG in a dataset split folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to convert (train or val).",
    )
    parser.add_argument(
        "--remove-source",
        action="store_true",
        help="Delete the original TIFF files after conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    folder = project_root / "datasets" / "images" / args.split

    try:
        count = convert_tiff_folder(folder, remove_source=args.remove_source)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"\n[INFO] Converted {count} TIFF files to JPG in: {folder}")


if __name__ == "__main__":
    main()
