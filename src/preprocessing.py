"""Dataset preprocessing utilities.

Provides CLAHE contrast enhancement, duplicate bounding-box removal,
and NMS-based label cleaning. Each function operates on standard YOLO
dataset directory layouts.

Usage (as module):
    from src.preprocessing import apply_clahe, clean_duplicate_boxes, clean_nms_boxes

Usage (CLI):
    python -m src.preprocessing --clahe
    python -m src.preprocessing --dedup
    python -m src.preprocessing --nms
    python -m src.preprocessing --all
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.paths import PROJECT_ROOT, TRAIN_LABELS, VAL_LABELS
from src.bounding_box import calculate_iou


# ── CLAHE Contrast Enhancement ───────────────────────────────────────

def apply_clahe(
    images_dir: Path | str | None = None,
    clip_limit: float = 2.5,
    tile_size: int = 8,
    backup: bool = True,
) -> int:
    """Apply CLAHE to all images in a directory.

    Args:
        images_dir: Directory containing images. Defaults to datasets/images.
        clip_limit: CLAHE clip limit.
        tile_size: CLAHE tile grid size.
        backup: If True, back up originals to images_original/.

    Returns:
        Number of images processed.
    """
    if images_dir is None:
        images_dir = PROJECT_ROOT / "datasets" / "images"
    images_dir = Path(images_dir)

    if backup:
        backup_dir = images_dir.parent / f"{images_dir.name}_original"
        if not backup_dir.exists():
            print(f"[INFO] Creating backup of {images_dir} to {backup_dir}...")
            shutil.copytree(images_dir, backup_dir)
        else:
            print(f"[INFO] Backup {backup_dir} already exists. Proceeding...")

    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(images_dir.rglob(ext))

    if not image_paths:
        print(f"[WARN] No images found in {images_dir}!")
        return 0

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))

    count = 0
    for path in image_paths:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_clahe = clahe.apply(l_channel)
        lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        cv2.imwrite(str(path), img_clahe)
        count += 1

    print(f"[INFO] Successfully applied CLAHE to {count} images in {images_dir}")
    return count


# ── Duplicate Box Removal (IoU-based) ────────────────────────────────

def _clean_file_iou(txt_path: Path, threshold: float = 0.6) -> int:
    """Remove overlapping boxes from a single label file. Returns count removed."""
    with open(txt_path, "r") as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            coords = list(map(float, parts[1:5]))
            boxes.append((int(parts[0]), coords, line))

    if len(boxes) < 2:
        return 0

    to_remove = set()
    for i in range(len(boxes)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(boxes)):
            if j in to_remove:
                continue
            if calculate_iou(boxes[i][1], boxes[j][1]) > threshold:
                to_remove.add(j)

    if to_remove:
        clean_boxes = [b for i, b in enumerate(boxes) if i not in to_remove]
        with open(txt_path, "w") as f:
            for b in clean_boxes:
                f.write(b[2])

    return len(to_remove)


def clean_duplicate_boxes(
    labels_dirs: list[Path] | None = None,
    threshold: float = 0.6,
) -> int:
    """Remove duplicate bounding boxes from YOLO label files using IoU.

    Args:
        labels_dirs: List of label directories to scan. Defaults to train + val.
        threshold: IoU threshold above which a box is considered a duplicate.

    Returns:
        Total number of boxes removed.
    """
    if labels_dirs is None:
        labels_dirs = [TRAIN_LABELS, VAL_LABELS]

    total_removed = 0
    total_files = 0

    for labels_dir in labels_dirs:
        if not labels_dir.exists():
            continue
        for txt_path in labels_dir.glob("*.txt"):
            total_files += 1
            removed = _clean_file_iou(txt_path, threshold)
            if removed > 0:
                print(f"[CLEANED] {txt_path.name} -> Removed {removed} duplicates.")
                total_removed += removed

    print(f"\nScan Complete! Scanned {total_files} files.")
    print(f"Purged {total_removed} duplicate bounding boxes across the entire dataset.")
    return total_removed


# ── NMS-based Box Cleaning ───────────────────────────────────────────

def clean_nms_boxes(
    labels_dirs: list[Path] | None = None,
    iou_threshold: float = 0.45,
) -> int:
    """Remove overlapping boxes using OpenCV's NMS algorithm.

    Args:
        labels_dirs: List of label directories to scan. Defaults to train + val.
        iou_threshold: NMS IoU threshold.

    Returns:
        Total number of boxes removed.
    """
    if labels_dirs is None:
        labels_dirs = [TRAIN_LABELS, VAL_LABELS]

    total_removed = 0

    for labels_dir in labels_dirs:
        if not Path(labels_dir).exists():
            continue

        txt_files = list(Path(labels_dir).glob("*.txt"))

        for txt_path in txt_files:
            with open(txt_path, "r") as f:
                lines = f.readlines()

            if not lines:
                continue

            boxes = []
            original_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    xc, yc, w, h = map(float, parts[1:5])
                    left = xc - w / 2
                    top = yc - h / 2
                    boxes.append([left, top, w, h])
                    original_lines.append(line)

            if not boxes:
                continue

            scores = [1.0] * len(boxes)
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)

            if len(indices) < len(boxes):
                kept_indices = indices.flatten()
                new_lines = [original_lines[i] for i in kept_indices]

                with open(txt_path, "w") as f:
                    f.writelines(new_lines)

                removed = len(boxes) - len(kept_indices)
                total_removed += removed
                print(f"Cleaned {txt_path.name}: Removed {removed} overlapping boxes.")

        print(f"---")
        print(f"Finished processing {labels_dir}.")

    print(f"Total overlapping boxes removed: {total_removed}")
    return total_removed


# ── CLI entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dataset preprocessing utilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast enhancement")
    parser.add_argument("--dedup", action="store_true", help="Remove duplicate boxes (IoU-based)")
    parser.add_argument("--nms", action="store_true", help="Remove overlapping boxes (NMS-based)")
    parser.add_argument("--all", action="store_true", help="Run all preprocessing steps")
    args = parser.parse_args()

    if not any([args.clahe, args.dedup, args.nms, args.all]):
        parser.print_help()
        return

    if args.clahe or args.all:
        print("=" * 50)
        print("  CLAHE Contrast Enhancement")
        print("=" * 50)
        apply_clahe()

    if args.dedup or args.all:
        print("\n" + "=" * 50)
        print("  Duplicate Box Removal")
        print("=" * 50)
        clean_duplicate_boxes()

    if args.nms or args.all:
        print("\n" + "=" * 50)
        print("  NMS Box Cleaning")
        print("=" * 50)
        clean_nms_boxes()


if __name__ == "__main__":
    main()
