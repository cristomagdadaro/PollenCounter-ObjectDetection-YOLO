"""Remove duplicate bounding boxes from YOLO label files.

Scans train/ and val/ label directories and removes boxes with IoU > 0.6.

Usage:
    python scripts/clean_duplicate_points.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'from src...' works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import TRAIN_LABELS, VAL_LABELS
from src.bounding_box import calculate_iou

LABELS_DIRS = [TRAIN_LABELS, VAL_LABELS]


def clean_file(txt_path: Path, threshold: float = 0.6) -> int:
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


def main():
    total_removed = 0
    total_files = 0

    for labels_dir in LABELS_DIRS:
        if not labels_dir.exists():
            continue

        for txt_path in labels_dir.glob("*.txt"):
            total_files += 1
            removed = clean_file(txt_path)
            if removed > 0:
                print(f"[CLEANED] {txt_path.name} -> Removed {removed} duplicates.")
                total_removed += removed

    print(f"\nScan Complete! Scanned {total_files} files.")
    print(f"Purged {total_removed} duplicate bounding boxes across the entire dataset.")


if __name__ == "__main__":
    main()
