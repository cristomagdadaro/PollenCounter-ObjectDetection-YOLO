import argparse
import sys
import tkinter as tk
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gui.app import AnnotationApp
from src.paths import TRAIN_IMAGES, TRAIN_LABELS

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate pollen images with bounding boxes.")
    parser.add_argument("--images", type=str, default=str(TRAIN_IMAGES), help="Folder of images to annotate.")
    parser.add_argument("--labels", type=str, default=str(TRAIN_LABELS), help="Folder to save YOLO labels.")
    parser.add_argument("--compare-labels", type=str, default=None, help="Folder containing secondary labels to overlay for comparison.")
    return parser.parse_args()

def main():
    args = parse_args()

    root = tk.Tk()
    root.state("zoomed")  # Start maximised on Windows

    app = AnnotationApp(
        root,
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        compare_labels_dir=Path(args.compare_labels) if args.compare_labels else None,
    )

    root.mainloop()

    # Final report
    labels_dir = Path(args.labels)
    label_count = sum(1 for f in labels_dir.glob("*.txt") if f.stat().st_size > 0)
    print(f"\n[INFO] Annotation complete  {label_count} label file(s) saved to {labels_dir}")

if __name__ == "__main__":
    main()
