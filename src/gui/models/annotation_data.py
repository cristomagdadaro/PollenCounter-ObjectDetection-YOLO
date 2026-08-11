from pathlib import Path
from typing import List, Optional
import shutil

from src.paths import (
    TRAIN_IMAGES, TRAIN_LABELS,
    VAL_IMAGES, VAL_LABELS,
    EXCLUDED_IMAGES, EXCLUDED_LABELS,
    IMAGE_EXTS
)
from src.bounding_box import BoundingBox

class AnnotationData:
    """Model responsible for managing state: images, datasets, and bounding boxes."""

    def __init__(self, images_dir: Path, labels_dir: Path):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        
        # Datasets
        self.set_paths = {
            "Train": (TRAIN_IMAGES, TRAIN_LABELS),
            "Validation": (VAL_IMAGES, VAL_LABELS),
            "Excluded": (EXCLUDED_IMAGES, EXCLUDED_LABELS)
        }
        
        self.current_set = "Custom"
        for name, (img_dir, _) in self.set_paths.items():
            if self.images_dir.resolve() == img_dir.resolve():
                self.current_set = name
                break

        # Images list and index
        self.image_paths: List[Path] = []
        self.current_idx: int = 0
        
        # Bounding boxes
        self.boxes: List[BoundingBox] = []
        self.compare_boxes: List[BoundingBox] = []
        
        self.refresh_image_list()

    def refresh_image_list(self):
        """Reload the list of images from the current directory."""
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )

    def change_dataset(self, new_set: str) -> bool:
        """Change to a different dataset (Train, Val, Excluded). Returns True if changed."""
        if new_set == self.current_set:
            return False
            
        self.save_labels()
        
        self.current_set = new_set
        self.images_dir, self.labels_dir = self.set_paths[self.current_set]
        
        self.refresh_image_list()
        self.current_idx = 0
        self.boxes.clear()
        self.compare_boxes.clear()
        return True

    def get_current_image_path(self) -> Optional[Path]:
        """Get the absolute path to the current image."""
        if not self.image_paths:
            return None
        return self.image_paths[self.current_idx]

    def label_path(self) -> Optional[Path]:
        """Get the path to the current image's corresponding .txt label file."""
        img_path = self.get_current_image_path()
        if not img_path:
            return None
        return self.labels_dir / f"{img_path.stem}.txt"

    def next_image(self) -> bool:
        """Advance to the next image. Returns True if advanced."""
        if self.image_paths and self.current_idx < len(self.image_paths) - 1:
            self.save_labels()
            self.current_idx += 1
            return True
        return False

    def prev_image(self) -> bool:
        """Go back to the previous image. Returns True if changed."""
        if self.image_paths and self.current_idx > 0:
            self.save_labels()
            self.current_idx -= 1
            return True
        return False

    def load_labels(self):
        """Read YOLO format labels from disk for the current image."""
        self.boxes.clear()
        lbl_path = self.label_path()
        if lbl_path and lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    box = BoundingBox.from_yolo_line(line)
                    if box:
                        self.boxes.append(box)

    def save_labels(self):
        """Write current bounding boxes to YOLO format .txt file."""
        lbl_path = self.label_path()
        if not lbl_path:
            return

        if not self.boxes:
            if lbl_path.exists():
                lbl_path.unlink()
            return

        # Ensure directory exists
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(lbl_path, "w") as f:
            for box in self.boxes:
                # Limit values to [0, 1] bounds for valid YOLO format
                cx = max(0.0, min(1.0, box.cx))
                cy = max(0.0, min(1.0, box.cy))
                w = max(0.0, min(1.0, box.w))
                h = max(0.0, min(1.0, box.h))
                # Class is always 0 (pollen)
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
