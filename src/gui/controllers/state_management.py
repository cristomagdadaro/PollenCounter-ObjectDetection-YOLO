from __future__ import annotations
from src.paths import *
from src.bounding_box import BoundingBox, calculate_iou_corners
from src.settings import load_settings, save_settings
from src.model_utils import get_latest_weights, collect_images
from src.theme import *


from typing import *
from pathlib import Path
import tkinter as tk
import cv2
import numpy as np
import shutil
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw


class StateManagementMixin:
    def _label_path(self) -> Path:
        """Get the label file path for the current image."""
        img_name = self.image_paths[self.current_idx].stem
        return self.labels_dir / f"{img_name}.txt"

    def _save_labels(self):
        """Save current boxes to YOLO .txt file."""
        if not self.image_paths:
            return
            
        self._clean_duplicates()
        label_path = self._label_path()
        
        # Only save if there are boxes, or delete the file if there are none to clean up
        if not self.boxes:
            if label_path.exists():
                label_path.unlink()
            return
            
        with open(label_path, "w") as f:
            for box in self.boxes:
                f.write(box.to_yolo_line() + "\n")

    # ════════════════════════════════════════════════════════════════
    #  NAVIGATION
    # ════════════════════════════════════════════════════════════════

    def _next_image(self):
        mode_val = self.view_mode.get() if hasattr(self, 'view_mode') else "Full"
        
        if mode_val == "Q1":
            self.view_mode.set("Q2")
            self._render_image()
            return
        elif mode_val == "Q2":
            self.view_mode.set("Q3")
            self._render_image()
            return
        elif mode_val == "Q3":
            self.view_mode.set("Q4")
            self._render_image()
            return
            
        if self.image_paths and self.current_idx < len(self.image_paths) - 1:
            self._save_labels()
            self.current_idx += 1
            if mode_val in ["Q1", "Q2", "Q3", "Q4"]:
                self.view_mode.set("Q1")
            self._load_image()

    def _prev_image(self):
        mode_val = self.view_mode.get() if hasattr(self, 'view_mode') else "Full"
        
        if mode_val == "Q4":
            self.view_mode.set("Q3")
            self._render_image()
            return
        elif mode_val == "Q3":
            self.view_mode.set("Q2")
            self._render_image()
            return
        elif mode_val == "Q2":
            self.view_mode.set("Q1")
            self._render_image()
            return
            
        if self.image_paths and self.current_idx > 0:
            self._save_labels()
            self.current_idx -= 1
            if mode_val in ["Q1", "Q2", "Q3", "Q4"]:
                self.view_mode.set("Q4")
            self._load_image()

    def _on_combo_jump(self, event=None):
        if not self.image_paths: return
        selected_name = self.image_combo.get()
        try:
            idx = next(i for i, p in enumerate(self.image_paths) if p.name == selected_name)
            if idx != self.current_idx:
                self._save_labels()
                self.current_idx = idx
                self._load_image()
        except StopIteration:
            pass

    def _delete_image(self):
        if not self.image_paths: return
        
        current_img_path = self.image_paths[self.current_idx]
        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to PERMANENTLY delete:\n\n{current_img_path.name}\n\nThis cannot be undone!", parent=self.root):
            return

        current_lbl_path = self._label_path()
        if current_lbl_path.exists():
            current_lbl_path.unlink()
            
        try:
            current_img_path.unlink()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete image: {e}")
            return
            
        del self.image_paths[self.current_idx]
        self._update_ui()
        
        if not self.image_paths:
            self.canvas.delete("all")
            self.pil_img = None
            self.boxes.clear()
            self._update_ui()
            self.status.config(text=" Deleted last image.")
            return

        if self.current_idx >= len(self.image_paths):
            self.current_idx = len(self.image_paths) - 1
            
        self._load_image()
        self.status.config(text=f" Permanently deleted image")

    # ════════════════════════════════════════════════════════════════
    #  ACTIONS
    # ════════════════════════════════════════════════════════════════

    def _undo(self):
        if self.boxes:
            self.boxes.pop()
            self._redraw_boxes()
            self._save_labels()
            self._update_ui()
            self.status.config(text=f"Undo boxes: {len(self.boxes)}")

    def _clear_boxes(self):
        if self.boxes:
            if messagebox.askyesno("Clear All", "Delete all boxes on this image?"):
                self.boxes.clear()
                self._redraw_boxes()
                self._save_labels()
                self._update_ui()
                self.status.config(text=" All boxes cleared")

    def _change_dataset(self, event):
        new_set = self.dataset_combo.get()
        if new_set == self.current_set:
            return
        
        if self.image_paths:
            self._save_labels()
            
        self.current_set = new_set
        self.images_dir, self.labels_dir = self.set_paths[self.current_set]
        
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        
        self.current_idx = 0
        if not self.image_paths:
            self.canvas.delete("all")
            self.pil_img = None
            self.boxes.clear()
            self._update_ui()
            return
            
        self._load_image()

    def _move_image(self, target_set):
        """Move the current image (and its label) to the selected set."""
        if not self.image_paths or self.current_set == target_set:
            return

        img_path = self.image_paths[self.current_idx]
        label_path = self._label_path()
        name = img_path.name
        
        target_img_dir, target_label_dir = self.set_paths[target_set]
        
        # Check for overwrite
        target_img_path = target_img_dir / name
        if target_img_path.exists():
            if not messagebox.askyesno(
                "Overwrite Warning", 
                f"The file '{name}' already exists in the {target_set} set!\n\nIf you proceed, the existing file and its labels will be permanently overwritten.\n\nDo you want to overwrite it?",
                icon="warning"
            ):
                return

        if messagebox.askyesno(f"Move to {target_set}", f"Move '{name}' to {target_set}?"):
            # Move image
            dest_img = target_img_dir / img_path.name
            
            # Windows file lock prevention: clear all image references before moving
            self.orig_pil_img = None
            self.pil_img = None
            if hasattr(self, 'tk_image'):
                self.tk_image = None
            if hasattr(self, 'overlay_tk'):
                self.overlay_tk = None
            self.canvas.delete("all")
            
            # Force garbage collection to release any dangling Windows file handles
            import gc
            gc.collect()
            
            try:
                shutil.move(str(img_path), str(dest_img))
            except Exception as e:
                messagebox.showerror("File Error", f"Could not move image. It might be locked by another process.\n\nError: {e}")
                return

            # Move label if exists
            if label_path.exists():
                dest_label = target_label_dir / label_path.name
                shutil.move(str(label_path), str(dest_label))
                
            # Move CLAHE image if exists
            clahe_img_path = Path(str(img_path).replace("images", "images_clahe"))
            if clahe_img_path.exists():
                dest_clahe_img = Path(str(dest_img).replace("images", "images_clahe"))
                dest_clahe_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(clahe_img_path), str(dest_clahe_img))

            # Remove from list
            self.image_paths.pop(self.current_idx)

            if not self.image_paths:
                self.canvas.delete("all")
                self.pil_img = None
                self.boxes.clear()
                self._update_ui()
                messagebox.showinfo("Empty Set", f"All images in {self.current_set} have been moved.")
                return

            if self.current_idx >= len(self.image_paths):
                self.current_idx = len(self.image_paths) - 1

            self._load_image()
            self.status.config(text=f"'{name}' moved to {target_set}")

    def _import_images(self):
        """Prompt user for images and import them into the current set."""
        if self.current_set == "All":
            messagebox.showinfo("Select a Set", "Please select 'Train' or 'Validation' from the dropdown to import images into.")
            return
            
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp"),
            ("All files", "*.*")
        ]
        
        filepaths = filedialog.askopenfilenames(
            title=f"Select Images to Import into {self.current_set}",
            filetypes=filetypes
        )
        
        if not filepaths:
            return
            
        target_img_dir, _ = self.set_paths[self.current_set]
        imported_count = 0
        
        for fp in filepaths:
            src = Path(fp)
            dest = target_img_dir / src.name
            if not dest.exists():
                try:
                    shutil.copy2(src, dest)
                    imported_count += 1
                except Exception as e:
                    messagebox.showerror("Error", f"Could not copy {src.name}:\n{e}")
            else:
                pass # Already exists
                
        if imported_count > 0:
            self._scan_dataset()
            # Jump to the last newly imported image (end of list)
            self.current_idx = max(0, len(self.image_paths) - 1)
            self._load_image()
            self._update_ui()
            self.status.config(text=f"Successfully imported {imported_count} images into {self.current_set}")
            messagebox.showinfo("Import Complete", f"Successfully imported {imported_count} images into {self.current_set}.")
        else:
            self.status.config(text="No new images imported.")

    # ════════════════════════════════════════════════════════════════
    #  UI UPDATES
    # ════════════════════════════════════════════════════════════════

