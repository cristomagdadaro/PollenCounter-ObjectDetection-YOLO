from __future__ import annotations
from src.gui.views.ui_builder import UIBuilderMixin
from src.gui.controllers.canvas_events import CanvasEventsMixin
from src.gui.controllers.rendering import RenderingMixin
from src.gui.controllers.active_learning import ActiveLearningMixin
from src.gui.controllers.box_snapping import BoxSnappingMixin
from src.gui.controllers.batch_tools import BatchToolsMixin
from src.gui.controllers.state_management import StateManagementMixin

#!/usr/bin/env python3
"""Pollen grain bounding box annotation tool.

A tkinter GUI for drawing YOLO-format bounding boxes on pollen microscopy images.

Usage:
    python scripts/annotate.py
    python scripts/annotate.py --images datasets/images/train --labels datasets/labels/train
"""


import argparse
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageTk, ImageDraw

from src.paths import (
    PROJECT_ROOT, TRAIN_IMAGES, TRAIN_LABELS,
    VAL_IMAGES, VAL_LABELS, EXCLUDED_IMAGES, EXCLUDED_LABELS, IMAGE_EXTS,
    RUNS_DETECT, RAW_IMAGES, DEFAULT_OUTPUT,
)
from src.bounding_box import BoundingBox, calculate_iou_corners
from src.settings import load_settings, save_settings
from src.model_utils import get_latest_weights, collect_images
from src.theme import (
    BOX_COLOR, BOX_COLOR_HOVER, ACTIVE_BOX_COLOR,
    BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR, PROGRESS_DONE, PROGRESS_TODO,
    OVERLAP_80_HEX, OVERLAP_80_RGB, OVERLAP_50_HEX, OVERLAP_50_RGB,
    OVERLAP_0_HEX, OVERLAP_0_RGB, AUTO_BOX_HEX, AUTO_BOX_RGB, BOX_RGB,
    FONT_LABEL, FONT_INPUT, HUMAN_COLOR, MODEL_COLOR,
)

# Aliases for backward compatibility within this file
DEFAULT_IMAGES = TRAIN_IMAGES
DEFAULT_LABELS = TRAIN_LABELS


# BoundingBox is now imported from src.bounding_box


class AnnotationApp(UIBuilderMixin, CanvasEventsMixin, RenderingMixin, ActiveLearningMixin, BoxSnappingMixin, BatchToolsMixin, StateManagementMixin):
    """Main tkinter annotation application."""

    # Maximum canvas display size
    MAX_CANVAS_W = 960
    MAX_CANVAS_H = 720

    def __init__(self, root: tk.Tk, images_dir: Path, labels_dir: Path, compare_labels_dir: Optional[Path] = None):
        self.root = root
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.compare_labels_dir = compare_labels_dir
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Ensure val dirs exist
        VAL_IMAGES.mkdir(parents=True, exist_ok=True)
        VAL_LABELS.mkdir(parents=True, exist_ok=True)
        EXCLUDED_IMAGES.mkdir(parents=True, exist_ok=True)
        EXCLUDED_LABELS.mkdir(parents=True, exist_ok=True)

        self.set_paths = {
            "Train": (DEFAULT_IMAGES, DEFAULT_LABELS),
            "Validation": (VAL_IMAGES, VAL_LABELS),
            "Excluded": (EXCLUDED_IMAGES, EXCLUDED_LABELS)
        }
        
        self.current_set = "Custom"
        for name, (img_dir, _) in self.set_paths.items():
            if self.images_dir.resolve() == img_dir.resolve():
                self.current_set = name
                break

        # ── Image list ──────────────────────────────────────────────
        self.image_paths: list[Path] = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            print(f"[WARN] No images found in {self.images_dir}")

        self.current_idx = 0
        self.boxes: list[BoundingBox] = []
        self.compare_boxes: list[BoundingBox] = []
        self.canvas_ids: list[int] = []  # canvas rectangle IDs

        # ── Drawing state ───────────────────────────────────────────
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.temp_rect: Optional[int] = None

        # ── Image display state ─────────────────────────────────────
        self.display_scale = 1.0
        self.zoom_level = 1.0
        self.orig_w = 0
        self.orig_h = 0
        self.pil_img: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.yolo_model = None
        self.loaded_model_name = None
        
        self._calculate_default_box_size()

        # ── Build UI ────────────────────────────────────────────────
        self._build_ui()
        self._load_settings()
        self._load_image()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ════════════════════════════════════════════════════════════════

    def _on_key(self, event):
        # Ignore keyboard shortcuts if a text entry or combobox has focus
        if isinstance(event.widget, (tk.Entry, ttk.Combobox)):
            return
            
        char = event.char.lower()
        if not char:
            return
            
        if char == 'w':
            self._fit_width()
        elif char == 'h':
            self._fit_height()
        elif char == 'f':
            self.view_mode.set("Full")
            self._render_image()
        elif char == '1':
            self.view_mode.set("Q1")
            self._render_image()
        elif char == '2':
            self.view_mode.set("Q2")
            self._render_image()
        elif char == '3':
            self.view_mode.set("Q3")
            self._render_image()
        elif char == '4':
            self.view_mode.set("Q4")
            self._render_image()
        elif char == 'r':
            self.show_red.set(not self.show_red.get())
            self._redraw_boxes()
        elif char == 'o':
            self.show_orange.set(not self.show_orange.get())
            self._redraw_boxes()
        elif char == 'y':
            self.show_yellow.set(not self.show_yellow.get())
            self._redraw_boxes()
        elif char == 'g':
            self.show_green.set(not self.show_green.get())
            self._redraw_boxes()
        elif char == 'v':
            self.show_violet.set(not self.show_violet.get())
            self._redraw_boxes()

    # ════════════════════════════════════════════════════════════════
    #  IMAGE LOADING
    # ════════════════════════════════════════════════════════════════

    def _update_size_entries(self):
        if not hasattr(self, 'full_w') or not self.full_w: return
        w_px = int(self.default_w * self.full_w)
        h_px = int(self.default_h * self.full_h)
        self.entry_w.delete(0, tk.END)
        self.entry_w.insert(0, str(w_px))
        self.entry_h.delete(0, tk.END)
        self.entry_h.insert(0, str(h_px))

    def _zoom_in(self):
        self.zoom_level = min(self.zoom_level * 1.25, 10.0)
        if hasattr(self, 'fit_mode'): self.fit_mode.set("none")
        self._render_image()

    def _zoom_out(self):
        self.zoom_level = max(self.zoom_level / 1.25, 0.2)
        if hasattr(self, 'fit_mode'): self.fit_mode.set("none")
        self._render_image()
        
    def _on_mousewheel(self, event):
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _canvas_to_norm(self, cx: float, cy: float) -> tuple[float, float]:
        """Canvas pixel → normalised image coordinates."""
        ix = (cx - self.img_offset_x) / self.display_scale
        iy = (cy - self.img_offset_y) / self.display_scale
        local_nx = ix / self.orig_w
        local_ny = iy / self.orig_h
        if hasattr(self, 'view_w_scale'):
            return (local_nx * self.view_w_scale) + self.view_x_offset, (local_ny * self.view_h_scale) + self.view_y_offset
        return local_nx, local_ny

    def _norm_to_canvas(self, nx: float, ny: float) -> tuple[int, int]:
        """Normalised image coords → canvas pixel."""
        if hasattr(self, 'view_w_scale'):
            local_nx = (nx - self.view_x_offset) / self.view_w_scale
            local_ny = (ny - self.view_y_offset) / self.view_h_scale
        else:
            local_nx, local_ny = nx, ny
        ix = local_nx * self.orig_w * self.display_scale
        iy = local_ny * self.orig_h * self.display_scale
        return int(ix + self.img_offset_x), int(iy + self.img_offset_y)

    # ════════════════════════════════════════════════════════════════
    #  DRAWING EVENTS
    # ════════════════════════════════════════════════════════════════

    def _on_space(self, event):
        if isinstance(event.widget, tk.Entry):
            return  # Allow typing spaces in text boxes (though not needed for numbers)
        if not hasattr(self, 'last_mouse_x'):
            return
        self._create_auto_box(self.last_mouse_x, self.last_mouse_y, "(spacebar)")

    def _on_double_click(self, event):
        """Auto-create a box of default size centered at the double-click."""
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        self._create_auto_box(cx, cy, "")
        
        # Cancel the drag-box that was started by the first click of the double-click
        self.drawing = False
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

    def _create_auto_box(self, cx: float, cy: float, suffix: str):
        nx, ny = self._canvas_to_norm(cx, cy)
        
        # Clamp coordinates so box doesn't go off screen
        nx = max(self.default_w / 2, min(1.0 - self.default_w / 2, nx))
        ny = max(self.default_h / 2, min(1.0 - self.default_h / 2, ny))
        
        box = BoundingBox(nx, ny, self.default_w, self.default_h, class_id=0)
        if self.auto_snap.get():
            self._snap_single_box(box)
        self.boxes.append(box)
        
        self._redraw_boxes()
        self._save_labels()
        self._update_ui()
        self.status.config(text=f" Auto-box added {suffix}  total: {len(self.boxes)}")

    def _calculate_default_box_size(self):
        """Find the median width and height of a sample of existing annotations."""
        widths = []
        heights = []
        import itertools
        
        # Sample at most 100 label files to prevent slow startup on massive sliced datasets
        label_files = itertools.islice(self.labels_dir.glob("*.txt"), 100)
        for label_file in label_files:
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        box = BoundingBox.from_yolo_line(line)
                        if box:
                            widths.append(box.w)
                            heights.append(box.h)
            except Exception:
                pass
        
        if widths and heights:
            widths.sort()
            heights.sort()
            self.default_w = widths[len(widths) // 2]
            self.default_h = heights[len(heights) // 2]
        else:
            self.default_w = 0.05
            self.default_h = 0.05

    def _run_batch(self, op, target, param, parent_win):
        # Gather labels
        targets = []
        if target == "dataset":
            targets = [p for p in self.labels_dir.glob("*.txt") if p.is_file()]
        else:
            for d in [TRAIN_LABELS, VAL_LABELS, EXCLUDED_LABELS]:
                if d.exists():
                    targets.extend([p for p in d.glob("*.txt") if p.is_file()])
                    
        if not targets:
            messagebox.showinfo("Info", "No label files found to process.", parent=parent_win)
            return
            
        scale_factor = 1.0
        if op == "scale":
            try:
                scale_factor = float(param.replace(',', '.'))
                if scale_factor <= 0: raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Invalid scale factor.", parent=parent_win)
                return

        if not messagebox.askyesno("Confirm", f"Are you sure you want to {op} on {len(targets)} files?\nThis cannot be undone easily.", parent=parent_win):
            return
            
        pb_var = tk.DoubleVar()
        pb = ttk.Progressbar(parent_win, variable=pb_var, maximum=len(targets))
        pb.pack(fill=tk.X, padx=15, pady=10)
        parent_win.update()

        affected = 0
        import cv2 # import locally to avoid cluttering global namespace if not needed
        for i, txt_path in enumerate(targets):
            if op == "clear":
                txt_path.unlink()
                affected += 1
            elif op == "scale" or op == "snap":
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                    
                new_lines = []
                img_bgr = None
                img_h, img_w = 0, 0
                
                if op == "snap":
                    # find the corresponding image
                    labels_folder = txt_path.parent
                    images_folder = labels_folder.parent.parent / "images" / labels_folder.name
                    img_path = None
                    for ext in IMAGE_EXTS:
                        possible = images_folder / f"{txt_path.stem}{ext}"
                        if possible.exists():
                            img_path = possible
                            break
                    if img_path:
                        img_bgr = cv2.imread(str(img_path))
                        if img_bgr is not None:
                            img_h, img_w = img_bgr.shape[:2]
                
                changed_this_file = False
                for line in lines:
                    box = BoundingBox.from_yolo_line(line)
                    if box:
                        if op == "scale":
                            box.w *= scale_factor
                            box.h *= scale_factor
                            changed_this_file = True
                        elif op == "snap" and img_bgr is not None:
                            # Use logic similar to _snap_single_box
                            x1 = int((box.x_center - box.w / 2) * img_w)
                            y1 = int((box.y_center - box.h / 2) * img_h)
                            x2 = int((box.x_center + box.w / 2) * img_w)
                            y2 = int((box.y_center + box.h / 2) * img_h)
                            
                            # Temporarily override self.orig_pil_img for the snap method
                            self.orig_pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                            if self._snap_single_box(box, img_bgr=img_bgr):
                                changed_this_file = True
                                    
                        new_lines.append(box.to_yolo_line() + "\n")
                        
                if new_lines and changed_this_file:
                    with open(txt_path, "w") as f:
                        f.writelines(new_lines)
                    affected += 1
                elif not new_lines and op == "scale": # only delete if scaling
                    txt_path.unlink()
            
            pb_var.set(i + 1)
            if i % 10 == 0:
                parent_win.update()
            
        pb.destroy()
        messagebox.showinfo("Done", f"Processed {affected} files successfully.", parent=parent_win)
        self._load_image()  # Refresh the canvas to show changes
        parent_win.destroy()

    def _clean_duplicates(self):
        """Remove boxes that overlap by more than 80% with another box."""
        if len(self.boxes) < 2:
            return
            
        to_remove = set()
        for i in range(len(self.boxes)):
            if i in to_remove: continue
            for j in range(i + 1, len(self.boxes)):
                if j in to_remove: continue
                if self.boxes[i].iou(self.boxes[j]) > 0.6:
                    to_remove.add(j)
                    
        if to_remove:
            self.boxes = [b for i, b in enumerate(self.boxes) if i not in to_remove]
            print(f"[INFO] Removed {len(to_remove)} highly overlapping duplicate boxes.")
            
    def _load_settings(self):
        """Restore UI control values from config/inference_settings.json."""
        config = load_settings()
        if not config:
            return
        try:
            if "model" in config and hasattr(self, 'model_combo'):
                if config["model"] in self.model_combo['values']:
                    self.model_combo.set(config["model"])
            if "threshold" in config and hasattr(self, 'conf_entry'):
                self.conf_entry.delete(0, tk.END)
                self.conf_entry.insert(0, str(config["threshold"]))
            if "opacity" in config and hasattr(self, 'opacity_var'):
                self.opacity_var.set(config["opacity"])
            if "box_width" in config and hasattr(self, 'entry_w'):
                self.entry_w.delete(0, tk.END)
                self.entry_w.insert(0, str(config["box_width"]))
            if "box_height" in config and hasattr(self, 'entry_h'):
                self.entry_h.delete(0, tk.END)
                self.entry_h.insert(0, str(config["box_height"]))
            if "thickness" in config and hasattr(self, 'thickness_var'):
                self.thickness_var.set(str(config["thickness"]))
            if "scale" in config and hasattr(self, 'scale_var'):
                self.scale_var.set(str(config["scale"]))
            if "show_compare" in config and hasattr(self, 'show_compare'):
                self.show_compare.set(config["show_compare"])
            if "smart_recount" in config and hasattr(self, 'smart_recount_var'):
                self.smart_recount_var.set(config["smart_recount"])
            if "force_recount" in config and hasattr(self, 'force_recount_var'):
                self.force_recount_var.set(config["force_recount"])
            if "use_sahi" in config and hasattr(self, 'use_sahi_var'):
                self.use_sahi_var.set(config["use_sahi"])
            if "iou" in config and hasattr(self, 'iou_entry'):
                self.iou_entry.delete(0, tk.END)
                self.iou_entry.insert(0, str(config["iou"]))
            if "auto_snap" in config and hasattr(self, 'auto_snap'):
                self.auto_snap.set(config["auto_snap"])
            if "snap_method" in config and hasattr(self, 'snap_method'):
                self.snap_method.set(config["snap_method"])
            if "fit_mode" in config and hasattr(self, 'fit_mode'):
                self.fit_mode.set(config["fit_mode"])
            if "show_violet" in config and hasattr(self, 'show_violet'):
                self.show_violet.set(config["show_violet"])
            if "embed_annotations" in config and hasattr(self, 'embed_annotations'):
                self.embed_annotations.set(config["embed_annotations"])
                
            if "current_image" in config and getattr(self, 'image_paths', None):
                target = config["current_image"]
                for i, p in enumerate(self.image_paths):
                    if p.name == target:
                        self.current_idx = i
                        self._load_image()
                        break
        except Exception as e:
            print(f"[WARNING] Failed to load UI settings: {e}")

    def _save_settings(self):
        """Persist all UI control values to config/inference_settings.json."""
        updates = {}
        if hasattr(self, 'model_combo'):
            updates["model"] = self.model_combo.get()
        if hasattr(self, 'conf_entry'):
            updates["threshold"] = self.conf_entry.get()
        if hasattr(self, 'opacity_var'):
            try:
                updates["opacity"] = float(self.opacity_var.get().replace(',', '.'))
            except Exception: pass
        if hasattr(self, 'entry_w'):
            updates["box_width"] = self.entry_w.get()
        if hasattr(self, 'entry_h'):
            updates["box_height"] = self.entry_h.get()
        if hasattr(self, 'thickness_var'):
            try:
                updates["thickness"] = int(self.thickness_var.get())
            except Exception: pass
        if hasattr(self, 'scale_var'):
            try:
                updates["scale"] = float(self.scale_var.get().replace(',', '.'))
            except Exception: pass
        if hasattr(self, 'show_compare'):
            updates["show_compare"] = self.show_compare.get()
        if hasattr(self, 'smart_recount_var'):
            updates["smart_recount"] = self.smart_recount_var.get()
        if hasattr(self, 'force_recount_var'):
            updates["force_recount"] = self.force_recount_var.get()
        if hasattr(self, 'use_sahi_var'):
            updates["use_sahi"] = self.use_sahi_var.get()
        if hasattr(self, 'iou_entry'):
            updates["iou"] = self.iou_entry.get()
        if hasattr(self, 'auto_snap'):
            updates["auto_snap"] = self.auto_snap.get()
        if hasattr(self, 'snap_method'):
            updates["snap_method"] = self.snap_method.get()
        if hasattr(self, 'fit_mode'):
            updates["fit_mode"] = self.fit_mode.get()
        if hasattr(self, 'show_violet'):
            updates["show_violet"] = self.show_violet.get()
        if hasattr(self, 'embed_annotations'):
            updates["embed_annotations"] = self.embed_annotations.get()
        if hasattr(self, 'image_paths') and hasattr(self, 'current_idx') and self.image_paths and self.current_idx < len(self.image_paths):
            updates["current_image"] = self.image_paths[self.current_idx].name
        save_settings(updates)

    def _on_close(self):
        self._save_settings()
        self.root.destroy()

    # ════════════════════════════════════════════════════════════════
    #  SEARCH FUNCTION
    # ════════════════════════════════════════════════════════════════
    def _search_image(self, event=None):
        query = self.search_var.get().strip().lower()
        if not query: return
            
        for i, path in enumerate(self.image_paths):
            if query in path.name.lower():
                self.current_idx = i
                self._load_image()
                self.search_entry.config(bg="#10B981", fg="white")
                self.root.after(500, lambda: self.search_entry.config(bg="#FFFFFF", fg="black"))
                self.canvas.focus_set()
                return
                
        # Not found
        self.search_entry.config(bg="#EF4444", fg="white")
        self.root.after(500, lambda: self.search_entry.config(bg="#FFFFFF", fg="black"))

    # ════════════════════════════════════════════════════════════════
    #  TOOLS MENU — Integrated Child Windows
    # ════════════════════════════════════════════════════════════════

