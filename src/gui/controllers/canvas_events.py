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
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw


class CanvasEventsMixin:
    def _adjust_scale(self, delta):
        scale_factor = 1.0 + delta
        if not self.boxes: return
        for box in self.boxes:
            box.w *= scale_factor
            box.h *= scale_factor
        self._redraw_boxes()
        self._save_labels()
        
        orig_bg = self.scale_entry.cget("bg")
        self.scale_entry.config(bg="#4ADE80" if delta > 0 else "#F87171")
        self.root.after(200, lambda: self.scale_entry.config(bg=orig_bg))
    def _on_canvas_resize(self, event):
        """Re-render when canvas is resized."""
        if self.image_paths and self.pil_img:
            self._render_image()

    # ════════════════════════════════════════════════════════════════
    #  COORDINATE TRANSFORMS
    # ════════════════════════════════════════════════════════════════

    def _on_press(self, event):
        self.canvas.focus_set()  # Remove focus from any text entries
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.temp_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline=ACTIVE_BOX_COLOR, width=2, dash=(4, 4)
        )

    def _on_drag(self, event):
        if self.drawing and self.temp_rect:
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            self.canvas.coords(self.temp_rect, self.start_x, self.start_y, cx, cy)

    def _on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False

        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

        # ── Compute box in normalised coords ────────────────────────
        x1, y1 = self.start_x, self.start_y
        x2 = self.canvas.canvasx(event.x)
        y2 = self.canvas.canvasy(event.y)

        # Reject tiny boxes (accidental clicks)
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return

        # Normalise
        nx1, ny1 = self._canvas_to_norm(min(x1, x2), min(y1, y2))
        nx2, ny2 = self._canvas_to_norm(max(x1, x2), max(y1, y2))

        # Clamp to [0, 1]
        nx1 = max(0.0, min(1.0, nx1))
        ny1 = max(0.0, min(1.0, ny1))
        nx2 = max(0.0, min(1.0, nx2))
        ny2 = max(0.0, min(1.0, ny2))

        xc = (nx1 + nx2) / 2
        yc = (ny1 + ny2) / 2
        w = nx2 - nx1
        h = ny2 - ny1

        if w <= 0 or h <= 0:
            return

        if getattr(self, 'smart_recount_var', None) and self.smart_recount_var.get():
            self._regional_recount(nx1, ny1, nx2, ny2)
        else:
            box = BoundingBox(xc, yc, w, h, class_id=0)
            if self.auto_snap.get():
                self._snap_single_box(box)
            self.boxes.append(box)
            self.status.config(text=f" Box added  total: {len(self.boxes)}")

        self._redraw_boxes()
        self._save_labels()
        self._update_ui()

    def _on_right_click(self, event):
        """Delete the box closest to the right-click position."""
        if not self.boxes:
            return

        # Find which box the click is inside
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        click_nx, click_ny = self._canvas_to_norm(cx, cy)

        for i, box in reversed(list(enumerate(self.boxes))):
            half_w = box.w / 2
            half_h = box.h / 2
            if (box.x_center - half_w <= click_nx <= box.x_center + half_w and
                    box.y_center - half_h <= click_ny <= box.y_center + half_h):
                self.boxes.pop(i)
                self._redraw_boxes()
                self._save_labels()
                self._update_ui()
                self.status.config(text=f" Box deleted  total: {len(self.boxes)}")
                return

    def _on_mouse_move(self, event):
        self.last_mouse_x = self.canvas.canvasx(event.x)
        self.last_mouse_y = self.canvas.canvasy(event.y)

