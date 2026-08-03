from __future__ import annotations
from src.paths import *
from src.bounding_box import BoundingBox, calculate_iou_corners
from src.settings import load_settings, save_settings
from src.model_utils import get_latest_weights, collect_images
from src.theme import *


from typing import *
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw


class RenderingMixin:
    def _load_image(self):
        """Load current image and its existing labels."""
        if not self.image_paths:
            return
            
        path = self.image_paths[self.current_idx]

        # Load with PIL
        with Image.open(path) as img:
            self.orig_pil_img = img.copy()
        self.full_w, self.full_h = self.orig_pil_img.size
        
        self.view_x_offset = 0.0
        self.view_y_offset = 0.0
        self.view_w_scale = 1.0
        self.view_h_scale = 1.0
        self.zoom_level = 1.0

        # ── Load existing labels ────────────────────────────────────
        self.boxes.clear()
        label_path = self._label_path()
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    box = BoundingBox.from_yolo_line(line)
                    if box:
                        self.boxes.append(box)
                        
        # ── Load comparison labels ──────────────────────────────────
        self.compare_boxes.clear()
        if hasattr(self, 'compare_labels_dir') and self.compare_labels_dir:
            comp_path = self.compare_labels_dir / f"{path.stem}.txt"
            if comp_path.exists():
                with open(comp_path, "r") as f:
                    for line in f:
                        box = BoundingBox.from_yolo_line(line)
                        if box:
                            self.compare_boxes.append(box)

        self._update_size_entries()
        self._render_image()
        
        # Apply auto fit if set
        if hasattr(self, 'fit_mode'):
            fm = self.fit_mode.get()
            if fm == "W":
                self._fit_width()
            elif fm == "H":
                self._fit_height()
                
        self._update_ui()
        
    def _render_image(self):
        """Scale and display the image on the canvas."""
        if not hasattr(self, 'orig_pil_img') or not self.orig_pil_img:
            return

        self.canvas.delete("all")
        self.canvas_ids.clear()
        
        # Determine viewport
        mode = getattr(self, 'view_mode', None)
        mode_val = mode.get() if mode else "Full"
        
        # Update UI colors for buttons
        if hasattr(self, 'btn_full'):
            for b, m in [(self.btn_full, "Full"), (self.btn_q1, "Q1"), (self.btn_q2, "Q2"), (self.btn_q3, "Q3"), (self.btn_q4, "Q4")]:
                if mode_val == m:
                    b.config(bg=ACCENT, activebackground="#6D28D9")
                else:
                    b.config(bg="#888888", activebackground="#666666")
        
        fw, fh = self.full_w, self.full_h
        if mode_val == "Q1":
            self.view_x_offset = 0.0
            self.view_y_offset = 0.0
            self.view_w_scale = 0.5
            self.view_h_scale = 0.5
            crop_box = (0, 0, fw//2, fh//2)
        elif mode_val == "Q2":
            self.view_x_offset = 0.5
            self.view_y_offset = 0.0
            self.view_w_scale = 0.5
            self.view_h_scale = 0.5
            crop_box = (fw//2, 0, fw, fh//2)
        elif mode_val == "Q3":
            self.view_x_offset = 0.0
            self.view_y_offset = 0.5
            self.view_w_scale = 0.5
            self.view_h_scale = 0.5
            crop_box = (0, fh//2, fw//2, fh)
        elif mode_val == "Q4":
            self.view_x_offset = 0.5
            self.view_y_offset = 0.5
            self.view_w_scale = 0.5
            self.view_h_scale = 0.5
            crop_box = (fw//2, fh//2, fw, fh)
        else: # Full
            self.view_x_offset = 0.0
            self.view_y_offset = 0.0
            self.view_w_scale = 1.0
            self.view_h_scale = 1.0
            crop_box = (0, 0, fw, fh)
            
        self.pil_img = self.orig_pil_img.crop(crop_box)
        self.orig_w, self.orig_h = self.pil_img.size

        # Calculate scale to fit canvas
        canvas_w = max(self.canvas.winfo_width(), 400)
        canvas_h = max(self.canvas.winfo_height(), 400)
        scale_w = canvas_w / self.orig_w
        scale_h = canvas_h / self.orig_h
        base_scale = min(scale_w, scale_h, 1.0)
        self.display_scale = base_scale * self.zoom_level

        disp_w = int(self.orig_w * self.display_scale)
        disp_h = int(self.orig_h * self.display_scale)

        resized = self.pil_img.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.img_offset_x = max((canvas_w - disp_w) // 2, 0)
        self.img_offset_y = max((canvas_h - disp_h) // 2, 0)

        # Update scrollregion
        scroll_w = max(disp_w, canvas_w)
        scroll_h = max(disp_h, canvas_h)
        self.canvas.config(scrollregion=(0, 0, scroll_w, scroll_h))

        self.canvas.create_image(
            self.img_offset_x, self.img_offset_y,
            anchor=tk.NW, image=self.tk_image, tags="image"
        )
        self._redraw_boxes()

    def _redraw_boxes(self):
        """Clear and redraw all bounding boxes on the canvas."""
        for cid in self.canvas_ids:
            self.canvas.delete(cid)
        self.canvas_ids.clear()
        
        box_colors = []
        box_color_strs = []
        box_line_widths = []
        
        if self.boxes:
            boxes_arr = np.array([
                [b.x_center - b.w/2, b.y_center - b.h/2, b.x_center + b.w/2, b.y_center + b.h/2]
                for b in self.boxes
            ])
            x1_arr, y1_arr, x2_arr, y2_arr = boxes_arr[:,0], boxes_arr[:,1], boxes_arr[:,2], boxes_arr[:,3]
            
            ix1 = np.maximum(x1_arr[:, None], x1_arr[None, :])
            iy1 = np.maximum(y1_arr[:, None], y1_arr[None, :])
            ix2 = np.minimum(x2_arr[:, None], x2_arr[None, :])
            iy2 = np.minimum(y2_arr[:, None], y2_arr[None, :])
            
            inter_w = np.maximum(0.0, ix2 - ix1)
            inter_h = np.maximum(0.0, iy2 - iy1)
            inter_area = inter_w * inter_h
            
            areas = (x2_arr - x1_arr) * (y2_arr - y1_arr)
            min_areas = np.minimum(areas[:, None], areas[None, :])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                overlap_matrix = inter_area / min_areas
                overlap_matrix[np.isnan(overlap_matrix)] = 0.0
                overlap_matrix[np.isinf(overlap_matrix)] = 0.0
                
            np.fill_diagonal(overlap_matrix, 0.0)
            max_overlaps = np.max(overlap_matrix, axis=1)
        else:
            max_overlaps = []
            
        if hasattr(self, 'chk_red'):
            if self.boxes and len(max_overlaps) > 0:
                is_auto_arr = np.array([getattr(b, 'is_auto', False) for b in self.boxes])
                
                # Exclude auto boxes from standard color counting
                red_mask = (max_overlaps >= 0.8) & ~is_auto_arr
                org_mask = (max_overlaps >= 0.5) & (max_overlaps < 0.8) & ~is_auto_arr
                yel_mask = (max_overlaps > 0.0) & (max_overlaps < 0.5) & ~is_auto_arr
                grn_mask = (max_overlaps == 0.0) & ~is_auto_arr
                
                red_count = int(np.sum(red_mask))
                orange_count = int(np.sum(org_mask))
                yellow_count = int(np.sum(yel_mask))
                green_count = int(np.sum(grn_mask))
                violet_count = int(np.sum(is_auto_arr))
                
                self.chk_red.config(text=f"Red ({red_count})")
                self.chk_orange.config(text=f"Org ({orange_count})")
                self.chk_yellow.config(text=f"Yel ({yellow_count})")
                self.chk_green.config(text=f"Grn ({green_count})")
                if hasattr(self, 'chk_violet'):
                    self.chk_violet.config(text=f"Vio ({violet_count})")
            else:
                self.chk_red.config(text="Red (0)")
                self.chk_orange.config(text="Org (0)")
                self.chk_yellow.config(text="Yel (0)")
                self.chk_green.config(text="Grn (0)")
                if hasattr(self, 'chk_violet'):
                    self.chk_violet.config(text="Vio (0)")
            
        for i, box in enumerate(self.boxes):
            x1_n = box.x_center - box.w / 2
            y1_n = box.y_center - box.h / 2
            x2_n = box.x_center + box.w / 2
            y2_n = box.y_center + box.h / 2
            
            # Check viewport intersection
            if hasattr(self, 'view_x_offset'):
                vw_x2 = self.view_x_offset + self.view_w_scale
                vw_y2 = self.view_y_offset + self.view_h_scale
                if x2_n < self.view_x_offset or x1_n > vw_x2 or y2_n < self.view_y_offset or y1_n > vw_y2:
                    box_colors.append(None)
                    box_color_strs.append(None)
                    continue

            is_massive = box.w > 0.5 or box.h > 0.5
            if is_massive:
                if hasattr(self, 'show_red') and not self.show_red.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append((255, 0, 0))
                    box_color_strs.append("#FF0000")
                continue
                
            max_overlap = max_overlaps[i] if i < len(max_overlaps) else 0.0
                        
            if getattr(box, 'is_auto', False):
                if hasattr(self, 'show_violet') and not self.show_violet.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(AUTO_BOX_RGB)
                    box_color_strs.append(AUTO_BOX_HEX)
            elif max_overlap >= 0.8:
                if hasattr(self, 'show_red') and not self.show_red.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(OVERLAP_80_RGB)
                    box_color_strs.append(OVERLAP_80_HEX)
            elif max_overlap >= 0.5:
                if hasattr(self, 'show_orange') and not self.show_orange.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(OVERLAP_50_RGB)
                    box_color_strs.append(OVERLAP_50_HEX)
            elif max_overlap > 0:
                if hasattr(self, 'show_yellow') and not self.show_yellow.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(OVERLAP_0_RGB)
                    box_color_strs.append(OVERLAP_0_HEX)
            else:
                if hasattr(self, 'show_green') and not self.show_green.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(BOX_RGB)
                    box_color_strs.append(BOX_COLOR)

        # Draw comparison boxes if enabled
        if hasattr(self, 'compare_labels_dir') and self.compare_labels_dir and self.show_compare.get():
            for box in self.compare_boxes:
                x1_n = box.x_center - box.w / 2
                y1_n = box.y_center - box.h / 2
                x2_n = box.x_center + box.w / 2
                y2_n = box.y_center + box.h / 2

                cx1, cy1 = self._norm_to_canvas(x1_n, y1_n)
                cx2, cy2 = self._norm_to_canvas(x2_n, y2_n)

                rect_id = self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2,
                    outline="#00FFFF", width=2, dash=(4, 4)
                )
                self.canvas_ids.append(rect_id)

        # Draw primary editable boxes
        try:
            opacity = int(float(self.opacity_var.get().replace(',', '.')) * 255)
        except Exception:
            opacity = 50
        if opacity > 0:
            disp_w = int(self.orig_w * self.display_scale)
            disp_h = int(self.orig_h * self.display_scale)
            overlay_pil = Image.new("RGBA", (disp_w, disp_h), (0,0,0,0))
            draw = ImageDraw.Draw(overlay_pil, "RGBA")

            try:
                base_width = int(self.thickness_var.get())
            except Exception:
                base_width = 3
            
            for i, box in enumerate(self.boxes):
                x1_n = box.x_center - box.w / 2
                y1_n = box.y_center - box.h / 2
                x2_n = box.x_center + box.w / 2
                y2_n = box.y_center + box.h / 2
                
                cx1, cy1 = self._norm_to_canvas(x1_n, y1_n)
                cx2, cy2 = self._norm_to_canvas(x2_n, y2_n)
                
                lx1 = cx1 - self.img_offset_x
                ly1 = cy1 - self.img_offset_y
                lx2 = cx2 - self.img_offset_x
                ly2 = cy2 - self.img_offset_y
                
                color = box_colors[i]
                if color is None: continue
                
                # Make massive/overlap boxes slightly thicker so they stand out
                is_massive = box.w > 0.5 or box.h > 0.5
                outline_w = base_width + 1 if is_massive else base_width
                
                draw.rectangle([lx1, ly1, lx2, ly2], outline=(*color, opacity), width=outline_w)

            self.overlay_tk = ImageTk.PhotoImage(overlay_pil)
            overlay_id = self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor=tk.NW, image=self.overlay_tk)
            self.canvas_ids.append(overlay_id)

        for i, box in enumerate(self.boxes):
            x1_n = box.x_center - box.w / 2
            y1_n = box.y_center - box.h / 2
            x2_n = box.x_center + box.w / 2
            y2_n = box.y_center + box.h / 2

            cx1, cy1 = self._norm_to_canvas(x1_n, y1_n)
            cx2, cy2 = self._norm_to_canvas(x2_n, y2_n)



            color_str = box_color_strs[i]
            if color_str is None: continue
            # Small label
            label_id = self.canvas.create_text(
                cx1 + 3, cy1 - 10,
                text=f"#{i + 1}", anchor=tk.NW,
                font=("Consolas", 8, "bold"), fill=color_str
            )
            self.canvas_ids.append(label_id)

    # ════════════════════════════════════════════════════════════════
    #  SAVE / LOAD / EXPORT
    # ════════════════════════════════════════════════════════════════

    def _update_ui(self):
        total = len(self.image_paths)
        if total == 0:
            self.image_combo.set("")
            self.image_combo.config(values=[])
            self.size_label.config(text="")
            self.count_label.config(text="Boxes: 0")
            self.progress_label.config(text=f"0/0    0 annotated")
            self.root.title(f" Pollen Annotator  {self.current_set} (Empty)")
            
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            if hasattr(self, 'move_train_btn'):
                self.move_train_btn.config(state=tk.DISABLED)
                self.move_val_btn.config(state=tk.DISABLED)
                self.exclude_btn.config(state=tk.DISABLED)
            return

        img_path = self.image_paths[self.current_idx]
        idx = self.current_idx + 1

        # Count how many images already have labels
        annotated = sum(
            1 for p in self.image_paths
            if (self.labels_dir / f"{p.stem}.txt").exists()
            and (self.labels_dir / f"{p.stem}.txt").stat().st_size > 0
        )

        current_names = [p.name for p in self.image_paths]
        if list(self.image_combo["values"]) != current_names:
            self.image_combo.config(values=current_names)
        self.image_combo.set(img_path.name)
        
        self.size_label.config(text=f"{self.orig_w} × {self.orig_h} px")
        self.count_label.config(text=f"Boxes: {len(self.boxes)}")
        self.progress_label.config(text=f"Image {idx}/{total}    {annotated} annotated")
        self.root.title(f" Pollen Annotator  {self.current_set}: {img_path.name} [{idx}/{total}]")

        # Button states
        self.prev_btn.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_idx < total - 1 else tk.DISABLED)
        
        if hasattr(self, 'move_train_btn'):
            self.move_train_btn.config(state=tk.DISABLED if self.current_set == "Train" else tk.NORMAL)
            self.move_val_btn.config(state=tk.DISABLED if self.current_set == "Validation" else tk.NORMAL)
            self.exclude_btn.config(state=tk.DISABLED if self.current_set == "Excluded" else tk.NORMAL)

    # ════════════════════════════════════════════════════════════════
    #  STATE PERSISTENCE
    # ════════════════════════════════════════════════════════════════

