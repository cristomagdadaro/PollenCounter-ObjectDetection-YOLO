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
import pandas as pd
from datetime import datetime
import shutil
import random


class BatchToolsMixin:
    def export_dataset(self):
        if not self.image_paths:
            messagebox.showinfo("Info", "No images to export in the current dataset.")
            return

        out_dir_str = filedialog.askdirectory(parent=getattr(self, "root", None), title="Select Output Directory for Export")
        if not out_dir_str:
            return
            
        out_dir = Path(out_dir_str)
        embed = self.embed_annotations.get()
        
        if embed:
            images_out = out_dir / "annotated"
        else:
            images_out = out_dir / "images"
            
        images_out.mkdir(parents=True, exist_ok=True)

        summary_rows = []
        detail_rows = []

        total = len(self.image_paths)
        for idx, img_path in enumerate(self.image_paths, start=1):
            lbl_path = self.labels_dir / f"{img_path.stem}.txt"
            
            # Read boxes
            boxes = []
            if lbl_path.exists():
                try:
                    with open(lbl_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(float(parts[0]))
                                cx, cy, w, h = map(float, parts[1:5])
                                conf = float(parts[5]) if len(parts) >= 6 else 1.0
                                box = BoundingBox(cx, cy, w, h, class_id=cls_id)
                                box.conf = conf
                                boxes.append(box)
                except Exception as e:
                    print(f"Error reading {lbl_path}: {e}")

            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            img_h, img_w = img.shape[:2]
            n_detections = len(boxes)
            confidences = [b.conf for b in boxes]
            
            summary_rows.append({
                "filename": img_path.name,
                "pollen_count": n_detections,
                "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
                "min_confidence": round(min(confidences), 4) if confidences else 0.0,
                "max_confidence": round(max(confidences), 4) if confidences else 0.0,
                "image_width": img_w,
                "image_height": img_h,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

            if n_detections > 0:
                for det_idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.to_pixel(img_w, img_h)
                    detail_rows.append({
                        "filename": img_path.name,
                        "detection_id": det_idx + 1,
                        "x_center": round((x1 + x2) / 2, 2),
                        "y_center": round((y1 + y2) / 2, 2),
                        "width": round(x2 - x1, 2),
                        "height": round(y2 - y1, 2),
                        "confidence": round(box.conf, 4) if box.conf is not None else 1.0,
                    })

            if embed:
                # Get styles from UI
                try:
                    opacity = int(float(self.opacity_var.get().replace(',', '.')) * 255)
                except Exception:
                    opacity = 255
                    
                try:
                    base_width = int(self.thickness_var.get())
                except Exception:
                    base_width = 1

                show_red = getattr(self, 'show_red', None)
                show_org = getattr(self, 'show_orange', None)
                show_yel = getattr(self, 'show_yellow', None)
                show_grn = getattr(self, 'show_green', None)
                show_vio = getattr(self, 'show_violet', None)

                show_red_val = show_red.get() if show_red else True
                show_org_val = show_org.get() if show_org else True
                show_yel_val = show_yel.get() if show_yel else True
                show_grn_val = show_grn.get() if show_grn else True
                show_vio_val = show_vio.get() if show_vio else True

                # Calculate overlaps to match GUI colors
                max_overlaps = []
                if boxes:
                    boxes_arr = np.array([
                        [b.x_center - b.w/2, b.y_center - b.h/2, b.x_center + b.w/2, b.y_center + b.h/2]
                        for b in boxes
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

                pil_img = Image.open(str(img_path)).convert("RGBA")
                overlay_pil = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay_pil, "RGBA")
                
                try:
                    from PIL import ImageFont
                    font = ImageFont.truetype("arial.ttf", size=14)
                except Exception:
                    font = None

                drawn_count = 0
                for det_idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.to_pixel(img_w, img_h)
                    
                    max_ov = max_overlaps[det_idx] if det_idx < len(max_overlaps) else 0.0
                    is_auto = getattr(box, 'is_auto', False)
                    is_massive = box.w > 0.5 or box.h > 0.5
                    
                    color = None
                    if is_massive:
                        if show_red_val: color = (255, 0, 0)
                    elif is_auto:
                        if show_vio_val: color = AUTO_BOX_RGB
                    elif max_ov >= 0.8:
                        if show_red_val: color = OVERLAP_80_RGB
                    elif max_ov >= 0.5:
                        if show_org_val: color = OVERLAP_50_RGB
                    elif max_ov > 0.0:
                        if show_yel_val: color = OVERLAP_0_RGB
                    else:
                        if show_grn_val: color = BOX_RGB
                        
                    if color is not None:
                        drawn_count += 1
                        outline_w = base_width + 1 if is_massive else base_width
                        draw.rectangle([x1, y1, x2, y2], outline=(*color, opacity), width=outline_w)
                        text = f"#{det_idx + 1}"
                        if font:
                            draw.text((x1, y1 - 15), text, fill=(*color, opacity), font=font)
                        else:
                            draw.text((x1, y1 - 10), text, fill=(*color, opacity))

                out_img = Image.alpha_composite(pil_img, overlay_pil).convert("RGB")
                
                # Add total count
                cv_img = np.array(out_img)
                # Convert RGB to BGR for cv2
                cv_img = cv_img[:, :, ::-1].copy()
                text = str(drawn_count)
                cv_font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(text, cv_font, 1.36, 3)
                
                # Top right corner with 30px padding
                text_x = img_w - text_w - 30
                text_y = text_h + 30
                
                cv2.putText(cv_img, text, (text_x, text_y), cv_font, 1.36, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(cv_img, text, (text_x, text_y), cv_font, 1.36, (255, 255, 255), 3, cv2.LINE_AA)
                
                cv2.imwrite(str(images_out / img_path.name), cv_img)
            else:
                shutil.copy2(img_path, images_out / img_path.name)

        with pd.ExcelWriter(str(out_dir / "pollen_counts.xlsx"), engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            if detail_rows:
                pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Detections", index=False)
                
        messagebox.showinfo("Export Complete", f"Dataset exported successfully to:\n{out_dir}")

    def _fit_width(self):
        if not self.pil_img: return
        canvas_w = max(self.canvas.winfo_width(), 400)
        canvas_h = max(self.canvas.winfo_height(), 400)
        scale_w = canvas_w / self.orig_w
        scale_h = canvas_h / self.orig_h
        base_scale = min(scale_w, scale_h, 1.0)
        self.zoom_level = scale_w / base_scale
        if hasattr(self, 'fit_mode'): self.fit_mode.set("W")
        self._render_image()

    def _fit_height(self):
        if not self.pil_img: return
        canvas_w = max(self.canvas.winfo_width(), 400)
        canvas_h = max(self.canvas.winfo_height(), 400)
        scale_w = canvas_w / self.orig_w
        scale_h = canvas_h / self.orig_h
        base_scale = min(scale_w, scale_h, 1.0)
        self.zoom_level = scale_h / base_scale
        if hasattr(self, 'fit_mode'): self.fit_mode.set("H")
        self._render_image()

    def _clean_overlapping_boxes(self):
        if not self.boxes: return
        try:
            threshold = float(self.clean_threshold_var.get()) / 100.0
        except ValueError:
            return
            
        self.boxes_backup = list(self.boxes) # Save for undo
        
        boxes_arr = np.array([
            [b.x_center - b.w/2, b.y_center - b.h/2, b.x_center + b.w/2, b.y_center + b.h/2]
            for b in self.boxes
        ])
        
        x1, y1 = boxes_arr[:, 0], boxes_arr[:, 1]
        x2, y2 = boxes_arr[:, 2], boxes_arr[:, 3]
        
        ix1 = np.maximum(x1[:, None], x1[None, :])
        iy1 = np.maximum(y1[:, None], y1[None, :])
        ix2 = np.minimum(x2[:, None], x2[None, :])
        iy2 = np.minimum(y2[:, None], y2[None, :])
        
        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h
        
        areas = (x2 - x1) * (y2 - y1)
        min_areas = np.minimum(areas[:, None], areas[None, :])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            overlap_matrix = inter_area / min_areas
            overlap_matrix[np.isnan(overlap_matrix)] = 0.0
            overlap_matrix[np.isinf(overlap_matrix)] = 0.0
            
        np.fill_diagonal(overlap_matrix, 0.0)
        
        to_remove = set()
        for i in range(len(self.boxes)):
            if i in to_remove: continue
            for j in range(i+1, len(self.boxes)):
                if j in to_remove: continue
                if overlap_matrix[i, j] > threshold:
                    if areas[j] < areas[i]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        if to_remove:
            self.boxes = [b for idx, b in enumerate(self.boxes) if idx not in to_remove]
            self._redraw_boxes()
            self._save_labels()
            self._update_ui()
            self.status.config(text=f" Cleaned {len(to_remove)} boxes with >{int(threshold*100)}% overlap.")
        else:
            self.status.config(text=" No boxes found exceeding overlap threshold.")

    def _undo_clean_boxes(self):
        if hasattr(self, 'boxes_backup') and self.boxes_backup:
            self.boxes = list(self.boxes_backup)
            self._redraw_boxes()
            self._save_labels()
            self._update_ui()
            self.status.config(text=" Restored boxes from backup.")

    # ════════════════════════════════════════════════════════════════
    #  BOX RENDERING
    # ════════════════════════════════════════════════════════════════

    def _export_jpg(self):
        """Export the current image and boxes to a JPG file."""
        if not hasattr(self, 'orig_pil_img') or not self.orig_pil_img or not self.image_paths:
            return
            
        path = self.image_paths[self.current_idx]
        
        from tkinter import filedialog
        initial_file = f"annotated_{path.name}"
        if not initial_file.lower().endswith(".jpg"):
            initial_file = str(Path(initial_file).with_suffix(".jpg"))
            
        out_path_str = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Annotated Image",
            initialfile=initial_file,
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )
        
        if not out_path_str:
            return
            
        out_path = Path(out_path_str)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        export_img = self.orig_pil_img.copy().convert("RGBA")
        overlay = Image.new("RGBA", export_img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        
        try:
            opacity = int(float(self.opacity_var.get().replace(',', '.')) * 255)
        except Exception:
            opacity = 50
        
        for i, box in enumerate(self.boxes):
            x1_n = box.x_center - box.w / 2
            y1_n = box.y_center - box.h / 2
            x2_n = box.x_center + box.w / 2
            y2_n = box.y_center + box.h / 2
            
            x1 = x1_n * self.full_w
            y1 = y1_n * self.full_h
            x2 = x2_n * self.full_w
            y2 = y2_n * self.full_h
            
            is_massive = box.w > 0.5 or box.h > 0.5
            
            is_overlap = False
            if not is_massive:
                for j, other in enumerate(self.boxes):
                    if i == j: continue
                    ox1 = other.x_center - other.w / 2
                    oy1 = other.y_center - other.h / 2
                    ox2 = other.x_center + other.w / 2
                    oy2 = other.y_center + other.h / 2
                    if not (x2_n <= ox1 or x1_n >= ox2 or y2_n <= oy1 or y1_n >= oy2):
                        is_overlap = True
                        break
                        
            if is_massive:
                color = (255, 0, 0)
            elif is_overlap:
                color = (245, 158, 11)
            else:
                color = (0, 255, 136)
            
            try:
                base_width = int(self.thickness_var.get())
            except Exception:
                base_width = 3
            outline_w = base_width + 1 if is_massive else base_width
            
            draw.rectangle([x1, y1, x2, y2], outline=(*color, opacity), width=outline_w)
            # optional text
            draw.text((x1 + 3, y1 - 15), f"#{i+1}", fill=(*color, 255))
            
        export_img = Image.alpha_composite(export_img, overlay).convert("RGB")
        out_path = out_dir / path.name
        export_img.save(out_path, quality=95)
        self.status.config(text=f" Exported JPG to {out_path.name}")

    def _open_batch_tools(self):
        batch_win = tk.Toplevel(self.root)
        batch_win.title("Batch Tools")
        batch_win.geometry("450x300")
        batch_win.configure(bg=BG_COLOR)
        batch_win.transient(self.root)
        batch_win.grab_set()

        tk.Label(batch_win, text="Batch Edit", font=("Segoe UI", 14, "bold"), bg=BG_COLOR, fg=ACCENT).pack(pady=10)

        # Target Selection
        target_frame = tk.LabelFrame(batch_win, text="Target Scope", bg=BG_COLOR, font=("Segoe UI", 10, "bold"), fg=TEXT_COLOR)
        target_frame.pack(fill=tk.X, padx=15, pady=5)
        
        target_var = tk.StringVar(value="dataset")
        tk.Radiobutton(target_frame, text="Current Dataset (e.g., all Train)", variable=target_var, value="dataset", bg=BG_COLOR, fg=TEXT_COLOR, selectcolor=SIDEBAR_BG).pack(anchor=tk.W, padx=5, pady=2)
        tk.Radiobutton(target_frame, text="All Datasets (Train + Val + Excluded)", variable=target_var, value="all", bg=BG_COLOR, fg=TEXT_COLOR, selectcolor=SIDEBAR_BG).pack(anchor=tk.W, padx=5, pady=2)

        # Operations
        ops_frame = tk.LabelFrame(batch_win, text="Operations", bg=BG_COLOR, font=("Segoe UI", 10, "bold"), fg=TEXT_COLOR)
        ops_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Scale
        scale_frame = tk.Frame(ops_frame, bg=BG_COLOR)
        scale_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(scale_frame, text="Scale Factor:", bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        scale_entry = tk.Entry(scale_frame, width=8, font=("Consolas", 10))
        scale_entry.insert(0, "1.1")
        scale_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(scale_frame, text="Apply Scale", bg=ACCENT, fg="white", bd=0, cursor="hand2", command=lambda: self._run_batch("scale", target_var.get(), scale_entry.get(), batch_win)).pack(side=tk.RIGHT)
        
        # Snap
        snap_frame = tk.Frame(ops_frame, bg=BG_COLOR)
        snap_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(snap_frame, text="Snap all boxes to edges:", bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        tk.Button(snap_frame, text="Auto-Snap All", bg="#8B5CF6", fg="white", bd=0, cursor="hand2", command=lambda: self._run_batch("snap", target_var.get(), None, batch_win)).pack(side=tk.RIGHT)
        
        # Clear
        clear_frame = tk.Frame(ops_frame, bg=BG_COLOR)
        clear_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(clear_frame, text="Clear all boxes from images:", bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        tk.Button(clear_frame, text="Clear All", bg="#EF4444", fg="white", bd=0, cursor="hand2", command=lambda: self._run_batch("clear", target_var.get(), None, batch_win)).pack(side=tk.RIGHT)

    def _open_inference(self):
        """Launch the Inference tool as a child window."""
        try:
            import subprocess
            subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "inference.py")])
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch Inference tool: {e}")

    def _open_compare(self):
        """Launch the Compare tool as a child window."""
        try:
            import subprocess
            subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "compare_val.py")])
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch Compare tool: {e}")

    def _open_live_video(self):
        """Launch the Live Video tool."""
        weights = get_latest_weights()
        if not weights or not weights.exists():
            messagebox.showerror("Error", "No trained model weights found.")
            return
        try:
            import subprocess
            subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "live_video.py"), "--model", str(weights), "--source", "0"])
        except Exception as e:
            messagebox.showerror("Error", f"Could not launch Live Video: {e}")


