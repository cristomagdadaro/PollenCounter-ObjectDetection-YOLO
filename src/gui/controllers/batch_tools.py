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


class BatchToolsMixin:
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


