import os
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
from ultralytics import YOLO
import csv

# ─── Project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAL_IMAGES = PROJECT_ROOT / "datasets" / "images" / "val"
VAL_LABELS = PROJECT_ROOT / "datasets" / "labels" / "val"

# Find latest YOLO11n weights
def get_latest_weights():
    detect_dir = PROJECT_ROOT / "runs" / "detect"
    if not detect_dir.exists():
        return None
    weight_files = list(detect_dir.rglob("weights/best.pt"))
    if not weight_files:
        return None
    # Prioritize yolo11n since that is what the user was using
    n_weights = [p for p in weight_files if "11n" in str(p)]
    if n_weights:
        return max(n_weights, key=lambda p: p.stat().st_mtime)
    return max(weight_files, key=lambda p: p.stat().st_mtime)

MODEL_PATH = get_latest_weights()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ─── Colours (Light Mode Theme) ─────────────────────────────────────
HUMAN_COLOR = "#00AA00"       # Green
MODEL_COLOR = "#FF0000"       # Red
BG_COLOR = "#FFFFFF"
SIDEBAR_BG = "#F0F0F0"
ACCENT = "#0000FF"
TEXT_COLOR = "#000000"

class CompareGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Validation Comparison Tool")
        self.root.geometry("1400x800")
        self.root.configure(bg=BG_COLOR)
        
        self.image_paths = sorted(p for p in VAL_IMAGES.glob("*") if p.suffix.lower() in IMAGE_EXTS)
        self.current_idx = 0
        
        # Load Model
        self.model = None
        if MODEL_PATH and MODEL_PATH.exists():
            try:
                self.model = YOLO(MODEL_PATH)
            except Exception as e:
                print(f"Error loading model: {e}")
                
        self._build_ui()
        
        if not self.image_paths:
            messagebox.showwarning("No Images", f"No validation images found in:\n{VAL_IMAGES}")
        else:
            self.image_combo.config(values=[p.name for p in self.image_paths])
            self._load_image()

    def _build_ui(self):
        # Top toolbar
        toolbar = tk.Frame(self.root, bg=SIDEBAR_BG, height=50)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        btn_style = {"font": ("Segoe UI", 10, "bold"), "bd": 0, "cursor": "hand2", "padx": 15, "pady": 5}
        
        self.prev_btn = tk.Button(toolbar, text="◀ Previous", bg="#888888", fg="white", activebackground="#666666", command=self._prev_image, **btn_style)
        self.prev_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.next_btn = tk.Button(toolbar, text="Next ▶", bg=ACCENT, fg="white", activebackground="#0000AA", command=self._next_image, **btn_style)
        self.next_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Label(toolbar, text="Image:", font=("Segoe UI", 10), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=(20, 5))
        
        self.image_combo = ttk.Combobox(toolbar, state="readonly", font=("Consolas", 10), width=35)
        self.image_combo.pack(side=tk.LEFT, padx=5, pady=10)
        self.image_combo.bind("<<ComboboxSelected>>", self._on_combo_jump)
        
        self.report_btn = tk.Button(toolbar, text="📊 Generate Excel Report", bg="#00AA00", fg="white", activebackground="#008800", command=self._generate_report, **btn_style)
        self.report_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.stats_lbl = tk.Label(toolbar, text="Stats: -", font=("Segoe UI", 10, "bold"), bg=SIDEBAR_BG, fg=TEXT_COLOR)
        self.stats_lbl.pack(side=tk.RIGHT, padx=20)
        
        # Main Canvas area
        self.canvas = tk.Canvas(self.root, bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        self.canvas.bind("<Configure>", lambda e: self._resize_image())
        
        self.display_img = None
        self.tk_img = None

    def _prev_image(self):
        if self.image_paths and self.current_idx > 0:
            self.current_idx -= 1
            self._load_image()

    def _next_image(self):
        if self.image_paths and self.current_idx < len(self.image_paths) - 1:
            self.current_idx += 1
            self._load_image()

    def _on_combo_jump(self, event=None):
        if not self.image_paths: return
        selected = self.image_combo.get()
        try:
            self.current_idx = next(i for i, p in enumerate(self.image_paths) if p.name == selected)
            self._load_image()
        except StopIteration:
            pass

    def _load_image(self):
        if not self.image_paths: return
        img_path = self.image_paths[self.current_idx]
        self.image_combo.set(img_path.name)
        
        # Update buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_idx < len(self.image_paths) - 1 else tk.DISABLED)
        
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None: return
            
            human_img = img_bgr.copy()
            pred_img = img_bgr.copy()
            h, w = img_bgr.shape[:2]
            
            # Draw Human Labels
            human_count = 0
            lbl_path = VAL_LABELS / f"{img_path.stem}.txt"
            if lbl_path.exists():
                with open(lbl_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cx, cy, bw, bh = map(float, parts[1:5])
                            x1 = int((cx - bw/2) * w)
                            y1 = int((cy - bh/2) * h)
                            x2 = int((cx + bw/2) * w)
                            y2 = int((cy + bh/2) * h)
                            cv2.rectangle(human_img, (x1, y1), (x2, y2), (0, 170, 0), 2)
                            human_count += 1
            
            # Draw Model Predictions
            model_count = 0
            if self.model:
                results = self.model(str(img_path), max_det=5000, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cx, cy, bw, bh = box.xywhn[0].tolist()
                        conf = box.conf[0].item()
                        x1 = int((cx - bw/2) * w)
                        y1 = int((cy - bh/2) * h)
                        x2 = int((cx + bw/2) * w)
                        y2 = int((cy + bh/2) * h)
                        cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(pred_img, f"{conf:.2f}", (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        model_count += 1
            
            # Add Titles
            cv2.putText(human_img, f"Human (Boxes: {human_count})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 170, 0), 3)
            cv2.putText(pred_img, f"Model (Boxes: {model_count})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            self.stats_lbl.config(text=f"Image {self.current_idx + 1}/{len(self.image_paths)} | Human: {human_count} | Model: {model_count}")
            
            # Concatenate
            side_by_side = np.hstack((human_img, pred_img))
            side_by_side_rgb = cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB)
            self.display_img = Image.fromarray(side_by_side_rgb)
            
            self._resize_image()
            
        except Exception as e:
            print(f"Error loading image: {e}")

    def _resize_image(self):
        if self.display_img is None: return
        
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10: return
        
        img_w, img_h = self.display_img.size
        scale = min(cw / img_w, ch / img_h, 1.0)
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        resized = self.display_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, anchor=tk.CENTER, image=self.tk_img)

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def _generate_report(self):
        if not self.model or not self.image_paths:
            messagebox.showerror("Error", "Model or images not loaded.")
            return
            
        out_path = PROJECT_ROOT / "runs" / "detect" / "validation_report.csv"
        
        self.report_btn.config(state=tk.DISABLED, text="⏳ Generating...")
        self.root.update()
        
        try:
            with open(out_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Image Name", "Human Count", "Model Count", "True Positives (TP)", "False Positives (FP)", "False Negatives (FN)", "True Negatives (TN)", "Average Confidence", "Precision", "Recall", "F1-Score"])
                
                for img_path in self.image_paths:
                    img_bgr = cv2.imread(str(img_path))
                    if img_bgr is None: continue
                    h, w = img_bgr.shape[:2]
                    
                    human_boxes = []
                    lbl_path = VAL_LABELS / f"{img_path.stem}.txt"
                    if lbl_path.exists():
                        with open(lbl_path, "r") as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cx, cy, bw, bh = map(float, parts[1:5])
                                    x1, y1 = (cx - bw/2)*w, (cy - bh/2)*h
                                    x2, y2 = (cx + bw/2)*w, (cy + bh/2)*h
                                    human_boxes.append([x1, y1, x2, y2])
                                    
                    model_boxes = []
                    confidences = []
                    results = self.model(str(img_path), max_det=5000, verbose=False)
                    for r in results:
                        for box in r.boxes:
                            cx, cy, bw, bh = box.xywhn[0].tolist()
                            confidences.append(box.conf[0].item())
                            x1, y1 = (cx - bw/2)*w, (cy - bh/2)*h
                            x2, y2 = (cx + bw/2)*w, (cy + bh/2)*h
                            model_boxes.append([x1, y1, x2, y2])
                            
                    tp, fp, fn = 0, 0, 0
                    matched_human = set()
                    
                    for m_idx, m_box in enumerate(model_boxes):
                        best_iou = 0
                        best_h_idx = -1
                        for h_idx, h_box in enumerate(human_boxes):
                            if h_idx in matched_human: continue
                            iou = self._calculate_iou(m_box, h_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_h_idx = h_idx
                                
                        if best_iou >= 0.5:
                            tp += 1
                            matched_human.add(best_h_idx)
                        else:
                            fp += 1
                            
                    fn = len(human_boxes) - len(matched_human)
                    
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    writer.writerow([img_path.name, len(human_boxes), len(model_boxes), tp, fp, fn, "N/A", f"{avg_conf:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}"])
            
            messagebox.showinfo("Report Generated", f"Excel report successfully generated at:\n{out_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")
            
        finally:
            self.report_btn.config(state=tk.NORMAL, text="📊 Generate Excel Report")

def main():
    root = tk.Tk()
    app = CompareGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
