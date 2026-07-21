#!/usr/bin/env python3
"""
inference.py — Unified YOLO Inference (GUI & CLI)
=================================================

This script replaces `count_pollen.py` and `auto_annotate.py`.
It provides two operating modes:
  1. Count & Analyze: Counts objects, exports to .xlsx, and saves images with dots.
  2. Auto-Annotate: Saves raw YOLO .txt labels for active learning in annotate.py.

Usage:
    GUI: python scripts/inference.py
"""

from __future__ import annotations

import argparse
import sys
import json
import threading
import random
import shutil
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ImageTk

# ─── Project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "detect" / "results"

def get_latest_weights():
    detect_dir = PROJECT_ROOT / "runs" / "detect"
    if not detect_dir.exists():
        return detect_dir / "train" / "weights" / "best.pt"
    weight_files = list(detect_dir.rglob("weights/best.pt"))
    if not weight_files:
        return detect_dir / "train" / "weights" / "best.pt"
    return max(weight_files, key=lambda p: p.stat().st_mtime)

DEFAULT_WEIGHTS = get_latest_weights()
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def collect_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

def run_inference(
    mode: str,
    model: YOLO,
    image_paths: list[Path],
    out_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
    save_annotated_imgs: bool = True,
    progress_cb=None,
    log_cb=None
):
    """
    mode: "Count & Analyze" or "Auto-Annotate"
    """
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup subfolders depending on mode
    if mode == "Count & Analyze":
        annotated_dir = out_dir / "annotated"
        if save_annotated_imgs:
            annotated_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Auto-Annotate mode
        aa_img_dir = out_dir / "images" / "review"
        aa_lbl_dir = out_dir / "labels" / "review"
        aa_img_dir.mkdir(parents=True, exist_ok=True)
        aa_lbl_dir.mkdir(parents=True, exist_ok=True)

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        if progress_cb: progress_cb(idx, total, f"Processing [{idx}/{total}]: {img_path.name}")
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        img_h, img_w = img.shape[:2]

        results = model.predict(source=str(img_path), conf=conf, iou=0.5, imgsz=imgsz, device=device, max_det=5000, verbose=False)
        boxes = results[0].boxes
        n_detections = len(boxes)
        confidences = boxes.conf.cpu().tolist() if n_detections > 0 else []

        if mode == "Count & Analyze":
            # ── COUNT MODE LOGIC ──
            summary_rows.append({
                "filename": img_path.name,
                "pollen_count": n_detections,
                "avg_confidence": round(sum(confidences)/len(confidences), 4) if confidences else 0.0,
                "min_confidence": round(min(confidences), 4) if confidences else 0.0,
                "max_confidence": round(max(confidences), 4) if confidences else 0.0,
                "image_width": img_w,
                "image_height": img_h,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })

            if n_detections > 0:
                for det_idx, (box, conf_val) in enumerate(zip(boxes.xyxy.cpu().tolist(), confidences)):
                    x1, y1, x2, y2 = box
                    detail_rows.append({
                        "filename": img_path.name,
                        "detection_id": det_idx + 1,
                        "x_center": round((x1 + x2) / 2, 2),
                        "y_center": round((y1 + y2) / 2, 2),
                        "width": round(x2 - x1, 2),
                        "height": round(y2 - y1, 2),
                        "confidence": round(conf_val, 4),
                    })

            if save_annotated_imgs:
                annotated_img = img.copy()
                if n_detections > 0:
                    for box in boxes.xyxy.cpu().tolist():
                        x1, y1, x2, y2 = box
                        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                        color = (random.randint(50, 255), random.randint(100, 255), random.randint(50, 255))
                        cv2.rectangle(annotated_img, (x1_i, y1_i), (x2_i, y2_i), color=color, thickness=3)

                overlay = annotated_img.copy()
                text = str(n_detections)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(text, font, 1.36, 3)
                text_x, text_y = (img_w - text_w) // 2, (img_h + text_h) // 2
                cv2.putText(overlay, text, (text_x, text_y), font, 1.36, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(overlay, text, (text_x, text_y), font, 1.36, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.5, annotated_img, 0.5, 0, annotated_img)
                cv2.imwrite(str(annotated_dir / img_path.name), annotated_img)

        else:
            # ── AUTO-ANNOTATE MODE LOGIC ──
            shutil.copy2(img_path, aa_img_dir / img_path.name)
            label_path = aa_lbl_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                if n_detections > 0:
                    cls_ids = boxes.cls.cpu().tolist()
                    xyxy = boxes.xyxy.cpu().tolist()
                    
                    for cls_id, box in zip(cls_ids, xyxy):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Clamp to image boundaries
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(img_w, x2)
                        y2 = min(img_h, y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                            
                        # Extract the crop (Region of Interest)
                        roi = img[y1:y2, x1:x2]
                        
                        # Apply OpenCV Contours to snap perfectly to the pollen
                        try:
                            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                            
                            # Otsu's thresholding (assumes dark pollen on bright background)
                            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            if contours:
                                largest_contour = max(contours, key=cv2.contourArea)
                                cx, cy, cw, ch = cv2.boundingRect(largest_contour)
                                
                                # Convert local contour rect back to global image coordinates
                                new_x1 = x1 + cx
                                new_y1 = y1 + cy
                                new_x2 = new_x1 + cw
                                new_y2 = new_y1 + ch
                                
                                # Convert to YOLO normalized format
                                w_norm = (new_x2 - new_x1) / img_w
                                h_norm = (new_y2 - new_y1) / img_h
                                x_center_norm = (new_x1 + new_x2) / 2.0 / img_w
                                y_center_norm = (new_y1 + new_y2) / 2.0 / img_h
                            else:
                                raise ValueError("No contour")
                        except Exception:
                            # Fallback to the original YOLO prediction if OpenCV fails
                            w_norm = (x2 - x1) / img_w
                            h_norm = (y2 - y1) / img_h
                            x_center_norm = (x1 + x2) / 2.0 / img_w
                            y_center_norm = (y1 + y2) / 2.0 / img_h
                            
                        f.write(f"{int(cls_id)} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    if mode == "Count & Analyze":
        with pd.ExcelWriter(str(out_dir / "pollen_counts.xlsx"), engine="openpyxl") as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
            if detail_rows:
                pd.DataFrame(detail_rows).to_excel(writer, sheet_name="Detections", index=False)
        return summary_rows
    else:
        return []

def show_results_viewer(parent, summary_rows, in_dir, out_dir):
    if not summary_rows: return
    viewer = tk.Toplevel(parent)
    viewer.title("Results Viewer")
    viewer.geometry("1000x600")
    
    current_idx = [0]
    top_frame = ttk.Frame(viewer)
    top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    
    btn_prev = ttk.Button(top_frame, text="<< Previous")
    btn_prev.pack(side=tk.LEFT, padx=10)
    
    zoom_enabled = tk.BooleanVar(value=True)
    ttk.Checkbutton(top_frame, text="Zoom on Hover", variable=zoom_enabled).pack(side=tk.LEFT, padx=10)
    
    lbl_info = ttk.Label(top_frame, text="", font=("TkDefaultFont", 12, "bold"))
    lbl_info.pack(side=tk.LEFT, expand=True)
    
    btn_next = ttk.Button(top_frame, text="Next >>")
    btn_next.pack(side=tk.RIGHT, padx=10)
    
    img_frame = ttk.Frame(viewer)
    img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)
    
    lbl_orig = ttk.Label(img_frame, text="Original")
    lbl_orig.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    
    lbl_annot = ttk.Label(img_frame, text="Annotated")
    lbl_annot.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
    
    bot_frame = ttk.Frame(viewer)
    bot_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
    lbl_meta = ttk.Label(bot_frame, text="", font=("TkDefaultFont", 11))
    lbl_meta.pack()
    
    viewer_state = {"orig_full": None, "annot_full": None, "orig_tk": None, "annot_tk": None, "orig_scale": 1.0, "annot_scale": 1.0, "zoom_tk": None}
    
    zoom_win = tk.Toplevel(viewer)
    zoom_win.withdraw()
    zoom_win.overrideredirect(True)
    zoom_lbl = tk.Label(zoom_win, bd=2, relief="solid", bg="black")
    zoom_lbl.pack()
    
    def handle_hover(event, img_type, lbl):
        if not zoom_enabled.get(): return zoom_win.withdraw()
        full_img, tk_img = viewer_state.get(f"{img_type}_full"), viewer_state.get(f"{img_type}_tk")
        if not full_img or not tk_img: return
        
        offset_x = (lbl.winfo_width() - tk_img.width()) // 2
        offset_y = (lbl.winfo_height() - tk_img.height()) // 2
        img_x, img_y = event.x - offset_x, event.y - offset_y
        if img_x < 0 or img_x > tk_img.width() or img_y < 0 or img_y > tk_img.height():
            return zoom_win.withdraw()
            
        scale = viewer_state[f"{img_type}_scale"]
        full_x, full_y = int(img_x * scale), int(img_y * scale)
        
        crop_size = 200
        box = (full_x - crop_size//2, full_y - crop_size//2, full_x + crop_size//2, full_y + crop_size//2)
        crop = full_img.crop(box).resize((crop_size * 2, crop_size * 2), getattr(Image, 'Resampling', Image).LANCZOS)
        
        viewer_state["zoom_tk"] = ImageTk.PhotoImage(crop)
        zoom_lbl.config(image=viewer_state["zoom_tk"])
        zoom_win.geometry(f"+{event.x_root + 20}+{event.y_root + 20}")
        zoom_win.deiconify()
        zoom_win.lift()
        
    lbl_orig.bind("<Motion>", lambda e: handle_hover(e, "orig", lbl_orig))
    lbl_orig.bind("<Leave>", lambda e: zoom_win.withdraw())
    lbl_annot.bind("<Motion>", lambda e: handle_hover(e, "annot", lbl_annot))
    lbl_annot.bind("<Leave>", lambda e: zoom_win.withdraw())
    
    def update_view():
        idx = current_idx[0]
        row = summary_rows[idx]
        lbl_info.config(text=f"Image {idx+1} of {len(summary_rows)}: {row['filename']}")
        lbl_meta.config(text=f"Count: {row['pollen_count']}   |   Avg Conf: {row['avg_confidence']:.2f}")
        
        orig_path = in_dir / row['filename']
        annot_path = out_dir / "annotated" / row['filename']
        
        viewer.update_idletasks()
        w, h = max(400, (viewer.winfo_width() - 40) // 2), max(400, viewer.winfo_height() - 150)
        
        def load_img(path, img_type):
            try:
                full = Image.open(path)
                viewer_state[f"{img_type}_full"] = full
                copy = full.copy()
                copy.thumbnail((w, h), getattr(Image, 'Resampling', Image).LANCZOS)
                viewer_state[f"{img_type}_tk"] = ImageTk.PhotoImage(copy)
                (lbl_orig if img_type=="orig" else lbl_annot).config(image=viewer_state[f"{img_type}_tk"], text="")
                viewer_state[f"{img_type}_scale"] = full.width / copy.width
            except Exception:
                (lbl_orig if img_type=="orig" else lbl_annot).config(image="", text="Not found")
        
        load_img(orig_path, "orig")
        load_img(annot_path, "annot")
        
        btn_prev.config(state=tk.NORMAL if idx > 0 else tk.DISABLED)
        btn_next.config(state=tk.NORMAL if idx < len(summary_rows)-1 else tk.DISABLED)
        
    btn_prev.config(command=lambda: (current_idx.__setitem__(0, current_idx[0]-1), update_view()))
    btn_next.config(command=lambda: (current_idx.__setitem__(0, current_idx[0]+1), update_view()))
    viewer.after(100, update_view)

def run_gui():
    root = tk.Tk()
    root.title("Unified Inference Tool")
    root.geometry("650x700")
    root.configure(bg="#1E1E2E")
    
    FONT_LABEL = ("Segoe UI", 10, "bold")
    FONT_INPUT = ("Consolas", 10)
    BG_COLOR = "#1E1E2E"
    FG_COLOR = "#CDD6F4"
    ACCENT_COLOR = "#89B4FA"
    BTN_BG = "#313244"
    
    tk.Label(root, text="✨ Unified YOLO Inference", font=("Segoe UI", 14, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR).pack(pady=10)
    
    CONFIG_FILE = PROJECT_ROOT / "config" / "inference_settings.json"
    saved_settings = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                saved_settings = json.load(f)
        except Exception:
            pass
            
    var_mode = tk.StringVar(value=saved_settings.get("mode", "Count & Analyze"))
    var_input = tk.StringVar(value=saved_settings.get("input", str(DEFAULT_INPUT)))
    var_out = tk.StringVar(value=saved_settings.get("output", str(DEFAULT_OUTPUT)))
    var_weights = tk.StringVar(value=saved_settings.get("weights", str(DEFAULT_WEIGHTS)))
    var_conf = tk.StringVar(value=str(saved_settings.get("conf", "0.03")))
    var_imgsz = tk.StringVar(value=str(saved_settings.get("imgsz", "1024")))
    var_device = tk.StringVar(value=str(saved_settings.get("device", "0")))
    
    def make_row(parent, label_text, var, browse_func=None, is_combo=False, combo_vals=None):
        frame = tk.Frame(parent, bg=BG_COLOR)
        frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame, text=label_text, font=FONT_LABEL, bg=BG_COLOR, fg=FG_COLOR, width=15, anchor="w").pack(side=tk.LEFT)
        
        if is_combo:
            cb = ttk.Combobox(frame, textvariable=var, values=combo_vals, state="readonly", font=FONT_INPUT)
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        else:
            tk.Entry(frame, textvariable=var, font=FONT_INPUT, bg="#181825", fg="#A6ADC8", insertbackground="white", bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=5)
            
        if browse_func:
            tk.Button(frame, text="Browse", command=browse_func, bg=BTN_BG, fg=FG_COLOR, bd=0, cursor="hand2").pack(side=tk.RIGHT, ipadx=5, ipady=2)
            
    make_row(root, "Mode:", var_mode, is_combo=True, combo_vals=["Count & Analyze", "Auto-Annotate"])
    tk.Frame(root, height=1, bg=BTN_BG).pack(fill=tk.X, padx=20, pady=10)
    
    make_row(root, "Input Folder:", var_input, lambda: var_input.set(filedialog.askdirectory() or var_input.get()))
    make_row(root, "Output Folder:", var_out, lambda: var_out.set(filedialog.askdirectory() or var_out.get()))
    make_row(root, "Weights (.pt):", var_weights, lambda: var_weights.set(filedialog.askopenfilename(filetypes=[("PyTorch Weights", "*.pt"), ("All Files", "*.*")]) or var_weights.get()))
    
    make_row(root, "Confidence:", var_conf)
    make_row(root, "Image Size:", var_imgsz)
    make_row(root, "Device:", var_device)
    
    # Pack the bottom frame FIRST so it always claims the bottom of the window
    bot_frame = tk.Frame(root, bg=BG_COLOR)
    bot_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=15)
    
    # Now pack the log frame which will expand to fill the REMAINING space
    log_frame = tk.Frame(root, bg="#11111B")
    log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    log_text = tk.Text(log_frame, bg="#11111B", fg="#A6ADC8", font=FONT_INPUT, bd=0, state=tk.DISABLED)
    log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def log_cb(message):
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, message + "\n")
        log_text.see(tk.END)
        log_text.config(state=tk.DISABLED)

    # Progress bar and Buttons inside the bottom frame
    progress_var = tk.DoubleVar()
    pb = ttk.Progressbar(bot_frame, variable=progress_var, maximum=100)
    pb.pack(fill=tk.X, pady=(0, 10))
    
    def progress_cb(current, total, msg):
        def update():
            progress_var.set((current / total) * 100)
            log_cb(msg)
        root.after(0, update)
        
    btn_frame = tk.Frame(bot_frame, bg=BG_COLOR)
    btn_frame.pack(fill=tk.X)
    
    btn_view = tk.Button(btn_frame, text="🔍 View Results", font=("Segoe UI", 11, "bold"), bg="#F9E2AF", fg="#11111B", bd=0, cursor="hand2", state=tk.DISABLED)
    btn_view.pack(side=tk.RIGHT, ipady=6, ipadx=10)
    
    def start_processing():
        btn_run.config(state=tk.DISABLED, text="Running...")
        btn_view.config(state=tk.DISABLED)
        log_text.config(state=tk.NORMAL)
        log_text.delete(1.0, tk.END)
        log_text.config(state=tk.DISABLED)
        progress_var.set(0)
        
        try:
            settings_to_save = {
                "mode": var_mode.get(),
                "input": var_input.get(),
                "output": var_out.get(),
                "weights": var_weights.get(),
                "conf": var_conf.get(),
                "imgsz": var_imgsz.get(),
                "device": var_device.get()
            }
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, "w") as f:
                json.dump(settings_to_save, f)
        except Exception as e:
            log_cb(f"Warning: Failed to save settings: {e}")
            
        def _thread():
            try:
                in_dir, out_dir = Path(var_input.get()), Path(var_out.get())
                images = collect_images(in_dir)
                if not images:
                    root.after(0, lambda: messagebox.showinfo("Info", "No images found."))
                    return
                
                root.after(0, lambda: log_cb(f"Found {len(images)} images.\nLoading model..."))
                model = YOLO(var_weights.get())
                
                summary = run_inference(
                    mode=var_mode.get(), model=model, image_paths=images,
                    out_dir=out_dir, conf=float(var_conf.get()), imgsz=int(var_imgsz.get()),
                    device=var_device.get(), progress_cb=progress_cb
                )
                
                if var_mode.get() == "Count & Analyze" and summary:
                    root.after(0, lambda: btn_view.config(text="🔍 View Results", state=tk.NORMAL, command=lambda: show_results_viewer(root, summary, in_dir, out_dir)))
                elif var_mode.get() == "Auto-Annotate":
                    import subprocess
                    def launch_annotator():
                        subprocess.Popen([sys.executable, "scripts/annotate.py", "--images", str(out_dir / "images" / "review"), "--labels", str(out_dir / "labels" / "review")])
                    root.after(0, lambda: btn_view.config(text="✏️ Open Annotator", state=tk.NORMAL, command=launch_annotator))
                
                root.after(0, lambda: log_cb(f"\nDone! Processed {len(images)} images.\nSaved to {out_dir}"))
            except Exception as e:
                root.after(0, lambda m=str(e): log_cb(f"\n[ERROR] {m}"))
            finally:
                root.after(0, lambda: btn_run.config(state=tk.NORMAL, text="🚀 Run Inference"))
                
        threading.Thread(target=_thread, daemon=True).start()

    btn_run = tk.Button(btn_frame, text="🚀 Run Inference", font=("Segoe UI", 11, "bold"), bg="#A6E3A1", fg="#11111B", bd=0, cursor="hand2", command=start_processing)
    btn_run.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10), ipady=6)

    root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Please run without arguments to use the GUI.")
    else:
        run_gui()
