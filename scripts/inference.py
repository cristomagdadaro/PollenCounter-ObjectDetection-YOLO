#!/usr/bin/env python3
"""Unified YOLO Inference (GUI).

Usage:
    python scripts/inference.py
"""

from __future__ import annotations

import random
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO

from src.paths import PROJECT_ROOT, RAW_IMAGES, DEFAULT_OUTPUT, IMAGE_EXTS
from src.model_utils import get_latest_weights, collect_images
from src.settings import load_settings, save_settings
from src.theme import BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR, FONT_MAIN, FONT_HEADER, FONT_LABEL

DEFAULT_WEIGHTS = get_latest_weights()


def run_inference(
    model: YOLO,
    image_paths: list[Path],
    out_dir: Path,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    save_annotated_imgs: bool = True,
    box_opacity: float = 0.3,
    progress_cb=None,
    log_cb=None,
    use_sahi: bool = False,
    sahi_model=None,
    sahi_slice_size: int = 512,
):
    """Run YOLO inference in either 'Count & Analyze' or 'Auto-Annotate' mode."""
    out_dir.mkdir(parents=True, exist_ok=True)

    aa_img_dir = out_dir / "images" / "review"
    aa_lbl_dir = out_dir / "labels" / "review"
    aa_img_dir.mkdir(parents=True, exist_ok=True)
    aa_lbl_dir.mkdir(parents=True, exist_ok=True)

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        if progress_cb:
            progress_cb(idx, total, f"Processing [{idx}/{total}]: {img_path.name}")

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        if use_sahi and sahi_model is not None:
            from sahi.predict import get_sliced_prediction
            sahi_result = get_sliced_prediction(
                str(img_path),
                sahi_model,
                slice_height=sahi_slice_size,
                slice_width=sahi_slice_size,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2,
                postprocess_type="NMM",
                postprocess_match_metric="IOU",
                postprocess_match_threshold=iou,
            )
            obj_list = sahi_result.object_prediction_list
            n_detections = len(obj_list)
            confidences = [obj.score.value for obj in obj_list] if n_detections > 0 else []
            xyxy_list = [[obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy] for obj in obj_list]
            cls_ids = [obj.category.id for obj in obj_list]
        else:
            results = model.predict(
                source=str(img_path), conf=conf, iou=iou, agnostic_nms=True,
                imgsz=imgsz, device=device, max_det=5000, verbose=False,
            )
            boxes = results[0].boxes
            n_detections = len(boxes)
            confidences = boxes.conf.cpu().tolist() if n_detections > 0 else []
            xyxy_list = boxes.xyxy.cpu().tolist() if n_detections > 0 else []
            cls_ids = boxes.cls.cpu().tolist() if n_detections > 0 else []

        # ── Auto-annotate mode ───────────────────────────────────
        shutil.copy2(img_path, aa_img_dir / img_path.name)
        label_path = aa_lbl_dir / f"{img_path.stem}.txt"
        with open(label_path, "w") as f:
            if n_detections > 0:
                for cls_id, box, conf_val in zip(cls_ids, xyxy_list, confidences):
                    x1, y1, x2, y2 = box
                    w_norm = (x2 - x1) / img_w
                    h_norm = (y2 - y1) / img_h
                    x_center_norm = (x1 + x2) / 2.0 / img_w
                    y_center_norm = (y1 + y2) / 2.0 / img_h
                    f.write(f"{int(cls_id)} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f} {conf_val:.6f}\n")





def run_gui():
    root = tk.Tk()
    root.title("Unified Inference Tool")
    root.geometry("650x700")
    root.configure(bg=BG_COLOR)

    tk.Label(root, text="Unified YOLO Inference", font=FONT_HEADER, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=(20, 10))

    saved = load_settings()

    detect_dir = PROJECT_ROOT / "runs" / "detect"
    available_models = [d.name for d in detect_dir.iterdir() if d.is_dir() and (d / "weights" / "best.pt").exists()]
    if not available_models:
        available_models = ["No models found"]
    
    default_weight_name = DEFAULT_WEIGHTS.parent.parent.name if DEFAULT_WEIGHTS else (available_models[-1] if available_models else "")

    default_weight_name = DEFAULT_WEIGHTS.parent.parent.name if DEFAULT_WEIGHTS else (available_models[-1] if available_models else "")
    var_input = tk.StringVar(value=saved.get("input", str(RAW_IMAGES)))
    var_out = tk.StringVar(value=saved.get("output", str(DEFAULT_OUTPUT)))
    
    saved_weights = saved.get("weights", "")
    if saved_weights not in available_models and available_models:
        saved_weights = default_weight_name
    var_weights = tk.StringVar(value=saved_weights)
    
    var_conf = tk.StringVar(value=str(saved.get("conf", "0.03")))
    var_iou = tk.StringVar(value=str(saved.get("iou", "0.40")))
    var_imgsz = tk.StringVar(value=str(saved.get("imgsz", "1024")))
    var_device = tk.StringVar(value=str(saved.get("device", "0")))
    var_opacity = tk.DoubleVar(value=float(saved.get("opacity", 0.3)))
    var_use_sahi = tk.BooleanVar(value=saved.get("use_sahi", False))
    var_sahi_slice = tk.StringVar(value=str(saved.get("sahi_slice", "512")))

    def make_row(parent, label_text, var, browse_func=None, is_combo=False, combo_vals=None):
        frame = tk.Frame(parent, bg=BG_COLOR)
        frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame, text=label_text, font=FONT_LABEL, bg=BG_COLOR, fg=TEXT_COLOR, width=15, anchor="w").pack(side=tk.LEFT)
        if is_combo:
            cb = ttk.Combobox(frame, textvariable=var, values=combo_vals, state="readonly", font=FONT_MAIN)
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        else:
            tk.Entry(frame, textvariable=var, font=FONT_MAIN, bg=SIDEBAR_BG, fg=TEXT_COLOR, insertbackground=TEXT_COLOR, bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=5)
        if browse_func:
            tk.Button(frame, text="Browse", command=browse_func, bg="#E2E8F0", fg=TEXT_COLOR, font=FONT_MAIN, bd=0, cursor="hand2").pack(side=tk.RIGHT, ipadx=10, ipady=4)

    make_row(root, "Input Folder:", var_input, lambda: var_input.set(filedialog.askdirectory() or var_input.get()))
    make_row(root, "Output Folder:", var_out, lambda: var_out.set(filedialog.askdirectory() or var_out.get()))
    make_row(root, "Weights (.pt):", var_weights, is_combo=True, combo_vals=available_models)

    make_row(root, "Confidence:", var_conf)
    make_row(root, "IoU Threshold:", var_iou)
    make_row(root, "Image Size:", var_imgsz)
    make_row(root, "Device:", var_device)

    # SAHI Frame
    frame_sahi = tk.Frame(root, bg=BG_COLOR)
    frame_sahi.pack(fill=tk.X, padx=20, pady=5)
    tk.Label(frame_sahi, text="SAHI (High Acc):", font=FONT_LABEL, bg=BG_COLOR, fg=TEXT_COLOR, width=15, anchor="w").pack(side=tk.LEFT)
    tk.Checkbutton(frame_sahi, text="Enable Sliced Inference", variable=var_use_sahi, bg=BG_COLOR, fg=TEXT_COLOR, activebackground=BG_COLOR, selectcolor=BG_COLOR, font=FONT_MAIN).pack(side=tk.LEFT)
    tk.Label(frame_sahi, text="Slice Size:", font=FONT_LABEL, bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=(10, 5))
    tk.Entry(frame_sahi, textvariable=var_sahi_slice, font=FONT_MAIN, bg=SIDEBAR_BG, fg=TEXT_COLOR, width=8, bd=0).pack(side=tk.LEFT, ipady=4)

    frame_op = tk.Frame(root, bg=BG_COLOR)
    frame_op.pack(fill=tk.X, padx=20, pady=5)
    tk.Label(frame_op, text="Box Opacity:", font=FONT_LABEL, bg=BG_COLOR, fg=TEXT_COLOR, width=15, anchor="w").pack(side=tk.LEFT)
    tk.Scale(frame_op, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=var_opacity, bg=BG_COLOR, fg=TEXT_COLOR, highlightthickness=0, bd=0, activebackground=ACCENT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    # Bottom frame (buttons + progress)
    bot_frame = tk.Frame(root, bg=BG_COLOR)
    bot_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=15)

    # Log frame
    log_frame = tk.Frame(root, bg=BG_COLOR)
    log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    log_text = tk.Text(log_frame, bg=SIDEBAR_BG, fg=TEXT_COLOR, font=FONT_MAIN, bd=1, relief="solid", highlightthickness=0, state=tk.DISABLED)
    log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def log_cb(message):
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, message + "\n")
        log_text.see(tk.END)
        log_text.config(state=tk.DISABLED)

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

    def open_annotator():
        import subprocess
        out = Path(var_out.get())
        subprocess.Popen([sys.executable, "scripts/annotate.py", "--images", str(out / "images" / "review"), "--labels", str(out / "labels" / "review")])

    btn_open_annot = tk.Button(btn_frame, text="Annotator", font=FONT_LABEL, bg="#F59E0B", fg="white", bd=0, cursor="hand2", command=open_annotator)
    btn_open_annot.pack(side=tk.RIGHT, ipady=8, ipadx=10, padx=(0, 10))

    def start_processing():
        btn_run.config(state=tk.DISABLED, text="Running...")
        log_text.config(state=tk.NORMAL)
        log_text.delete(1.0, tk.END)
        log_text.config(state=tk.DISABLED)
        progress_var.set(0)

        # Save settings
        save_settings({
            "input": var_input.get(),
            "output": var_out.get(),
            "weights": var_weights.get(),
            "conf": var_conf.get(),
            "iou": var_iou.get(),
            "imgsz": var_imgsz.get(),
            "device": var_device.get(),
            "opacity": var_opacity.get(),
            "use_sahi": var_use_sahi.get(),
            "sahi_slice": var_sahi_slice.get(),
        })

        def _thread():
            try:
                in_dir, out_dir = Path(var_input.get()), Path(var_out.get())
                images = collect_images(in_dir)
                if not images:
                    root.after(0, lambda: messagebox.showinfo("Info", "No images found."))
                    return

                root.after(0, lambda: log_cb(f"Found {len(images)} images.\nLoading model..."))
                weight_path = str(PROJECT_ROOT / "runs" / "detect" / var_weights.get() / "weights" / "best.pt")
                model = YOLO(weight_path)

                use_sahi = var_use_sahi.get()
                sahi_model = None
                if use_sahi:
                    from sahi import AutoDetectionModel
                    sahi_model = AutoDetectionModel.from_pretrained(
                        model_type="yolov8",
                        model_path=weight_path,
                        confidence_threshold=float(var_conf.get()),
                        device=var_device.get(),
                    )

                summary = run_inference(
                    model=model, image_paths=images,
                    out_dir=out_dir, conf=float(var_conf.get()), iou=float(var_iou.get()), imgsz=int(var_imgsz.get()),
                    device=var_device.get(),
                    box_opacity=float(var_opacity.get()), progress_cb=progress_cb,
                    use_sahi=use_sahi, sahi_model=sahi_model, sahi_slice_size=int(var_sahi_slice.get())
                )

                root.after(0, lambda: log_cb(f"\nDone! Processed {len(images)} images.\nSaved to {out_dir}"))
            except Exception as e:
                root.after(0, lambda m=str(e): log_cb(f"\n[ERROR] {m}"))
            finally:
                root.after(0, lambda: btn_run.config(state=tk.NORMAL, text="Run Inference"))

        threading.Thread(target=_thread, daemon=True).start()

    btn_run = tk.Button(btn_frame, text="Run Inference", font=FONT_LABEL, bg=ACCENT, fg="white", bd=0, cursor="hand2", command=start_processing)
    btn_run.pack(side=tk.RIGHT, ipady=8, ipadx=20)

    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Please run without arguments to use the GUI.")
    else:
        run_gui()
