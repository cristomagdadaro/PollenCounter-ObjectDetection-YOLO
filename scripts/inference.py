#!/usr/bin/env python3
"""Unified YOLO Inference (GUI).

Two modes:
  1. Count & Analyze — counts objects, exports .xlsx, saves annotated images.
  2. Auto-Annotate — saves raw YOLO .txt labels for active learning.

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
from src.theme import FONT_LABEL, FONT_INPUT

DEFAULT_WEIGHTS = get_latest_weights()


def run_inference(
    mode: str,
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
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup subfolders
    if mode == "Count & Analyze":
        annotated_dir = out_dir / "annotated"
        if save_annotated_imgs:
            annotated_dir.mkdir(parents=True, exist_ok=True)
    else:
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

        if mode == "Count & Analyze":
            # ── Count mode ───────────────────────────────────────────
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
                for det_idx, (box, conf_val) in enumerate(zip(xyxy_list, confidences)):
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
                    box_overlay = img.copy()
                    for box in xyxy_list:
                        x1, y1, x2, y2 = box
                        color = (random.randint(50, 255), random.randint(100, 255), random.randint(50, 255))
                        cv2.rectangle(box_overlay, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=3)
                    cv2.addWeighted(box_overlay, box_opacity, annotated_img, 1.0 - box_opacity, 0, annotated_img)

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
            # ── Auto-annotate mode ───────────────────────────────────
            shutil.copy2(img_path, aa_img_dir / img_path.name)
            label_path = aa_lbl_dir / f"{img_path.stem}.txt"
            with open(label_path, "w") as f:
                if n_detections > 0:
                    for cls_id, box in zip(cls_ids, xyxy_list):
                        x1, y1, x2, y2 = map(int, box)
                        pad_x = int((x2 - x1) * 0.3)
                        pad_y = int((y2 - y1) * 0.3)
                        
                        rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                        rx2, ry2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)
                        
                        if rx2 <= rx1 or ry2 <= ry1:
                            continue

                        # Snap to pollen edges via OpenCV contours
                        roi = img[ry1:ry2, rx1:rx2]
                        try:
                            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            
                            kernel = np.ones((3, 3), np.uint8)
                            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
                            
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            if contours:
                                orig_cx_roi = (x1 + x2) / 2.0 - rx1
                                orig_cy_roi = (y1 + y2) / 2.0 - ry1
                                
                                best_contour = None
                                min_dist = float('inf')
                                
                                for cnt in contours:
                                    if cv2.contourArea(cnt) < 10: continue
                                    br_x, br_y, br_w, br_h = cv2.boundingRect(cnt)
                                    cnt_cx = br_x + br_w / 2.0
                                    cnt_cy = br_y + br_h / 2.0
                                    
                                    dist = (cnt_cx - orig_cx_roi)**2 + (cnt_cy - orig_cy_roi)**2
                                    if dist < min_dist:
                                        min_dist = dist
                                        best_contour = cnt
                                
                                if best_contour is None:
                                    raise ValueError("No valid contour")
                                
                                cx, cy, cw, ch = cv2.boundingRect(best_contour)
                                pad_w, pad_h = int(cw * 0.08), int(ch * 0.08)
                                new_x1 = max(0, rx1 + cx - pad_w)
                                new_y1 = max(0, ry1 + cy - pad_h)
                                new_x2 = min(img_w, rx1 + cx + cw + pad_w)
                                new_y2 = min(img_h, ry1 + cy + ch + pad_h)

                                w_norm = (new_x2 - new_x1) / img_w
                                h_norm = (new_y2 - new_y1) / img_h
                                x_center_norm = (new_x1 + new_x2) / 2.0 / img_w
                                y_center_norm = (new_y1 + new_y2) / 2.0 / img_h
                            else:
                                raise ValueError("No contour")
                        except Exception:
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
    """Post-inference image viewer with zoom-on-hover."""
    if not summary_rows:
        return
    viewer = tk.Toplevel(parent)
    viewer.title("Results Viewer")
    viewer.geometry("1000x600")

    current_idx = [0]
    top_frame = ttk.Frame(viewer)
    top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

    btn_prev = ttk.Button(top_frame, text="<< Previous")
    btn_prev.pack(side=tk.LEFT, padx=10)

    def save_current_image():
        idx = current_idx[0]
        row = summary_rows[idx]
        img_name = row["filename"]
        annot_path = out_dir / "annotated" / img_name

        if not annot_path.exists():
            messagebox.showerror("Error", "Annotated image not found!")
            return

        default_name = f"{Path(img_name).stem}_annotated.jpg"
        save_path = filedialog.asksaveasfilename(
            title="Save Annotated Image",
            initialfile=default_name,
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")],
        )
        if save_path:
            shutil.copy(annot_path, save_path)

    btn_save = ttk.Button(top_frame, text="Save Image", command=save_current_image)
    btn_save.pack(side=tk.LEFT, padx=5)

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

    viewer_state = {
        "orig_full": None, "annot_full": None,
        "orig_tk": None, "annot_tk": None,
        "orig_scale": 1.0, "annot_scale": 1.0,
        "zoom_tk": None,
    }

    zoom_win = tk.Toplevel(viewer)
    zoom_win.withdraw()
    zoom_win.overrideredirect(True)
    zoom_lbl = tk.Label(zoom_win, bd=2, relief="solid", bg="#CCCCCC")
    zoom_lbl.pack()

    def handle_hover(event, img_type, lbl):
        if not zoom_enabled.get():
            return zoom_win.withdraw()
        full_img = viewer_state.get(f"{img_type}_full")
        tk_img = viewer_state.get(f"{img_type}_tk")
        if not full_img or not tk_img:
            return

        offset_x = (lbl.winfo_width() - tk_img.width()) // 2
        offset_y = (lbl.winfo_height() - tk_img.height()) // 2
        img_x, img_y = event.x - offset_x, event.y - offset_y
        if img_x < 0 or img_x > tk_img.width() or img_y < 0 or img_y > tk_img.height():
            return zoom_win.withdraw()

        scale = viewer_state[f"{img_type}_scale"]
        full_x, full_y = int(img_x * scale), int(img_y * scale)

        crop_size = 200
        box = (full_x - crop_size // 2, full_y - crop_size // 2, full_x + crop_size // 2, full_y + crop_size // 2)
        crop = full_img.crop(box).resize((crop_size * 2, crop_size * 2), getattr(Image, "Resampling", Image).LANCZOS)

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
        lbl_info.config(text=f"Image {idx + 1} of {len(summary_rows)}: {row['filename']}")
        lbl_meta.config(text=f"Count: {row['pollen_count']}   |   Avg Conf: {row['avg_confidence']:.2f}")

        orig_path = in_dir / row["filename"]
        annot_path = out_dir / "annotated" / row["filename"]

        viewer.update_idletasks()
        w, h = max(400, (viewer.winfo_width() - 40) // 2), max(400, viewer.winfo_height() - 150)

        def load_img(path, img_type):
            try:
                full = Image.open(path)
                viewer_state[f"{img_type}_full"] = full
                copy = full.copy()
                copy.thumbnail((w, h), getattr(Image, "Resampling", Image).LANCZOS)
                viewer_state[f"{img_type}_tk"] = ImageTk.PhotoImage(copy)
                target = lbl_orig if img_type == "orig" else lbl_annot
                target.config(image=viewer_state[f"{img_type}_tk"], text="")
                viewer_state[f"{img_type}_scale"] = full.width / copy.width
            except Exception:
                target = lbl_orig if img_type == "orig" else lbl_annot
                target.config(image="", text="Not found")

        load_img(orig_path, "orig")
        load_img(annot_path, "annot")

        btn_prev.config(state=tk.NORMAL if idx > 0 else tk.DISABLED)
        btn_next.config(state=tk.NORMAL if idx < len(summary_rows) - 1 else tk.DISABLED)

    btn_prev.config(command=lambda: (current_idx.__setitem__(0, current_idx[0] - 1), update_view()))
    btn_next.config(command=lambda: (current_idx.__setitem__(0, current_idx[0] + 1), update_view()))
    viewer.after(100, update_view)


def run_gui():
    root = tk.Tk()
    root.title("Unified Inference Tool")
    root.geometry("650x700")
    root.configure(bg="#1E1E2E")

    BG = "#FFFFFF"
    FG = "#000000"
    ACCENT_CLR = "#0000FF"
    BTN_BG = "#CCCCCC"

    tk.Label(root, text="Unified YOLO Inference", font=("Segoe UI", 14, "bold"), bg=BG, fg=ACCENT_CLR).pack(pady=10)

    saved = load_settings()

    detect_dir = PROJECT_ROOT / "runs" / "detect"
    available_models = [d.name for d in detect_dir.iterdir() if d.is_dir() and (d / "weights" / "best.pt").exists()]
    if not available_models:
        available_models = ["No models found"]
    
    default_weight_name = DEFAULT_WEIGHTS.parent.parent.name if DEFAULT_WEIGHTS else (available_models[-1] if available_models else "")

    var_mode = tk.StringVar(value=saved.get("mode", "Count & Analyze"))
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
        frame = tk.Frame(parent, bg=BG)
        frame.pack(fill=tk.X, padx=20, pady=5)
        tk.Label(frame, text=label_text, font=FONT_LABEL, bg=BG, fg=FG, width=15, anchor="w").pack(side=tk.LEFT)
        if is_combo:
            cb = ttk.Combobox(frame, textvariable=var, values=combo_vals, state="readonly", font=FONT_INPUT)
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        else:
            tk.Entry(frame, textvariable=var, font=FONT_INPUT, bg="#F0F0F0", fg="#000000", insertbackground="black", bd=0).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=5)
        if browse_func:
            tk.Button(frame, text="Browse", command=browse_func, bg=BTN_BG, fg=FG, bd=0, cursor="hand2").pack(side=tk.RIGHT, ipadx=5, ipady=2)

    make_row(root, "Mode:", var_mode, is_combo=True, combo_vals=["Count & Analyze", "Auto-Annotate"])
    tk.Frame(root, height=1, bg=BTN_BG).pack(fill=tk.X, padx=20, pady=10)

    make_row(root, "Input Folder:", var_input, lambda: var_input.set(filedialog.askdirectory() or var_input.get()))
    make_row(root, "Output Folder:", var_out, lambda: var_out.set(filedialog.askdirectory() or var_out.get()))
    make_row(root, "Weights (.pt):", var_weights, is_combo=True, combo_vals=available_models)

    make_row(root, "Confidence:", var_conf)
    make_row(root, "IoU Threshold:", var_iou)
    make_row(root, "Image Size:", var_imgsz)
    make_row(root, "Device:", var_device)

    # SAHI Frame
    frame_sahi = tk.Frame(root, bg=BG)
    frame_sahi.pack(fill=tk.X, padx=20, pady=5)
    tk.Label(frame_sahi, text="SAHI (High Acc):", font=FONT_LABEL, bg=BG, fg=FG, width=15, anchor="w").pack(side=tk.LEFT)
    tk.Checkbutton(frame_sahi, text="Enable Sliced Inference", variable=var_use_sahi, bg=BG, fg=FG, activebackground=BG, selectcolor=BG).pack(side=tk.LEFT)
    tk.Label(frame_sahi, text="Slice Size:", font=FONT_LABEL, bg=BG, fg=FG).pack(side=tk.LEFT, padx=(10, 5))
    tk.Entry(frame_sahi, textvariable=var_sahi_slice, font=FONT_INPUT, bg="#F0F0F0", fg="#000000", width=8, bd=0).pack(side=tk.LEFT, ipady=4)

    frame_op = tk.Frame(root, bg=BG)
    frame_op.pack(fill=tk.X, padx=20, pady=5)
    tk.Label(frame_op, text="Box Opacity:", font=FONT_LABEL, bg=BG, fg=FG, width=15, anchor="w").pack(side=tk.LEFT)
    tk.Scale(frame_op, from_=0.0, to=1.0, resolution=0.1, orient=tk.HORIZONTAL, variable=var_opacity, bg=BG, fg=FG, highlightthickness=0, bd=0, activebackground=BTN_BG).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    # Bottom frame (buttons + progress)
    bot_frame = tk.Frame(root, bg=BG)
    bot_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=15)

    # Log frame
    log_frame = tk.Frame(root, bg="#F0F0F0")
    log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    log_text = tk.Text(log_frame, bg="#F0F0F0", fg="#000000", font=FONT_INPUT, bd=0, state=tk.DISABLED)
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

    btn_frame = tk.Frame(bot_frame, bg=BG)
    btn_frame.pack(fill=tk.X)

    def open_annotator():
        import subprocess
        out = Path(var_out.get())
        subprocess.Popen([sys.executable, "scripts/annotate.py", "--images", str(out / "images" / "review"), "--labels", str(out / "labels" / "review")])

    def open_viewer():
        out = Path(var_out.get())
        in_dir = Path(var_input.get())
        csv_path = out / "pollen_counts.xlsx"
        if not csv_path.exists():
            messagebox.showinfo("Not Found", "No pollen_counts.xlsx found.\nRun 'Count & Analyze' first!")
            return
        try:
            summary = pd.read_excel(csv_path, sheet_name="Summary").to_dict("records")
            show_results_viewer(root, summary, in_dir, out)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load results: {e}")

    btn_open_annot = tk.Button(btn_frame, text="Annotator", font=("Segoe UI", 11, "bold"), bg="#F9E2AF", fg="#11111B", bd=0, cursor="hand2", command=open_annotator)
    btn_open_annot.pack(side=tk.RIGHT, ipady=6, ipadx=5)

    btn_open_view = tk.Button(btn_frame, text="Viewer", font=("Segoe UI", 11, "bold"), bg="#F9E2AF", fg="#11111B", bd=0, cursor="hand2", command=open_viewer)
    btn_open_view.pack(side=tk.RIGHT, ipady=6, ipadx=5, padx=(0, 10))

    def start_processing():
        btn_run.config(state=tk.DISABLED, text="Running...")
        log_text.config(state=tk.NORMAL)
        log_text.delete(1.0, tk.END)
        log_text.config(state=tk.DISABLED)
        progress_var.set(0)

        # Save settings
        save_settings({
            "mode": var_mode.get(),
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
                    mode=var_mode.get(), model=model, image_paths=images,
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

    btn_run = tk.Button(btn_frame, text="Run Inference", font=("Segoe UI", 11, "bold"), bg="#A6E3A1", fg="#11111B", bd=0, cursor="hand2", command=start_processing)
    btn_run.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 10), ipady=6)

    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Please run without arguments to use the GUI.")
    else:
        run_gui()
