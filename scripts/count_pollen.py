#!/usr/bin/env python3
"""
count_pollen.py — High-Volume Automated Pollen Counting (GUI & CLI)
===================================================================

Batch-inference script that:
  1. Loads a trained YOLOv11s model.
  2. Iterates over every image in the input folder.
  3. Counts YOLO-generated bounding boxes per image.
  4. Exports a structured data log to an .xlsx spreadsheet.
  5. Optionally saves annotated images with colored dots on detections.

If run without arguments, launches a Tkinter GUI.
"""

from __future__ import annotations

import argparse
import sys
import threading
import random
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
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "real"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "detect" / "real"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Count pollen grains in microscopy images using YOLOv11s.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT),
        help="Folder containing images to process.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Folder to save the .xlsx report (and annotated images).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained YOLO weights (.pt file).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Inference image size (pixels).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: '0' for GPU, 'cpu' for CPU inference.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save annotated images with bounding boxes drawn.",
    )
    parser.add_argument(
        "--xlsx-name",
        type=str,
        default="pollen_counts.xlsx",
        help="Output spreadsheet filename.",
    )
    return parser.parse_args()


def collect_images(folder: Path) -> list[Path]:
    """Recursively collect all image files in *folder*."""
    images = sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    return images


def run_inference(
    model: YOLO,
    image_paths: list[Path],
    conf: float,
    imgsz: int,
    device: str,
    save_annotated: bool,
    annotated_dir: Path,
    progress_cb=None,
) -> tuple[list[dict], list[dict]]:
    """
    Run YOLO inference on each image.
    """
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        msg = f"Processing [{idx}/{total}]: {img_path.name}"
        if progress_cb:
            progress_cb(idx, total, msg)
        else:
            print(f"  [{idx}/{total}] Processing: {img_path.name} … ", end="", flush=True)

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            if not progress_cb: print("SKIPPED (unreadable)")
            continue
        img_h, img_w = img.shape[:2]

        # Inference
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=0.5,  # stricter IOU suppression for overlapping grains
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes

        n_detections = len(boxes)
        confidences = boxes.conf.cpu().tolist() if n_detections > 0 else []

        # Summary
        summary_rows.append(
            {
                "filename": img_path.name,
                "pollen_count": n_detections,
                "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
                "min_confidence": round(min(confidences), 4) if confidences else 0.0,
                "max_confidence": round(max(confidences), 4) if confidences else 0.0,
                "image_width": img_w,
                "image_height": img_h,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )

        # Details & Annotation
        if n_detections > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            for det_idx, (box, conf_val) in enumerate(zip(xyxy, confidences)):
                x1, y1, x2, y2 = box
                detail_rows.append(
                    {
                        "filename": img_path.name,
                        "detection_id": det_idx + 1,
                        "x_center": round((x1 + x2) / 2, 2),
                        "y_center": round((y1 + y2) / 2, 2),
                        "width": round(x2 - x1, 2),
                        "height": round(y2 - y1, 2),
                        "confidence": round(conf_val, 4),
                    }
                )

        if save_annotated:
            annotated_img = img.copy()
            if n_detections > 0:
                xyxy = boxes.xyxy.cpu().tolist()
                for box in xyxy:
                    x1, y1, x2, y2 = box
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    # Random bright color
                    color = (random.randint(50, 255), random.randint(100, 255), random.randint(50, 255))
                    # Draw a filled circle (dot) and a white outline
                    cv2.circle(annotated_img, (cx, cy), radius=6, color=color, thickness=-1)
                    cv2.circle(annotated_img, (cx, cy), radius=6, color=(255, 255, 255), thickness=1)

            # Draw the count text with 50% opacity at the center
            overlay = annotated_img.copy()
            text = str(n_detections)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # In OpenCV, a font_scale of 1.0 is ~22 pixels high.
            # For ~30px font, scale = 30 / 22 = 1.36
            font_scale = 1.36
            thickness = 3
            
            # Get text size to center it perfectly
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (img_w - text_w) // 2
            text_y = (img_h + text_h) // 2
            
            # Draw on overlay (black outline for visibility, then white text)
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
            # Blend with 50% opacity
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0, annotated_img)

            out_path = annotated_dir / img_path.name
            cv2.imwrite(str(out_path), annotated_img)

        if not progress_cb:
            print(f"{n_detections} pollen grain(s) detected")

    return summary_rows, detail_rows


def export_xlsx(
    summary_rows: list[dict],
    detail_rows: list[dict],
    output_path: Path,
) -> None:
    """Write summary and detail DataFrames to a single .xlsx workbook."""
    df_summary = pd.DataFrame(summary_rows)
    df_detail = pd.DataFrame(detail_rows)

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        if not df_detail.empty:
            df_detail.to_excel(writer, sheet_name="Detections", index=False)


def show_results_viewer(parent, summary_rows, in_dir, out_dir):
    if not summary_rows:
        messagebox.showinfo("Info", "No results to display.")
        return
        
    viewer = tk.Toplevel(parent)
    viewer.title("Results Viewer")
    viewer.geometry("1000x600")
    
    current_idx = [0]
    
    top_frame = ttk.Frame(viewer)
    top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
    
    btn_prev = ttk.Button(top_frame, text="<< Previous")
    btn_prev.pack(side=tk.LEFT, padx=10)
    
    zoom_enabled = tk.BooleanVar(value=True)
    chk_zoom = ttk.Checkbutton(top_frame, text="Zoom on Hover", variable=zoom_enabled)
    chk_zoom.pack(side=tk.LEFT, padx=10)
    
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
        "orig_full": None,
        "annot_full": None,
        "orig_tk": None,
        "annot_tk": None,
        "orig_scale": 1.0,
        "annot_scale": 1.0,
        "zoom_tk": None,
    }
    
    zoom_win = tk.Toplevel(viewer)
    zoom_win.withdraw()
    zoom_win.overrideredirect(True)
    zoom_lbl = tk.Label(zoom_win, bd=2, relief="solid", bg="black")
    zoom_lbl.pack()
    
    def handle_hover(event, img_type, lbl):
        if not zoom_enabled.get():
            zoom_win.withdraw()
            return
            
        full_img = viewer_state.get(f"{img_type}_full")
        tk_img = viewer_state.get(f"{img_type}_tk")
        if not full_img or not tk_img:
            return
            
        lbl_w = lbl.winfo_width()
        lbl_h = lbl.winfo_height()
        
        img_w = tk_img.width()
        img_h = tk_img.height()
        
        offset_x = (lbl_w - img_w) // 2
        offset_y = (lbl_h - img_h) // 2
        
        img_x = event.x - offset_x
        img_y = event.y - offset_y
        
        if img_x < 0 or img_x > img_w or img_y < 0 or img_y > img_h:
            zoom_win.withdraw()
            return
            
        scale = viewer_state[f"{img_type}_scale"]
        full_x = int(img_x * scale)
        full_y = int(img_y * scale)
        
        crop_size = 200
        box = (full_x - crop_size//2, full_y - crop_size//2, full_x + crop_size//2, full_y + crop_size//2)
        crop = full_img.crop(box)
        
        resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
        crop = crop.resize((crop_size * 2, crop_size * 2), resample_filter)
        
        viewer_state["zoom_tk"] = ImageTk.PhotoImage(crop)
        zoom_lbl.config(image=viewer_state["zoom_tk"])
        
        cx = event.x_root + 20
        cy = event.y_root + 20
        zoom_win.geometry(f"+{cx}+{cy}")
        zoom_win.deiconify()
        zoom_win.lift()
        
    lbl_orig.bind("<Motion>", lambda e: handle_hover(e, "orig", lbl_orig))
    lbl_orig.bind("<Leave>", lambda e: zoom_win.withdraw())
    
    lbl_annot.bind("<Motion>", lambda e: handle_hover(e, "annot", lbl_annot))
    lbl_annot.bind("<Leave>", lambda e: zoom_win.withdraw())
    
    def update_view():
        idx = current_idx[0]
        row = summary_rows[idx]
        filename = row["filename"]
        
        lbl_info.config(text=f"Image {idx+1} of {len(summary_rows)}: {filename}")
        
        meta_text = (f"Count: {row['pollen_count']}   |   "
                     f"Avg Conf: {row['avg_confidence']:.2f}   |   "
                     f"Min Conf: {row['min_confidence']:.2f}   |   "
                     f"Max Conf: {row['max_confidence']:.2f}")
        lbl_meta.config(text=meta_text)
        
        orig_path = in_dir / filename
        annot_path = out_dir / "annotated" / filename
        
        viewer.update_idletasks()
        w = max(400, (viewer.winfo_width() - 40) // 2)
        h = max(400, viewer.winfo_height() - 150)
        
        resample_filter = getattr(Image, 'Resampling', Image).LANCZOS
        
        try:
            viewer_state["orig_full"] = Image.open(orig_path)
            orig_copy = viewer_state["orig_full"].copy()
            orig_copy.thumbnail((w, h), resample_filter)
            viewer_state["orig_tk"] = ImageTk.PhotoImage(orig_copy)
            lbl_orig.config(image=viewer_state["orig_tk"], text="")
            viewer_state["orig_scale"] = viewer_state["orig_full"].width / orig_copy.width
        except Exception as e:
            lbl_orig.config(image="", text="Original not found")
            viewer_state["orig_full"] = None
            
        try:
            viewer_state["annot_full"] = Image.open(annot_path)
            annot_copy = viewer_state["annot_full"].copy()
            annot_copy.thumbnail((w, h), resample_filter)
            viewer_state["annot_tk"] = ImageTk.PhotoImage(annot_copy)
            lbl_annot.config(image=viewer_state["annot_tk"], text="")
            viewer_state["annot_scale"] = viewer_state["annot_full"].width / annot_copy.width
        except Exception as e:
            lbl_annot.config(image="", text="Annotated not found")
            viewer_state["annot_full"] = None
            
        btn_prev.config(state=tk.NORMAL if idx > 0 else tk.DISABLED)
        btn_next.config(state=tk.NORMAL if idx < len(summary_rows)-1 else tk.DISABLED)
        
    def go_prev():
        if current_idx[0] > 0:
            current_idx[0] -= 1
            update_view()
            
    def go_next():
        if current_idx[0] < len(summary_rows) - 1:
            current_idx[0] += 1
            update_view()
            
    btn_prev.config(command=go_prev)
    btn_next.config(command=go_next)
    
    viewer.after(100, update_view)


def run_cli(args):
    """Run via command line"""
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    weights_path = Path(args.weights)

    if not input_dir.is_dir():
        print(f"[ERROR] Input folder not found: {input_dir}")
        sys.exit(1)

    image_paths = collect_images(input_dir)
    if not image_paths:
        print(f"[ERROR] No images found in {input_dir}")
        sys.exit(1)

    if weights_path.exists():
        model = YOLO(str(weights_path))
    else:
        model = YOLO("yolo26n.pt")

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"
    if args.save_images:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Processing {len(image_paths)} image(s) …\n")
    summary_rows, detail_rows = run_inference(
        model=model,
        image_paths=image_paths,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save_annotated=args.save_images,
        annotated_dir=annotated_dir,
    )

    xlsx_path = output_dir / args.xlsx_name
    export_xlsx(summary_rows, detail_rows, xlsx_path)


def run_gui():
    """Run with Tkinter GUI"""
    root = tk.Tk()
    root.title("Pollen Counter (YOLOv11s)")
    root.geometry("600x480")
    
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')
        
    in_var = tk.StringVar(value=str(DEFAULT_INPUT))
    out_var = tk.StringVar(value=str(DEFAULT_OUTPUT))
    conf_var = tk.DoubleVar(value=0.25)
    save_img_var = tk.BooleanVar(value=True)
    device_var = tk.StringVar(value="0") # GPU default
    
    frame = ttk.Frame(root, padding="15")
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Input
    ttk.Label(frame, text="Input Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
    ttk.Entry(frame, textvariable=in_var, width=45).grid(row=0, column=1, padx=5)
    ttk.Button(frame, text="Browse", command=lambda: in_var.set(filedialog.askdirectory() or in_var.get())).grid(row=0, column=2)
    
    # Output
    ttk.Label(frame, text="Output Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
    ttk.Entry(frame, textvariable=out_var, width=45).grid(row=1, column=1, padx=5)
    ttk.Button(frame, text="Browse", command=lambda: out_var.set(filedialog.askdirectory() or out_var.get())).grid(row=1, column=2)
    
    # Confidence
    ttk.Label(frame, text="Confidence:").grid(row=2, column=0, sticky=tk.W, pady=5)
    conf_frame = ttk.Frame(frame)
    conf_frame.grid(row=2, column=1, sticky=tk.W, padx=5)
    ttk.Scale(conf_frame, from_=0.01, to=1.0, variable=conf_var, orient=tk.HORIZONTAL, length=200).pack(side=tk.LEFT)
    conf_label = ttk.Label(conf_frame, text="0.25")
    conf_label.pack(side=tk.LEFT, padx=5)
    def update_conf_lbl(*args): conf_label.config(text=f"{conf_var.get():.2f}")
    conf_var.trace_add("write", update_conf_lbl)
    
    # Checkboxes
    opt_frame = ttk.Frame(frame)
    opt_frame.grid(row=3, column=1, sticky=tk.W, pady=5)
    ttk.Checkbutton(opt_frame, text="Save Annotated Images (Dots Only)", variable=save_img_var).pack(side=tk.LEFT, padx=(0, 15))
    
    # Device
    ttk.Label(frame, text="Device:").grid(row=4, column=0, sticky=tk.W, pady=5)
    device_cb = ttk.Combobox(frame, textvariable=device_var, values=["0", "cpu"], state="readonly", width=10)
    device_cb.grid(row=4, column=1, sticky=tk.W, padx=5)
    ttk.Label(frame, text="(0 = GPU, cpu = CPU)").grid(row=4, column=1, sticky=tk.W, padx=(100, 0))
    
    # Log text
    log_text = tk.Text(frame, height=10, width=60, state=tk.DISABLED, bg="#f0f0f0")
    log_text.grid(row=5, column=0, columnspan=3, pady=10, sticky=tk.EW)
    
    def log(msg):
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)
        log_text.config(state=tk.DISABLED)
        
    # Progress
    progress_var = tk.DoubleVar(value=0)
    pb = ttk.Progressbar(frame, variable=progress_var, maximum=100)
    pb.grid(row=6, column=0, columnspan=3, sticky=tk.EW, pady=5)
    
    def progress_cb(current, total, msg):
        def update():
            progress_var.set((current / total) * 100)
            log(msg)
        root.after(0, update)
        
    # Run logic
    def run_process():
        in_dir = Path(in_var.get())
        out_dir = Path(out_var.get())
        conf = conf_var.get()
        save_img = save_img_var.get()
        dev = device_var.get()
        
        if not in_dir.exists():
            messagebox.showerror("Error", "Input directory does not exist.")
            return
            
        btn_run.config(state=tk.DISABLED)
        btn_view.config(state=tk.DISABLED)
        log_text.config(state=tk.NORMAL)
        log_text.delete(1.0, tk.END)
        log_text.config(state=tk.DISABLED)
        progress_var.set(0)
        
        def task():
            try:
                log("Starting pollen counting...")
                images = collect_images(in_dir)
                if not images:
                    root.after(0, lambda: messagebox.showinfo("Info", "No images found."))
                    return
                    
                log(f"Found {len(images)} images.")
                weights_path = DEFAULT_WEIGHTS
                if not weights_path.exists():
                    log(f"Warning: {weights_path.name} not found. Using pretrained yolo26n.pt")
                    weights_path = "yolo26n.pt"
                    
                actual_dev = dev
                import torch
                if actual_dev == "0" and not torch.cuda.is_available():
                    log("Warning: GPU not available. Falling back to CPU.")
                    actual_dev = "cpu"
                    
                model = YOLO(str(weights_path))
                
                out_dir.mkdir(parents=True, exist_ok=True)
                annotated_dir = out_dir / "annotated"
                if save_img:
                    annotated_dir.mkdir(parents=True, exist_ok=True)
                    
                summary, detail = run_inference(
                    model=model,
                    image_paths=images,
                    conf=conf,
                    imgsz=1024,
                    device=actual_dev,
                    save_annotated=save_img,
                    annotated_dir=annotated_dir,
                    progress_cb=progress_cb
                )
                
                xlsx_path = out_dir / "pollen_counts.xlsx"
                export_xlsx(summary, detail, xlsx_path)
                
                def on_finish():
                    log(f"\nFinished! Processed {len(images)} images.\nResults saved to: {out_dir}")
                    btn_view.config(state=tk.NORMAL, command=lambda: show_results_viewer(root, summary, in_dir, out_dir))
                
                root.after(0, on_finish)
            except Exception as e:
                err_msg = str(e)
                root.after(0, lambda m=err_msg: messagebox.showerror("Error", m))
                root.after(0, lambda m=err_msg: log(f"Error: {m}"))
            finally:
                root.after(0, lambda: btn_run.config(state=tk.NORMAL))
                
        threading.Thread(target=task, daemon=True).start()
        
    btn_frame = ttk.Frame(frame)
    btn_frame.grid(row=7, column=0, columnspan=3, pady=10)
    
    btn_run = ttk.Button(btn_frame, text="Run Count", command=run_process)
    btn_run.pack(side=tk.LEFT, padx=10)
    
    btn_view = ttk.Button(btn_frame, text="View Results", state=tk.DISABLED)
    btn_view.pack(side=tk.LEFT, padx=10)
    
    root.mainloop()


def main():
    if len(sys.argv) > 1:
        run_cli(parse_args())
    else:
        run_gui()


if __name__ == "__main__":
    main()
