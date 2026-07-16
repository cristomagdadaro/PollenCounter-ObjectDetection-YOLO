#!/usr/bin/env python3
"""
count_pollen.py — High-Volume Automated Pollen Counting
========================================================

Batch-inference script that:
  1. Loads a trained YOLOv26 model (best.pt from training, or yolo26n.pt).
  2. Iterates over every image in the input folder.
  3. Counts YOLO-generated bounding boxes per image.
  4. Exports a structured data log to an .xlsx spreadsheet with two sheets:
       • **Summary**    — one row per image (filename, pollen_count, confidence stats).
       • **Detections** — one row per bounding box (coordinates, confidence).
  5. Optionally saves annotated images with drawn bounding boxes.

Why YOLOv26?
------------
YOLOv26's NMS-free (Non-Maximum Suppression-free) end-to-end design is
essential for pollen counting.  Traditional NMS would delete overlapping
bounding boxes when pollen grains are clumped together, leading to
systematic under-counting.  YOLO26 avoids this entirely.

The C2PSA (Cross-Stage Partial Spatial Attention) mechanism inherited
and refined from YOLO11 acts as a digital spectrometer — it spatially
attends to the vibrant anthocyanin pigmentation of pollen grains while
suppressing slide debris, air bubbles, and background noise.

Usage
-----
    python scripts/count_pollen.py                               # defaults
    python scripts/count_pollen.py --input data/slides --conf 0.3
    python scripts/count_pollen.py --weights runs/detect/train/weights/best.pt
    python scripts/count_pollen.py --save-images                 # annotated output
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


# ─── Project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "input_images"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

# Supported image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Count pollen grains in microscopy images using YOLOv26.",
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
        default=640,
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
) -> tuple[list[dict], list[dict]]:
    """
    Run YOLO inference on each image.

    Returns
    -------
    summary_rows : list[dict]
        One dict per image with aggregate stats.
    detail_rows : list[dict]
        One dict per detected bounding box.
    """
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []

    total = len(image_paths)
    for idx, img_path in enumerate(image_paths, start=1):
        print(f"  [{idx}/{total}] Processing: {img_path.name} … ", end="", flush=True)

        # Read image dimensions via OpenCV (fast header read)
        img = cv2.imread(str(img_path))
        if img is None:
            print("SKIPPED (unreadable)")
            continue
        img_h, img_w = img.shape[:2]

        # ── YOLOv26 inference (NMS-free, end-to-end) ───────────────
        results = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )

        result = results[0]
        boxes = result.boxes  # ultralytics Boxes object

        n_detections = len(boxes)
        confidences = boxes.conf.cpu().tolist() if n_detections > 0 else []

        # ── Summary row ─────────────────────────────────────────────
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

        # ── Detail rows (one per bounding box) ─────────────────────
        if n_detections > 0:
            xyxy = boxes.xyxy.cpu().tolist()       # [[x1,y1,x2,y2], …]
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

        # ── Optionally save annotated image ─────────────────────────
        if save_annotated:
            annotated_img = result.plot()  # numpy array with boxes drawn
            out_path = annotated_dir / img_path.name
            cv2.imwrite(str(out_path), annotated_img)

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

    # ── Print a quick table to the terminal ─────────────────────────
    total_pollen = df_summary["pollen_count"].sum()
    total_images = len(df_summary)
    print()
    print("=" * 60)
    print(f"  Pollen Counting Report")
    print(f"  ─────────────────────────────────")
    print(f"  Images processed : {total_images}")
    print(f"  Total pollen     : {total_pollen}")
    if total_images > 0:
        print(f"  Avg per image    : {total_pollen / total_images:.1f}")
    print(f"  Spreadsheet      : {output_path}")
    print("=" * 60)


def main() -> None:
    """Entry-point for batch pollen counting."""
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    weights_path = Path(args.weights)

    # ── Validate input directory ────────────────────────────────────
    if not input_dir.is_dir():
        print(f"[ERROR] Input folder not found: {input_dir}")
        sys.exit(1)

    image_paths = collect_images(input_dir)
    if not image_paths:
        print(f"[ERROR] No images found in {input_dir}")
        print(f"        Supported formats: {', '.join(sorted(IMAGE_EXTS))}")
        sys.exit(1)

    # ── Resolve model weights ───────────────────────────────────────
    if weights_path.exists():
        print(f"[INFO] Loading trained weights: {weights_path}")
        model = YOLO(str(weights_path))
    else:
        fallback = "yolo26n.pt"
        print(f"[WARN] Trained weights not found at {weights_path}")
        print(f"       Falling back to pretrained: {fallback}")
        model = YOLO(fallback)

    # ── Prepare output directory ────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated"
    if args.save_images:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    # ── Run inference ───────────────────────────────────────────────
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

    # ── Export to .xlsx ─────────────────────────────────────────────
    xlsx_path = output_dir / args.xlsx_name
    export_xlsx(summary_rows, detail_rows, xlsx_path)

    if args.save_images:
        print(f"  Annotated images : {annotated_dir}")
    print()


if __name__ == "__main__":
    main()
