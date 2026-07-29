#!/usr/bin/env python3
"""CPU Inference Server Script (Auto-Annotator).
Designed to run on a headless server (e.g., AMD EPYC) to automatically
pseudo-label incoming raw images for active learning.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2

def process_image(img_path: Path, output_img_dir: Path, output_lbl_dir: Path, model_path: str, conf: float, iou: float, slice_size: int):
    """Processes a single image using SAHI and saves the YOLO .txt label."""
    try:
        # Import inside worker to prevent pickling issues with multiprocessing
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        
        # Load model inside worker process
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=conf,
            device="cpu", # Force CPU
        )

        img = cv2.imread(str(img_path))
        if img is None:
            return f"[ERROR] Could not read {img_path.name}"
            
        img_h, img_w = img.shape[:2]

        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMM",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=iou,
            verbose=0
        )

        obj_list = result.object_prediction_list
        
        # Save image to review folder
        out_img_path = output_img_dir / img_path.name
        cv2.imwrite(str(out_img_path), img)

        # Write YOLO .txt file
        out_lbl_path = output_lbl_dir / f"{img_path.stem}.txt"
        with open(out_lbl_path, "w") as f:
            for obj in obj_list:
                # Convert to normalized YOLO format (class x_center y_center width height)
                cls_id = obj.category.id
                box = obj.bbox
                x_c = (box.minx + box.maxx) / 2.0 / img_w
                y_c = (box.miny + box.maxy) / 2.0 / img_h
                w = (box.maxx - box.minx) / img_w
                h = (box.maxy - box.miny) / img_h
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
        # Delete original image so it's not processed again
        img_path.unlink()
        
        return f"[SUCCESS] Processed {img_path.name} ({len(obj_list)} detections)"
        
    except Exception as e:
        return f"[ERROR] Failed processing {img_path.name}: {e}"

def main():
    parser = argparse.ArgumentParser(description="CPU Inference Server")
    parser.add_argument("--model", type=str, required=True, help="Path to weights (e.g., best.pt or best_openvino_model)")
    parser.add_argument("--input", type=str, default="datasets/server_input", help="Directory to monitor for incoming images")
    parser.add_argument("--output", type=str, default="datasets", help="Root dataset directory to output into images/review and labels/review")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of CPU threads to use")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--slice-size", type=int, default=512, help="SAHI slice size")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_img_dir = Path(args.output).resolve() / "images" / "review"
    output_lbl_dir = Path(args.output).resolve() / "labels" / "review"

    for d in [input_dir, output_img_dir, output_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f" CPU Inference Server Started")
    print(f" Model:   {args.model}")
    print(f" Workers: {args.workers}")
    print(f" Input:   {input_dir}")
    print(f" Output:  {output_img_dir}")
    print(f"============================================================")
    print("Waiting for images... (Press Ctrl+C to exit)\n")

    valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            while True:
                # Find all images in input_dir
                images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
                
                if images:
                    print(f"[{time.strftime('%H:%M:%S')}] Found {len(images)} new images. Processing...")
                    
                    futures = []
                    for img_path in images:
                        futures.append(
                            executor.submit(
                                process_image,
                                img_path,
                                output_img_dir,
                                output_lbl_dir,
                                args.model,
                                args.conf,
                                args.iou,
                                args.slice_size
                            )
                        )
                    
                    for future in as_completed(futures):
                        print(future.result())
                        
                    print(f"[{time.strftime('%H:%M:%S')}] Batch complete. Waiting for new images...")
                
                time.sleep(5) # Poll every 5 seconds
                
    except KeyboardInterrupt:
        print("\nShutting down server.")

if __name__ == "__main__":
    main()
