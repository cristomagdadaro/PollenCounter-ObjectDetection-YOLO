#!/usr/bin/env python3
"""
auto_annotate.py — Active Learning Auto-Annotator
=================================================

This script bridges the gap between your trained YOLO model and the manual
annotation GUI (`annotate.py`). 

It takes a folder of raw, unseen images, runs inference using your best weights,
and outputs the results as YOLO `.txt` label files in a "review" folder.
You can then open these pre-annotated images in `annotate.py` to fix the
model's mistakes and add them to your training dataset.

Usage:
    python scripts/auto_annotate.py --input datasets/raw
    
Then review them:
    python scripts/annotate.py --images datasets/images/review --labels datasets/labels/review
"""

import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"

# Default workflow folders
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "raw"
DEFAULT_OUT_IMG = PROJECT_ROOT / "datasets" / "images" / "review"
DEFAULT_OUT_LBL = PROJECT_ROOT / "datasets" / "labels" / "review"

def parse_args():
    parser = argparse.ArgumentParser(description="Auto-annotate images for Active Learning")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT), help="Folder containing raw, unlabeled images")
    parser.add_argument("--out-images", type=str, default=str(DEFAULT_OUT_IMG), help="Destination for review images")
    parser.add_argument("--out-labels", type=str, default=str(DEFAULT_OUT_LBL), help="Destination for auto-generated YOLO labels")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS), help="Path to your trained best.pt weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input)
    out_img_dir = Path(args.out_images)
    out_lbl_dir = Path(args.out_labels)
    weights = Path(args.weights)
    
    # Ensure the input directory exists so users know where to drop files
    if not input_dir.exists():
        print(f"\n[INFO] Creating raw input folder: {input_dir}")
        input_dir.mkdir(parents=True, exist_ok=True)
        print("Please place some unlabeled microscopy images in this folder and run the script again!")
        return
        
    image_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}]
    
    if not image_paths:
        print(f"\n[ERROR] No images found in {input_dir}")
        print("Please drop your raw images there first.")
        return
        
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nLoading model from {weights}...")
    model = YOLO(str(weights))
    
    print(f"\nProcessing {len(image_paths)} images...")
    
    for img_path in image_paths:
        # Run standard inference with NMS settings tuned for pollen
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=0.5,
            uoimgsz=args.imgsz,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes
        
        # 1. Copy the image to the review folder
        dest_img = out_img_dir / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # 2. Save the bounding boxes as a YOLO .txt file
        label_path = out_lbl_dir / f"{img_path.stem}.txt"
        
        with open(label_path, "w") as f:
            if len(boxes) > 0:
                cls_ids = boxes.cls.cpu().tolist()
                xywhn = boxes.xywhn.cpu().tolist() # Normalized format for YOLO
                
                for cls_id, box in zip(cls_ids, xywhn):
                    x, y, w, h = box
                    f.write(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                    
        print(f"  -> {img_path.name}: Auto-annotated {len(boxes)} grains.")
        
    print("\n" + "="*70)
    print("✨ AUTO-ANNOTATION COMPLETE ✨")
    print(f"Images copied to: {out_img_dir}")
    print(f"Labels saved to:  {out_lbl_dir}")
    print("="*70)
    print("\nNEXT STEP:")
    print("Run the command below to open these images in the GUI and correct the model's mistakes:\n")
    print(f"    python scripts/annotate.py --images \"{out_img_dir}\" --labels \"{out_lbl_dir}\"")
    print("\n(When you are done fixing them, you can move them into your main train/val folders using the GUI's sidebar buttons!)")
    
if __name__ == "__main__":
    main()
