#!/usr/bin/env python3
"""Slice YOLO dataset into overlapping patches for high-accuracy training."""

import os
import cv2
import argparse
from pathlib import Path

def get_yolo_boxes(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_c = float(parts[1]) * img_w
                y_c = float(parts[2]) * img_h
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                x1 = x_c - w / 2
                y1 = y_c - h / 2
                x2 = x_c + w / 2
                y2 = y_c + h / 2
                boxes.append([cls_id, x1, y1, x2, y2])
    return boxes

def calculate_intersection_and_format(box, win_x, win_y, win_size, min_area_ratio=0.5):
    cls_id, bx1, by1, bx2, by2 = box
    
    # Calculate intersection box
    ix1 = max(bx1, win_x)
    iy1 = max(by1, win_y)
    ix2 = min(bx2, win_x + win_size)
    iy2 = min(by2, win_y + win_size)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    
    inter_area = (ix2 - ix1) * (iy2 - iy1)
    orig_area = (bx2 - bx1) * (by2 - by1)
    
    # Discard if the box is heavily truncated
    if inter_area / orig_area < min_area_ratio:
        return None
    
    # Translate coordinates relative to the window
    rx1 = ix1 - win_x
    ry1 = iy1 - win_y
    rx2 = ix2 - win_x
    ry2 = iy2 - win_y
    
    # Convert back to YOLO normalized format [x_center, y_center, w, h]
    rw = (rx2 - rx1) / win_size
    rh = (ry2 - ry1) / win_size
    rcx = (rx1 + rx2) / 2.0 / win_size
    rcy = (ry1 + ry2) / 2.0 / win_size
    
    return f"{cls_id} {rcx:.6f} {rcy:.6f} {rw:.6f} {rh:.6f}"

def process_image(img_path, label_path, out_img_dir, out_lbl_dir, slice_size, overlap, min_area_ratio):
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    img_h, img_w = img.shape[:2]
    
    boxes = get_yolo_boxes(label_path, img_w, img_h)
    step = int(slice_size * (1 - overlap))
    
    # Calculate grid start coordinates
    y_starts = list(range(0, img_h - slice_size + 1, step))
    if img_h > slice_size and y_starts[-1] + slice_size < img_h:
        y_starts.append(img_h - slice_size)
        
    x_starts = list(range(0, img_w - slice_size + 1, step))
    if img_w > slice_size and x_starts[-1] + slice_size < img_w:
        x_starts.append(img_w - slice_size)
        
    if not y_starts: y_starts = [0]
    if not x_starts: x_starts = [0]

    saved_slices = 0
    
    for y in y_starts:
        for x in x_starts:
            slice_boxes = []
            for box in boxes:
                fmt_box = calculate_intersection_and_format(box, x, y, slice_size, min_area_ratio)
                if fmt_box:
                    slice_boxes.append(fmt_box)
            
            # Only save the patch if it contains at least one object
            if slice_boxes:
                crop_y2 = min(y + slice_size, img_h)
                crop_x2 = min(x + slice_size, img_w)
                crop = img[y:crop_y2, x:crop_x2]
                
                # Pad if the image is smaller than the slice size (edges)
                if crop.shape[0] < slice_size or crop.shape[1] < slice_size:
                    pad_h = slice_size - crop.shape[0]
                    pad_w = slice_size - crop.shape[1]
                    crop = cv2.copyMakeBorder(crop, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                
                base_name = img_path.stem
                slice_name = f"{base_name}_{x}_{y}"
                
                cv2.imwrite(str(out_img_dir / f"{slice_name}.jpg"), crop)
                with open(out_lbl_dir / f"{slice_name}.txt", "w") as f:
                    f.write("\n".join(slice_boxes) + "\n")
                
                saved_slices += 1
                
    return saved_slices

def main():
    parser = argparse.ArgumentParser(description="Slice YOLO dataset into overlapping patches.")
    parser.add_argument("--input", type=str, default="datasets", help="Input dataset directory")
    parser.add_argument("--output", type=str, default="datasets_sliced", help="Output sliced dataset directory")
    parser.add_argument("--slice-size", type=int, default=512, help="Patch size (pixels)")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio between patches (0 to 1)")
    parser.add_argument("--min-area", type=float, default=0.5, help="Minimum remaining area ratio to keep a bounding box")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    
    total_slices = 0
    for split in ["train", "val"]:
        in_img_dir = in_dir / "images" / split
        in_lbl_dir = in_dir / "labels" / split
        
        if not in_img_dir.exists():
            continue
            
        out_img_dir = out_dir / "images" / split
        out_lbl_dir = out_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        img_paths = list(in_img_dir.glob("*.*"))
        print(f"Slicing {len(img_paths)} {split} images...")
        
        for img_path in img_paths:
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                continue
            lbl_path = in_lbl_dir / f"{img_path.stem}.txt"
            slices = process_image(img_path, lbl_path, out_img_dir, out_lbl_dir, args.slice_size, args.overlap, args.min_area)
            total_slices += slices
            
    print(f"\n[SUCCESS] Generated {total_slices} patched images in {out_dir}")

if __name__ == "__main__":
    main()
