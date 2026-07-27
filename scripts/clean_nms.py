import cv2
import glob
import os
from pathlib import Path
import numpy as np

def clean_labels(labels_dir, iou_threshold=0.45):
    txt_files = list(Path(labels_dir).glob("*.txt"))
    total_removed = 0
    
    for txt_path in txt_files:
        with open(txt_path, "r") as f:
            lines = f.readlines()
            
        if not lines:
            continue
            
        boxes = []
        original_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                # Convert normalized [xc, yc, w, h] to [left, top, w, h]
                left = xc - w/2
                top = yc - h/2
                boxes.append([left, top, w, h])
                original_lines.append(line)
                
        if not boxes:
            continue
            
        scores = [1.0] * len(boxes)
        
        # cv2.dnn.NMSBoxes returns indices of boxes to keep
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
        
        if len(indices) < len(boxes):
            kept_indices = indices.flatten()
            new_lines = [original_lines[i] for i in kept_indices]
            
            with open(txt_path, "w") as f:
                f.writelines(new_lines)
                
            removed = len(boxes) - len(kept_indices)
            total_removed += removed
            print(f"Cleaned {txt_path.name}: Removed {removed} overlapping boxes.")
            
    print(f"---")
    print(f"Finished processing {labels_dir}.")
    print(f"Total overlapping boxes removed: {total_removed}")

if __name__ == "__main__":
    base_dir = Path("d:/CBC-Apps/PollenCounter-ObjectDetection-YOLO")
    train_labels = base_dir / "datasets" / "labels" / "train"
    val_labels = base_dir / "datasets" / "labels" / "val"
    
    print("Starting NMS Cleanup...")
    if train_labels.exists():
        clean_labels(train_labels, iou_threshold=0.45)
    if val_labels.exists():
        clean_labels(val_labels, iou_threshold=0.45)
