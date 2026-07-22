import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS_DIRS = [
    PROJECT_ROOT / "datasets" / "labels" / "train",
    PROJECT_ROOT / "datasets" / "labels" / "val"
]

def calculate_iou(boxA, boxB):
    # box format is YOLO: cx, cy, w, h
    cx1, cy1, w1, h1 = boxA
    cx2, cy2, w2, h2 = boxB
    
    # Convert to x1, y1, x2, y2
    x1A = cx1 - w1/2
    y1A = cy1 - h1/2
    x2A = cx1 + w1/2
    y2A = cy1 + h1/2
    
    x1B = cx2 - w2/2
    y1B = cy2 - h2/2
    x2B = cx2 + w2/2
    y2B = cy2 + h2/2
    
    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
        
    boxAArea = (x2A - x1A) * (y2A - y1A)
    boxBArea = (x2B - x1B) * (y2B - y1B)
    
    return interArea / float(boxAArea + boxBArea - interArea)

def clean_file(txt_path, threshold=0.6):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            coords = list(map(float, parts[1:5]))
            boxes.append((class_id, coords, line))
            
    if len(boxes) < 2:
        return 0
        
    to_remove = set()
    for i in range(len(boxes)):
        if i in to_remove: continue
        for j in range(i + 1, len(boxes)):
            if j in to_remove: continue
            iou = calculate_iou(boxes[i][1], boxes[j][1])
            if iou > threshold:
                to_remove.add(j)
                
    if to_remove:
        clean_boxes = [b for i, b in enumerate(boxes) if i not in to_remove]
        with open(txt_path, 'w') as f:
            for b in clean_boxes:
                f.write(b[2])
                
    return len(to_remove)

def main():
    total_removed = 0
    total_files = 0
    
    for labels_dir in LABELS_DIRS:
        if not labels_dir.exists():
            continue
            
        for txt_path in labels_dir.glob("*.txt"):
            total_files += 1
            removed = clean_file(txt_path)
            if removed > 0:
                print(f"[CLEANED] {txt_path.name} -> Removed {removed} duplicates.")
                total_removed += removed
                
    print(f"\n✅ Scan Complete! Scanned {total_files} files.")
    print(f"🗑️  Purged {total_removed} duplicate bounding boxes across the entire dataset.")

if __name__ == "__main__":
    main()
