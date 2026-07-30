import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class YOLODatasetRCNN(Dataset):
    """
    Reads YOLO format txt labels (normalized cx, cy, w, h)
    and converts them to Faster R-CNN format (absolute xmin, ymin, xmax, ymax).
    """
    def __init__(self, list_file, transforms=None):
        self.transforms = transforms
        with open(list_file, 'r') as f:
            self.image_files = [line.strip() for line in f.readlines() if line.strip()]
            
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback if image fails to load
            img = np.zeros((512, 512, 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Convert image to float32 tensor [C, H, W] in range [0, 1]
        img_tensor = torch.as_tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        
        # Build label path (e.g. replace images/train/X.jpg with labels/train/X.txt)
        label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
        
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        
                        xmin = (xc - bw / 2) * w
                        ymin = (yc - bh / 2) * h
                        xmax = (xc + bw / 2) * w
                        ymax = (yc + bh / 2) * h
                        
                        # Faster R-CNN expects 0 to be background, so we map class 0 -> 1
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id + 1)
                        
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            # Clip boxes to image boundaries
            boxes[:, 0] = boxes[:, 0].clamp(min=0, max=w)
            boxes[:, 1] = boxes[:, 1].clamp(min=0, max=h)
            boxes[:, 2] = boxes[:, 2].clamp(min=0, max=w)
            boxes[:, 3] = boxes[:, 3].clamp(min=0, max=h)
            
            # Remove invalid boxes where max <= min
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            labels = labels[keep]
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros((0,))
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        # Apply custom transforms if provided (e.g. Albumentations later)
        if self.transforms is not None:
            # We skip advanced transforms for the initial baseline plan
            pass
            
        return img_tensor, target
