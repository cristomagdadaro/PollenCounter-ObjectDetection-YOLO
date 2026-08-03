import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from src.rcnn_dataset import YOLODatasetRCNN

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    val_txt = PROJECT_ROOT / "datasets_sliced" / "val.txt"
    if not val_txt.exists():
        print(f"Error: Could not find {val_txt}")
        return
        
    print("Loading validation dataset...")
    val_dataset = YOLODatasetRCNN(str(val_txt))
    
    # Use persistent_workers=True to avoid shm.dll crashes while keeping it fast
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, persistent_workers=True, collate_fn=collate_fn)
    
    print("Initializing Faster R-CNN ResNet50-FPN V2...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=None)
    
    # Replace the classifier
    num_classes = 2  # 1 class (pollen) + background (class 0)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Load best weights
    weights_path = PROJECT_ROOT / "runs" / "detect" / "rcnn_best.pth"
    if not weights_path.exists():
        print(f"Error: Could not find {weights_path}. Please run train_rcnn.py first.")
        return
        
    print(f"Loading weights from {weights_path}...")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Initialize metric
    metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    
    print("Starting Evaluation...")
    start_time = time.time()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = list(image.to(device) for image in images)
            
            # Forward pass
            preds = model(images)
            
            # Format predictions and targets for torchmetrics
            formatted_preds = []
            for pred in preds:
                formatted_preds.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
                
            formatted_targets = []
            for target in targets:
                formatted_targets.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
                
            metric.update(formatted_preds, formatted_targets)
            
            if i % 10 == 0:
                print(f"Evaluating Batch {i}/{len(val_loader)}...")
                
    print("\nComputing Final Metrics (this may take a moment)...")
    results = metric.compute()
    
    mAP50 = results['map_50'].item() * 100
    mAP50_95 = results['map'].item() * 100
    
    print("=========================================")
    print(f"Evaluation Complete in {time.time() - start_time:.1f} seconds")
    print(f"mAP50:     {mAP50:.2f}% (Object Detection Accuracy)")
    print(f"mAP50-95:  {mAP50_95:.2f}% (Bounding Box Tightness)")
    print("=========================================")
    
if __name__ == "__main__":
    main()
