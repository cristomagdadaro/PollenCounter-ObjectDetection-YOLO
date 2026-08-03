import sys
import os
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.rcnn_dataset import YOLODatasetRCNN

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_txt = PROJECT_ROOT / "datasets_sliced" / "train.txt"
    val_txt = PROJECT_ROOT / "datasets_sliced" / "val.txt"
    
    if not train_txt.exists():
        print(f"Error: Could not find {train_txt}")
        return
        
    print("Loading datasets...")
    train_dataset = YOLODatasetRCNN(str(train_txt))
    val_dataset = YOLODatasetRCNN(str(val_txt))
    
    # Maximize the RTX 3090 (24GB VRAM)
    # Use persistent_workers=True to prevent Windows shm.dll crash while keeping background loading fast
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, collate_fn=collate_fn)
    
    print("Initializing Faster R-CNN ResNet50-FPN V2...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    
    # Replace the classifier
    num_classes = 2  # 1 class (pollen) + background (class 0)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.to(device)
    
    # Optimizer (SGD is the academic gold standard for Faster R-CNN)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    
    num_epochs = 65
    print(f"Starting training for {num_epochs} epochs...")
    
    total_steps = num_epochs * len(train_loader)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
            
            current_step = epoch * len(train_loader) + i + 1
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time_per_step = elapsed / current_step
                steps_left = total_steps - current_step
                eta_secs = steps_left * avg_time_per_step
                
                m, s = divmod(int(eta_secs), 60)
                h, m = divmod(m, 60)
                
                print(f"Epoch {epoch} | Step {i}/{len(train_loader)} | Loss: {losses.item():.4f} | ETA: {h}h {m}m {s}s")
                
        print(f"--- Epoch {epoch} finished. Average Loss: {epoch_loss / len(train_loader):.4f} ---")
        
    print("Saving model weights...")
    runs_dir = PROJECT_ROOT / "runs" / "detect"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = runs_dir / "rcnn_best.pth"
    torch.save(model.state_dict(), str(save_path))
    print(f"Model saved successfully to {save_path}")

if __name__ == "__main__":
    main()