"""Faster R-CNN training and validation for pollen detection.

Trains a Faster R-CNN ResNet50-FPN V2 model on the YOLO-format dataset,
or evaluates a previously saved checkpoint.

Usage:
    python scripts/train_rcnn.py                 # Train for N epochs
    python scripts/train_rcnn.py --epochs 100    # Override epoch count
    python scripts/train_rcnn.py --val-only      # Evaluate saved weights only
"""

import sys
import os
import time
import argparse
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


def _build_model(num_classes: int = 2, pretrained: bool = True):
    """Create and return a Faster R-CNN ResNet50-FPN V2 model."""
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(num_epochs: int = 65, batch_size: int = 16):
    """Train the Faster R-CNN model."""
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

    # Use persistent_workers=True to prevent Windows shm.dll crash
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, persistent_workers=True, collate_fn=collate_fn)

    print("Initializing Faster R-CNN ResNet50-FPN V2...")
    model = _build_model(num_classes=2, pretrained=True)
    model.to(device)

    # Optimizer (SGD is the academic gold standard for Faster R-CNN)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)

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


def validate(batch_size: int = 8):
    """Evaluate the saved Faster R-CNN weights on the validation set."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    val_txt = PROJECT_ROOT / "datasets_sliced" / "val.txt"
    if not val_txt.exists():
        print(f"Error: Could not find {val_txt}")
        return

    print("Loading validation dataset...")
    val_dataset = YOLODatasetRCNN(str(val_txt))

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, persistent_workers=True, collate_fn=collate_fn)

    print("Initializing Faster R-CNN ResNet50-FPN V2...")
    model = _build_model(num_classes=2, pretrained=False)

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

            preds = model(images)

            formatted_preds = [
                {'boxes': p['boxes'].cpu(), 'scores': p['scores'].cpu(), 'labels': p['labels'].cpu()}
                for p in preds
            ]
            formatted_targets = [
                {'boxes': t['boxes'].cpu(), 'labels': t['labels'].cpu()}
                for t in targets
            ]

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


def main():
    parser = argparse.ArgumentParser(
        description="Faster R-CNN training and validation for pollen detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=65, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--val-only", action="store_true", help="Run validation only (skip training)")
    args = parser.parse_args()

    if args.val_only:
        validate(batch_size=args.batch)
    else:
        train(num_epochs=args.epochs, batch_size=args.batch)


if __name__ == "__main__":
    main()