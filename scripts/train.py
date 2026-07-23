#!/usr/bin/env python3
"""
train.py  YOLOv11s Pollen Detection Training Script
=====================================================

Trains a YOLO model on annotated pollen microscopy images.

Usage
-----
    python scripts/train.py                          # defaults
    python scripts/train.py --epochs 300 --batch 8   # override
    python scripts/train.py --resume                 # resume interrupted run
    python scripts/train.py --kfold 5                # run 5-fold cross validation
"""

from __future__ import annotations

import argparse
import sys
import random
import yaml
import shutil
from pathlib import Path

# pyrefly: ignore [missing-import]
from ultralytics import YOLO


# ─── Project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "config" / "pollen_dataset.yaml"
DEFAULT_MODEL = "yolo11n.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11s for pollen grain detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Base YOLO model (e.g. yolo26n.pt, yolo26s.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATASET_YAML),
        help="Path to the dataset YAML config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1024,
        help="Input image size (pixels).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (reduce for lower VRAM).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device: '0' for GPU 0, 'cpu' for CPU.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(PROJECT_ROOT / "runs" / "detect"),
        help="Directory to save training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Run name inside the project directory.",
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=0,
        help="Number of folds for cross-validation (0 to disable).",
    )
    return parser.parse_args()


def _generate_annotated_image_lists(dataset_root: Path, batch_size: int) -> None:
    """Generate train.txt and val.txt containing only images that have annotations."""
    for split in ["train", "val"]:
        labels_dir = dataset_root / "labels" / split
        images_dir = dataset_root / "images" / split
        list_file = dataset_root / f"{split}.txt"
        
        if not labels_dir.exists():
            continue
            
        annotated_images = []
        for label_file in labels_dir.glob("*.txt"):
            if label_file.stat().st_size > 0:
                # Find matching image
                for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]:
                    img_path = images_dir / f"{label_file.stem}{ext}"
                    if img_path.exists():
                        annotated_images.append(str(img_path.absolute()).replace("\\", "/"))
                        break
                        
        # Pad to prevent BatchNorm crash if remainder is 1
        if split == "train" and len(annotated_images) > 0 and (len(annotated_images) % batch_size == 1):
            annotated_images.append(annotated_images[0])
            print(f"[INFO] Padded train dataset with 1 image to prevent BatchNorm crash.")
                        
        with open(list_file, "w") as f:
            for img in annotated_images:
                f.write(f"{img}\n")
        print(f"[INFO] Generated {split}.txt with {len(annotated_images)} annotated images.")


def main() -> None:
    """Entry-point: configure and launch YOLOv11s training."""
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        sys.exit(1)

    dataset_root = PROJECT_ROOT / "datasets"
    
    if args.kfold > 1:
        print(f"\n[INFO] Starting {args.kfold}-Fold Cross Validation...")
        _run_kfold(args, dataset_root)
    else:
        _generate_annotated_image_lists(dataset_root, args.batch)
        _run_standard_training(args, data_path)


def _run_standard_training(args, data_path):
    if args.resume:
        last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"[ERROR] No checkpoint found at {last_ckpt} for --resume.")
            sys.exit(1)
        print(f"[INFO] Resuming training from {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        print(f"[INFO] Loading base model: {args.model}")
        model = YOLO(args.model)

    print("[INFO] Starting YOLOv11s training …")
    print(f"       Dataset : {data_path}")
    print(f"       Epochs  : {args.epochs}")
    print(f"       ImgSize : {args.imgsz}")
    print(f"       Batch   : {args.batch}")
    print(f"       Device  : {args.device}")
    print()

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        optimizer="auto",
        degrees=180.0,
        fliplr=0.5,
        flipud=0.5,
        scale=0.5,
        translate=0.3,
        mosaic=1.0,
        hsv_h=0.03,
        hsv_s=0.4,
        hsv_v=0.4,
        mixup=0.1,
        copy_paste=0.0,
        workers=0,
        max_det=2000,
        close_mosaic=0,
        multi_scale=True,
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best weights saved to: {best_weights}")
    print("=" * 60)

    print("\n[INFO] Running data validation on the best weights...")
    val_model = YOLO(str(best_weights))
    metrics = val_model.val(
        data=str(data_path), 
        split="val", 
        device=args.device, 
        iou=0.5, 
        agnostic_nms=True, 
        max_det=1000,
        project=str(Path(args.project) / args.name),
        name="val_results"
    )
    
    print("\n" + "=" * 60)
    print("  Validation complete!")
    if hasattr(metrics, 'box'):
        print(f"  Final mAP50: {metrics.box.map50:.4f}")
    print("=" * 60)

    # Automatically open the training visualization graphs
    results_png = Path(args.project) / args.name / "results.png"
    if results_png.exists():
        print(f"\n[INFO] Opening training visualization: {results_png}")
        try:
            import os
            os.startfile(results_png)
        except Exception as e:
            print(f"[WARNING] Could not auto-open results.png: {e}")



def _run_kfold(args, dataset_root):
    # 1. Gather all annotated images
    all_annotated_images = []
    for split in ["train", "val"]:
        labels_dir = dataset_root / "labels" / split
        images_dir = dataset_root / "images" / split
        if not labels_dir.exists(): continue
        for label_file in labels_dir.glob("*.txt"):
            if label_file.stat().st_size > 0:
                for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]:
                    img_path = images_dir / f"{label_file.stem}{ext}"
                    if img_path.exists():
                        all_annotated_images.append(str(img_path.absolute()).replace("\\", "/"))
                        break
                        
    if not all_annotated_images:
        print("[ERROR] No annotated images found!")
        sys.exit(1)
        
    random.seed(42)
    random.shuffle(all_annotated_images)
    
    folds = args.kfold
    fold_size = len(all_annotated_images) // folds
    map50_scores = []
    
    with open(args.data, 'r') as f:
        base_data = yaml.safe_load(f)
        
    for i in range(folds):
        print(f"\n{'='*60}\n  STARTING FOLD {i+1}/{folds}\n{'='*60}")
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < folds - 1 else len(all_annotated_images)
        
        val_images = all_annotated_images[val_start:val_end]
        train_images = all_annotated_images[:val_start] + all_annotated_images[val_end:]
        
        # Pad to prevent BatchNorm crash if remainder is 1
        if len(train_images) > 0 and (len(train_images) % args.batch == 1):
            train_images.append(train_images[0])
        
        train_txt = dataset_root / f"fold_{i}_train.txt"
        val_txt = dataset_root / f"fold_{i}_val.txt"
        
        with open(train_txt, 'w') as f:
            for img in train_images: f.write(f"{img}\n")
        with open(val_txt, 'w') as f:
            for img in val_images: f.write(f"{img}\n")
            
        fold_yaml_path = dataset_root / f"fold_{i}.yaml"
        fold_data = base_data.copy()
        fold_data['train'] = str(train_txt)
        fold_data['val'] = str(val_txt)
        with open(fold_yaml_path, 'w') as f:
            yaml.dump(fold_data, f)
            
        fold_name = f"{args.name}_fold_{i}"
        
        model = YOLO(args.model)
        model.train(
            data=str(fold_yaml_path),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=fold_name,
            exist_ok=True,
            optimizer="auto",
            degrees=180.0,
            fliplr=0.5,
            flipud=0.5,
            scale=0.2,
            translate=0.1,
            mosaic=1.0,
            hsv_h=0.015,
            hsv_s=0.2,
            hsv_v=0.2,
            mixup=0.1,
            copy_paste=0.0,
            workers=0,
            max_det=1000,
            close_mosaic=0,
            multi_scale=True,
        )
        
        best_weights = Path(args.project) / fold_name / "weights" / "best.pt"
        val_model = YOLO(str(best_weights))
        metrics = val_model.val(
            data=str(fold_yaml_path), 
            split="val", 
            device=args.device, 
            iou=0.5, 
            agnostic_nms=True, 
            max_det=1000,
            project=str(Path(args.project) / fold_name),
            name="val_results"
        )
        
        score = metrics.box.map50 if hasattr(metrics, 'box') else 0
        map50_scores.append(score)
        print(f"[INFO] Fold {i+1} mAP50: {score:.4f}")
        
    print("\n" + "="*60)
    print("  K-FOLD CROSS VALIDATION COMPLETE")
    for i, s in enumerate(map50_scores):
        print(f"  Fold {i+1}: {s:.4f}")
    if map50_scores:
        avg_score = sum(map50_scores) / len(map50_scores)
        print(f"  Average mAP50: {avg_score:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
