#!/usr/bin/env python3
"""Train a YOLO model on annotated pollen microscopy images.

Hyperparameters are loaded from config/training.yaml.

Usage:
    python scripts/train.py                          # defaults
    python scripts/train.py --epochs 300 --batch 8   # override
    python scripts/train.py --resume                 # resume interrupted run
    python scripts/train.py --kfold 5                # 5-fold cross validation
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'from src...' works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
from ultralytics import YOLO

from src.paths import PROJECT_ROOT, DATASET_YAML, DATASET_ROOT, TRAINING_YAML, DEFAULT_MODEL


def _load_training_config() -> dict:
    """Load hyperparameters from config/training.yaml. Returns {} if missing."""
    if not TRAINING_YAML.exists():
        print(f"[WARNING] {TRAINING_YAML} not found, using YOLO defaults.")
        return {}
    with open(TRAINING_YAML, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLO for pollen grain detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Base YOLO model (e.g. pretrained_models/yolo26n.pt).")
    parser.add_argument("--data", type=str, default=str(DATASET_YAML),
                        help="Path to the dataset YAML config.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total training epochs.")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Input image size (pixels).")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (reduce for lower VRAM).")
    parser.add_argument("--device", type=str, default="0",
                        help="Device: '0' for GPU 0, 'cpu' for CPU.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last checkpoint.")
    parser.add_argument("--project", type=str,
                        default=str(PROJECT_ROOT / "runs" / "detect"),
                        help="Directory to save training runs.")
    parser.add_argument("--name", type=str, default="train",
                        help="Run name inside the project directory.")
    parser.add_argument("--kfold", type=int, default=0,
                        help="Number of folds for cross-validation (0 to disable).")
    return parser.parse_args()


def _generate_annotated_image_lists(dataset_root: Path, batch_size: int) -> None:
    """Write train.txt / val.txt containing only images that have non-empty labels."""
    for split in ["train", "val"]:
        labels_dir = dataset_root / "labels" / split
        images_dir = dataset_root / "images" / split
        list_file = dataset_root / f"{split}.txt"

        if not labels_dir.exists():
            continue

        annotated_images = []
        for label_file in labels_dir.glob("*.txt"):
            if label_file.stat().st_size > 0:
                for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"]:
                    img_path = images_dir / f"{label_file.stem}{ext}"
                    if img_path.exists():
                        annotated_images.append(str(img_path.absolute()).replace("\\", "/"))
                        break

        # Pad to prevent BatchNorm crash when remainder is 1
        if split == "train" and annotated_images and (len(annotated_images) % batch_size == 1):
            annotated_images.append(annotated_images[0])
            print("[INFO] Padded train dataset with 1 image to prevent BatchNorm crash.")

        with open(list_file, "w") as f:
            for img in annotated_images:
                f.write(f"{img}\n")
        print(f"[INFO] Generated {split}.txt with {len(annotated_images)} annotated images.")


def main() -> None:
    """Entry-point: configure and launch YOLO training."""
    args = parse_args()

    # Route standard models to pretrained_models directory
    model_lower = args.model.lower()
    if model_lower in ["yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", 
                       "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]:
        if not model_lower.endswith(".pt"):
            model_lower += ".pt"
        Path("pretrained_models").mkdir(parents=True, exist_ok=True)
        args.model = f"pretrained_models/{model_lower}"
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        sys.exit(1)

    if args.kfold > 1:
        print(f"\n[INFO] Starting {args.kfold}-Fold Cross Validation...")
        _run_kfold(args, DATASET_ROOT)
    else:
        _generate_annotated_image_lists(DATASET_ROOT, args.batch)
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

    # Load hyperparameters from config/training.yaml
    hyp = _load_training_config()

    print("[INFO] Starting YOLO training ...")
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
        **hyp,  # Unpack all hyperparameters from training.yaml
    )

    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Best weights saved to: {best_weights}")
    print("=" * 60)

    # Post-training validation
    print("\n[INFO] Running data validation on the best weights...")
    val_model = YOLO(str(best_weights))
    metrics = val_model.val(
        data=str(data_path),
        split="val",
        device=args.device,
        iou=0.5,
        agnostic_nms=True,
        max_det=1500,
        project=str(Path(args.project) / args.name),
        name="val_results",
    )

    print("\n" + "=" * 60)
    print("  Validation complete!")
    if hasattr(metrics, "box"):
        print(f"  Final mAP50: {metrics.box.map50:.4f}")
    print("=" * 60)

    # ── Auto-rename folder: i{N}_{trainCount}T_{valCount}V_{MODEL}_{P}P_{R}R_{mAP50}A
    try:
        dataset_root = Path(__file__).parent.parent / "datasets"
        train_txt = dataset_root / "train.txt"
        val_txt = dataset_root / "val.txt"
        train_count = sum(1 for _ in open(train_txt)) if train_txt.exists() else 0
        val_count = sum(1 for _ in open(val_txt)) if val_txt.exists() else 0

        model_name = Path(args.model).stem.upper()

        if hasattr(metrics, "box"):
            p = int(metrics.box.mp * 100)
            r = int(metrics.box.mr * 100)
            map50 = int(metrics.box.map50 * 100)
        else:
            p, r, map50 = 0, 0, 0

        detect_dir = Path(args.project)
        iteration = len([d for d in detect_dir.iterdir() if d.is_dir() and d.name.startswith("i")]) + 1

        new_name = f"i{iteration}_{train_count}T_{val_count}V_{model_name}_{p}P_{r}R_{map50}A"
        old_dir = detect_dir / args.name
        new_dir = detect_dir / new_name

        if old_dir.exists():
            os.rename(str(old_dir), str(new_dir))
            print(f"\n[INFO] Renamed training folder to: {new_name}")
            args.name = new_name
    except Exception as e:
        print(f"\n[WARNING] Could not automatically rename folder: {e}")

    # Open results visualization
    results_png = Path(args.project) / args.name / "results.png"
    if results_png.exists():
        print(f"\n[INFO] Opening training visualization: {results_png}")
        try:
            os.startfile(results_png)
        except Exception as e:
            print(f"[WARNING] Could not auto-open results.png: {e}")


def _run_kfold(args, dataset_root):
    # Gather all annotated images from train + val
    all_annotated_images = []
    for split in ["train", "val"]:
        labels_dir = dataset_root / "labels" / split
        images_dir = dataset_root / "images" / split
        if not labels_dir.exists():
            continue
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

    with open(args.data, "r") as f:
        base_data = yaml.safe_load(f)

    # Load shared hyperparameters
    hyp = _load_training_config()

    for i in range(folds):
        print(f"\n{'=' * 60}\n  STARTING FOLD {i + 1}/{folds}\n{'=' * 60}")
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < folds - 1 else len(all_annotated_images)

        val_images = all_annotated_images[val_start:val_end]
        train_images = all_annotated_images[:val_start] + all_annotated_images[val_end:]

        # Pad to prevent BatchNorm crash
        if train_images and (len(train_images) % args.batch == 1):
            train_images.append(train_images[0])

        train_txt = dataset_root / f"fold_{i}_train.txt"
        val_txt = dataset_root / f"fold_{i}_val.txt"

        with open(train_txt, "w") as f:
            for img in train_images:
                f.write(f"{img}\n")
        with open(val_txt, "w") as f:
            for img in val_images:
                f.write(f"{img}\n")

        fold_yaml_path = dataset_root / f"fold_{i}.yaml"
        fold_data = base_data.copy()
        fold_data["train"] = str(train_txt)
        fold_data["val"] = str(val_txt)
        with open(fold_yaml_path, "w") as f:
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
            **hyp,
        )

        best_weights = Path(args.project) / fold_name / "weights" / "best.pt"
        val_model = YOLO(str(best_weights))
        metrics = val_model.val(
            data=str(fold_yaml_path),
            split="val",
            device=args.device,
            iou=0.5,
            agnostic_nms=True,
            max_det=1500,
            project=str(Path(args.project) / fold_name),
            name="val_results",
        )

        score = metrics.box.map50 if hasattr(metrics, "box") else 0
        map50_scores.append(score)
        print(f"[INFO] Fold {i + 1} mAP50: {score:.4f}")

    print("\n" + "=" * 60)
    print("  K-FOLD CROSS VALIDATION COMPLETE")
    for i, s in enumerate(map50_scores):
        print(f"  Fold {i + 1}: {s:.4f}")
    if map50_scores:
        avg_score = sum(map50_scores) / len(map50_scores)
        print(f"  Average mAP50: {avg_score:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
