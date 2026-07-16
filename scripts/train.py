#!/usr/bin/env python3
"""
train.py — YOLOv26 Pollen Detection Training Script
=====================================================

Trains a YOLOv26-Nano (yolo26n.pt) model on annotated pollen microscopy
images.  Key design choices:

  • NMS-free end-to-end inference — overlapping pollen in clumps are NOT
    suppressed, which is critical for high-density slides.

  • MuSGD optimizer (hybrid Muon-SGD) — YOLO26's native optimizer that
    delivers rapid convergence on small scientific datasets.

  • Geometric-ONLY augmentation — rotation, flipping, scaling, and mosaic
    are enabled.  ALL colour augmentations (hsv_h, hsv_s, hsv_v, mixup)
    are DISABLED to preserve anthocyanin pigmentation data that the
    C2PSA (Cross-Stage Partial Spatial Attention) mechanism relies on to
    discriminate pollen from background debris.

Usage
-----
    python scripts/train.py                          # defaults
    python scripts/train.py --epochs 300 --batch 8   # override
    python scripts/train.py --resume                 # resume interrupted run

Prerequisites
-------------
    1. Place training images   → datasets/images/train/
    2. Place training labels   → datasets/labels/train/
    3. Place validation images → datasets/images/val/
    4. Place validation labels → datasets/labels/val/
    Labels must be in YOLO .txt format (class x_center y_center w h).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


# ─── Project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_YAML = PROJECT_ROOT / "config" / "pollen_dataset.yaml"
DEFAULT_MODEL = "yolo26n.pt"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv26-Nano for pollen grain detection.",
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
        default=150,
        help="Total training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (pixels).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
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
    return parser.parse_args()


def main() -> None:
    """Entry-point: configure and launch YOLOv26 training."""
    args = parse_args()

    # ── Validate dataset config exists ──────────────────────────────
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        sys.exit(1)

    # ── Load model ──────────────────────────────────────────────────
    if args.resume:
        # Resume from the last saved checkpoint
        last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            print(f"[ERROR] No checkpoint found at {last_ckpt} for --resume.")
            sys.exit(1)
        print(f"[INFO] Resuming training from {last_ckpt}")
        model = YOLO(str(last_ckpt))
    else:
        print(f"[INFO] Loading base model: {args.model}")
        model = YOLO(args.model)

    # ── Launch training ─────────────────────────────────────────────
    #
    # AUGMENTATION PHILOSOPHY
    # -----------------------
    # ✅ Geometric augmentations (rotation, flip, scale, translate, mosaic)
    #    → pollen grains are orientation-invariant; these increase
    #      positional diversity without corrupting colour.
    #
    # ❌ Colour / intensity augmentations (hsv_h, hsv_s, hsv_v, mixup)
    #    → DISABLED to preserve the anthocyanin pigmentation signal.
    #      The C2PSA (Cross-Stage Partial Spatial Attention) mechanism
    #      in YOLOv26 acts as a *digital spectrometer*, isolating the
    #      vibrant purple pigmentation of pollen from background noise.
    #      Altering brightness, saturation, or hue would corrupt the
    #      very colour features C2PSA depends on.
    #
    print("[INFO] Starting YOLOv26 training …")
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
        # ── Optimizer ───────────────────────────────────────────────
        optimizer="MuSGD",            # YOLO26 native hybrid Muon-SGD
        # ── Geometric augmentations (ENABLED) ───────────────────────
        degrees=180.0,                # full rotation (pollen is symmetric)
        fliplr=0.5,                   # horizontal flip
        flipud=0.5,                   # vertical flip  (microscopy has no vertical preference)
        scale=0.2,                    # ±20 % zoom to simulate magnification variance
        translate=0.1,                # ±10 % translation
        mosaic=1.0,                   # mosaic composition (geometric, no colour change)
        # ── Colour augmentations (DISABLED — protect anthocyanin) ───
        hsv_h=0.0,                    # hue shift OFF
        hsv_s=0.0,                    # saturation shift OFF
        hsv_v=0.0,                    # brightness shift OFF
        mixup=0.0,                    # mixup OFF (would blend colours)
    )

    # ── Report ──────────────────────────────────────────────────────
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print()
    print("=" * 60)
    print("  Training complete!")
    print(f"  Best weights saved to: {best_weights}")
    print("=" * 60)


if __name__ == "__main__":
    main()
