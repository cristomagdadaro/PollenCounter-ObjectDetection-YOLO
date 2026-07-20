# 🔬 PollenCounter — YOLO11 Object Detection

> **High-Volume Automated Pollen Counting** using YOLO11.

Automate the counting of microscopic pollen grains across massive image datasets containing overlapping clusters and slide debris. The pipeline trains a custom YOLO11-Nano detector and exports per-image pollen counts to a structured `.xlsx` spreadsheet, while generating annotated images.

---

## 🚀 Model Evolution & Best Practices (Living Document)

This project has evolved through continuous testing to optimise accuracy. Below are the key discoveries and techniques to improve the model over time.

### 1. Model Choice: Why YOLO11 Nano?
- We are currently using **`yolo11n.pt` (Nano)**. 
- **Why not YOLO11 Extra Large (`yolo11x.pt`)?** With a small dataset (e.g., 16 images), large models instantly memorise the dataset (catastrophic overfitting). They also run much slower. 
- **When to upgrade:** Once you have manually annotated **300 to 500 images** using the GUI annotator, you can safely upgrade to `yolo11s.pt` (Small) or `yolo11m.pt` (Medium) for a significant accuracy boost.

### 2. Preventing Double-Counting (IoU Tuning)
- Overlapping pollen grains can easily be double-counted if the model draws two boxes on the exact same grain.
- **Solution:** We tuned the Non-Maximum Suppression (NMS) `iou` threshold to `0.25` in `count_pollen.py`. This ensures that if two boxes overlap by more than 25%, the less confident one is deleted, preventing duplicate counts on a single grain.

### 3. Resolution and VRAM Limits
- **`imgsz=1024` vs `imgsz=2048`**: We found that training at `1024` actually produced a higher `mAP50` (35.8%) than `2048` (33.1%).
- **Why?** At 2048x2048, the background details (microscopic dirt, slide scratches) become too sharp, distracting the model from the macroscopic pollen shape. 
- **Hardware limit:** `imgsz 2048` at `batch 24` exceeds the 24GB VRAM of an RTX 3090. If you ever must train at 2048, you must drop the batch size to `4` or `8`. The sweet spot for this project is `imgsz=1024, batch=20`.

### 4. Overfitting & Epochs
- **Question:** Does increasing epochs (e.g. 150 to 300) increase accuracy?
- **Answer:** Only up to a point. If the dataset is too small (e.g., 16 images), training for too long results in "Fitness Collapse" (NaN losses) where the model overfits and breaks. To get `mAP50` to 90%+, you must annotate more images, not just increase epochs.

### 5. Augmentation Strategy
- **Geometric (Enabled):** Rotations (±180°), flips, and scaling simulate magnification variance and orientation since pollen is symmetrical.
- **Colour (Enabled):** `hsv_h`, `hsv_s`, `hsv_v`, and `mixup` are **enabled**. This is critical because lighting conditions and I2KI staining shades vary wildly between different microscope slides. Colour augmentations force the model to look at the *shape* of the pollen, rather than memorising the exact shade of purple.

---

## 📊 Training History & Results Log

To ensure continuous improvement, log the results of every major training run here to compare and contrast how different hyperparameter combinations affect the `mAP50`.

| Date | Dataset Size | Model | Resolution (`imgsz`) | Batch Size | Epochs | mAP50 | Notes / Insights |
|---|---|---|---|---|---|---|---|
| **Jul 16** | 16 Train / 2 Val | `yolo11n.pt` | 1024 | 16 | 150 | **35.8%** | *Initial YOLO11 baseline. Good balance of speed and detail.* |
| **Jul 16** | 16 Train / 2 Val | `yolo11n.pt` | 2048 | 4 | 150 | **33.1%** | *Massive VRAM usage (caused OOM at batch 24). Accuracy dropped due to microscopic noise and artifacts distracting the model.* |
| **Jul 16** | 16 Train / 2 Val | `yolo11n.pt` | 768 | 24 | 150 | **31.1%** | *Downscaling too far caused loss of critical pollen grain details.* |
| **Jul 20** | 16 Train / 2 Val | `yolo11n.pt` | 1024 | 20 | 100 | **TBD** | *Enabled HSV and Mixup augmentations to combat lighting variance across slides. Changed optimizer to `auto`.* |

*Remember to update this table every time a new dataset batch is annotated or a major training setting is changed!*

---

## 🎨 Annotation GUI Tool

A built-in Tkinter GUI is provided to rapidly build your dataset.

```bash
python scripts/annotate.py
```

### Dataset Management Features
- **Auto-Box:** Double-click or press `Spacebar` to instantly place a default-sized box at your cursor.
- **Dataset Switcher:** Use the dropdown in the sidebar to switch between viewing your `Train`, `Validation`, and `Excluded` image sets.
- **Move/Exclude:** Use the sidebar buttons to instantly move an image (and its label data) between the Train, Validation, or Excluded folders. Excluded images are safely hidden and ignored during training.

---

## 🏋️ Training Pipeline

```bash
python scripts/train.py --model yolo11n.pt --device 0 --epochs 100 --batch 20 --imgsz 1024
```

### Key Options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 100 | Training epochs (Stop early if it overfits) |
| `--batch` | 20 | Batch size (Lower if CUDA OutOfMemoryError occurs) |
| `--imgsz` | 1024 | Input resolution (1024 is the current sweet spot) |
| `--device` | `0` | GPU device (`cpu` for CPU-only) |

*Note: The script automatically runs a Validation step using the best weights after training completes.*

---

## 🔬 Inference & Counting

Place your un-annotated microscopy images in `input_images/`, then run:

```bash
python scripts/count_pollen.py
```

The script will:
1. Detect all pollen grains.
2. Draw a semi-transparent text overlay with the total count in the center of the output image.
3. Generate random-colored bounding boxes for clear visibility of overlapping grains (no text labels on boxes).
4. Save the annotated images and a comprehensive `pollen_counts.xlsx` report to `output/`.

---

## 📂 Project Structure

```text
PollenCounter-ObjectDetection-YOLO/
├── config/
│   └── pollen_dataset.yaml       # Dataset paths & class names
├── datasets/
│   ├── images/
│   │   ├── train/                
│   │   ├── val/                  
│   │   └── excluded/             # Images hidden from training
│   └── labels/
│       ├── train/                
│       ├── val/                  
│       └── excluded/             
├── input_images/                 # Images for batch inference
├── output/                       # XLSX reports & annotated images
├── scripts/
│   ├── train.py                  # Training launcher
│   ├── count_pollen.py           # Batch inference + XLSX export
│   └── annotate.py               # GUI Annotation Tool
└── README.md
```
