# 🔬 PollenCounter — YOLOv26 Object Detection

> **High-Volume Automated Pollen Counting** using YOLOv26's NMS-free, end-to-end architecture.

Automate the counting of microscopic pollen grains across massive image datasets containing overlapping clusters and slide debris.  The pipeline trains a custom YOLOv26-Nano detector and exports per-image pollen counts to a structured `.xlsx` spreadsheet.

---

## Why YOLOv26?

| Feature | Benefit for Pollen Counting |
|---|---|
| **NMS-Free Inference** | Traditional Non-Maximum Suppression _deletes_ overlapping bounding boxes. When pollen grains are clumped, NMS causes systematic under-counting. YOLO26 eliminates NMS entirely, preserving every detection. |
| **C2PSA (Cross-Stage Partial Spatial Attention)** | Acts as a _digital spectrometer_ — the spatial attention mechanism isolates the vibrant purple anthocyanin pigmentation of pollen grains while suppressing slide debris, air bubbles, and background noise. |
| **MuSGD Optimizer** | Hybrid Muon-SGD optimizer delivers rapid convergence, critical when training on smaller scientific annotation sets. |
| **STAL (Small-Target-Aware Label Assignment)** | Improves detection of tiny, distant pollen grains that appear at low magnification. |

---

## Project Structure

```
PollenCounter-ObjectDetection-YOLO/
├── config/
│   └── pollen_dataset.yaml       # Dataset paths & class names
├── datasets/
│   ├── images/
│   │   ├── train/                # Your annotated training images
│   │   └── val/                  # Your annotated validation images
│   └── labels/
│       ├── train/                # YOLO .txt labels for training
│       └── val/                  # YOLO .txt labels for validation
├── input_images/                 # Images for batch inference
├── output/                       # XLSX reports & annotated images
├── scripts/
│   ├── train.py                  # Training launcher
│   └── count_pollen.py           # Batch inference + XLSX export
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `ultralytics` (with PyTorch), `opencv-python`, `pandas`, and `openpyxl`.

---

## Dataset Preparation

### Annotation Tools

Use **[CVAT](https://www.cvat.ai/)** or **[Roboflow](https://roboflow.com/)** to draw bounding boxes around every pollen grain.

### Export Format

Export annotations in **YOLO format** — one `.txt` file per image with the same base-name:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalised to `[0, 1]`.  For single-class detection, `class_id` is always `0`.

### Folder Layout

```
datasets/
├── images/
│   ├── train/
│   │   ├── slide_001.jpg
│   │   ├── slide_002.jpg
│   │   └── ...
│   └── val/
│       ├── slide_050.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── slide_001.txt
    │   ├── slide_002.txt
    │   └── ...
    └── val/
        ├── slide_050.txt
        └── ...
```

---

## Training

```bash
python scripts/train.py
```

### Key Options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 150 | Training epochs |
| `--batch` | 16 | Batch size (lower for small GPUs) |
| `--imgsz` | 640 | Input image resolution |
| `--device` | `0` | GPU device (`cpu` for CPU-only) |
| `--resume` | — | Resume from last checkpoint |

### Augmentation Strategy

> **Geometric augmentations are enabled; colour augmentations are disabled.**

| Augmentation | Setting | Rationale |
|---|---|---|
| Rotation (±180°) | ✅ | Pollen is rotationally symmetric |
| Horizontal flip | ✅ 0.5 | Orientation-invariant |
| Vertical flip | ✅ 0.5 | Microscopy has no vertical preference |
| Scale (±20%) | ✅ | Simulates magnification variance |
| Mosaic | ✅ 1.0 | Geometric composition |
| HSV hue | ❌ 0.0 | Preserves anthocyanin colour data |
| HSV saturation | ❌ 0.0 | Preserves anthocyanin colour data |
| HSV brightness | ❌ 0.0 | Prevents washing out colour cues |
| Mixup | ❌ 0.0 | Would blend colours, corrupting signal |

Brightness and contrast alterations are **strictly excluded** because the C2PSA spatial attention mechanism depends on authentic anthocyanin pigmentation to distinguish pollen from debris.

---

## Inference & Counting

Place your microscopy images in `input_images/`, then run:

```bash
python scripts/count_pollen.py
```

### Key Options

| Flag | Default | Description |
|---|---|---|
| `--input` | `input_images/` | Folder of images to count |
| `--output` | `output/` | Destination for XLSX & annotated images |
| `--weights` | `runs/detect/train/weights/best.pt` | Trained model weights |
| `--conf` | 0.25 | Confidence threshold |
| `--save-images` | — | Save images with bounding boxes drawn |

### Output: `pollen_counts.xlsx`

**Summary Sheet** — one row per image:

| filename | pollen_count | avg_confidence | min_confidence | max_confidence | image_width | image_height | timestamp |
|---|---|---|---|---|---|---|---|
| slide_001.jpg | 47 | 0.8821 | 0.3012 | 0.9734 | 2048 | 1536 | 2026-07-16T11:30:00 |

**Detections Sheet** — one row per bounding box:

| filename | detection_id | x_center | y_center | width | height | confidence |
|---|---|---|---|---|---|---|
| slide_001.jpg | 1 | 512.5 | 384.0 | 28.3 | 26.7 | 0.9734 |

---

## License

This project is provided as-is for academic and research use.
