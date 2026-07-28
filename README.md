# Pollen Counter & Annotator (YOLOv11n)

> **High-Volume Automated Pollen Counting** using YOLO11.

Automate the counting of microscopic pollen grains across massive image datasets containing overlapping clusters and slide debris. The pipeline trains a custom YOLO11-Nano detector and exports per-image pollen counts to a structured `.xlsx` spreadsheet, while generating annotated images.

---

##  Concept Primer: How the AI Works

To understand the architecture of this project, it helps to understand how the underlying technologies interact:

### 1. Machine Learning (The Broad Category)
Machine Learning (ML) is a subset of Artificial Intelligence where computers are taught to recognize patterns in data without being explicitly programmed with rules. 
- **How it works:** Instead of writing an `if/else` statement like *"if a pixel is dark yellow, it's pollen,"* we give the computer examples of pollen and let it figure out the rules on its own.

### 2. Deep Learning (The Advanced Technique)
Deep Learning (DL) is a highly specialized subfield *inside* Machine Learning. It uses **Artificial Neural Networks** inspired by the human brain, containing many layers. 
- **How it works:** It can handle incredibly complex, unstructured data (like raw microscopy images) that traditional Machine Learning struggles with. It automatically discovers the most important features (edges, shapes, textures) rather than requiring a human to extract them first.
- **Application:** Our **YOLO11** model is a Deep Learning convolutional neural network.

### 3. Computer Vision (The Application)
Computer Vision (CV) is the specific field of teaching computers to "see" and interpret the visual world. It is the *goal*, rather than the algorithm.
- **How it works:** Historically, CV was done using manual math and geometry (like the OpenCV contour-snapping algorithm used in `annotate.py` to perfectly fit bounding boxes). Today, modern CV relies almost entirely on Deep Learning to achieve high accuracy.
- **Putting it together:** We are solving a **Computer Vision** problem (counting pollen) by using a **Machine Learning** approach, specifically powered by a **Deep Learning** network (YOLO11).

---

##  Model Evolution & Best Practices (Living Document)

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
- **Mosaic (Always On):** YOLO disables mosaic augmentation for the last 10 epochs by default (`close_mosaic=10`). Because our dataset is small (e.g. 63 images), we use `close_mosaic=0` to keep this critical augmentation active through the entire training run!

### 6. Cross-Validation (K-Fold)
- **Problem:** When a dataset is extremely small (e.g., 60 images), evaluating accuracy on a fixed 20% validation split is heavily biased by which images happen to land in that 20%. A "lucky" split might give you 90% mAP, while an "unlucky" split gives 30%.
- **Solution:** We introduced 5-Fold Cross Validation (`--kfold 5`). This trains 5 separate models on different 80/20 splits and averages the results, providing a much more robust and trustworthy metric. In our tests, switching to 5-Fold training resulted in a massive jump from **31.8% to 49.8% mAP50**, proving its effectiveness for small microscopy datasets.

---

### 7. Optimizers (AdamW vs SGD)
- **AdamW (Default for Small Datasets):** Converges incredibly fast and rarely gets stuck. Highly recommended for small datasets (under a few thousand images). We use a smaller learning rate (`lr0=0.001`) to prevent it from crashing out of the bounds.
- **SGD (The Tortoise):** Learns much slower and bounces violently, but often finds a better final global minimum. Recommended when training for long periods (e.g. 500 epochs). Standard learning rate applies (`lr0=0.01`).

### 8. Scale Invariance (The AI is Literal)
- If you train the model exclusively on images taken with a 10x magnification lens, the AI will learn that "pollen" is a very specific physical pixel size (e.g., 30x30 pixels). 
- If you feed it an image with a massive 200x200 pixel pollen grain, it will ignore it because it's "too big to be pollen."
- **Solution:** If you intend to use different microscope magnifications, you must increase the `scale` augmentation in `training.yaml` to force the AI to learn pollen at all zoom levels!

---

##  Training History & Results Log

### Comprehensive Model Performance (i1 to i17)

Below is a complete matrix of all recorded model iterations (stored in `runs/detect/`) from the first to the latest, allowing for direct comparison of Precision, Recall, and mAP50 as the dataset grew.

| Run ID | Architecture | Train Imgs | Val Imgs | Precision | Recall | mAP50 | Remarks |
|---|---|---|---|---|---|---|---|
| `i1` | YOLO11N | 30 | 2 | 45% | 43% | 35% | Initial baseline |
| `i2` | YOLO11M | 57 | 10 | 63% | 56% | 51% | Upgraded to Medium (overfit) |
| `i3` | YOLO11S | 57 | 10 | 57% | 61% | 49% | Upgraded to Small |
| `i4` | YOLO11N | 57 | 10 | 64% | 60% | 53% | Reverted to Nano (performed best) |
| `i5` | YOLO11N | 79 | 10 | 63% | 58% | 53% | Dataset scaling |
| `i6` | YOLO11N | 86 | 13 | 63% | 63% | 53% | Dataset scaling |
| `i7` | YOLO11N | 164 | 20 | 69% | 67% | 61% | Breakthrough: First time breaking 60% mAP50 |
| `i8` | YOLO11S | 164 | 20 | 69% | 65% | 57% | YOLO11S test (peaked early) |
| `i9` | YOLO11N | 192 | 14 | 64% | 64% | 52% | Dataset purged of duplicate boxes (True baseline) |
| `i10` | YOLO11N | 192 | 14 | 70% | 65% | 65% | Extreme Augmentations |
| `i11` | YOLO11N | 200 | 16 | 79% | 70% | 68% | Data Leakage Detected (DLeak) |
| `i12` | YOLO11N | 200 | 16 | 74% | 66% | 59% | Data Leakage Detected (DLeak) |
| `i13` | YOLO11N | 200 | 16 | 79% | 71% | 69% | Data Leakage Detected (DLeak) |
| `i14` | YOLO11N | 200 | 16 | 80% | 73% | 73% | Data Leakage Detected (DLeak) |
| `i15` | YOLO11N | 220 | 16 | 75% | 65% | 59% | Unseen Data Baseline (Strict isolation) |
| `i16` | YOLO11N | 220 | 16 | 78% | 66% | 66% | Bootcamp augmentations (mosaic=1.0, translate=0.5) |
| `i17` | YOLO11N | 221 | 15 | 82% | 69% | 72% | SGD Optimizer + Bootcamp (Current Best) |

### Major Milestones & Insights

To ensure continuous improvement, log the results of every major training run here to compare and contrast how different hyperparameter combinations affect the `mAP50`.

| Date | Dataset Size | Model | Resolution (`imgsz`) | Batch Size | Epochs | mAP50 | Notes / Insights |
|---|---|---|---|---|---|---|---|
| **Jul 16** | 16 Train / 2 Val | `yolo11n.pt` | 1024 | 16 | 150 | **35.8%** | *Initial YOLO11 baseline. Good balance of speed and detail.* |
| **Jul 16** | 16 Train / 2 Val | `yolo11n.pt` | 2048 | 4 | 150 | **33.1%** | *Massive VRAM usage (caused OOM at batch 24). Accuracy dropped due to microscopic noise and artifacts distracting the model.* |
| **Jul 16** | 16 Train / 2 Val | `yolo11n.pt` | 768 | 24 | 150 | **31.1%** | *Downscaling too far caused loss of critical pollen grain details.* |
| **Jul 20** | ~50 Train / ~13 Val | `yolo11s.pt` | 1024 | 4 | 150 | **31.8%** | *Standard run (no K-Fold). Model struggled to generalize.* |
| **Jul 20** | 63 (K-Fold=5) | `yolo11s.pt` | 1024 | 4 | 150 | **49.8%** | *Upgraded to K-Fold Cross Validation. Massive ~18% improvement in accuracy due to robust dataset splitting.* |
| **Jul 20** | 79 Train / 10 Val | `yolo11n.pt` | 1024 | - | 150 | **55.3%** | *Active learning dataset scaling! Massive jump in accuracy by simply feeding the model more corrected data.* |
| **Jul 21** | 165 Train / 20 Val | `yolo11n.pt` | 1024 | 4 | 150 | **61.3%** | *Breakthrough! First time breaking 60%. Precision (69.7%) and Recall (67.4%) are incredibly balanced. Nano model is now reaching its limits.* || **Jul 21** | 164 Train / 20 Val | `yolo11s.pt` | 1024 | 4 | 150 | **57.7%** | *Attempted to upgrade to YOLO11s, but accuracy dropped compared to Nano. Model peaked too early (Epoch 50). Conclusion: ~165 images is still not enough data for the larger 'Small' model. Reverting to Nano until dataset hits 300+.* |
| **Jul 22** | 164 Train / 20 Val | `yolo11n.pt` | 1024 | 4 | 150 | **54.4%** | *Pre-purge baseline. High score was artificially inflated due to overlapping duplicate bounding boxes in the dataset.* |
| **Jul 22** | 192 Train / 14 Val | `yolo11n.pt` | 1024 | 4 | 150 | **41.1%** | *True Baseline! Dataset was purged of hundreds of corrupted duplicate boxes using `clean_all_duplicates.py`. This is the first honest, un-inflated metric on a fully sanitized dataset.* |
| **Jul 22** | 192 Train / 14 Val | `yolo11n.pt` | 1024 | 4 | 100 | **50.0%** | *Extreme Augmentation Run (`scale=0.5, hsv=0.4, translate=0.3`). By forcing the model to learn on chaotic, heavily augmented images, accuracy surged from 41% to 50% (peaking at 51.3% mid-training).* |
| **Jul 28** | 220 Train / 16 Val | `yolo11n.pt` | 1024 | 4 | 258 | **59.9%** | *`i15` Model. Unseen Data Baseline. Old leaked validation data was moved to training, and a 100% brand-new unseen validation set of 16 images was created to prevent data leakage. Hit 59.9% mAP50 and 75.6% Precision.* |
| **Jul 28** | 220 Train / 16 Val | `yolo11n.pt` | 1024 | 4 | 295 | **66.3%** | *`i16` Model. Bootcamp Augmentations. Cranked `mosaic: 1.0` and `translate: 0.5`. Training was violently hard, so `patience` was bumped to 150. Model broke through at Epoch 145 and scored an incredibly robust 66.3% mAP50 and 78.0% Precision on totally unseen data. Best model to date!* |
| **Jul 28** | 220 Train / 16 Val | `yolo11n.pt` | 1024 | 4 | 256 | **72.1%** | *`i17` Model (SGD). Switched optimizer to SGD (`lr0=0.01`). Model oscillated violently but eventually found a massive global minimum at Epoch 106, scoring **72.1% mAP50** and an unprecedented **82.4% Precision**. SGD + Bootcamp Augmentations is the current winning formula!* |

*Remember to update this table every time a new dataset batch is annotated or a major training setting is changed!*

---

##  Interactive GUI Tools

This project features three custom-built Tkinter desktop applications to make managing, evaluating, and utilizing your dataset incredibly fast.

### 1. Dataset Annotator (`scripts/annotate.py`)
Rapidly build your dataset manually or fix auto-annotated active-learning images.
- **OpenCV Auto-Snapping:** Perfectly "shrink-wraps" your drawn bounding boxes around the dark pollen grain automatically (with an 8% padding to preserve blurry edges). Restored the original, simpler, and highly accurate edge-snapping algorithm.
- **Visual Error Warnings:** Boxes are color-coded in real-time. **Green** (Normal), **Orange** (Overlapping), **Red** (Oversized).
- **Move/Exclude:** Instantly move images between Train, Validation, or Excluded folders. Automatically unloads images to bypass Windows file locks and syncs CLAHE equivalents dynamically.

### 2. Validation Comparator (`scripts/compare_val.py`)
Visually analyze exactly where your model is making mistakes.
- **Side-by-Side Canvas:** Displays your human Ground Truth annotations (Green) directly next to the YOLO Model predictions (Red).
- **Real-Time Inference:** Instantly runs your `best.pt` model dynamically when you switch images.
- **Metrics Engine:** Click "Generate Excel Report" to automatically calculate True Positives (TP), False Positives (FP), False Negatives (FN), Precision, Recall, and F1-Score for every single image using IoU matching!

### 3. Unified Inference Tool (`scripts/inference.py`)
Run your model on thousands of new images.
- **Count & Analyze:** Detects all pollen grains, draws bounding boxes, and generates a massive `pollen_counts.xlsx` spreadsheet.
- **Auto-Annotate:** Feeds raw images through the model and generates YOLO `.txt` labels (using OpenCV to snap them perfectly) so you can pull them into `annotate.py` to fix mistakes and expand your training data.

---

##  Training Pipeline

```bash
python scripts/train.py --model yolo11n.pt --device 0 --epochs 100 --batch 20 --imgsz 1024 --kfolds 5
```

### Key Options

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 100 | Training epochs (Stop early if it overfits) |
| `--batch` | 20 | Batch size (Lower if CUDA OutOfMemoryError occurs) |
| `--imgsz` | 1024 | Input resolution|
| `--device` | `0` | GPU device (`cpu` for CPU-only) |
| `--kfolds` | 1 | Number of folds for K-fold cross-validation (Default: 1 for standard training, 5 for robust evaluation. Warning! the higher the value multiply to waiting time) |

*Note: The script automatically runs a Validation step using the best weights after training completes.*

---

##  Moving Forward (Next Steps)

Now that the foundational architecture, robust K-Fold training, and unified UI tools are complete, the sole focus should shift to **Data Scaling via Active Learning**. 

### The Active Learning Loop:
1. **Acquire Raw Data:** Take a new batch of 50-100 raw microscopy images.
2. **Auto-Annotate:** Run `scripts/inference.py` in **Auto-Annotate** mode on the raw folder.
3. **Review & Correct:** Open `scripts/annotate.py`, point it to the output `review` folders, and rapidly fix any missed or falsely-detected grains.
4. **Merge:** Move those corrected images/labels into your main `datasets/images` and `datasets/labels` folders.
5. **Retrain:** Run `scripts/train.py` to retrain the model on the newly expanded dataset.

*Once you reach **300-500 high-quality annotated images**, you will be able to upgrade the model to YOLO11 Medium (`yolo11m.pt`) and easily achieve 90%+ accuracy.*

---

##  Project Architecture (Config-Driven Modular Pipeline)

The project has recently been refactored from isolated monolithic scripts into a modular pipeline with a strong Separation of Concerns (SoC).

### Shared Library (`src/`)
Instead of duplicating code, all scripts now import shared logic from the `src/` Python package:
- **`src.paths`**: Centralized definitions for all folders and file extensions.
- **`src.bounding_box`**: Unified `BoundingBox` class and IoU calculations.
- **`src.model_utils`**: Helpers for auto-discovering the latest weights and collecting images.
- **`src.settings`**: A robust JSON reader/writer that syncs UI state across all tools (saved in `config/inference_settings.json`).
- **`src.theme`**: Centralized colors and fonts for all Tkinter GUIs.

### Externalized Training Config
Over 20 YOLO hyperparameters (learning rate, augmentations, optimizer) have been extracted from `train.py` into **`config/training.yaml`**. You can now tweak training strategies without touching any Python code.

---

##  Project Structure

```text
PollenCounter-ObjectDetection-YOLO/
├── config/
│   ├── pollen_dataset.yaml       # Dataset paths & class names
│   ├── training.yaml             # Externalized YOLO hyperparameters
│   └── inference_settings.json   # Auto-saved GUI states
├── datasets/
│   ├── images/
│   │   ├── train/                
│   │   ├── val/                  
│   │   └── excluded/             # Images hidden from training
│   └── labels/
│       ├── train/                
│       ├── val/                  
│       └── excluded/             
├── src/                          # Shared Library (Core Logic)
│   ├── bounding_box.py           
│   ├── model_utils.py            
│   ├── paths.py                  
│   ├── settings.py               
│   └── theme.py                  
├── scripts/                      # Entry-Point Tools
│   ├── train.py                  # CLI Training launcher (K-Fold supported)
│   ├── inference.py              # Unified Batch inference & Auto-Annotation
│   ├── annotate.py               # GUI Annotation Tool
│   ├── compare_val.py            # GUI Validation visualizer & reporting
│   └── clean_duplicate_points.py # Utility to purge overlapping bounding boxes
└── README.md
```
