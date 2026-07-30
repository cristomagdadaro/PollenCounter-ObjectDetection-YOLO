# Developer Notes

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

### 9. Patch-Based Training & SAHI (The Breakthrough)
- **The Problem:** Compressing massive 2048x2048 microscope images down to standard YOLO sizes (1024x1024) permanently destroys microscopic features, capping accuracy regardless of dataset size or augmentation.
- **The Solution:** By using **Slicing Aided Hyper Inference (SAHI)** and **Patch-Based Training**, we slice the raw high-resolution images into overlapping 512x512 patches (`scripts/slice_dataset.py`). 
- **The Result:** Training on these unscaled patches caused an explosive jump in accuracy (from ~70% to **96.3% mAP50** in the `i19` model). During inference, the GUI physically slices new images on the fly and stitches the bounding boxes back together using OpenCV.

### 10. The P2 Architecture (Micro-Object Detection)
- **The Problem:** Even with patches, standard YOLO architecture (`P3, P4, P5`) shrinks feature maps by 8x. A tiny 20px pollen grain becomes a 2.5px smudge, making it impossible for the AI to draw a tight box (low `mAP50-95`).
- **The Solution:** We created a custom `config/yolo11s-p2.yaml` architecture that adds a high-resolution **P2 layer** (stride 4, or 4x shrinkage). This acts like a magnifying glass for the neural network.
- **VRAM Warning:** The P2 layer creates massive intermediate feature maps that consume massive amounts of VRAM. An RTX 3090 (24GB) will easily crash if `batch` is set to auto or 24. **For P2 training on 24GB VRAM, `batch` MUST be hardcoded to 16 maximum** (8 or 12 are safer) in `training.yaml`.

---

##  Training History & Major Milestones

Below is a complete matrix of all recorded model iterations (`i1` to `i17`), highlighting how hyperparameter tuning, dataset scaling, and active learning evolved the model's accuracy over time.

| Run ID (Date) | Architecture | Train / Val Imgs | Res / Batch / Epochs | Precision | Recall | mAP50 | mAP50-95 | Remarks & Insights |
|---|---|---|---|---|---|---|---|---|
| `i1` (Jul 16) | YOLO11N | 30 / 2 | 1024 / 16 / 150 | 45.0% | 43.0% | 35.8% | 10.9% | *Initial YOLO11 baseline. Good balance of speed and detail.* |
| *(Test)* (Jul 16) | YOLO11N | 16 / 2 | 2048 / 4 / 150 | - | - | 33.1% | - | *Massive VRAM usage (caused OOM at batch 24). Accuracy dropped due to microscopic noise and artifacts distracting the model.* |
| *(Test)* (Jul 16) | YOLO11N | 16 / 2 | 768 / 24 / 150 | - | - | 31.1% | - | *Downscaling too far caused loss of critical pollen grain details.* |
| `i2` | YOLO11M | 57 / 10 | - | 63.0% | 56.0% | 51.0% | 12.5% | *Upgraded to Medium (overfit).* |
| `i3` | YOLO11S | 57 / 10 | - | 57.0% | 61.0% | 49.0% | 12.0% | *Upgraded to Small.* |
| `i4` | YOLO11N | 57 / 10 | - | 64.0% | 60.0% | 53.0% | 13.5% | *Reverted to Nano (performed best).* |
| *(Test)* (Jul 20) | YOLO11S | ~50 / ~13 | 1024 / 4 / 150 | - | - | 31.8% | - | *Standard run (no K-Fold). Model struggled to generalize.* |
| *(Test)* (Jul 20) | YOLO11S | 63 (K=5) | 1024 / 4 / 150 | - | - | 49.8% | - | *Upgraded to K-Fold Cross Validation. Massive ~18% improvement in accuracy due to robust dataset splitting.* |
| `i5` (Jul 20) | YOLO11N | 79 / 10 | 1024 / - / 150 | 63.0% | 58.0% | 55.3% | 13.7% | *Active learning dataset scaling! Massive jump in accuracy by simply feeding the model more corrected data.* |
| `i6` | YOLO11N | 86 / 13 | - | 63.0% | 63.0% | 53.0% | 14.9% | *Dataset scaling.* |
| `i7` (Jul 21) | YOLO11N | 164 / 20 | 1024 / 4 / 150 | 69.7% | 67.4% | 61.3% | 16.8% | *Breakthrough! First time breaking 60%. Precision (69.7%) and Recall (67.4%) are incredibly balanced. Nano model is now reaching its limits.* |
| `i8` (Jul 21) | YOLO11S | 164 / 20 | 1024 / 4 / 150 | 69.0% | 65.0% | 57.7% | 15.6% | *Attempted to upgrade to YOLO11s, but accuracy dropped compared to Nano. Model peaked too early (Epoch 50). Conclusion: ~165 images is still not enough data for the larger 'Small' model. Reverting to Nano until dataset hits 300+.* |
| *(Test)* (Jul 22) | YOLO11N | 164 / 20 | 1024 / 4 / 150 | - | - | 54.4% | - | *Pre-purge baseline. High score was artificially inflated due to overlapping duplicate bounding boxes in the dataset.* |
| `i9` (Jul 22) | YOLO11N | 192 / 14 | 1024 / 4 / 150 | 64.0% | 64.0% | 41.1% | 14.6% | *True Baseline! Dataset was purged of hundreds of corrupted duplicate boxes using `clean_all_duplicates.py`. This is the first honest, un-inflated metric on a fully sanitized dataset.* |
| `i10` (Jul 22)| YOLO11N | 192 / 14 | 1024 / 4 / 100 | 70.0% | 65.0% | 50.0% | 19.7% | *Extreme Augmentation Run (`scale=0.5, hsv=0.4, translate=0.3`). By forcing the model to learn on chaotic, heavily augmented images, accuracy surged from 41% to 50% (peaking at 51.3% mid-training).* |
| `i11` | YOLO11N | 200 / 16 | - | 79.0% | 70.0% | 68.0% | 20.6% | *Data Leakage Detected (DLeak).* |
| `i12` | YOLO11N | 200 / 16 | - | 74.0% | 66.0% | 59.0% | 18.2% | *Data Leakage Detected (DLeak).* |
| `i13` | YOLO11N | 200 / 16 | - | 79.0% | 71.0% | 69.0% | 21.8% | *Data Leakage Detected (DLeak).* |
| `i14` | YOLO11N | 200 / 16 | - | 80.0% | 73.0% | 73.0% | 23.9% | *Data Leakage Detected (DLeak).* |
| `i15` (Jul 28)| YOLO11N | 220 / 16 | 1024 / 4 / 258 | 75.6% | 65.0% | 59.9% | 16.2% | *`i15` Model. Unseen Data Baseline. Old leaked validation data was moved to training, and a 100% brand-new unseen validation set of 16 images was created to prevent data leakage. Hit 59.9% mAP50 and 75.6% Precision.* |
| `i16` (Jul 28)| YOLO11N | 220 / 16 | 1024 / 4 / 295 | 78.0% | 66.0% | 66.3% | 18.4% | *`i16` Model. Bootcamp Augmentations. Cranked `mosaic: 1.0` and `translate: 0.5`. Training was violently hard, so `patience` was bumped to 150. Model broke through at Epoch 145 and scored an incredibly robust 66.3% mAP50 and 78.0% Precision on totally unseen data. Best model to date!* |
| `i17` (Jul 28)| YOLO11N | 221 / 15 | 1024 / 4 / 256 | 82.4% | 69.0% | 72.1% | 23.6% | *`i17` Model (SGD). Switched optimizer to SGD (`lr0=0.01`). Model oscillated violently but eventually found a massive global minimum at Epoch 106, scoring **72.1% mAP50** and an unprecedented **82.4% Precision**. SGD + Bootcamp Augmentations is the current winning formula!* |
| `i18` (Jul 28)| YOLO11N | 252 / 16 | 1024 / 4 / 263 | 82.8% | 70.0% | 70.4% | 21.9% | *`i18` Model. Reverted workers to 2 for thermal limits. Continued dataset scaling pushed precision to 82.8% but mAP50 settled at 70.4%.* |
| `i19` (Jul 29)| YOLO11N | 252 / 16 | 512 / 4 / 74 | 93.5% | 95.6% | 96.3% | 47.6% | *`i19` Model (Patch-Based Training). Sliced the dataset into 6,000+ 512x512 patches. Set `imgsz=512`, `patience=50`, `scale=0.1`. Accuracy exploded to a staggering **96.3% mAP50**, proving that processing large images without downscaling fixes the small-object detection issue perfectly!* |
| `i20` (Jul 29)| YOLO11S | 260 / 16 | 512 / Auto / 500 | ~94% | ~95% | 97.5% | 48.3% | *Upgraded to YOLO11S (Small) and heavily increased augmentations (`dropout=0.3, scale=0.3, mixup=0.2`). Added capacity allowed mAP50 to hit **97.5%**, but `mAP50-95` hovered around 45% due to architectural limits on tiny objects.* |
| `i21` (Jul 29)| YOLO11N-P2 | 260 / 16 | 512 / 16 / 198 | 93.8% | 94.5% | 97.9% | 49.0% | *`i21` Model (Nano-P2). Added a P2 stride-4 head and 150 patience. Highest score of all time (97.9% mAP50 / 49.0% mAP50-95). Ultimate microscopic detection architecture!* |
| `i22` (Jul 30)| YOLO11N-P2 | 260 / 16 | 512 / 16 / 166 | 94.9% | 94.6% | 98.1% | 53.4% | *`i22` Model (AdamW). Switched optimizer to AdamW, pushing mAP50-95 to 53.4% (vastly tighter bounding boxes). However, the default learning rate (`0.01`) was slightly unstable, causing slower convergence.* |
| `i23` (Jul 30)| YOLO11N-P2 | 260 / 16 | 512 / 16 / 97 | 94.7% | 95.3% | 98.3% | 53.3% | *`i23` Model (AdamW Optimized). Lowered `lr0` to `0.001` to stabilize AdamW, increased `box` loss penalty to `4.0`. Model converged blisteringly fast (peaked at Epoch 45) achieving the highest accuracy and tightest bounding boxes to date!* |

*Remember to update this table every time a new dataset batch is annotated or a major training setting is changed!*

---

##  Moving Forward (Next Steps)

Now that the foundational architecture, robust K-Fold training, and unified UI tools are complete, the sole focus should shift to **Data Scaling via Active Learning**. 

### The Active Learning Loop:
1. **Acquire Raw Data:** Take a new batch of 50-100 raw microscopy images.
2. **Auto-Annotate:** Run `scripts/inference.py` in **Auto-Annotate** mode on the raw folder.
3. **Review & Correct:** Open `scripts/annotate.py`, point it to the output `review` folders, and rapidly fix any missed or falsely-detected grains.
4. **Merge:** Move those corrected images/labels into your main `datasets/images` and `datasets/labels` folders.
5. **Retrain:** Run `scripts/train.py` to retrain the model on the newly expanded dataset.

*Once you reach **300-500 raw, full-resolution annotated microscope images** (which will slice into 10,000+ patches), you will be able to upgrade the model to YOLO11 Medium (`yolo11m.pt`) and push accuracy even higher.*


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


