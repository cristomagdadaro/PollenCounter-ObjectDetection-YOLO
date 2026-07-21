#  PollenCounter  YOLO11 Object Detection

> **High-Volume Automated Pollen Counting** using YOLO11.

Automate the counting of microscopic pollen grains across massive image datasets containing overlapping clusters and slide debris. The pipeline trains a custom YOLO11-Nano detector and exports per-image pollen counts to a structured `.xlsx` spreadsheet, while generating annotated images.

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

##  Training History & Results Log

To ensure continuous improvement, log the results of every major training run here to compare and contrast how different hyperparameter combinations affect the `mAP50`.

| Date | Dataset Size | Model | Resolution (`imgsz`) | Batch Size | Epochs | mAP50 | Notes / Insights |
|---|---|---|---|---|---|---|---|
| **Jul 16** | 16 Train / 2 Val | `yolo11n.pt` | 1024 | 16 | 150 | **35.8%** | *Initial YOLO11 baseline. Good balance of speed and detail.* |
| **Jul 16** | 16 Train / 2 Val | `yolo11n.pt` | 2048 | 4 | 150 | **33.1%** | *Massive VRAM usage (caused OOM at batch 24). Accuracy dropped due to microscopic noise and artifacts distracting the model.* |
| **Jul 16** | 16 Train / 2 Val | `yolo11n.pt` | 768 | 24 | 150 | **31.1%** | *Downscaling too far caused loss of critical pollen grain details.* |
| **Jul 20** | ~50 Train / ~13 Val | `yolo11s.pt` | 1024 | 4 | 150 | **31.8%** | *Standard run (no K-Fold). Model struggled to generalize.* |
| **Jul 20** | 63 (K-Fold=5) | `yolo11s.pt` | 1024 | 4 | 150 | **49.8%** | *Upgraded to K-Fold Cross Validation. Massive ~18% improvement in accuracy due to robust dataset splitting.* |
| **Jul 20** | 79 Train / 10 Val | `yolo11n.pt` | 1024 | - | 150 | **55.3%** | *Active learning dataset scaling! Massive jump in accuracy by simply feeding the model more corrected data. Highest score yet.* |

*Remember to update this table every time a new dataset batch is annotated or a major training setting is changed!*

---

##  Annotation GUI Tool

A built-in Tkinter GUI is provided to rapidly build your dataset.

```bash
python scripts/annotate.py
```

### Dataset Management & Annotation Features
- **OpenCV Auto-Snapping:** Just draw a rough box and release your mousethe tool will automatically use OpenCV to perfectly "shrink-wrap" the bounding box around the pollen grain (with an 8% padding to preserve blurry edges). You can also click " Snap Boxes to Edges" to run this on all boxes in an image.
- **Visual Error Warnings:** Bounding boxes are color-coded in real-time.  **Green** means normal.  **Orange** warns you that boxes are overlapping (e.g. clumped pollen).  **Red** means a box is massively oversized.
- **Box Opacity Slider:** Adjust the transparency of the bounding box borders to clearly see the edges of the pollen underneath.
- **Export to JPG:** Use the  Export button to instantly save a flattened image of the current frame and its bounding boxes for reports.
- **Auto-Box:** Double-click or press `Spacebar` to instantly place a default-sized box at your cursor (which also automatically snaps to edges!).
- **Dataset Switcher:** Use the dropdown in the sidebar to switch between viewing your `Train`, `Validation`, and `Excluded` image sets.
- **Move/Exclude:** Use the sidebar buttons to instantly move an image (and its label data) between the Train, Validation, or Excluded folders. Includes an overwrite safety warning. Excluded images are safely hidden and ignored during training.

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
| `--kfolds` | 1 | Number of folds for K-fold cross-validation (Default: 1 for standard training, 5 for robust evaluation. Warning! the higher the value multiply to waiting time)

*Note: The script automatically runs a Validation step using the best weights after training completes.*

---

##  Unified Inference Tool (Count & Auto-Annotate)

We have combined all inference logic into a single, easy-to-use GUI. 

```bash
python scripts/inference.py
```

The GUI offers two **Modes**:
1. ** Count & Analyze:** 
   - Detects all pollen grains.
   - Generates random-colored bounding box dots for clear visibility of overlapping grains.
   - Draws a semi-transparent text overlay with the total count in the center of the output image.
   - Saves the annotated images and a comprehensive `pollen_counts.xlsx` report to the output folder.
   - Unlocks a built-in Result Viewer to quickly flip through results (includes an individual  Save Image button for each slide).

2. ** Auto-Annotate:** 
   - Used for **Active Learning**.
   - Feeds raw, unlabelled images through the model and saves YOLO `.txt` labels.
   - **OpenCV Auto-Snapping:** Automatically utilizes Otsu's thresholding and Contour Detection to perfectly snap the predicted YOLO bounding box to the exact circular edge of the dark pollen grain (including an 8% padding to preserve delicate outer edges).
   - Supports extremely dense slides by raising the `max_det` limit from YOLO's default of 300 to **5,000** objects per image.
   - Copies the images into a `review` folder so you can open them in `scripts/annotate.py`, quickly delete orange overlapping boxes or red errors, and instantly add them to your dataset!

*Both modes feature a real-time **Box Opacity** slider in the unified launch GUI so you can perfectly tune the thickness and transparency of drawn borders.*

---

##  Moving Forward (Next Steps)

Now that the foundational architecture, robust K-Fold training, and unified UI tools are complete, the sole focus should shift to **Data Scaling via Active Learning**. 

### The Active Learning Loop:
1. **Acquire Raw Data:** Take a new batch of 50-100 raw microscopy images.
2. **Auto-Annotate:** Run `scripts/inference.py` in **Auto-Annotate** mode on the raw folder.
3. **Review & Correct:** Open `scripts/annotate.py`, point it to the output `review` folders, and rapidly fix any missed or falsely-detected grains.
4. **Merge:** Move those corrected images/labels into your main `datasets/images` and `datasets/labels` folders.
5. **Retrain:** Run `scripts/train_gui.py` to retrain the model on the newly expanded dataset.

*Once you reach **300-500 high-quality annotated images**, you will be able to upgrade the model to YOLO11 Medium (`yolo11m.pt`) and easily achieve 90%+ accuracy.*

---

##  Project Structure

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
│   ├── train_gui.py              # GUI Training launcher
│   ├── train.py                  # CLI Training launcher
│   ├── inference.py              # Unified Batch inference & Auto-Annotation
│   └── annotate.py               # GUI Annotation Tool
└── README.md
```
