# Precision Quantification of Rice Pollen: Integrating P2-Enhanced YOLO11, Slicing Aided Hyper Inference (SAHI), and Active Learning for High-Fidelity Detection

> Automate the counting of microscopic **rice pollen** grains across massive image datasets containing overlapping clusters and slide debris. Because high-quality, open-source rice pollen image datasets are incredibly rare in the scientific community, this repository provides a dedicated dataset alongside an end-to-end YOLO11 detection pipeline. The system exports per-image pollen counts to a structured `.xlsx` spreadsheet while generating perfectly annotated inference images.

## Citation

If you use this software, pipeline, or dataset in your research, please cite it as follows:

**APA Format:**
Magdadaro, C. R. C., Diocton, R. F. D., et al. (2026). *Precision Quantification of Rice Pollen: Integrating P2-Enhanced YOLO11, Slicing Aided Hyper Inference (SAHI), and Active Learning for High-Fidelity Detection* [Computer software]. GitHub. <https://github.com/cristomagdadaro/PollenCounter-ObjectDetection-YOLO>

**IEEE Format:**
C. R. C. Magdadaro, R. F. D. Diocton, et al., *Precision Quantification of Rice Pollen: Integrating P2-Enhanced YOLO11, Slicing Aided Hyper Inference (SAHI), and Active Learning for High-Fidelity Detection*. GitHub, 2026. [Source Code]. Available: <https://github.com/cristomagdadaro/PollenCounter-ObjectDetection-YOLO>

## Features

- **Automated High-Speed Pollen Counting:** Instantly detect and count thousands of pollen grains across massive image sets, automatically exporting the results to an Excel spreadsheet.
- **Model-Assisted Smart Annotation:** Rapidly build your dataset using a GUI that features auto-snapping bounding boxes and an *Active Learning* pipeline (where the model pre-annotates images for you to simply approve or correct).
- **Microscopy-Optimized Workflow:** Navigate massive, dense slide images easily with Quadrant-View slicing and targeted Regional Recounting for highly clustered areas.
- **Robust Model Training:** Train custom YOLO models with advanced features like K-Fold cross-validation for small datasets, automatic data augmentation, and easy run-resuming.
- **SAHI (Slicing Aided Hyper Inference):** Built-in support for Patch-Based Training and Sliced Inference to seamlessly detect microscopic pollen grains on massive, ultra-high-resolution microscopy imagery without downscaling.
- **Batch Processing Utilities:** Apply powerful operations—like automatic overlapping box cleanup (NMS), edge-snapping, dataset slicing, and scaling—across your entire dataset with a single click.

## Environment Setup

To run training, inference, and the GUI annotation tools, you must set up your Python environment with the required dependencies (such as OpenCV, Ultralytics YOLO, and Pandas).

### 1. Create a Virtual Environment (Recommended)

It is highly recommended to use a Python virtual environment to prevent package conflicts.

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

Once your virtual environment is active, install all required packages:

```bash
pip install -r requirements.txt
```

*(Note: If you plan to train the model on a GPU, ensure you have the appropriate NVIDIA CUDA drivers and a CUDA-enabled PyTorch version installed for hardware acceleration).*

## How to Use

The pipeline is split into three core scripts. Run them from the project root:

### 1. Training the Model (`train.py`)

Train or retrain a YOLO11 model using your annotated dataset.

```bash
python scripts/train.py --model yolo11n.pt --epochs 150 --batch 20 --imgsz 1024
```

**Key Arguments:**

- `--model`: Base model weights (e.g., `yolo11n.pt`)
- `--data`: Path to the dataset YAML config
- `--epochs` / `--batch` / `--imgsz`: Standard YOLO training hyperparameters
- `--kfold`: Set to `5` to enable 5-fold cross-validation for robust accuracy metrics
- `--resume`: Add this flag to resume an interrupted training run

### 2. Manual & Model-Assisted Annotation (`annotate.py`)

Launch the interactive desktop GUI to manually label data or correct model predictions.

```bash
python scripts/annotate.py
```

**Key Capabilities:**

- **Active Learning:** Use your trained model to auto-annotate or correct existing labels on the fly.
- **Regional Recounting:** Drag a box over a dense cluster and let the model recount that specific region dynamically.
- **GUI Tuning:** Adjust confidence and IoU (overlap) thresholds directly in the interface to fine-tune detections.
- **Dataset Management:** Instantly sort images into Training, Validation, or Excluded sets.

### 3. Inference & Counting (`inference.py`)

Run your trained model on new, unseen images to generate counts.

```bash
python scripts/inference.py
```

**Operation Modes:**

- **Count & Analyze:** Automatically counts all pollen grains, draws bounding boxes on the output images, and exports a final `.xlsx` spreadsheet report.
- **Auto-Annotate:** Pre-annotates raw images and exports YOLO `.txt` labels to be later imported into `annotate.py` for Active Learning.
- **Sliced Inference (SAHI):** Enable SAHI inside the GUI to physically slice massive images into `512x512` patches during inference, completely eliminating accuracy loss from YOLO image downscaling.

### 4. Patch-Based Training Preparation (`slice_dataset.py`)

To train your model to identify microscopic details perfectly, use this utility to generate a high-resolution, patch-based dataset.

```bash
python scripts/slice_dataset.py
```

This physically crops your raw high-res images and their bounding box `.txt` labels into overlapping `512x512` patches, saving them into a `datasets_sliced` directory. You can then train a wildly accurate model by pointing the trainer to `config/pollen_dataset_sliced.yaml`.

## Interactive GUI Tools

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

### 4. Real-Time Training Monitor (`scripts/monitor.py`)

Visually track your YOLO model's training progress in real-time.

- **Live Dynamic Graphs:** Automatically reads `results.csv` every 5 seconds and renders interactive Matplotlib graphs for `mAP50`, `mAP50-95`, and `box_loss`.
- **Status Dashboard:** Instantly see your Max Accuracy and Current Epoch without needing to parse the scrolling terminal output.
- **Process Management:** Includes a dedicated "Stop Training" button that gracefully hooks into the Windows process tree to safely terminate training if things go wrong.

## Project Structure

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
