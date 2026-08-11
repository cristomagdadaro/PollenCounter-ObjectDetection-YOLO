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
- **Dual Architecture Training:** Train custom YOLO11 models with K-Fold cross-validation, or experiment with Faster R-CNN (ResNet50-FPN V2) for academic comparison — all from a single codebase.
- **SAHI (Slicing Aided Hyper Inference):** Built-in support for Patch-Based Training and Sliced Inference to seamlessly detect microscopic pollen grains on massive, ultra-high-resolution microscopy imagery without downscaling.
- **Batch Processing Utilities:** Apply powerful operations — like automatic overlapping box cleanup (NMS), CLAHE contrast enhancement, and dataset slicing — across your entire dataset with a single command.

## Quick Start (Pre-compiled Release)

If you are an end-user (e.g., a scientist or researcher) who simply wants to count pollen without touching any code, you can use the pre-compiled Windows executable:

1. Go to the **[Releases](https://github.com/cristomagdadaro/PollenCounter-ObjectDetection-YOLO/releases)** page on the right side of this repository.
2. Download the latest `PollenCounterStudio.zip` file.
3. Extract the ZIP file to a folder on your computer.
4. Double-click **`PollenCounterStudio.exe`** to launch the unified dashboard.
5. From the dashboard, launch **Batch Inference** to automatically count pollen across an entire folder of images, or launch the **Smart Annotator** to visually inspect and manage your dataset.

## Developer Environment Setup

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

The pipeline is organized into clean scripts (entry-point tools) and shared library modules.

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

### 2. Training Faster R-CNN (`train_rcnn.py`)

Train or evaluate a Faster R-CNN ResNet50-FPN V2 model for academic comparison.

```bash
python scripts/train_rcnn.py                   # Train for 65 epochs
python scripts/train_rcnn.py --epochs 100      # Override epoch count
python scripts/train_rcnn.py --val-only        # Evaluate saved weights (mAP50 / mAP50-95)
```

### 3. The Unified Launcher (`launcher.py`)

Launch the unified desktop GUI to access all tools from a single dashboard.

```bash
python launcher.py
```

The launcher provides access to:
- **Smart Annotator** — Draw and correct bounding boxes with auto-snapping, Regional Recounting, and dataset management (Train/Val/Excluded). Also includes a new **Import Images** tool.
- **Batch Inference** — Run your trained model on new images (Count & Analyze or Auto-Annotate mode), with optional SAHI for high-res sliced inference.
- **Error Logs View** — View application crash and error logs (caught gracefully by the new Global Error Logger).

*(Note: When compiled to a standalone `.exe`, developer-only tools like the Neural Net Visualizer, Training Monitor, and Export Model are automatically hidden to keep the interface clean for end-users).*

### 4. Dataset Preprocessing (`src/preprocessing` & `src/slicer`)

Prepare and clean your dataset using importable library modules:

```bash
# Apply CLAHE contrast enhancement to all images
python -m src.preprocessing --clahe

# Remove duplicate bounding boxes (IoU-based)
python -m src.preprocessing --dedup

# Remove overlapping boxes (NMS-based)
python -m src.preprocessing --nms

# Run all preprocessing steps
python -m src.preprocessing --all

# Slice full-resolution images into 512x512 overlapping patches
python -m src.slicer
python -m src.slicer --slice-size 512 --overlap 0.2
```

### 5. Live Video Feed (`live_video.py`)

Run real-time YOLO inference on a webcam or microscope USB camera:

```bash
python scripts/live_video.py --model "runs/detect/i24_.../weights/best.pt" --source 0
```

### 6. Real-Time Training Monitor (`monitor.py`)

Visually track YOLO training progress in real-time (auto-launched by `train.py`).

```bash
python scripts/monitor.py
```

- **Live Dynamic Graphs:** Automatically reads `results.csv` every 5 seconds and renders interactive Matplotlib graphs for `mAP50`, `mAP50-95`, and `box_loss`.
- **Status Dashboard:** Instantly see your Max Accuracy, Current Epoch, and adjust the **Patience** parameter directly from the UI.
- **Process Management:** Includes a dedicated "Stop Training" button that gracefully hooks into the Windows process tree to safely terminate training if things go wrong.

### 7. Neural Network Visualizer (`visualize.py`)

Launch a specialized GUI to visually demonstrate how the YOLO model processes an image step-by-step.

```bash
.venv\Scripts\python.exe scripts/visualize.py
```

This tool intercepts the internal forward pass of the neural network. You can visually step through every convolutional layer and SPPF block to see exactly how the AI extracts edges, shapes, and pollen grains from raw pixels. It features:
- **Pretrained vs Trained:** Easily swap between raw YOLO models and your fine-tuned models to compare feature extraction.
- **Adjustable Thresholds:** Tweak Confidence and IoU directly in the GUI.
- **Custom Bounding Boxes:** The final layer draws clean, transparent bounding boxes specifically designed for high-visibility visual analysis.

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
│   ├── bounding_box.py           # BoundingBox class & IoU utilities
│   ├── model_utils.py            # Auto-discover latest weights
│   ├── paths.py                  # Centralized path constants
│   ├── preprocessing.py          # CLAHE, duplicate cleaning, NMS
│   ├── rcnn_dataset.py           # YOLO→R-CNN dataset adapter
│   ├── settings.py               # JSON settings reader/writer
│   ├── slicer.py                 # Dataset slicing into patches
│   └── theme.py                  # Centralized colors & fonts
├── scripts/                      # Entry-Point Tools
│   ├── annotate.py               # Unified GUI (Annotate + Inference + Compare tabs)
│   ├── train.py                  # YOLO training (K-Fold supported)
│   ├── train_rcnn.py             # Faster R-CNN training + validation
│   ├── monitor.py                # Real-time training monitor GUI
│   ├── live_video.py             # Live webcam/microscope inference
│   └── epyc_inference.py         # Headless CPU server auto-annotator
└── README.md
```
