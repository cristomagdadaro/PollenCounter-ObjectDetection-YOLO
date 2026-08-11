#!/usr/bin/env python3
"""PollenCounter YOLO Augmentation Previewer.

Visually preview the effects of Ultralytics YOLO augmentations defined in training.yaml
before actually training the model.
"""

import os
import sys
import yaml
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image, ImageTk

# Ensure albumentations is installed
try:
    import albumentations as A
except ImportError:
    print("Installing missing albumentations library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "albumentations>=1.4.0"])
    import albumentations as A

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROJECT_ROOT, DATASET_ROOT
from src.theme import BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR

TRAINING_YAML = PROJECT_ROOT / "config" / "training.yaml"

class AugmentationPreviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Augmentation Previewer")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG_COLOR)
        
        self.yaml_data = {}
        self.image_files = []
        self.current_image_path = None
        self.original_image = None
        self.original_bboxes = []
        self.original_labels = []
        self.tk_img = None
        
        self._load_yaml()
        self._load_dataset_images()
        self._build_ui()
        self._select_random_image()
        
    def _load_yaml(self):
        if TRAINING_YAML.exists():
            with open(TRAINING_YAML, 'r', encoding='utf-8') as f:
                self.yaml_data = yaml.safe_load(f) or {}
        else:
            self.yaml_data = {}

    def _save_yaml(self):
        # Update yaml data from sliders
        for key, var in self.sliders.items():
            try:
                self.yaml_data[key] = float(var.get())
            except ValueError:
                pass
                
        # Make sure directory exists
        TRAINING_YAML.parent.mkdir(parents=True, exist_ok=True)
        
        # Save while preserving order/comments if possible (ruamel.yaml is better, but pyyaml is standard)
        # To avoid destroying comments, we'll just regex replace the specific lines if they exist
        if TRAINING_YAML.exists():
            with open(TRAINING_YAML, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                for key in self.sliders:
                    if line.strip().startswith(f"{key}:"):
                        # Extract comment
                        parts = line.split('#', 1)
                        comment = f" #{parts[1]}" if len(parts) > 1 else ""
                        lines[i] = f"{key}: {self.yaml_data[key]}{comment}\n"
                        
            with open(TRAINING_YAML, 'w', encoding='utf-8') as f:
                f.writelines(lines)
        else:
            with open(TRAINING_YAML, 'w', encoding='utf-8') as f:
                yaml.dump(self.yaml_data, f, default_flow_style=False)
                
        self.status_lbl.config(text="Saved to training.yaml!", fg="#10B981")
        self.root.after(3000, lambda: self.status_lbl.config(text=""))

    def _load_dataset_images(self):
        train_dir = DATASET_ROOT / "images" / "train"
        if train_dir.exists():
            self.image_files = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))

    def _build_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=350)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Augmentation Tuning", font=("Segoe UI", 16, "bold"), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(pady=(20, 10))
        
        # Image Selection
        tk.Label(sidebar, text="Base Image:", bg=SIDEBAR_BG, fg=TEXT_COLOR, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=15, pady=(5, 2))
        
        btn_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        tk.Button(btn_frame, text="Random Image", bg="#4B5563", fg="white", bd=0, cursor="hand2", command=self._select_random_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        tk.Button(btn_frame, text="Browse...", bg="#4B5563", fg="white", bd=0, cursor="hand2", command=self._browse_image).pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.img_lbl = tk.Label(sidebar, text="No image selected", bg=SIDEBAR_BG, fg="#9CA3AF", font=("Segoe UI", 8), wraplength=320, justify=tk.LEFT)
        self.img_lbl.pack(anchor=tk.W, padx=15, pady=(0, 15))

        # Sliders Frame (Scrollable)
        canvas_scroll = tk.Canvas(sidebar, bg=SIDEBAR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=canvas_scroll.yview)
        scrollable_frame = tk.Frame(canvas_scroll, bg=SIDEBAR_BG)

        scrollable_frame.bind("<Configure>", lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")))
        canvas_scroll.create_window((0, 0), window=scrollable_frame, anchor="nw", width=330)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)
        
        canvas_scroll.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Sliders definition
        self.sliders = {}
        
        params = [
            ("hsv_h", "HSV-Hue (Fraction 0-1)", 0.0, 1.0, 0.015),
            ("hsv_s", "HSV-Saturation (Fraction 0-1)", 0.0, 1.0, 0.3),
            ("hsv_v", "HSV-Value (Fraction 0-1)", 0.0, 1.0, 0.3),
            ("degrees", "Rotation (+/- deg)", 0.0, 180.0, 180.0),
            ("translate", "Translate (+/- fraction)", 0.0, 0.5, 0.1),
            ("scale", "Scale (+/- gain)", 0.0, 0.9, 0.1),
            ("shear", "Shear (+/- deg)", 0.0, 45.0, 0.0),
            ("perspective", "Perspective (Fraction)", 0.0, 0.001, 0.0),
            ("flipud", "Flip Up-Down (Prob)", 0.0, 1.0, 0.5),
            ("fliplr", "Flip Left-Right (Prob)", 0.0, 1.0, 0.5),
        ]
        
        for key, label, vmin, vmax, default in params:
            frame = tk.Frame(scrollable_frame, bg=SIDEBAR_BG)
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            # Read from yaml if exists
            val = self.yaml_data.get(key, default)
            
            header = tk.Frame(frame, bg=SIDEBAR_BG)
            header.pack(fill=tk.X)
            tk.Label(header, text=label, bg=SIDEBAR_BG, fg=TEXT_COLOR, font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
            
            val_var = tk.StringVar(value=f"{val:.3f}")
            val_lbl = tk.Label(header, textvariable=val_var, bg=SIDEBAR_BG, fg=ACCENT, font=("Segoe UI", 8))
            val_lbl.pack(side=tk.RIGHT)
            
            scale = ttk.Scale(frame, from_=vmin, to=vmax, orient=tk.HORIZONTAL, command=lambda v, var=val_var, k=key: self._on_slider_change(v, var, k))
            scale.set(val)
            scale.pack(fill=tk.X, pady=(2, 0))
            
            self.sliders[key] = scale

        # Footer Actions
        footer = tk.Frame(sidebar, bg=SIDEBAR_BG)
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        tk.Button(footer, text="Generate New Variation (Space)", bg="#4F46E5", fg="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", command=self._apply_augmentation).pack(fill=tk.X, padx=15, pady=5)
        self.root.bind("<space>", lambda e: self._apply_augmentation())
        
        tk.Button(footer, text="Save to training.yaml", bg="#10B981", fg="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", command=self._save_yaml).pack(fill=tk.X, padx=15, pady=5)
        
        self.status_lbl = tk.Label(footer, text="", bg=SIDEBAR_BG, fg="#10B981", font=("Segoe UI", 9, "bold"))
        self.status_lbl.pack(pady=5)

        # Main Canvas Area
        self.main_area = tk.Frame(self.root, bg=BG_COLOR)
        self.main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Split into original and augmented
        self.top_bar = tk.Frame(self.main_area, bg=BG_COLOR)
        self.top_bar.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(self.top_bar, text="Original Image", bg=BG_COLOR, fg="#9CA3AF", font=("Segoe UI", 12)).pack(side=tk.LEFT, expand=True)
        tk.Label(self.top_bar, text="Augmented Result", bg=BG_COLOR, fg="#9CA3AF", font=("Segoe UI", 12)).pack(side=tk.RIGHT, expand=True)
        
        self.canvas_frame = tk.Frame(self.main_area, bg=BG_COLOR)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.canvas_orig = tk.Canvas(self.canvas_frame, bg="#111827", highlightthickness=0)
        self.canvas_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.canvas_aug = tk.Canvas(self.canvas_frame, bg="#111827", highlightthickness=0)
        self.canvas_aug.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.canvas_orig.bind("<Configure>", lambda e: self._render_images())
        self.canvas_aug.bind("<Configure>", lambda e: self._render_images())
        
    def _on_slider_change(self, val, var, key):
        var.set(f"{float(val):.3f}")
        self._apply_augmentation()

    def _browse_image(self):
        start_dir = str(DATASET_ROOT / "images" / "train") if DATASET_ROOT.exists() else str(PROJECT_ROOT)
        path = filedialog.askopenfilename(initialdir=start_dir, title="Select Image", filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self._load_image(Path(path))

    def _select_random_image(self):
        if not self.image_files:
            return
        path = random.choice(self.image_files)
        self._load_image(path)
        
    def _load_image(self, path):
        self.current_image_path = path
        self.img_lbl.config(text=path.name)
        
        img = cv2.imread(str(path))
        if img is None: return
        self.original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load YOLO labels
        self.original_bboxes = []
        self.original_labels = []
        
        label_dir = DATASET_ROOT / "labels" / path.parent.name
        label_path = label_dir / (path.stem + ".txt")
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        
                        # Clip to ensure Albumentations doesn't throw errors for float precision
                        x_min = max(0.0, xc - w/2)
                        x_max = min(1.0, xc + w/2)
                        y_min = max(0.0, yc - h/2)
                        y_max = min(1.0, yc + h/2)
                        
                        # Recompute bounded values
                        w = x_max - x_min
                        h = y_max - y_min
                        xc = x_min + w/2
                        yc = y_min + h/2
                        
                        # Only add if the box still has area
                        if w > 0 and h > 0:
                            self.original_bboxes.append([xc, yc, w, h])
                            self.original_labels.append(cls_id)
                        
        self._apply_augmentation()
        
    def _apply_augmentation(self):
        if self.original_image is None: return
        
        # Build Albumentations pipeline based on current slider values
        # Note: Albumentations uses different parameter ranges than YOLO in some cases.
        hue = float(self.sliders["hsv_h"].get()) * 180  # YOLO frac of 180
        sat = float(self.sliders["hsv_s"].get()) * 255  # YOLO frac of 255
        val = float(self.sliders["hsv_v"].get()) * 255
        
        deg = float(self.sliders["degrees"].get())
        translate = float(self.sliders["translate"].get())
        scale = float(self.sliders["scale"].get())
        shear = float(self.sliders["shear"].get())
        perspective = float(self.sliders["perspective"].get())
        
        flipud = float(self.sliders["flipud"].get())
        fliplr = float(self.sliders["fliplr"].get())
        
        transform = A.Compose([
            A.ColorJitter(hue=hue/180.0, saturation=sat/255.0, brightness=val/255.0, contrast=0, p=1.0),
            A.Affine(
                scale=(max(0.1, 1.0 - scale), 1.0 + scale), 
                translate_percent={"x": (-translate, translate), "y": (-translate, translate)}, 
                rotate=(-deg, deg), 
                shear=(-shear, shear), 
                p=1.0
            ),
            # In Albumentations p=1 means ALWAYS flip. But we want probability of flipping.
            # To simulate randomness across "generations", we randomly apply it based on probability.
            A.HorizontalFlip(p=fliplr),
            A.VerticalFlip(p=flipud)
        ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.01, label_fields=['class_labels']))
        
        # Apply transform
        try:
            # Random seed is automatically used by Albumentations per call
            transformed = transform(image=self.original_image, bboxes=self.original_bboxes, class_labels=self.original_labels)
            self.aug_image = transformed['image']
            self.aug_bboxes = transformed['bboxes']
        except Exception as e:
            print(f"Augmentation Error: {e}")
            self.aug_image = self.original_image.copy()
            self.aug_bboxes = []
            
        self._render_images()
        
    def _render_images(self):
        if self.original_image is None: return
        
        self.root.update_idletasks()
        cw = max(self.canvas_orig.winfo_width(), 100)
        ch = max(self.canvas_orig.winfo_height(), 100)
        
        # Render Original
        img_orig = self._draw_bboxes(self.original_image, self.original_bboxes)
        self.tk_orig = self._scale_to_canvas(img_orig, cw, ch)
        self.canvas_orig.delete("all")
        self.canvas_orig.create_image(cw//2, ch//2, image=self.tk_orig, anchor=tk.CENTER)
        
        # Render Augmented
        if hasattr(self, 'aug_image'):
            img_aug = self._draw_bboxes(self.aug_image, self.aug_bboxes)
            self.tk_aug = self._scale_to_canvas(img_aug, cw, ch)
            self.canvas_aug.delete("all")
            self.canvas_aug.create_image(cw//2, ch//2, image=self.tk_aug, anchor=tk.CENTER)

    def _draw_bboxes(self, image_np, bboxes):
        img_copy = image_np.copy()
        h, w = img_copy.shape[:2]
        
        for box in bboxes:
            xc, yc, bw, bh = box
            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)
            
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        return Image.fromarray(img_copy)
        
    def _scale_to_canvas(self, pil_img, cw, ch):
        iw, ih = pil_img.size
        scale = min(cw/iw, ch/ih)
        new_w, new_h = int(iw * scale), int(ih * scale)
        if new_w > 0 and new_h > 0:
            pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = AugmentationPreviewer(root)
    root.mainloop()
