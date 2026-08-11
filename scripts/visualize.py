#!/usr/bin/env python3
"""PollenCounter YOLO Feature Map Visualizer.

This GUI tool runs a single image through a selected YOLO model and extracts
the internal convolutional feature maps (activations) to visually demonstrate
how the AI processes the image step-by-step.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw, ImageFont
import re
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import PROJECT_ROOT, RUNS_DETECT, DATASET_ROOT, SETTINGS_JSON
from src.theme import BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR

class FeatureVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Neural Network Visualizer")
        self.root.geometry("1200x800")
        self.root.configure(bg=BG_COLOR)

        # State
        self.image_path = None
        self.feature_maps = []  # List of PIL PhotoImage
        self.layer_names = []
        self.current_layer_idx = 0

        self._build_ui()
        self._load_models()
        self._load_settings()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

    def _build_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="NN Visualizer", font=("Segoe UI", 16, "bold"), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(pady=(20, 10))

        # Model Selection
        tk.Label(sidebar, text="Select Trained Model:", bg=SIDEBAR_BG, fg=TEXT_COLOR, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=15, pady=(10, 2))
        self.model_combo = ttk.Combobox(sidebar, state="readonly", font=("Segoe UI", 9))
        self.model_combo.pack(fill=tk.X, padx=15, pady=(0, 15))
        self.model_combo.bind("<<ComboboxSelected>>", lambda e: self._save_settings())

        # Image Selection
        tk.Label(sidebar, text="Input Image:", bg=SIDEBAR_BG, fg=TEXT_COLOR, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=15, pady=(5, 2))
        self.img_lbl = tk.Label(sidebar, text="No image selected", bg=SIDEBAR_BG, fg="#9CA3AF", font=("Segoe UI", 8), wraplength=270, justify=tk.LEFT)
        self.img_lbl.pack(anchor=tk.W, padx=15, pady=(0, 5))
        
        tk.Button(sidebar, text="Browse Image...", bg="#4B5563", fg="white", bd=0, cursor="hand2", command=self._browse_image).pack(fill=tk.X, padx=15, pady=(0, 15))

        # Parameters
        param_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        param_frame.pack(fill=tk.X, padx=15, pady=(0, 10))
        param_frame.columnconfigure(0, weight=1)
        
        tk.Label(param_frame, text="Conf Threshold:", bg=SIDEBAR_BG, fg=TEXT_COLOR, font=("Segoe UI", 9)).grid(row=0, column=0, sticky=tk.W, pady=2)
        self.conf_var = tk.StringVar(value="0.25")
        tk.Entry(param_frame, textvariable=self.conf_var, width=6).grid(row=0, column=1, sticky=tk.E, pady=2)
        
        tk.Label(param_frame, text="IoU Threshold:", bg=SIDEBAR_BG, fg=TEXT_COLOR, font=("Segoe UI", 9)).grid(row=1, column=0, sticky=tk.W, pady=2)
        self.iou_var = tk.StringVar(value="0.45")
        tk.Entry(param_frame, textvariable=self.iou_var, width=6).grid(row=1, column=1, sticky=tk.E, pady=2)

        # Generate Button
        self.generate_btn = tk.Button(sidebar, text="Generate Feature Maps", bg=ACCENT, fg="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", command=self._generate)
        self.generate_btn.pack(fill=tk.X, padx=15, pady=(5, 20))

        # Status
        self.status_lbl = tk.Label(sidebar, text="", bg=SIDEBAR_BG, fg="#10B981", font=("Segoe UI", 9, "bold"))
        self.status_lbl.pack(pady=10)
        
        # Layer Guide
        tk.Frame(sidebar, bg="#374151", height=1).pack(fill=tk.X, padx=15, pady=5)
        tk.Label(sidebar, text="Layer Guide:", bg=SIDEBAR_BG, fg=TEXT_COLOR, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=15)
        self.guide_lbl = tk.Label(sidebar, text="Select a model and image to begin.", bg=SIDEBAR_BG, fg="#9CA3AF", font=("Segoe UI", 9), wraplength=270, justify=tk.LEFT)
        self.guide_lbl.pack(anchor=tk.W, padx=15, pady=5, fill=tk.X)

        # Main Canvas Area
        self.main_area = tk.Frame(self.root, bg=BG_COLOR)
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Navigation header
        self.nav_frame = tk.Frame(self.main_area, bg=BG_COLOR)
        self.nav_frame.pack(fill=tk.X, pady=10)
        
        self.prev_btn = tk.Button(self.nav_frame, text="◄ Previous Layer", bg="#4B5563", fg="white", bd=0, cursor="hand2", state=tk.DISABLED, command=self._prev_layer)
        self.prev_btn.pack(side=tk.LEFT, padx=20)
        
        self.layer_lbl = tk.Label(self.nav_frame, text="Layer: N/A", bg=BG_COLOR, fg=TEXT_COLOR, font=("Segoe UI", 12, "bold"))
        self.layer_lbl.pack(side=tk.LEFT, expand=True)

        self.next_btn = tk.Button(self.nav_frame, text="Next Layer ►", bg="#4B5563", fg="white", bd=0, cursor="hand2", state=tk.DISABLED, command=self._next_layer)
        self.next_btn.pack(side=tk.RIGHT, padx=20)

        # Image display
        self.canvas = tk.Canvas(self.main_area, bg="#111827", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        # Bind resize event to scale image
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        
        # Bind arrow keys
        self.root.bind("<Left>", lambda e: self._prev_layer())
        self.root.bind("<Right>", lambda e: self._next_layer())

    def _load_models(self):
        self.model_paths = {}
        
        # Load trained models
        if RUNS_DETECT.exists():
            for d in RUNS_DETECT.iterdir():
                if d.is_dir():
                    weight_path = d / "weights" / "best.pt"
                    if weight_path.exists():
                        self.model_paths[f"trained: {d.name}"] = weight_path
                        
        # Load pretrained models
        pretrained_dir = PROJECT_ROOT / "pretrained_models"
        if pretrained_dir.exists():
            for f in pretrained_dir.glob("*.pt"):
                self.model_paths[f"pretrained: {f.name}"] = f
                
        if self.model_paths:
            self.model_combo['values'] = list(self.model_paths.keys())
            self.model_combo.set(list(self.model_paths.keys())[-1])
            
    def _browse_image(self):
        start_dir = str(DATASET_ROOT / "images" / "train") if DATASET_ROOT.exists() else str(PROJECT_ROOT)
        path = filedialog.askopenfilename(initialdir=start_dir, title="Select Image", filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            self.image_path = Path(path)
            self.img_lbl.config(text=self.image_path.name)
            self._save_settings()
            
    def _generate(self):
        selection = self.model_combo.get()
        if not selection or selection not in getattr(self, 'model_paths', {}):
            messagebox.showerror("Error", "Select a valid model first")
            return
        if not self.image_path:
            messagebox.showerror("Error", "Select an image first")
            return
            
        model_weights = self.model_paths[selection]
        
        self.status_lbl.config(text="Processing...", fg="#F59E0B")
        self.generate_btn.config(state=tk.DISABLED)
        self.root.update()
        
        try:
            from ultralytics import YOLO
            import glob
            
            # Predict and visualize
            model = YOLO(str(model_weights))
            
            # YOLO visualize=True saves to runs/detect/predict[n]
            # We will grab the latest predict folder after inference
            existing_predicts = set(RUNS_DETECT.glob("predict*"))
            
            try:
                conf = float(self.conf_var.get())
                iou = float(self.iou_var.get())
            except ValueError:
                conf, iou = 0.25, 0.45
                
            results = model.predict(source=str(self.image_path), visualize=True, save=False, imgsz=512, conf=conf, iou=iou)
            
            new_predicts = set(RUNS_DETECT.glob("predict*")) - existing_predicts
            if not new_predicts:
                # Fallback if it re-used a predict folder, find newest
                all_predicts = list(RUNS_DETECT.glob("predict*"))
                if all_predicts:
                    all_predicts.sort(key=lambda x: x.stat().st_mtime)
                    latest_dir = all_predicts[-1]
                else:
                    raise Exception("No prediction directory found.")
            else:
                latest_dir = list(new_predicts)[0]
                
            # Find all feature maps (stage*_features.png)
            feature_images = list(latest_dir.rglob("*_features.png"))
            
            if not feature_images:
                raise Exception(f"No feature maps found in {latest_dir}")
                
            # Sort by stage number
            def extract_stage(p):
                match = re.search(r'stage(\d+)_', p.name)
                return int(match.group(1)) if match else 999
                
            feature_images.sort(key=extract_stage)
            
            self.feature_maps = []
            self.layer_names = []
            
            for f in feature_images:
                # Store layer name (e.g., stage0_Conv)
                name = f.stem.replace("_features", "")
                self.layer_names.append(name)
                self.feature_maps.append(str(f))
                
            # Add custom final predicted image with 1px green, 50% opacity bounding boxes
            try:
                base_img = Image.open(self.image_path).convert("RGBA")
                overlay = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(overlay)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 10)
                except IOError:
                    font = ImageFont.load_default()
                    
                result = results[0]
                if result.boxes:
                    for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                        x1, y1, x2, y2 = box
                        # 1px green outline, 50% opacity (128 alpha)
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 128), width=1)
                        # Label with confidence
                        label = f"{conf:.2f}"
                        # Try to draw text just above the box
                        text_bbox = draw.textbbox((0, 0), label, font=font)
                        th = text_bbox[3] - text_bbox[1]
                        draw.text((x1, max(0, y1 - th - 2)), label, fill=(0, 255, 0, 128), font=font)
                        
                final_combined = Image.alpha_composite(base_img, overlay).convert("RGB")
                custom_final_path = latest_dir / "custom_final_output.jpg"
                final_combined.save(custom_final_path)
                
                self.layer_names.append("Final Output (Bounding Boxes)")
                self.feature_maps.append(str(custom_final_path))
            except Exception as e:
                print(f"Failed to create custom final image: {e}")
                
            self.current_layer_idx = 0
            self._display_current_layer()
            
            self.status_lbl.config(text="Success!", fg="#10B981")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate maps:\n{str(e)}")
            self.status_lbl.config(text="Failed", fg="#EF4444")
        finally:
            self.generate_btn.config(state=tk.NORMAL)

    def _display_current_layer(self):
        if not self.feature_maps: return
        
        layer_path = self.feature_maps[self.current_layer_idx]
        layer_name = self.layer_names[self.current_layer_idx]
        
        self.layer_lbl.config(text=f"Layer {self.current_layer_idx + 1}/{len(self.feature_maps)}: {layer_name}")
        self.guide_lbl.config(text=self._get_layer_explanation(layer_name))
        
        # Load and scale image to fit canvas
        img = Image.open(layer_path)
        
        # Update canvas geometry info first
        self.root.update()
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        
        iw, ih = img.size
        scale = min(cw/iw, ch/ih)
        
        new_w, new_h = int(iw * scale), int(ih * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, image=self.tk_img, anchor=tk.CENTER)
        
        # Update buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_layer_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_layer_idx < len(self.feature_maps)-1 else tk.DISABLED)

    def _on_canvas_resize(self, event):
        if self.feature_maps:
            # simple debounce or just re-render
            self._display_current_layer()

    def _prev_layer(self):
        if self.current_layer_idx > 0:
            self.current_layer_idx -= 1
            self._display_current_layer()
            
    def _next_layer(self):
        if self.current_layer_idx < len(self.feature_maps) - 1:
            self.current_layer_idx += 1
            self._display_current_layer()

    def _get_layer_explanation(self, layer_name):
        name = layer_name.lower()
        if "stage0" in name or "stage1" in name or "stage2" in name:
            return "Early Layers (Conv/Stem):\nThe AI is looking at raw pixels to find basic shapes, hard edges, corners, and color gradients. It doesn't know what pollen is yet; it just sees lines."
        elif "sppf" in name:
            return "SPPF (Spatial Pyramid Pooling):\nThe AI pools information at multiple scales (zoom levels). This helps it recognize pollen grains whether they are tiny smudges or massive blobs."
        elif "c3k2" in name or "c2f" in name or "c2psa" in name:
            return "CSP Bottleneck (Deep Features):\nThe network is combining simple edges into complex textures and patterns (e.g., the spiky or smooth surface of a pollen grain)."
        elif "upsample" in name or "concat" in name:
            return "FPN/PAN (Feature Fusion):\nThe AI is merging high-resolution spatial details (where things are) with deep semantic details (what things are) to accurately draw bounding boxes."
        elif "conv" in name:
            return "Convolutional Layer:\nA standard filter passing over the image to extract localized features and textures."
        elif "detect" in name:
            return "Detection Head:\nThe final step! The AI converts all the abstract patterns it found into dense grids of predictions (not quite boxes yet, but probabilities)."
        elif "final output" in name:
            return "Final Output (NMS & Bounding Boxes):\nThe AI takes the raw predictions, filters out the low-confidence ones, and uses Non-Maximum Suppression (NMS) to delete overlapping boxes. What remains are the final detected pollen grains!"
        else:
            return "Intermediate Layer:\nExtracting abstract mathematical features from the image to build a deeper understanding of the objects."

    def _load_settings(self):
        if not SETTINGS_JSON.exists(): return
        try:
            with open(SETTINGS_JSON, "r") as f:
                config = json.load(f)
                
            # Load visualizer model
            saved_model = config.get("vis_model")
            if saved_model and saved_model in self.model_combo['values']:
                self.model_combo.set(saved_model)
                
            # Load visualizer image
            saved_img = config.get("vis_image")
            if saved_img and Path(saved_img).exists():
                self.image_path = Path(saved_img)
                self.img_lbl.config(text=self.image_path.name)
                
            # Load params
            saved_conf = config.get("vis_conf")
            if saved_conf: self.conf_var.set(str(saved_conf))
            saved_iou = config.get("vis_iou")
            if saved_iou: self.iou_var.set(str(saved_iou))
        except Exception as e:
            print(f"[WARNING] Failed to load visualizer settings: {e}")

    def _save_settings(self):
        config = {}
        if SETTINGS_JSON.exists():
            try:
                with open(SETTINGS_JSON, "r") as f:
                    config = json.load(f)
            except Exception: pass
            
        config["vis_model"] = self.model_combo.get()
        if self.image_path:
            config["vis_image"] = str(self.image_path)
            
        config["vis_conf"] = self.conf_var.get()
        config["vis_iou"] = self.iou_var.get()
            
        try:
            with open(SETTINGS_JSON, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"[WARNING] Failed to save visualizer settings: {e}")
            
    def _on_closing(self):
        self._save_settings()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureVisualizer(root)
    root.mainloop()
