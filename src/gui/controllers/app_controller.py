from __future__ import annotations
from src.paths import *
from src.bounding_box import BoundingBox, calculate_iou_corners
from src.settings import load_settings, save_settings
from src.model_utils import get_latest_weights, collect_images
from src.theme import *


from typing import *
from pathlib import Path
import tkinter as tk
import cv2
import numpy as np
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw


import tkinter as tk
import cv2
import numpy as np
from tkinter import ttk, messagebox, filedialog
from tkinter import messagebox
from pathlib import Path
import sys

from src.gui.models.annotation_data import AnnotationData
from src.gui.views.main_window import MainWindow
from src.paths import PROJECT_ROOT

class AppController:
    """Orchestrates interaction between AnnotationData and MainWindow."""

    def __init__(self, root: tk.Tk, images_dir: Path, labels_dir: Path):
        self.root = root
        
        # 1. Initialize Model
        self.model = AnnotationData(images_dir, labels_dir)
        
        # 2. Initialize View
        self.view = MainWindow(root)
        
        # 3. Bind UI Events to Controller Methods
        self._bind_events()
        
        # 4. Initial UI Update
        self._update_dataset_combo()
        self._load_current_image()

    def _bind_events(self):
        # Navigation
        self.view.next_btn.config(command=self.next_image)
        self.view.prev_btn.config(command=self.prev_image)
        self.view.image_combo.bind("<<ComboboxSelected>>", self.on_image_selected)
        
        # Datasets
        self.view.dataset_combo.bind("<<ComboboxSelected>>", self.on_dataset_changed)
        self.view.move_train_btn.config(command=lambda: self.move_image("Train"))
        self.view.move_val_btn.config(command=lambda: self.move_image("Validation"))
        self.view.exclude_btn.config(command=lambda: self.move_image("Excluded"))
        
        # Menu
        self.view.tools_menu.add_command(label="🔍 Inference (Count & Auto-Annotate)", command=self.open_inference)
        self.view.tools_menu.add_command(label="📊 Compare (Human vs Model)", command=self.open_compare)
        self.view.tools_menu.add_separator()
        self.view.tools_menu.add_command(label="📹 Live Video Feed", command=self.open_live_video)

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("a", lambda e: self.prev_image())
        self.root.bind("d", lambda e: self.next_image())

    def _update_dataset_combo(self):
        combo_values = ["Train", "Validation", "Excluded"]
        if self.model.current_set not in combo_values:
            combo_values.append(self.model.current_set)
        self.view.dataset_combo.config(values=combo_values)
        self.view.dataset_combo.set(self.model.current_set)

    def _load_current_image(self):
        # Update image combo
        image_names = [p.name for p in self.model.image_paths]
        self.view.image_combo.config(values=image_names)
        
        img_path = self.model.get_current_image_path()
        if not img_path:
            self.view.status.config(text="No images found.")
            self.view.image_combo.set("")
            self.view.count_label.config(text="Boxes: 0")
            return

        self.view.image_combo.set(img_path.name)
        self.model.load_labels()
        self.view.count_label.config(text=f"Boxes: {len(self.model.boxes)}")
        self.view.status.config(text=f"Loaded {img_path.name}")
        
        # TODO: Trigger CanvasController to draw the image and boxes

    def next_image(self):
        if self.model.next_image():
            self._load_current_image()

    def prev_image(self):
        if self.model.prev_image():
            self._load_current_image()

    def on_image_selected(self, event=None):
        selected_name = self.view.image_combo.get()
        try:
            idx = next(i for i, p in enumerate(self.model.image_paths) if p.name == selected_name)
            if idx != self.model.current_idx:
                self.model.save_labels()
                self.model.current_idx = idx
                self._load_current_image()
        except StopIteration:
            pass

    def on_dataset_changed(self, event=None):
        new_set = self.view.dataset_combo.get()
        if self.model.change_dataset(new_set):
            self._load_current_image()

    def move_image(self, target_set: str):
        # TODO: Implement move image logic using self.model
        pass

    def open_inference(self):
        import subprocess
        subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "inference.py")])

    def open_compare(self):
        import subprocess
        subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "compare_val.py")])

    def open_live_video(self):
        from src.model_utils import get_latest_weights
        weights = get_latest_weights()
        if not weights or not weights.exists():
            messagebox.showerror("Error", "No trained model weights found.")
            return
        import subprocess
        subprocess.Popen([sys.executable, str(PROJECT_ROOT / "scripts" / "live_video.py"), "--model", str(weights), "--source", "0"])
