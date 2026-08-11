#!/usr/bin/env python3
"""PollenCounter Dataset Analytics Dashboard.

Visualizes dataset statistics to identify class imbalances, edge cases,
and spatial distribution patterns.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.paths import DATASET_ROOT
from src.theme import BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR

class DatasetAnalytics:
    def __init__(self, root):
        self.root = root
        self.root.title("Dataset Analytics Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG_COLOR)
        
        self.split_var = tk.StringVar(value="train")
        
        self.data = {
            "widths": [],
            "heights": [],
            "areas": [],
            "aspect_ratios": [],
            "x_centers": [],
            "y_centers": [],
            "objects_per_image": [],
            "classes": []
        }
        
        self._build_ui()
        # Ensure matplotlib uses dark theme to match our UI
        plt.style.use('dark_background')
        
    def _build_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Dataset Analytics", font=("Segoe UI", 16, "bold"), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(pady=(20, 10))
        
        # Split Selection
        tk.Label(sidebar, text="Dataset Split:", bg=SIDEBAR_BG, fg=TEXT_COLOR, font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=15, pady=(10, 2))
        
        split_combo = ttk.Combobox(sidebar, textvariable=self.split_var, state="readonly", values=["train", "val"])
        split_combo.pack(fill=tk.X, padx=15, pady=(0, 20))
        
        # Action Buttons
        tk.Button(sidebar, text="Run Analytics", bg=ACCENT, fg="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", command=self._run_analytics).pack(fill=tk.X, padx=15, pady=5)
        
        # Status Label
        self.status_lbl = tk.Label(sidebar, text="Ready", bg=SIDEBAR_BG, fg="#9CA3AF", font=("Segoe UI", 9))
        self.status_lbl.pack(pady=20, padx=15, anchor=tk.W)
        
        # Stats summary text
        self.summary_text = tk.Text(sidebar, bg="#1F2937", fg="#D1D5DB", font=("Consolas", 9), height=15, bd=0, padx=10, pady=10)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        self.summary_text.insert(tk.END, "Summary statistics will\nappear here after run.")
        self.summary_text.config(state=tk.DISABLED)

        # Main Plot Area
        self.main_area = tk.Frame(self.root, bg=BG_COLOR)
        self.main_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_area)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
    def _run_analytics(self):
        split = self.split_var.get()
        label_dir = DATASET_ROOT / "labels" / split
        image_dir = DATASET_ROOT / "images" / split
        
        if not label_dir.exists() or not image_dir.exists():
            messagebox.showerror("Error", f"Dataset split '{split}' not found at {label_dir}")
            return
            
        self.status_lbl.config(text="Parsing dataset...", fg="#F59E0B")
        self.root.update()
        
        self.data = {
            "widths": [],
            "heights": [],
            "areas": [],
            "aspect_ratios": [],
            "x_centers": [],
            "y_centers": [],
            "objects_per_image": [],
            "classes": []
        }
        
        label_files = list(label_dir.glob("*.txt"))
        total_images = len(label_files)
        
        for lf in label_files:
            # Find matching image to get real dimensions
            img_path = image_dir / (lf.stem + ".jpg")
            if not img_path.exists():
                img_path = image_dir / (lf.stem + ".png")
                
            img_w, img_h = 512, 512 # Fallback
            if img_path.exists():
                try:
                    with Image.open(img_path) as im:
                        img_w, img_h = im.size
                except:
                    pass
            
            with open(lf, 'r') as f:
                lines = f.readlines()
                self.data["objects_per_image"].append(len(lines))
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xc, yc, w, h = map(float, parts[1:5])
                        
                        # Store normalized centers for heatmap
                        self.data["x_centers"].append(xc)
                        self.data["y_centers"].append(yc)
                        
                        # Store absolute pixels for sizing
                        abs_w = w * img_w
                        abs_h = h * img_h
                        
                        self.data["widths"].append(abs_w)
                        self.data["heights"].append(abs_h)
                        self.data["areas"].append(abs_w * abs_h)
                        self.data["aspect_ratios"].append(abs_w / (abs_h + 1e-6))
                        self.data["classes"].append(cls_id)
                        
        self._update_summary(total_images)
        self._plot_data()
        
        self.status_lbl.config(text="Analytics Complete!", fg="#10B981")

    def _update_summary(self, total_images):
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        
        total_objects = len(self.data["areas"])
        avg_objs = np.mean(self.data["objects_per_image"]) if total_images > 0 else 0
        
        summary = (
            f"Dataset Split: {self.split_var.get().upper()}\n"
            + "-"*25 + "\n"
            f"Total Images : {total_images}\n"
            f"Total Objects: {total_objects}\n"
            f"Avg Objs/Img : {avg_objs:.1f}\n\n"
        )
        
        if total_objects > 0:
            summary += (
                f"Avg Box W  : {np.mean(self.data['widths']):.1f} px\n"
                f"Avg Box H  : {np.mean(self.data['heights']):.1f} px\n"
                f"Avg Aspect : {np.mean(self.data['aspect_ratios']):.2f}\n"
            )
            
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state=tk.DISABLED)

    def _plot_data(self):
        self.fig.clf()
        
        if not self.data["areas"]:
            return
            
        # 2x2 Grid
        gs = gridspec.GridSpec(2, 2, figure=self.fig)
        
        # 1. Bounding Box Sizes (Area)
        ax1 = self.fig.add_subplot(gs[0, 0])
        ax1.hist(self.data["areas"], bins=50, color="#3B82F6", alpha=0.7)
        ax1.set_title("Bounding Box Areas (px²)")
        ax1.set_xlabel("Area (Pixels Squared)")
        ax1.set_ylabel("Frequency")
        
        # 2. Aspect Ratios
        ax2 = self.fig.add_subplot(gs[0, 1])
        ax2.scatter(self.data["widths"], self.data["heights"], alpha=0.5, c="#10B981", s=10)
        ax2.set_title("Box Width vs Height")
        ax2.set_xlabel("Width (px)")
        ax2.set_ylabel("Height (px)")
        
        # Draw 1:1 ratio line
        max_dim = max(max(self.data["widths"]), max(self.data["heights"]))
        ax2.plot([0, max_dim], [0, max_dim], 'r--', alpha=0.5, label='1:1 Aspect Ratio')
        ax2.legend()
        
        # 3. Spatial Heatmap
        ax3 = self.fig.add_subplot(gs[1, 0])
        h, xedges, yedges, image = ax3.hist2d(
            self.data["x_centers"], 
            self.data["y_centers"], 
            bins=20, 
            cmap='inferno',
            range=[[0, 1], [0, 1]]
        )
        ax3.invert_yaxis() # Image coordinates (y=0 is top)
        ax3.set_title("Spatial Location Heatmap")
        ax3.set_xlabel("Normalized X")
        ax3.set_ylabel("Normalized Y")
        
        # 4. Objects per Image
        ax4 = self.fig.add_subplot(gs[1, 1])
        ax4.hist(self.data["objects_per_image"], bins=min(50, max(10, len(set(self.data["objects_per_image"])))), color="#8B5CF6", alpha=0.7)
        ax4.set_title("Objects per Image Distribution")
        ax4.set_xlabel("Number of Objects")
        ax4.set_ylabel("Number of Images")
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetAnalytics(root)
    root.mainloop()
