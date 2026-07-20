#!/usr/bin/env python3
"""
annotate.py — Pollen Grain Bounding Box Annotation Tool
=========================================================

A local tkinter GUI for drawing bounding boxes on pollen microscopy
images and saving the annotations in YOLO format.

Features:
  • Click-and-drag to draw bounding boxes
  • Right-click a box to delete it
  • Navigate between images with Next/Previous or keyboard arrows
  • Auto-saves YOLO .txt label files to datasets/labels/train/
  • Shows box count and annotation progress
  • Supports splitting images into train/val sets

Usage:
    python scripts/annotate.py
    python scripts/annotate.py --images datasets/images/train --labels datasets/labels/train
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Optional

from PIL import Image, ImageTk

# ─── Project paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = PROJECT_ROOT / "datasets" / "images" / "train"
DEFAULT_LABELS = PROJECT_ROOT / "datasets" / "labels" / "train"
VAL_IMAGES = PROJECT_ROOT / "datasets" / "images" / "val"
VAL_LABELS = PROJECT_ROOT / "datasets" / "labels" / "val"
EXCLUDED_IMAGES = PROJECT_ROOT / "datasets" / "images" / "excluded"
EXCLUDED_LABELS = PROJECT_ROOT / "datasets" / "labels" / "excluded"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# ─── Colours ────────────────────────────────────────────────────────
BOX_COLOR = "#00FF88"
BOX_COLOR_HOVER = "#FF4444"
ACTIVE_BOX_COLOR = "#FFAA00"
BG_COLOR = "#1E1E2E"
SIDEBAR_BG = "#2A2A3C"
ACCENT = "#7C3AED"
TEXT_COLOR = "#E0E0E0"
PROGRESS_DONE = "#00FF88"
PROGRESS_TODO = "#444466"


class BoundingBox:
    """Represents a single YOLO-format bounding box."""

    def __init__(self, x_center: float, y_center: float, w: float, h: float, class_id: int = 0):
        self.x_center = x_center
        self.y_center = y_center
        self.w = w
        self.h = h
        self.class_id = class_id

    def to_yolo_line(self) -> str:
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.w:.6f} {self.h:.6f}"

    @classmethod
    def from_yolo_line(cls, line: str) -> Optional["BoundingBox"]:
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        try:
            cid, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            return cls(xc, yc, w, h, cid)
        except ValueError:
            return None

    def to_pixel(self, img_w: int, img_h: int) -> tuple[int, int, int, int]:
        """Convert normalised coords to pixel (x1, y1, x2, y2)."""
        px = self.x_center * img_w
        py = self.y_center * img_h
        pw = self.w * img_w
        ph = self.h * img_h
        return (
            int(px - pw / 2),
            int(py - ph / 2),
            int(px + pw / 2),
            int(py + ph / 2),
        )

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union (IoU) with another box."""
        x_left = max(self.x_center - self.w / 2, other.x_center - other.w / 2)
        y_top = max(self.y_center - self.h / 2, other.y_center - other.h / 2)
        x_right = min(self.x_center + self.w / 2, other.x_center + other.w / 2)
        y_bottom = min(self.y_center + self.h / 2, other.y_center + other.h / 2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = self.w * self.h
        area2 = other.w * other.h
        
        return intersection / (area1 + area2 - intersection)


class AnnotationApp:
    """Main tkinter annotation application."""

    # Maximum canvas display size
    MAX_CANVAS_W = 960
    MAX_CANVAS_H = 720

    def __init__(self, root: tk.Tk, images_dir: Path, labels_dir: Path, compare_labels_dir: Optional[Path] = None):
        self.root = root
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.compare_labels_dir = compare_labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Ensure val dirs exist
        VAL_IMAGES.mkdir(parents=True, exist_ok=True)
        VAL_LABELS.mkdir(parents=True, exist_ok=True)
        EXCLUDED_IMAGES.mkdir(parents=True, exist_ok=True)
        EXCLUDED_LABELS.mkdir(parents=True, exist_ok=True)

        self.set_paths = {
            "Train": (DEFAULT_IMAGES, DEFAULT_LABELS),
            "Validation": (VAL_IMAGES, VAL_LABELS),
            "Excluded": (EXCLUDED_IMAGES, EXCLUDED_LABELS)
        }
        
        self.current_set = "Custom"
        for name, (img_dir, _) in self.set_paths.items():
            if self.images_dir.resolve() == img_dir.resolve():
                self.current_set = name
                break

        # ── Image list ──────────────────────────────────────────────
        self.image_paths: list[Path] = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not self.image_paths:
            print(f"[WARN] No images found in {self.images_dir}")

        self.current_idx = 0
        self.boxes: list[BoundingBox] = []
        self.compare_boxes: list[BoundingBox] = []
        self.canvas_ids: list[int] = []  # canvas rectangle IDs

        # ── Drawing state ───────────────────────────────────────────
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.temp_rect: Optional[int] = None

        # ── Image display state ─────────────────────────────────────
        self.display_scale = 1.0
        self.zoom_level = 1.0
        self.orig_w = 0
        self.orig_h = 0
        self.pil_img: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        
        self._calculate_default_box_size()

        # ── Build UI ────────────────────────────────────────────────
        self._build_ui()
        self._load_image()

    # ════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.root.title("🔬 Pollen Annotator — YOLOv11s")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(1100, 700)

        # ── Top bar ─────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=ACCENT, height=48)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Label(
            top, text="🔬  Pollen Grain Annotator", font=("Segoe UI", 14, "bold"),
            bg=ACCENT, fg="white"
        ).pack(side=tk.LEFT, padx=16)

        self.progress_label = tk.Label(
            top, text="", font=("Segoe UI", 11), bg=ACCENT, fg="#DDD"
        )
        self.progress_label.pack(side=tk.RIGHT, padx=16)

        # ── Main area (canvas + sidebar) ────────────────────────────
        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill=tk.BOTH, expand=True)

        # Canvas
        canvas_frame = tk.Frame(main, bg="#000000", bd=2, relief=tk.SUNKEN)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)

        self.canvas = tk.Canvas(
            canvas_frame, bg="#111111", cursor="crosshair",
            highlightthickness=0
        )
        self.vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)

        self.hbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Sidebar
        sidebar = tk.Frame(main, bg=SIDEBAR_BG, width=260)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        sidebar.pack_propagate(False)

        # ── Sidebar: Dataset Set Selection ──────────────────────────
        tk.Label(
            sidebar, text="Dataset Set", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(16, 2))
        
        combo_values = ["Train", "Validation", "Excluded"]
        if self.current_set not in combo_values:
            combo_values.append(self.current_set)
            
        self.dataset_combo = ttk.Combobox(sidebar, values=combo_values, state="readonly", font=("Segoe UI", 10))
        self.dataset_combo.set(self.current_set)
        self.dataset_combo.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._change_dataset)

        tk.Frame(sidebar, bg="#444466", height=1).pack(fill=tk.X, padx=12, pady=4)

        # ── Sidebar: file info ──────────────────────────────────────
        tk.Label(
            sidebar, text="Current Image", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(8, 2))

        self.file_label = tk.Label(
            sidebar, text="", font=("Consolas", 10), bg=SIDEBAR_BG, fg=TEXT_COLOR,
            wraplength=230, justify=tk.LEFT
        )
        self.file_label.pack(anchor=tk.W, padx=12)

        self.size_label = tk.Label(
            sidebar, text="", font=("Consolas", 9), bg=SIDEBAR_BG, fg="#888"
        )
        self.size_label.pack(anchor=tk.W, padx=12, pady=(0, 8))

        # ── Sidebar: box count ──────────────────────────────────────
        tk.Frame(sidebar, bg="#444466", height=1).pack(fill=tk.X, padx=12, pady=4)

        self.count_label = tk.Label(
            sidebar, text="Boxes: 0", font=("Segoe UI", 20, "bold"),
            bg=SIDEBAR_BG, fg=BOX_COLOR
        )
        self.count_label.pack(pady=12)
        
        # ── Sidebar: comparison toggle ──────────────────────────────
        self.show_compare = tk.BooleanVar(value=True)
        self.chk_compare = tk.Checkbutton(
            sidebar, text="Show Comparison", variable=self.show_compare,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, selectcolor=BG_COLOR,
            activebackground=SIDEBAR_BG, activeforeground=TEXT_COLOR,
            command=self._redraw_boxes
        )
        self.chk_compare.pack(pady=(0, 12))
        if not hasattr(self, 'compare_labels_dir') or not self.compare_labels_dir:
            self.chk_compare.config(state=tk.DISABLED)

        # ── Sidebar: auto box size ──────────────────────────────────
        tk.Frame(sidebar, bg="#444466", height=1).pack(fill=tk.X, padx=12, pady=4)
        tk.Label(
            sidebar, text="Auto-Box Size (px)", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(4, 2))

        size_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        size_frame.pack(anchor=tk.W, padx=12, pady=(0, 4))

        tk.Label(size_frame, text="W:", font=("Consolas", 9), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.entry_w = tk.Entry(size_frame, width=5, font=("Consolas", 10), bg="#333", fg="white", insertbackground="white", bd=0)
        self.entry_w.pack(side=tk.LEFT, padx=(2, 8))

        tk.Label(size_frame, text="H:", font=("Consolas", 9), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.entry_h = tk.Entry(size_frame, width=5, font=("Consolas", 10), bg="#333", fg="white", insertbackground="white", bd=0)
        self.entry_h.pack(side=tk.LEFT, padx=(2, 0))
        self.entry_w.bind("<KeyRelease>", self._update_auto_size)
        self.entry_h.bind("<KeyRelease>", self._update_auto_size)
        
        # Remove focus on Enter or Escape
        self.entry_w.bind("<Return>", lambda e: self.canvas.focus_set())
        self.entry_h.bind("<Return>", lambda e: self.canvas.focus_set())
        self.entry_w.bind("<Escape>", lambda e: self.canvas.focus_set())
        self.entry_h.bind("<Escape>", lambda e: self.canvas.focus_set())

        # ── Sidebar: instructions ───────────────────────────────────
        tk.Frame(sidebar, bg="#444466", height=1).pack(fill=tk.X, padx=12, pady=4)

        instructions = [
            ("🖱 Drag", "Draw box"),
            ("Double-Click", "Auto-box at cursor"),
            ("Spacebar", "Auto-box at cursor"),
            ("Right-click", "Delete box"),
            ("← →  or  A/D", "Prev / Next"),
            ("Scroll or + / -", "Zoom In / Out"),
            ("Ctrl+Z", "Undo last box"),
            ("Ctrl+S", "Save labels"),
        ]
        tk.Label(
            sidebar, text="Controls", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(8, 4))

        for key, action in instructions:
            row = tk.Frame(sidebar, bg=SIDEBAR_BG)
            row.pack(anchor=tk.W, padx=12, pady=1)
            tk.Label(row, text=key, font=("Consolas", 9, "bold"), bg=SIDEBAR_BG, fg="#AAA", width=14, anchor=tk.W).pack(side=tk.LEFT)
            tk.Label(row, text=action, font=("Segoe UI", 9), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)

        # ── Sidebar: buttons ────────────────────────────────────────
        tk.Frame(sidebar, bg="#444466", height=1).pack(fill=tk.X, padx=12, pady=12)

        btn_style = {"font": ("Segoe UI", 10, "bold"), "width": 22, "cursor": "hand2", "bd": 0, "pady": 6}

        self.prev_btn = tk.Button(
            sidebar, text="◀  Previous", bg="#3A3A5C", fg="white",
            activebackground="#4A4A6C", command=self._prev_image, **btn_style
        )
        self.prev_btn.pack(pady=2)

        self.next_btn = tk.Button(
            sidebar, text="Next  ▶", bg=ACCENT, fg="white",
            activebackground="#6D28D9", command=self._next_image, **btn_style
        )
        self.next_btn.pack(pady=2)

        tk.Frame(sidebar, bg="#444466", height=1).pack(fill=tk.X, padx=12, pady=8)

        self.clear_btn = tk.Button(
            sidebar, text="🗑  Clear All Boxes", bg="#DC2626", fg="white",
            activebackground="#B91C1C", command=self._clear_boxes, **btn_style
        )
        self.clear_btn.pack(pady=2)

        tk.Frame(sidebar, bg="#444466", height=1).pack(fill=tk.X, padx=12, pady=8)

        tk.Label(
            sidebar, text="Dataset Management", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(4, 2))

        self.move_train_btn = tk.Button(
            sidebar, text="Move to Train", bg="#059669", fg="white",
            activebackground="#047857", command=lambda: self._move_image("Train"), **btn_style
        )
        self.move_train_btn.pack(pady=2)

        self.move_val_btn = tk.Button(
            sidebar, text="Move to Validation", bg="#2563EB", fg="white",
            activebackground="#1D4ED8", command=lambda: self._move_image("Validation"), **btn_style
        )
        self.move_val_btn.pack(pady=2)

        self.exclude_btn = tk.Button(
            sidebar, text="Exclude Image", bg="#9CA3AF", fg="black",
            activebackground="#6B7280", command=lambda: self._move_image("Excluded"), **btn_style
        )
        self.exclude_btn.pack(pady=2)

        # ── Status bar ──────────────────────────────────────────────
        self.status = tk.Label(
            self.root, text="Ready", font=("Segoe UI", 9),
            bg="#16161E", fg="#888", anchor=tk.W, padx=8
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

        # ── Bind events ────────────────────────────────────────────
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        self.canvas.bind("<Motion>", self._on_mouse_move)

        self.root.bind("<Left>", lambda e: self._prev_image())
        self.root.bind("<Right>", lambda e: self._next_image())
        self.root.bind("a", lambda e: self._prev_image())
        self.root.bind("d", lambda e: self._next_image())
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-s>", lambda e: self._save_labels())
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<plus>", lambda e: self._zoom_in())
        self.root.bind("<equal>", lambda e: self._zoom_in())
        self.root.bind("<minus>", lambda e: self._zoom_out())
        self.root.bind("<space>", self._on_space)

    # ════════════════════════════════════════════════════════════════
    #  IMAGE LOADING
    # ════════════════════════════════════════════════════════════════

    def _load_image(self):
        """Load current image and its existing labels."""
        if not self.image_paths:
            return
            
        path = self.image_paths[self.current_idx]

        # Load with PIL
        self.pil_img = Image.open(path)
        self.orig_w, self.orig_h = self.pil_img.size
        self.zoom_level = 1.0

        # ── Load existing labels ────────────────────────────────────
        self.boxes.clear()
        label_path = self._label_path()
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    box = BoundingBox.from_yolo_line(line)
                    if box:
                        self.boxes.append(box)
                        
        # ── Load comparison labels ──────────────────────────────────
        self.compare_boxes.clear()
        if hasattr(self, 'compare_labels_dir') and self.compare_labels_dir:
            comp_path = self.compare_labels_dir / f"{path.stem}.txt"
            if comp_path.exists():
                with open(comp_path, "r") as f:
                    for line in f:
                        box = BoundingBox.from_yolo_line(line)
                        if box:
                            self.compare_boxes.append(box)

        self._update_size_entries()
        self._render_image()
        self._update_ui()
        
    def _update_size_entries(self):
        if not self.orig_w or not self.orig_h: return
        w_px = int(self.default_w * self.orig_w)
        h_px = int(self.default_h * self.orig_h)
        self.entry_w.delete(0, tk.END)
        self.entry_w.insert(0, str(w_px))
        self.entry_h.delete(0, tk.END)
        self.entry_h.insert(0, str(h_px))

    def _update_auto_size(self, event=None):
        if not self.orig_w or not self.orig_h: return
        try:
            w_px = float(self.entry_w.get())
            h_px = float(self.entry_h.get())
            if w_px > 0 and h_px > 0:
                self.default_w = w_px / self.orig_w
                self.default_h = h_px / self.orig_h
        except ValueError:
            pass # Invalid input, ignore until valid
        
    def _render_image(self):
        """Scale and display the image on the canvas."""
        if not self.pil_img:
            return

        self.canvas.delete("all")
        self.canvas_ids.clear()

        # Calculate scale to fit canvas
        canvas_w = max(self.canvas.winfo_width(), 400)
        canvas_h = max(self.canvas.winfo_height(), 400)
        scale_w = canvas_w / self.orig_w
        scale_h = canvas_h / self.orig_h
        base_scale = min(scale_w, scale_h, 1.0)
        self.display_scale = base_scale * self.zoom_level

        disp_w = int(self.orig_w * self.display_scale)
        disp_h = int(self.orig_h * self.display_scale)

        resized = self.pil_img.resize((disp_w, disp_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.img_offset_x = max((canvas_w - disp_w) // 2, 0)
        self.img_offset_y = max((canvas_h - disp_h) // 2, 0)

        # Update scrollregion
        scroll_w = max(disp_w, canvas_w)
        scroll_h = max(disp_h, canvas_h)
        self.canvas.config(scrollregion=(0, 0, scroll_w, scroll_h))

        self.canvas.create_image(
            self.img_offset_x, self.img_offset_y,
            anchor=tk.NW, image=self.tk_image, tags="image"
        )
        self._redraw_boxes()

    def _zoom_in(self):
        self.zoom_level = min(self.zoom_level * 1.25, 10.0)
        self._render_image()

    def _zoom_out(self):
        self.zoom_level = max(self.zoom_level / 1.25, 0.2)
        self._render_image()

    def _on_mousewheel(self, event):
        if event.delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()

    def _label_path(self) -> Path:
        """Get the label file path for the current image."""
        img_name = self.image_paths[self.current_idx].stem
        return self.labels_dir / f"{img_name}.txt"

    def _on_canvas_resize(self, event):
        """Re-render when canvas is resized."""
        if self.image_paths and self.pil_img:
            self._render_image()

    # ════════════════════════════════════════════════════════════════
    #  COORDINATE TRANSFORMS
    # ════════════════════════════════════════════════════════════════

    def _canvas_to_norm(self, cx: float, cy: float) -> tuple[float, float]:
        """Canvas pixel → normalised image coordinates."""
        ix = (cx - self.img_offset_x) / self.display_scale
        iy = (cy - self.img_offset_y) / self.display_scale
        return ix / self.orig_w, iy / self.orig_h

    def _norm_to_canvas(self, nx: float, ny: float) -> tuple[int, int]:
        """Normalised image coords → canvas pixel."""
        ix = nx * self.orig_w * self.display_scale
        iy = ny * self.orig_h * self.display_scale
        return int(ix + self.img_offset_x), int(iy + self.img_offset_y)

    # ════════════════════════════════════════════════════════════════
    #  DRAWING EVENTS
    # ════════════════════════════════════════════════════════════════

    def _on_press(self, event):
        self.canvas.focus_set()  # Remove focus from any text entries
        self.drawing = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.temp_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline=ACTIVE_BOX_COLOR, width=2, dash=(4, 4)
        )

    def _on_drag(self, event):
        if self.drawing and self.temp_rect:
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            self.canvas.coords(self.temp_rect, self.start_x, self.start_y, cx, cy)

    def _on_release(self, event):
        if not self.drawing:
            return
        self.drawing = False

        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

        # ── Compute box in normalised coords ────────────────────────
        x1, y1 = self.start_x, self.start_y
        x2 = self.canvas.canvasx(event.x)
        y2 = self.canvas.canvasy(event.y)

        # Reject tiny boxes (accidental clicks)
        if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
            return

        # Normalise
        nx1, ny1 = self._canvas_to_norm(min(x1, x2), min(y1, y2))
        nx2, ny2 = self._canvas_to_norm(max(x1, x2), max(y1, y2))

        # Clamp to [0, 1]
        nx1 = max(0.0, min(1.0, nx1))
        ny1 = max(0.0, min(1.0, ny1))
        nx2 = max(0.0, min(1.0, nx2))
        ny2 = max(0.0, min(1.0, ny2))

        xc = (nx1 + nx2) / 2
        yc = (ny1 + ny2) / 2
        w = nx2 - nx1
        h = ny2 - ny1

        if w <= 0 or h <= 0:
            return

        box = BoundingBox(xc, yc, w, h, class_id=0)
        self.boxes.append(box)

        self._redraw_boxes()
        self._save_labels()
        self._update_ui()
        self.status.config(text=f"✅ Box added — total: {len(self.boxes)}")

    def _on_right_click(self, event):
        """Delete the box closest to the right-click position."""
        if not self.boxes:
            return

        # Find which box the click is inside
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        click_nx, click_ny = self._canvas_to_norm(cx, cy)

        for i, box in reversed(list(enumerate(self.boxes))):
            half_w = box.w / 2
            half_h = box.h / 2
            if (box.x_center - half_w <= click_nx <= box.x_center + half_w and
                    box.y_center - half_h <= click_ny <= box.y_center + half_h):
                self.boxes.pop(i)
                self._redraw_boxes()
                self._save_labels()
                self._update_ui()
                self.status.config(text=f"🗑 Box deleted — total: {len(self.boxes)}")
                return

    def _on_mouse_move(self, event):
        self.last_mouse_x = self.canvas.canvasx(event.x)
        self.last_mouse_y = self.canvas.canvasy(event.y)

    def _on_space(self, event):
        if isinstance(event.widget, tk.Entry):
            return  # Allow typing spaces in text boxes (though not needed for numbers)
        if not hasattr(self, 'last_mouse_x'):
            return
        self._create_auto_box(self.last_mouse_x, self.last_mouse_y, "(spacebar)")

    def _on_double_click(self, event):
        """Auto-create a box of default size centered at the double-click."""
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        self._create_auto_box(cx, cy, "")
        
        # Cancel the drag-box that was started by the first click of the double-click
        self.drawing = False
        if self.temp_rect:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None

    def _create_auto_box(self, cx: float, cy: float, suffix: str):
        nx, ny = self._canvas_to_norm(cx, cy)
        
        # Clamp coordinates so box doesn't go off screen
        nx = max(self.default_w / 2, min(1.0 - self.default_w / 2, nx))
        ny = max(self.default_h / 2, min(1.0 - self.default_h / 2, ny))
        
        box = BoundingBox(nx, ny, self.default_w, self.default_h, class_id=0)
        self.boxes.append(box)
        
        self._redraw_boxes()
        self._save_labels()
        self._update_ui()
        self.status.config(text=f"✅ Auto-box added {suffix} — total: {len(self.boxes)}")

    def _calculate_default_box_size(self):
        """Find the median width and height of all existing annotations."""
        widths = []
        heights = []
        for label_file in self.labels_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    box = BoundingBox.from_yolo_line(line)
                    if box:
                        widths.append(box.w)
                        heights.append(box.h)
        
        if widths and heights:
            widths.sort()
            heights.sort()
            self.default_w = widths[len(widths) // 2]
            self.default_h = heights[len(heights) // 2]
        else:
            self.default_w = 0.05
            self.default_h = 0.05

    # ════════════════════════════════════════════════════════════════
    #  BOX RENDERING
    # ════════════════════════════════════════════════════════════════

    def _redraw_boxes(self):
        """Clear and redraw all bounding boxes on the canvas."""
        for cid in self.canvas_ids:
            self.canvas.delete(cid)
        self.canvas_ids.clear()

        # Draw comparison boxes if enabled
        if hasattr(self, 'compare_labels_dir') and self.compare_labels_dir and self.show_compare.get():
            for box in self.compare_boxes:
                x1_n = box.x_center - box.w / 2
                y1_n = box.y_center - box.h / 2
                x2_n = box.x_center + box.w / 2
                y2_n = box.y_center + box.h / 2

                cx1, cy1 = self._norm_to_canvas(x1_n, y1_n)
                cx2, cy2 = self._norm_to_canvas(x2_n, y2_n)

                rect_id = self.canvas.create_rectangle(
                    cx1, cy1, cx2, cy2,
                    outline="#00FFFF", width=2, dash=(4, 4)
                )
                self.canvas_ids.append(rect_id)

        # Draw primary editable boxes
        for i, box in enumerate(self.boxes):
            x1_n = box.x_center - box.w / 2
            y1_n = box.y_center - box.h / 2
            x2_n = box.x_center + box.w / 2
            y2_n = box.y_center + box.h / 2

            cx1, cy1 = self._norm_to_canvas(x1_n, y1_n)
            cx2, cy2 = self._norm_to_canvas(x2_n, y2_n)

            # Sanity checks for annotation errors
            is_microscopic = box.w < 0.005 or box.h < 0.005
            is_massive = box.w > 0.5 or box.h > 0.5
            color = "#FF0000" if (is_microscopic or is_massive) else BOX_COLOR
            outline_w = 4 if (is_microscopic or is_massive) else 2

            rect_id = self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=color, width=outline_w
            )
            self.canvas_ids.append(rect_id)

            # Small label
            label_id = self.canvas.create_text(
                cx1 + 3, cy1 - 10,
                text=f"#{i + 1}", anchor=tk.NW,
                font=("Consolas", 8, "bold"), fill=color
            )
            self.canvas_ids.append(label_id)

    # ════════════════════════════════════════════════════════════════
    #  SAVE / LOAD
    # ════════════════════════════════════════════════════════════════

    def _clean_duplicates(self):
        """Remove boxes that overlap by more than 80% with another box."""
        if len(self.boxes) < 2:
            return
            
        to_remove = set()
        for i in range(len(self.boxes)):
            if i in to_remove: continue
            for j in range(i + 1, len(self.boxes)):
                if j in to_remove: continue
                if self.boxes[i].iou(self.boxes[j]) > 0.8:
                    to_remove.add(j)
                    
        if to_remove:
            self.boxes = [b for i, b in enumerate(self.boxes) if i not in to_remove]
            print(f"[INFO] Removed {len(to_remove)} highly overlapping duplicate boxes.")
            
    def _save_labels(self):
        """Save current boxes to YOLO .txt file."""
        if not self.image_paths:
            return
            
        self._clean_duplicates()
        label_path = self._label_path()
        
        # Only save if there are boxes, or delete the file if there are none to clean up
        if not self.boxes:
            if label_path.exists():
                label_path.unlink()
            return
            
        with open(label_path, "w") as f:
            for box in self.boxes:
                f.write(box.to_yolo_line() + "\n")

    # ════════════════════════════════════════════════════════════════
    #  NAVIGATION
    # ════════════════════════════════════════════════════════════════

    def _next_image(self):
        if self.image_paths and self.current_idx < len(self.image_paths) - 1:
            self._save_labels()
            self.current_idx += 1
            self._load_image()

    def _prev_image(self):
        if self.image_paths and self.current_idx > 0:
            self._save_labels()
            self.current_idx -= 1
            self._load_image()

    # ════════════════════════════════════════════════════════════════
    #  ACTIONS
    # ════════════════════════════════════════════════════════════════

    def _undo(self):
        if self.boxes:
            self.boxes.pop()
            self._redraw_boxes()
            self._save_labels()
            self._update_ui()
            self.status.config(text=f"↩ Undo — boxes: {len(self.boxes)}")

    def _clear_boxes(self):
        if self.boxes:
            if messagebox.askyesno("Clear All", "Delete all boxes on this image?"):
                self.boxes.clear()
                self._redraw_boxes()
                self._save_labels()
                self._update_ui()
                self.status.config(text="🗑 All boxes cleared")

    def _change_dataset(self, event):
        new_set = self.dataset_combo.get()
        if new_set == self.current_set:
            return
        
        if self.image_paths:
            self._save_labels()
            
        self.current_set = new_set
        self.images_dir, self.labels_dir = self.set_paths[self.current_set]
        
        self.image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        
        self.current_idx = 0
        if not self.image_paths:
            self.canvas.delete("all")
            self.pil_img = None
            self.boxes.clear()
            self._update_ui()
            return
            
        self._load_image()

    def _move_image(self, target_set):
        """Move the current image (and its label) to the selected set."""
        if not self.image_paths or self.current_set == target_set:
            return

        img_path = self.image_paths[self.current_idx]
        label_path = self._label_path()
        name = img_path.name
        
        target_img_dir, target_label_dir = self.set_paths[target_set]

        if messagebox.askyesno(f"Move to {target_set}", f"Move '{name}' to {target_set}?"):
            # Move image
            dest_img = target_img_dir / img_path.name
            shutil.move(str(img_path), str(dest_img))

            # Move label if exists
            if label_path.exists():
                dest_label = target_label_dir / label_path.name
                shutil.move(str(label_path), str(dest_label))

            # Remove from list
            self.image_paths.pop(self.current_idx)

            if not self.image_paths:
                self.canvas.delete("all")
                self.pil_img = None
                self.boxes.clear()
                self._update_ui()
                messagebox.showinfo("Empty Set", f"All images in {self.current_set} have been moved.")
                return

            if self.current_idx >= len(self.image_paths):
                self.current_idx = len(self.image_paths) - 1

            self._load_image()
            self.status.config(text=f"📦 '{name}' moved to {target_set}")

    # ════════════════════════════════════════════════════════════════
    #  UI UPDATES
    # ════════════════════════════════════════════════════════════════

    def _update_ui(self):
        total = len(self.image_paths)
        if total == 0:
            self.file_label.config(text="")
            self.size_label.config(text="")
            self.count_label.config(text="Boxes: 0")
            self.progress_label.config(text=f"0/0  •  0 annotated")
            self.root.title(f"🔬 Pollen Annotator — {self.current_set} (Empty)")
            
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)
            if hasattr(self, 'move_train_btn'):
                self.move_train_btn.config(state=tk.DISABLED)
                self.move_val_btn.config(state=tk.DISABLED)
                self.exclude_btn.config(state=tk.DISABLED)
            return

        img_path = self.image_paths[self.current_idx]
        idx = self.current_idx + 1

        # Count how many images already have labels
        annotated = sum(
            1 for p in self.image_paths
            if (self.labels_dir / f"{p.stem}.txt").exists()
            and (self.labels_dir / f"{p.stem}.txt").stat().st_size > 0
        )

        self.file_label.config(text=img_path.name)
        self.size_label.config(text=f"{self.orig_w} × {self.orig_h} px")
        self.count_label.config(text=f"Boxes: {len(self.boxes)}")
        self.progress_label.config(text=f"Image {idx}/{total}  •  {annotated} annotated")
        self.root.title(f"🔬 Pollen Annotator — {self.current_set}: {img_path.name} [{idx}/{total}]")

        # Button states
        self.prev_btn.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_idx < total - 1 else tk.DISABLED)
        
        if hasattr(self, 'move_train_btn'):
            self.move_train_btn.config(state=tk.DISABLED if self.current_set == "Train" else tk.NORMAL)
            self.move_val_btn.config(state=tk.DISABLED if self.current_set == "Validation" else tk.NORMAL)
            self.exclude_btn.config(state=tk.DISABLED if self.current_set == "Excluded" else tk.NORMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate pollen images with bounding boxes.")
    parser.add_argument("--images", type=str, default=str(DEFAULT_IMAGES), help="Folder of images to annotate.")
    parser.add_argument("--labels", type=str, default=str(DEFAULT_LABELS), help="Folder to save YOLO labels.")
    parser.add_argument("--compare-labels", type=str, default=None, help="Folder containing secondary labels to overlay for comparison.")
    return parser.parse_args()


def main():
    args = parse_args()

    root = tk.Tk()
    root.state("zoomed")  # Start maximised on Windows

    app = AnnotationApp(
        root,
        images_dir=Path(args.images),
        labels_dir=Path(args.labels),
        compare_labels_dir=Path(args.compare_labels) if args.compare_labels else None,
    )

    root.mainloop()

    # Final report
    labels_dir = Path(args.labels)
    label_count = sum(1 for f in labels_dir.glob("*.txt") if f.stat().st_size > 0)
    print(f"\n[INFO] Annotation complete — {label_count} label file(s) saved to {labels_dir}")


if __name__ == "__main__":
    main()
