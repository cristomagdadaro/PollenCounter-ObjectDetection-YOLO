#!/usr/bin/env python3
"""Pollen grain bounding box annotation tool.

A tkinter GUI for drawing YOLO-format bounding boxes on pollen microscopy images.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageTk, ImageDraw

from src.paths import (
    PROJECT_ROOT, TRAIN_IMAGES, TRAIN_LABELS,
    VAL_IMAGES, VAL_LABELS, EXCLUDED_IMAGES, EXCLUDED_LABELS, IMAGE_EXTS,
)
from src.bounding_box import BoundingBox
from src.settings import load_settings, save_settings
from src.theme import (
    BOX_COLOR, BOX_COLOR_HOVER, ACTIVE_BOX_COLOR,
    BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR, PROGRESS_DONE, PROGRESS_TODO,
    OVERLAP_80_HEX, OVERLAP_80_RGB, OVERLAP_50_HEX, OVERLAP_50_RGB,
    OVERLAP_0_HEX, OVERLAP_0_RGB, AUTO_BOX_HEX, AUTO_BOX_RGB, BOX_RGB
)

# Aliases for backward compatibility within this file
DEFAULT_IMAGES = TRAIN_IMAGES
DEFAULT_LABELS = TRAIN_LABELS


# BoundingBox is now imported from src.bounding_box


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
        self.yolo_model = None
        self.loaded_model_name = None
        
        self._calculate_default_box_size()

        # ── Build UI ────────────────────────────────────────────────
        self._build_ui()
        self._load_settings()
        self._load_image()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ════════════════════════════════════════════════════════════════
    #  UI CONSTRUCTION
    # ════════════════════════════════════════════════════════════════

    def _build_ui(self):
        self.root.title(" Pollen Annotator  YOLOv11s")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(1100, 700)

        # ── Top bar ─────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=ACCENT, height=60)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Label(
            top, text="  Pollen Grain Annotator", font=("Segoe UI", 14, "bold"),
            bg=ACCENT, fg="white"
        ).pack(side=tk.LEFT, padx=16)

        # ── Header Controls ──────────────────────────────────────────
        header_controls = tk.Frame(top, bg=ACCENT)
        header_controls.pack(side=tk.LEFT, padx=40, fill=tk.Y, pady=6)
        
        # 1. Auto-Box Size
        box_size_frame = tk.Frame(header_controls, bg=ACCENT)
        box_size_frame.pack(side=tk.LEFT, padx=10)
        tk.Label(box_size_frame, text="Default Box W:", font=("Consolas", 9), bg=ACCENT, fg="white").pack(side=tk.LEFT)
        self.entry_w = tk.Entry(box_size_frame, width=8, font=("Consolas", 10), bg="#FFFFFF", fg="black", insertbackground="black", bd=0)
        self.entry_w.pack(side=tk.LEFT, padx=2)
        tk.Label(box_size_frame, text="H:", font=("Consolas", 9), bg=ACCENT, fg="white").pack(side=tk.LEFT)
        self.entry_h = tk.Entry(box_size_frame, width=8, font=("Consolas", 10), bg="#FFFFFF", fg="black", insertbackground="black", bd=0)
        self.entry_h.pack(side=tk.LEFT, padx=2)
        
        self.entry_w.bind("<KeyRelease>", self._update_auto_size)
        self.entry_h.bind("<KeyRelease>", self._update_auto_size)
        self.entry_w.bind("<Return>", lambda e: self.canvas.focus_set())
        self.entry_h.bind("<Return>", lambda e: self.canvas.focus_set())
        self.entry_w.bind("<Escape>", lambda e: self.canvas.focus_set())
        self.entry_h.bind("<Escape>", lambda e: self.canvas.focus_set())

        # 2. Box Scale
        tk.Label(header_controls, text="Scale:", font=("Segoe UI", 9, "bold"), bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=(10, 2))
        self.scale_var = tk.StringVar(value="1.0")
        self.scale_entry = tk.Entry(
            header_controls, textvariable=self.scale_var, width=5, font=("Consolas", 10),
            bg="#FFFFFF", fg="black", insertbackground="black", bd=0
        )
        self.scale_entry.pack(side=tk.LEFT, padx=2)
        
        def apply_scale(event):
            try:
                # Safely parse float, replacing comma with dot for European locales
                val_str = self.scale_var.get().replace(',', '.')
                scale_factor = float(val_str)
                if scale_factor <= 0 or scale_factor == 1.0: return
                for box in self.boxes:
                    box.w *= scale_factor
                    box.h *= scale_factor
                self._redraw_boxes()
                self._save_labels()
                self.scale_var.set("1.0")
                self.canvas.focus_set()
            except Exception:
                pass

        self.scale_entry.bind("<Return>", apply_scale)
        self.scale_entry.bind("<Escape>", lambda e: self.canvas.focus_set())

        # 3. Box Opacity
        tk.Label(header_controls, text="Opacity:", font=("Segoe UI", 9, "bold"), bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=(10, 2))
        self.opacity_var = tk.StringVar(value="0.2")
        self.opacity_entry = tk.Entry(
            header_controls, textvariable=self.opacity_var, width=5, font=("Consolas", 10),
            bg="#FFFFFF", fg="black", insertbackground="black", bd=0
        )
        self.opacity_entry.pack(side=tk.LEFT, padx=2)
        
        def update_opacity(*args):
            try:
                val_str = self.opacity_var.get().replace(',', '.')
                val = float(val_str)
                if 0.0 <= val <= 1.0:
                    self._redraw_boxes()
            except Exception:
                pass
                
        self.opacity_var.trace_add("write", update_opacity)
        self.opacity_entry.bind("<Return>", lambda e: self.canvas.focus_set())
        self.opacity_entry.bind("<Escape>", lambda e: self.canvas.focus_set())
        
        # 4. Border Thickness
        tk.Label(header_controls, text="Thickness:", font=("Segoe UI", 9, "bold"), bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=(10, 2))
        self.thickness_var = tk.StringVar(value="3")
        self.thickness_entry = tk.Entry(
            header_controls, textvariable=self.thickness_var, width=5, font=("Consolas", 10),
            bg="#FFFFFF", fg="black", insertbackground="black", bd=0
        )
        self.thickness_entry.pack(side=tk.LEFT, padx=2)
        
        def update_thickness(*args):
            try:
                val = int(self.thickness_var.get())
                if val >= 1:
                    self._redraw_boxes()
            except Exception:
                pass
                
        self.thickness_var.trace_add("write", update_thickness)
        self.thickness_entry.bind("<Return>", lambda e: self.canvas.focus_set())
        self.thickness_entry.bind("<Escape>", lambda e: self.canvas.focus_set())

        # 5. Fit Options
        tk.Button(header_controls, text="Fit W", font=("Segoe UI", 8, "bold"), bg="#4B5563", fg="white", bd=0, cursor="hand2", padx=4, command=self._fit_width).pack(side=tk.LEFT, padx=(10, 2))
        tk.Button(header_controls, text="Fit H", font=("Segoe UI", 8, "bold"), bg="#4B5563", fg="white", bd=0, cursor="hand2", padx=4, command=self._fit_height).pack(side=tk.LEFT, padx=2)

        # 6. Auto-Snap Checkbox
        self.auto_snap = tk.BooleanVar(value=True)
        tk.Checkbutton(
            header_controls, text="Auto-Snap", variable=self.auto_snap,
            bg=ACCENT, fg="white", selectcolor="#FFFFFF",
            activebackground=ACCENT, activeforeground="white"
        ).pack(side=tk.LEFT, padx=(10, 2))

        # 7. Clean Overlaps
        tk.Label(header_controls, text="Max Overlap%:", font=("Segoe UI", 9, "bold"), bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=(10, 2))
        self.clean_threshold_var = tk.StringVar(value="80")
        self.clean_threshold_entry = tk.Entry(
            header_controls, textvariable=self.clean_threshold_var, width=4, font=("Consolas", 10),
            bg="#FFFFFF", fg="black", insertbackground="black", bd=0
        )
        self.clean_threshold_entry.pack(side=tk.LEFT, padx=2)
        tk.Button(header_controls, text="Clean", font=("Segoe UI", 8, "bold"), bg="#EF4444", fg="white", bd=0, cursor="hand2", padx=4, command=self._clean_overlapping_boxes).pack(side=tk.LEFT, padx=2)
        tk.Button(header_controls, text="Undo", font=("Segoe UI", 8, "bold"), bg="#F59E0B", fg="white", bd=0, cursor="hand2", padx=4, command=self._undo_clean_boxes).pack(side=tk.LEFT, padx=2)
        
        self.clean_threshold_var.trace_add("write", lambda *args: self._redraw_boxes())

        # 8. Visibility Filters
        self.show_red = tk.BooleanVar(value=True)
        self.show_orange = tk.BooleanVar(value=True)
        self.show_yellow = tk.BooleanVar(value=True)
        self.show_green = tk.BooleanVar(value=True)
        
        self.chk_red = tk.Checkbutton(header_controls, text="Red (0)", variable=self.show_red, bg=ACCENT, fg="#EF4444", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#EF4444", font=("Segoe UI", 8, "bold"), command=self._redraw_boxes)
        self.chk_red.pack(side=tk.LEFT, padx=(6, 2))
        self.chk_orange = tk.Checkbutton(header_controls, text="Org (0)", variable=self.show_orange, bg=ACCENT, fg="#F97316", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#F97316", font=("Segoe UI", 8, "bold"), command=self._redraw_boxes)
        self.chk_orange.pack(side=tk.LEFT, padx=2)
        self.chk_yellow = tk.Checkbutton(header_controls, text="Yel (0)", variable=self.show_yellow, bg=ACCENT, fg="#FACC15", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#FACC15", font=("Segoe UI", 8, "bold"), command=self._redraw_boxes)
        self.chk_yellow.pack(side=tk.LEFT, padx=2)
        self.chk_green = tk.Checkbutton(header_controls, text="Grn (0)", variable=self.show_green, bg=ACCENT, fg="#22C55E", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#22C55E", font=("Segoe UI", 8, "bold"), command=self._redraw_boxes)
        self.chk_green.pack(side=tk.LEFT, padx=2)

        self.progress_label = tk.Label(
            top, text="", font=("Segoe UI", 11), bg=ACCENT, fg="#FFFFFF"
        )
        self.progress_label.pack(side=tk.RIGHT, padx=16)

        # ── Main area (canvas + sidebar) ────────────────────────────
        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill=tk.BOTH, expand=True)

        # Canvas
        canvas_frame = tk.Frame(main, bg="#CCCCCC", bd=2, relief=tk.SUNKEN)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 4), pady=8)

        self.canvas = tk.Canvas(
            canvas_frame, bg="#FFFFFF", cursor="crosshair",
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

        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=4)

        # ── Sidebar: file info ──────────────────────────────────────
        tk.Label(
            sidebar, text="Current Image", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(8, 2))

        self.image_combo = ttk.Combobox(
            sidebar, state="readonly", font=("Consolas", 9)
        )
        self.image_combo.pack(fill=tk.X, padx=12, pady=(0, 2))
        self.image_combo.bind("<<ComboboxSelected>>", self._on_combo_jump)
        
        nav_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        nav_frame.pack(fill=tk.X, padx=12, pady=(4, 2))
        
        nav_btn_style = {"font": ("Segoe UI", 9, "bold"), "cursor": "hand2", "bd": 0, "pady": 4}
        
        self.prev_btn = tk.Button(
            nav_frame, text="◀ Prev", bg="#888888", fg="white",
            activebackground="#666666", command=self._prev_image, **nav_btn_style
        )
        self.prev_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.next_btn = tk.Button(
            nav_frame, text="Next ▶", bg=ACCENT, fg="white",
            activebackground="#6D28D9", command=self._next_image, **nav_btn_style
        )
        self.next_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self.size_label = tk.Label(
            sidebar, text="", font=("Consolas", 9), bg=SIDEBAR_BG, fg="#666666"
        )
        self.size_label.pack(anchor=tk.W, padx=12, pady=(0, 8))

        # ── Sidebar: box count ──────────────────────────────────────
        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=4)

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



        # ── Sidebar: Recount by Ai ────────────────────────────────
        tk.Label(
            sidebar, text="Recount by Ai", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(12, 2))

        ar_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        ar_frame.pack(anchor=tk.W, padx=12, pady=(0, 4))
        tk.Label(ar_frame, text="Threshold:", font=("Consolas", 9), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.conf_entry = tk.Entry(ar_frame, width=5, font=("Consolas", 10), bg="#FFFFFF", fg="black", insertbackground="black", bd=0)
        self.conf_entry.insert(0, "0.15")
        self.conf_entry.pack(side=tk.LEFT, padx=(2, 8))
        
        self.recount_btn = tk.Button(
            ar_frame, text="Find Missing", bg="#8B5CF6", fg="white",
            activebackground="#7C3AED", command=self._auto_recount, font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", padx=8, pady=2
        )
        self.recount_btn.pack(side=tk.LEFT)

        self.discard_recount_btn = tk.Button(
            ar_frame, text="✖", bg="#EF4444", fg="white",
            activebackground="#DC2626", command=self._discard_recount, font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", padx=6, pady=2
        )
        self.discard_recount_btn.pack(side=tk.LEFT, padx=(4, 0))

        # ── Sidebar: Batch Tools ────────────────────────────────────
        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=8)
        
        tk.Button(
            sidebar, text="⚙ Batch Tools", font=("Segoe UI", 10, "bold"),
            bg="#2563EB", fg="white", activebackground="#1D4ED8",
            bd=0, cursor="hand2", pady=4, command=self._open_batch_tools
        ).pack(fill=tk.X, padx=12, pady=(0, 4))

        # Determine available models for dropdown
        available_models = []
        try:
            detect_dir = PROJECT_ROOT / "runs" / "detect"
            if detect_dir.exists():
                runs = sorted([d for d in detect_dir.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                for run in runs:
                    if (run / "weights" / "best.pt").exists():
                        available_models.append(run.name)
        except Exception:
            pass
            
        if not available_models:
            available_models = ["None"]

        tk.Label(
            sidebar, text="Model:", font=("Consolas", 8),
            bg=SIDEBAR_BG, fg="#888888"
        ).pack(anchor=tk.W, padx=12, pady=(4, 0))
        
        self.model_combo = ttk.Combobox(sidebar, values=available_models, state="readonly", width=30)
        self.model_combo.set(available_models[0])
        self.model_combo.pack(anchor=tk.W, padx=12, pady=(0, 12))






        # ── Sidebar: buttons ────────────────────────────────────────
        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=12)

        btn_style = {"font": ("Segoe UI", 10, "bold"), "width": 22, "cursor": "hand2", "bd": 0, "pady": 6}

        self.snap_btn = tk.Button(
            sidebar, text=" Snap Boxes to Edges", bg="#059669", fg="white",
            activebackground="#047857", command=self._snap_boxes, **btn_style
        )
        self.snap_btn.pack(pady=2)

        self.export_btn = tk.Button(
            sidebar, text=" Export to JPG", bg="#D97706", fg="white",
            activebackground="#B45309", command=self._export_jpg, **btn_style
        )
        self.export_btn.pack(pady=2)



        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=8)

        self.clear_btn = tk.Button(
            sidebar, text="  Clear All Boxes", bg="#DC2626", fg="white",
            activebackground="#B91C1C", command=self._clear_boxes, **btn_style
        )
        self.clear_btn.pack(pady=2)

        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=8)

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

        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=4)

        self.delete_btn = tk.Button(
            sidebar, text=" Delete Permanently", bg="#7F1D1D", fg="white",
            activebackground="#450A0A", command=self._delete_image, **btn_style
        )
        self.delete_btn.pack(pady=2)

        # ── Status bar ──────────────────────────────────────────────
        self.status = tk.Label(
            self.root, text="Ready", font=("Segoe UI", 9),
            bg="#E0E0E0", fg="#666666", anchor=tk.W, padx=8
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
        
    def _fit_width(self):
        if not self.pil_img: return
        canvas_w = max(self.canvas.winfo_width(), 400)
        canvas_h = max(self.canvas.winfo_height(), 400)
        scale_w = canvas_w / self.orig_w
        scale_h = canvas_h / self.orig_h
        base_scale = min(scale_w, scale_h, 1.0)
        self.zoom_level = scale_w / base_scale
        self._render_image()

    def _fit_height(self):
        if not self.pil_img: return
        canvas_w = max(self.canvas.winfo_width(), 400)
        canvas_h = max(self.canvas.winfo_height(), 400)
        scale_w = canvas_w / self.orig_w
        scale_h = canvas_h / self.orig_h
        base_scale = min(scale_w, scale_h, 1.0)
        self.zoom_level = scale_h / base_scale
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
        if self.auto_snap.get():
            self._snap_single_box(box)
        self.boxes.append(box)

        self._redraw_boxes()
        self._save_labels()
        self._update_ui()
        self.status.config(text=f" Box added  total: {len(self.boxes)}")

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
                self.status.config(text=f" Box deleted  total: {len(self.boxes)}")
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
        if self.auto_snap.get():
            self._snap_single_box(box)
        self.boxes.append(box)
        
        self._redraw_boxes()
        self._save_labels()
        self._update_ui()
        self.status.config(text=f" Auto-box added {suffix}  total: {len(self.boxes)}")

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

    def _clean_overlapping_boxes(self):
        if not self.boxes: return
        try:
            threshold = float(self.clean_threshold_var.get()) / 100.0
        except ValueError:
            return
            
        self.boxes_backup = list(self.boxes) # Save for undo
        
        import numpy as np
        boxes_arr = np.array([
            [b.x_center - b.w/2, b.y_center - b.h/2, b.x_center + b.w/2, b.y_center + b.h/2]
            for b in self.boxes
        ])
        
        x1, y1 = boxes_arr[:, 0], boxes_arr[:, 1]
        x2, y2 = boxes_arr[:, 2], boxes_arr[:, 3]
        
        ix1 = np.maximum(x1[:, None], x1[None, :])
        iy1 = np.maximum(y1[:, None], y1[None, :])
        ix2 = np.minimum(x2[:, None], x2[None, :])
        iy2 = np.minimum(y2[:, None], y2[None, :])
        
        inter_w = np.maximum(0.0, ix2 - ix1)
        inter_h = np.maximum(0.0, iy2 - iy1)
        inter_area = inter_w * inter_h
        
        areas = (x2 - x1) * (y2 - y1)
        min_areas = np.minimum(areas[:, None], areas[None, :])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            overlap_matrix = inter_area / min_areas
            overlap_matrix[np.isnan(overlap_matrix)] = 0.0
            overlap_matrix[np.isinf(overlap_matrix)] = 0.0
            
        np.fill_diagonal(overlap_matrix, 0.0)
        
        to_remove = set()
        for i in range(len(self.boxes)):
            if i in to_remove: continue
            for j in range(i+1, len(self.boxes)):
                if j in to_remove: continue
                if overlap_matrix[i, j] > threshold:
                    if areas[j] < areas[i]:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break
        
        if to_remove:
            self.boxes = [b for idx, b in enumerate(self.boxes) if idx not in to_remove]
            self._redraw_boxes()
            self._save_labels()
            self._update_ui()
            self.status.config(text=f" Cleaned {len(to_remove)} boxes with >{int(threshold*100)}% overlap.")
        else:
            self.status.config(text=" No boxes found exceeding overlap threshold.")

    def _undo_clean_boxes(self):
        if hasattr(self, 'boxes_backup') and self.boxes_backup:
            self.boxes = list(self.boxes_backup)
            self._redraw_boxes()
            self._save_labels()
            self._update_ui()
            self.status.config(text=" Restored boxes from backup.")

    # ════════════════════════════════════════════════════════════════
    #  BOX RENDERING
    # ════════════════════════════════════════════════════════════════

    def _redraw_boxes(self):
        """Clear and redraw all bounding boxes on the canvas."""
        for cid in self.canvas_ids:
            self.canvas.delete(cid)
        self.canvas_ids.clear()
        
        box_colors = []
        box_color_strs = []
        box_line_widths = []
        
        import numpy as np
        if self.boxes:
            boxes_arr = np.array([
                [b.x_center - b.w/2, b.y_center - b.h/2, b.x_center + b.w/2, b.y_center + b.h/2]
                for b in self.boxes
            ])
            x1_arr, y1_arr, x2_arr, y2_arr = boxes_arr[:,0], boxes_arr[:,1], boxes_arr[:,2], boxes_arr[:,3]
            
            ix1 = np.maximum(x1_arr[:, None], x1_arr[None, :])
            iy1 = np.maximum(y1_arr[:, None], y1_arr[None, :])
            ix2 = np.minimum(x2_arr[:, None], x2_arr[None, :])
            iy2 = np.minimum(y2_arr[:, None], y2_arr[None, :])
            
            inter_w = np.maximum(0.0, ix2 - ix1)
            inter_h = np.maximum(0.0, iy2 - iy1)
            inter_area = inter_w * inter_h
            
            areas = (x2_arr - x1_arr) * (y2_arr - y1_arr)
            min_areas = np.minimum(areas[:, None], areas[None, :])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                overlap_matrix = inter_area / min_areas
                overlap_matrix[np.isnan(overlap_matrix)] = 0.0
                overlap_matrix[np.isinf(overlap_matrix)] = 0.0
                
            np.fill_diagonal(overlap_matrix, 0.0)
            max_overlaps = np.max(overlap_matrix, axis=1)
        else:
            max_overlaps = []
            
        if hasattr(self, 'chk_red'):
            if self.boxes and len(max_overlaps) > 0:
                red_count = int(np.sum(max_overlaps >= 0.8))
                orange_count = int(np.sum((max_overlaps >= 0.5) & (max_overlaps < 0.8)))
                yellow_count = int(np.sum((max_overlaps > 0.0) & (max_overlaps < 0.5)))
                green_count = int(np.sum(max_overlaps == 0.0))
                
                self.chk_red.config(text=f"Red ({red_count})")
                self.chk_orange.config(text=f"Org ({orange_count})")
                self.chk_yellow.config(text=f"Yel ({yellow_count})")
                self.chk_green.config(text=f"Grn ({green_count})")
            else:
                self.chk_red.config(text="Red (0)")
                self.chk_orange.config(text="Org (0)")
                self.chk_yellow.config(text="Yel (0)")
                self.chk_green.config(text="Grn (0)")
            
        for i, box in enumerate(self.boxes):
            is_massive = box.w > 0.5 or box.h > 0.5
            if is_massive:
                if hasattr(self, 'show_red') and not self.show_red.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append((255, 0, 0))
                    box_color_strs.append("#FF0000")
                continue
                
            max_overlap = max_overlaps[i] if i < len(max_overlaps) else 0.0
                        
            if max_overlap >= 0.8:
                if hasattr(self, 'show_red') and not self.show_red.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(OVERLAP_80_RGB)
                    box_color_strs.append(OVERLAP_80_HEX)
            elif max_overlap >= 0.5:
                if hasattr(self, 'show_orange') and not self.show_orange.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(OVERLAP_50_RGB)
                    box_color_strs.append(OVERLAP_50_HEX)
            elif max_overlap > 0:
                if hasattr(self, 'show_yellow') and not self.show_yellow.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(OVERLAP_0_RGB)
                    box_color_strs.append(OVERLAP_0_HEX)
            elif getattr(box, 'is_auto', False):
                if hasattr(self, 'show_green') and not self.show_green.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(AUTO_BOX_RGB)
                    box_color_strs.append(AUTO_BOX_HEX)
            else:
                if hasattr(self, 'show_green') and not self.show_green.get():
                    box_colors.append(None)
                    box_color_strs.append(None)
                else:
                    box_colors.append(BOX_RGB)
                    box_color_strs.append(BOX_COLOR)

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
        try:
            opacity = int(float(self.opacity_var.get().replace(',', '.')) * 255)
        except Exception:
            opacity = 50
        if opacity > 0:
            disp_w = int(self.orig_w * self.display_scale)
            disp_h = int(self.orig_h * self.display_scale)
            overlay_pil = Image.new("RGBA", (disp_w, disp_h), (0,0,0,0))
            draw = ImageDraw.Draw(overlay_pil, "RGBA")

            try:
                base_width = int(self.thickness_var.get())
            except Exception:
                base_width = 3
            
            for i, box in enumerate(self.boxes):
                x1_n = box.x_center - box.w / 2
                y1_n = box.y_center - box.h / 2
                x2_n = box.x_center + box.w / 2
                y2_n = box.y_center + box.h / 2
                
                cx1, cy1 = self._norm_to_canvas(x1_n, y1_n)
                cx2, cy2 = self._norm_to_canvas(x2_n, y2_n)
                
                lx1 = cx1 - self.img_offset_x
                ly1 = cy1 - self.img_offset_y
                lx2 = cx2 - self.img_offset_x
                ly2 = cy2 - self.img_offset_y
                
                color = box_colors[i]
                if color is None: continue
                
                # Make massive/overlap boxes slightly thicker so they stand out
                is_massive = box.w > 0.5 or box.h > 0.5
                outline_w = base_width + 1 if is_massive else base_width
                
                draw.rectangle([lx1, ly1, lx2, ly2], outline=(*color, opacity), width=outline_w)

            self.overlay_tk = ImageTk.PhotoImage(overlay_pil)
            overlay_id = self.canvas.create_image(self.img_offset_x, self.img_offset_y, anchor=tk.NW, image=self.overlay_tk)
            self.canvas_ids.append(overlay_id)

        for i, box in enumerate(self.boxes):
            x1_n = box.x_center - box.w / 2
            y1_n = box.y_center - box.h / 2
            x2_n = box.x_center + box.w / 2
            y2_n = box.y_center + box.h / 2

            cx1, cy1 = self._norm_to_canvas(x1_n, y1_n)
            cx2, cy2 = self._norm_to_canvas(x2_n, y2_n)



            color_str = box_color_strs[i]
            if color_str is None: continue
            # Small label
            label_id = self.canvas.create_text(
                cx1 + 3, cy1 - 10,
                text=f"#{i + 1}", anchor=tk.NW,
                font=("Consolas", 8, "bold"), fill=color_str
            )
            self.canvas_ids.append(label_id)

    # ════════════════════════════════════════════════════════════════
    #  SAVE / LOAD / EXPORT
    # ════════════════════════════════════════════════════════════════

    def _export_jpg(self):
        """Export the current image and boxes to a JPG file."""
        if not self.pil_img or not self.image_paths:
            return
            
        path = self.image_paths[self.current_idx]
        
        from tkinter import filedialog
        initial_file = f"annotated_{path.name}"
        if not initial_file.lower().endswith(".jpg"):
            initial_file = str(Path(initial_file).with_suffix(".jpg"))
            
        out_path_str = filedialog.asksaveasfilename(
            parent=self.root,
            title="Export Annotated Image",
            initialfile=initial_file,
            defaultextension=".jpg",
            filetypes=[("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )
        
        if not out_path_str:
            return
            
        out_path = Path(out_path_str)
        out_dir = out_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        export_img = self.pil_img.copy().convert("RGBA")
        overlay = Image.new("RGBA", export_img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        
        opacity = int(self.opacity_var.get() * 255)
        
        for i, box in enumerate(self.boxes):
            x1_n = box.x_center - box.w / 2
            y1_n = box.y_center - box.h / 2
            x2_n = box.x_center + box.w / 2
            y2_n = box.y_center + box.h / 2
            
            x1 = x1_n * self.orig_w
            y1 = y1_n * self.orig_h
            x2 = x2_n * self.orig_w
            y2 = y2_n * self.orig_h
            
            is_massive = box.w > 0.5 or box.h > 0.5
            
            is_overlap = False
            if not is_massive:
                for j, other in enumerate(self.boxes):
                    if i == j: continue
                    ox1 = other.x_center - other.w / 2
                    oy1 = other.y_center - other.h / 2
                    ox2 = other.x_center + other.w / 2
                    oy2 = other.y_center + other.h / 2
                    if not (x2_n <= ox1 or x1_n >= ox2 or y2_n <= oy1 or y1_n >= oy2):
                        is_overlap = True
                        break
                        
            if is_massive:
                color = (255, 0, 0)
            elif is_overlap:
                color = (245, 158, 11)
            else:
                color = (0, 255, 136)
            
            try:
                base_width = int(self.thickness_var.get())
            except Exception:
                base_width = 3
            outline_w = base_width + 1 if is_massive else base_width
            
            draw.rectangle([x1, y1, x2, y2], outline=(*color, opacity), width=outline_w)
            # optional text
            draw.text((x1 + 3, y1 - 15), f"#{i+1}", fill=(*color, 255))
            
        export_img = Image.alpha_composite(export_img, overlay).convert("RGB")
        out_path = out_dir / path.name
        export_img.save(out_path, quality=95)
        self.status.config(text=f" Exported JPG to {out_path.name}")

    def _snap_single_box(self, box, img_bgr=None):
        if not self.pil_img: return False
        import cv2, numpy as np
        if img_bgr is None:
            try:
                img_bgr = cv2.cvtColor(np.array(self.pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                return False
            
        img_h, img_w = img_bgr.shape[:2]
        x1 = int((box.x_center - box.w / 2) * img_w)
        y1 = int((box.y_center - box.h / 2) * img_h)
        x2 = int((box.x_center + box.w / 2) * img_w)
        y2 = int((box.y_center + box.h / 2) * img_h)
        
        pad_x = int((x2 - x1) * 0.3)
        pad_y = int((y2 - y1) * 0.3)
        
        rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        rx2, ry2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)
        
        if rx2 <= rx1 or ry2 <= ry1: return False
            
        roi = img_bgr[ry1:ry2, rx1:rx2]
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply morphological opening to disconnect slightly touching grains
            kernel = np.ones((3, 3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                orig_cx_roi = (x1 + x2) / 2.0 - rx1
                orig_cy_roi = (y1 + y2) / 2.0 - ry1
                
                best_contour = None
                min_dist = float('inf')
                
                for cnt in contours:
                    if cv2.contourArea(cnt) < 10: continue
                    br_x, br_y, br_w, br_h = cv2.boundingRect(cnt)
                    cnt_cx = br_x + br_w / 2.0
                    cnt_cy = br_y + br_h / 2.0
                    
                    dist = (cnt_cx - orig_cx_roi)**2 + (cnt_cy - orig_cy_roi)**2
                    if dist < min_dist:
                        min_dist = dist
                        best_contour = cnt
                
                if best_contour is None:
                    return False
                    
                cx, cy, cw, ch = cv2.boundingRect(best_contour)
                
                pad_w = int(cw * 0.08)
                pad_h = int(ch * 0.08)
                new_x1 = max(0, rx1 + cx - pad_w)
                new_y1 = max(0, ry1 + cy - pad_h)
                new_x2 = min(img_w, rx1 + cx + cw + pad_w)
                new_y2 = min(img_h, ry1 + cy + ch + pad_h)
                
                box.w = (new_x2 - new_x1) / img_w
                box.h = (new_y2 - new_y1) / img_h
                box.x_center = (new_x1 + new_x2) / 2.0 / img_w
                box.y_center = (new_y1 + new_y2) / 2.0 / img_h
                return True
        except Exception:
            pass
        return False

    def _snap_boxes(self):
        """Recompute bounding boxes to snap perfectly to pollen edges using OpenCV."""
        if not self.pil_img or not self.boxes: return
        import cv2, numpy as np
        import concurrent.futures
        
        try:
            img_bgr = cv2.cvtColor(np.array(self.pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return
            
        def snap(box):
            return self._snap_single_box(box, img_bgr=img_bgr)
            
        updated = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(snap, self.boxes))
            updated = sum(1 for r in results if r)
                
        self._redraw_boxes()
        self._save_labels()
        self.status.config(text=f" Snapped {updated} boxes to edges")

    def _discard_recount(self):
        """Remove all boxes generated by the last auto-recount."""
        if not self.boxes: return
        original_len = len(self.boxes)
        self.boxes = [b for b in self.boxes if not getattr(b, 'is_auto', False)]
        removed = original_len - len(self.boxes)
        if removed > 0:
            self._redraw_boxes()
            self._save_labels()
            self._update_ui()
            self.status.config(text=f" Discarded {removed} auto-recounted boxes.")

    def _auto_recount(self):
        """Lazy load YOLO, run inference, and add non-overlapping boxes."""
        if not self.pil_img or not self.image_paths: return
        
        try:
            conf_val = float(self.conf_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid confidence value")
            return
            
        selected_model = getattr(self, 'model_combo', None)
        if selected_model:
            selected_model = selected_model.get()
        else:
            selected_model = "None"

        if selected_model == "None":
            messagebox.showerror("Error", "No model selected.")
            return

        if self.yolo_model is not None and getattr(self, 'loaded_model_name', None) != selected_model:
            print(f"[INFO] Switching model to {selected_model}")
            self.yolo_model = None

        if self.yolo_model is None:
            self.status.config(text=f" Loading YOLO model: {selected_model}...")
            self.root.update()
            try:
                from ultralytics import YOLO
                
                model_path = PROJECT_ROOT / "runs" / "detect" / selected_model / "weights" / "best.pt"
                
                if not model_path.exists():
                    messagebox.showerror("Error", f"Could not find a trained best.pt model in {model_path}!")
                    self.status.config(text=" Model load failed")
                    return
                    
                print(f"[INFO] Lazy-loading model from: {model_path}")
                self.yolo_model = YOLO(model_path)
                self.loaded_model_name = selected_model
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load YOLO: {e}")
                self.status.config(text=" Model load failed")
                return

        self.status.config(text=f" Running inference at conf {conf_val}...")
        self.root.update()
        
        img_path = str(self.image_paths[self.current_idx])
        results = self.yolo_model.predict(
            source=img_path,
            conf=conf_val,
            iou=0.45,
            imgsz=1024,
            max_det=5000,
            device="0", # Keep it on GPU for speed
            verbose=False
        )
        
        if not results: return
        
        boxes_data = results[0].boxes
        if not boxes_data:
            self.status.config(text=" No new boxes found by YOLO.")
            return
            
        new_count = 0
        
        for box_data in boxes_data:
            xywhn = box_data.xywhn[0].cpu().numpy()
            xc, yc, w, h = xywhn
            
            new_box = BoundingBox(float(xc), float(yc), float(w), float(h), class_id=0, is_auto=True)
            
            # Filter: Check overlap against all existing boxes
            is_overlap = False
            for exist_box in self.boxes:
                if exist_box.iou(new_box) > 0.25:
                    is_overlap = True
                    break
                    
            if not is_overlap:
                self._snap_single_box(new_box)
                self.boxes.append(new_box)
                new_count += 1
                
        self._redraw_boxes()
        self._save_labels()
        self._update_ui()
        self.status.config(text=f" Auto-recount: Added {new_count} missing pollen grains.")

    def _open_batch_tools(self):
        batch_win = tk.Toplevel(self.root)
        batch_win.title("Batch Tools")
        batch_win.geometry("450x300")
        batch_win.configure(bg=BG_COLOR)
        batch_win.transient(self.root)
        batch_win.grab_set()

        tk.Label(batch_win, text="Batch Edit", font=("Segoe UI", 14, "bold"), bg=BG_COLOR, fg=ACCENT).pack(pady=10)

        # Target Selection
        target_frame = tk.LabelFrame(batch_win, text="Target Scope", bg=BG_COLOR, font=("Segoe UI", 10, "bold"), fg=TEXT_COLOR)
        target_frame.pack(fill=tk.X, padx=15, pady=5)
        
        target_var = tk.StringVar(value="dataset")
        tk.Radiobutton(target_frame, text="Current Dataset (e.g., all Train)", variable=target_var, value="dataset", bg=BG_COLOR, fg=TEXT_COLOR, selectcolor=SIDEBAR_BG).pack(anchor=tk.W, padx=5, pady=2)
        tk.Radiobutton(target_frame, text="All Datasets (Train + Val + Excluded)", variable=target_var, value="all", bg=BG_COLOR, fg=TEXT_COLOR, selectcolor=SIDEBAR_BG).pack(anchor=tk.W, padx=5, pady=2)

        # Operations
        ops_frame = tk.LabelFrame(batch_win, text="Operations", bg=BG_COLOR, font=("Segoe UI", 10, "bold"), fg=TEXT_COLOR)
        ops_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Scale
        scale_frame = tk.Frame(ops_frame, bg=BG_COLOR)
        scale_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(scale_frame, text="Scale Factor:", bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        scale_entry = tk.Entry(scale_frame, width=8, font=("Consolas", 10))
        scale_entry.insert(0, "1.1")
        scale_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(scale_frame, text="Apply Scale", bg=ACCENT, fg="white", bd=0, cursor="hand2", command=lambda: self._run_batch("scale", target_var.get(), scale_entry.get(), batch_win)).pack(side=tk.RIGHT)
        
        # Snap
        snap_frame = tk.Frame(ops_frame, bg=BG_COLOR)
        snap_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(snap_frame, text="Snap all boxes to edges:", bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        tk.Button(snap_frame, text="Auto-Snap All", bg="#8B5CF6", fg="white", bd=0, cursor="hand2", command=lambda: self._run_batch("snap", target_var.get(), None, batch_win)).pack(side=tk.RIGHT)
        
        # Clear
        clear_frame = tk.Frame(ops_frame, bg=BG_COLOR)
        clear_frame.pack(fill=tk.X, padx=5, pady=5)
        tk.Label(clear_frame, text="Clear all boxes from images:", bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        tk.Button(clear_frame, text="Clear All", bg="#EF4444", fg="white", bd=0, cursor="hand2", command=lambda: self._run_batch("clear", target_var.get(), None, batch_win)).pack(side=tk.RIGHT)

    def _run_batch(self, op, target, param, parent_win):
        # Gather labels
        targets = []
        if target == "dataset":
            targets = [p for p in self.labels_dir.glob("*.txt") if p.is_file()]
        else:
            for d in [TRAIN_LABELS, VAL_LABELS, EXCLUDED_LABELS]:
                if d.exists():
                    targets.extend([p for p in d.glob("*.txt") if p.is_file()])
                    
        if not targets:
            messagebox.showinfo("Info", "No label files found to process.", parent=parent_win)
            return
            
        scale_factor = 1.0
        if op == "scale":
            try:
                scale_factor = float(param.replace(',', '.'))
                if scale_factor <= 0: raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Invalid scale factor.", parent=parent_win)
                return

        if not messagebox.askyesno("Confirm", f"Are you sure you want to {op} on {len(targets)} files?\nThis cannot be undone easily.", parent=parent_win):
            return
            
        pb_var = tk.DoubleVar()
        pb = ttk.Progressbar(parent_win, variable=pb_var, maximum=len(targets))
        pb.pack(fill=tk.X, padx=15, pady=10)
        parent_win.update()

        affected = 0
        import cv2 # import locally to avoid cluttering global namespace if not needed
        for i, txt_path in enumerate(targets):
            if op == "clear":
                txt_path.unlink()
                affected += 1
            elif op == "scale" or op == "snap":
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                    
                new_lines = []
                img_bgr = None
                img_h, img_w = 0, 0
                
                if op == "snap":
                    # find the corresponding image
                    labels_folder = txt_path.parent
                    images_folder = labels_folder.parent.parent / "images" / labels_folder.name
                    img_path = None
                    for ext in IMAGE_EXTS:
                        possible = images_folder / f"{txt_path.stem}{ext}"
                        if possible.exists():
                            img_path = possible
                            break
                    if img_path:
                        img_bgr = cv2.imread(str(img_path))
                        if img_bgr is not None:
                            img_h, img_w = img_bgr.shape[:2]
                
                changed_this_file = False
                for line in lines:
                    box = BoundingBox.from_yolo_line(line)
                    if box:
                        if op == "scale":
                            box.w *= scale_factor
                            box.h *= scale_factor
                            changed_this_file = True
                        elif op == "snap" and img_bgr is not None:
                            # Use logic similar to _snap_single_box
                            x1 = int((box.x_center - box.w / 2) * img_w)
                            y1 = int((box.y_center - box.h / 2) * img_h)
                            x2 = int((box.x_center + box.w / 2) * img_w)
                            y2 = int((box.y_center + box.h / 2) * img_h)
                            
                            pad_x = int((x2 - x1) * 0.3)
                            pad_y = int((y2 - y1) * 0.3)
                            
                            rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                            rx2, ry2 = min(img_w, x2 + pad_x), min(img_h, y2 + pad_y)
                            
                            if rx2 > rx1 and ry2 > ry1:
                                roi = img_bgr[ry1:ry2, rx1:rx2]
                                try:
                                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                                    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                    if contours:
                                        largest = max(contours, key=cv2.contourArea)
                                        cx, cy, cw, ch = cv2.boundingRect(largest)
                                        pad_w, pad_h = int(cw * 0.08), int(ch * 0.08)
                                        new_x1 = max(0, rx1 + cx - pad_w)
                                        new_y1 = max(0, ry1 + cy - pad_h)
                                        new_x2 = min(img_w, rx1 + cx + cw + pad_w)
                                        new_y2 = min(img_h, ry1 + cy + ch + pad_h)
                                        box.w = (new_x2 - new_x1) / img_w
                                        box.h = (new_y2 - new_y1) / img_h
                                        box.x_center = (new_x1 + new_x2) / 2.0 / img_w
                                        box.y_center = (new_y1 + new_y2) / 2.0 / img_h
                                        changed_this_file = True
                                except Exception:
                                    pass
                                    
                        new_lines.append(box.to_yolo_line() + "\n")
                        
                if new_lines and changed_this_file:
                    with open(txt_path, "w") as f:
                        f.writelines(new_lines)
                    affected += 1
                elif not new_lines and op == "scale": # only delete if scaling
                    txt_path.unlink()
            
            pb_var.set(i + 1)
            if i % 10 == 0:
                parent_win.update()
            
        pb.destroy()
        messagebox.showinfo("Done", f"Processed {affected} files successfully.", parent=parent_win)
        self._load_image()  # Refresh the canvas to show changes
        parent_win.destroy()

    def _clean_duplicates(self):
        """Remove boxes that overlap by more than 80% with another box."""
        if len(self.boxes) < 2:
            return
            
        to_remove = set()
        for i in range(len(self.boxes)):
            if i in to_remove: continue
            for j in range(i + 1, len(self.boxes)):
                if j in to_remove: continue
                if self.boxes[i].iou(self.boxes[j]) > 0.6:
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

    def _on_combo_jump(self, event=None):
        if not self.image_paths: return
        selected_name = self.image_combo.get()
        try:
            idx = next(i for i, p in enumerate(self.image_paths) if p.name == selected_name)
            if idx != self.current_idx:
                self._save_labels()
                self.current_idx = idx
                self._load_image()
        except StopIteration:
            pass

    def _delete_image(self):
        if not self.image_paths: return
        
        current_img_path = self.image_paths[self.current_idx]
        if not messagebox.askyesno("Confirm Delete", f"Are you sure you want to PERMANENTLY delete:\n\n{current_img_path.name}\n\nThis cannot be undone!", parent=self.root):
            return

        current_lbl_path = self._label_path()
        if current_lbl_path.exists():
            current_lbl_path.unlink()
            
        try:
            current_img_path.unlink()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete image: {e}")
            return
            
        del self.image_paths[self.current_idx]
        self._update_ui()
        
        if not self.image_paths:
            self.canvas.delete("all")
            self.pil_img = None
            self.boxes.clear()
            self._update_ui()
            self.status.config(text=" Deleted last image.")
            return

        if self.current_idx >= len(self.image_paths):
            self.current_idx = len(self.image_paths) - 1
            
        self._load_image()
        self.status.config(text=f" Permanently deleted image")

    # ════════════════════════════════════════════════════════════════
    #  ACTIONS
    # ════════════════════════════════════════════════════════════════

    def _undo(self):
        if self.boxes:
            self.boxes.pop()
            self._redraw_boxes()
            self._save_labels()
            self._update_ui()
            self.status.config(text=f"Undo boxes: {len(self.boxes)}")

    def _clear_boxes(self):
        if self.boxes:
            if messagebox.askyesno("Clear All", "Delete all boxes on this image?"):
                self.boxes.clear()
                self._redraw_boxes()
                self._save_labels()
                self._update_ui()
                self.status.config(text=" All boxes cleared")

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
        
        # Check for overwrite
        target_img_path = target_img_dir / name
        if target_img_path.exists():
            if not messagebox.askyesno(
                "Overwrite Warning", 
                f"The file '{name}' already exists in the {target_set} set!\n\nIf you proceed, the existing file and its labels will be permanently overwritten.\n\nDo you want to overwrite it?",
                icon="warning"
            ):
                return

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
            self.status.config(text=f"'{name}' moved to {target_set}")

    # ════════════════════════════════════════════════════════════════
    #  UI UPDATES
    # ════════════════════════════════════════════════════════════════

    def _update_ui(self):
        total = len(self.image_paths)
        if total == 0:
            self.image_combo.set("")
            self.image_combo.config(values=[])
            self.size_label.config(text="")
            self.count_label.config(text="Boxes: 0")
            self.progress_label.config(text=f"0/0    0 annotated")
            self.root.title(f" Pollen Annotator  {self.current_set} (Empty)")
            
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

        current_names = [p.name for p in self.image_paths]
        if list(self.image_combo["values"]) != current_names:
            self.image_combo.config(values=current_names)
        self.image_combo.set(img_path.name)
        
        self.size_label.config(text=f"{self.orig_w} × {self.orig_h} px")
        self.count_label.config(text=f"Boxes: {len(self.boxes)}")
        self.progress_label.config(text=f"Image {idx}/{total}    {annotated} annotated")
        self.root.title(f" Pollen Annotator  {self.current_set}: {img_path.name} [{idx}/{total}]")

        # Button states
        self.prev_btn.config(state=tk.NORMAL if self.current_idx > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_idx < total - 1 else tk.DISABLED)
        
        if hasattr(self, 'move_train_btn'):
            self.move_train_btn.config(state=tk.DISABLED if self.current_set == "Train" else tk.NORMAL)
            self.move_val_btn.config(state=tk.DISABLED if self.current_set == "Validation" else tk.NORMAL)
            self.exclude_btn.config(state=tk.DISABLED if self.current_set == "Excluded" else tk.NORMAL)

    # ════════════════════════════════════════════════════════════════
    #  STATE PERSISTENCE
    # ════════════════════════════════════════════════════════════════

    def _load_settings(self):
        """Restore UI control values from config/inference_settings.json."""
        config = load_settings()
        if not config:
            return
        try:
            if "model" in config and hasattr(self, 'model_combo'):
                if config["model"] in self.model_combo['values']:
                    self.model_combo.set(config["model"])
            if "threshold" in config and hasattr(self, 'conf_entry'):
                self.conf_entry.delete(0, tk.END)
                self.conf_entry.insert(0, str(config["threshold"]))
            if "opacity" in config and hasattr(self, 'opacity_var'):
                self.opacity_var.set(config["opacity"])
            if "box_width" in config and hasattr(self, 'entry_w'):
                self.entry_w.delete(0, tk.END)
                self.entry_w.insert(0, str(config["box_width"]))
            if "box_height" in config and hasattr(self, 'entry_h'):
                self.entry_h.delete(0, tk.END)
                self.entry_h.insert(0, str(config["box_height"]))
            if "thickness" in config and hasattr(self, 'thickness_var'):
                self.thickness_var.set(str(config["thickness"]))
            if "scale" in config and hasattr(self, 'scale_var'):
                self.scale_var.set(str(config["scale"]))
            if "show_compare" in config and hasattr(self, 'show_compare'):
                self.show_compare.set(config["show_compare"])
        except Exception as e:
            print(f"[WARNING] Failed to load UI settings: {e}")

    def _save_settings(self):
        """Persist all UI control values to config/inference_settings.json."""
        updates = {}
        if hasattr(self, 'model_combo'):
            updates["model"] = self.model_combo.get()
        if hasattr(self, 'conf_entry'):
            updates["threshold"] = self.conf_entry.get()
        if hasattr(self, 'opacity_var'):
            try:
                updates["opacity"] = float(self.opacity_var.get().replace(',', '.'))
            except Exception: pass
        if hasattr(self, 'entry_w'):
            updates["box_width"] = self.entry_w.get()
        if hasattr(self, 'entry_h'):
            updates["box_height"] = self.entry_h.get()
        if hasattr(self, 'thickness_var'):
            try:
                updates["thickness"] = int(self.thickness_var.get())
            except Exception: pass
        if hasattr(self, 'scale_var'):
            try:
                updates["scale"] = float(self.scale_var.get().replace(',', '.'))
            except Exception: pass
        if hasattr(self, 'show_compare'):
            updates["show_compare"] = self.show_compare.get()
        save_settings(updates)

    def _on_close(self):
        self._save_settings()
        self.root.destroy()


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
    print(f"\n[INFO] Annotation complete  {label_count} label file(s) saved to {labels_dir}")


if __name__ == "__main__":
    main()
