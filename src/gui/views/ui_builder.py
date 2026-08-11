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


class UIBuilderMixin:
    def _build_ui(self):
        self.root.title(" Pollen Annotator  YOLOv11s")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(1100, 700)

        # ── Menu Bar ──────────────────────────────────────────────
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # ── Navigation Menu (Mirrors Launcher) ────────────────────────────────
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # We spawn other tools using the same runpy logic as the launcher
        def spawn_tool(script_name):
            import subprocess
            import sys
            # Pass the tool name to launcher.py or run it directly
            cmd = [sys.executable, str(PROJECT_ROOT / "launcher.py"), script_name]
            subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
            
        tools_menu.add_command(label="Smart Annotator", command=lambda: spawn_tool("annotate"))
        tools_menu.add_command(label="Batch Inference", command=lambda: spawn_tool("inference"))
        tools_menu.add_command(label="Import Images", command=self._import_images)
        tools_menu.add_command(label="Error Logs View", command=lambda: spawn_tool("view_logs"))
        
        if not getattr(sys, 'frozen', False):
            tools_menu.add_separator()
            tools_menu.add_command(label="Training Dashboard", command=lambda: spawn_tool("monitor"))
            tools_menu.add_command(label="Dataset Analytics", command=lambda: spawn_tool("dataset_analytics"))
            tools_menu.add_command(label="Augmentation Preview", command=lambda: spawn_tool("augment_preview"))
            tools_menu.add_command(label="Export Model", command=lambda: spawn_tool("export"))

        # ── Top bar ─────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=ACCENT, height=60)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Label(
            top, text="  Pollen Grain Annotator", font=FONT_HEADER,
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
        self.fit_mode = tk.StringVar(value="H")
        tk.Button(header_controls, text="Fit W", font=("Segoe UI", 8, "bold"), bg="#4B5563", fg="white", bd=0, cursor="hand2", padx=4, command=self._fit_width).pack(side=tk.LEFT, padx=(10, 2))
        tk.Button(header_controls, text="Fit H", font=("Segoe UI", 8, "bold"), bg="#4B5563", fg="white", bd=0, cursor="hand2", padx=4, command=self._fit_height).pack(side=tk.LEFT, padx=2)

        # 6. Auto-Snap Checkbox & Technique Combobox
        self.auto_snap = tk.BooleanVar(value=True)
        tk.Checkbutton(
            header_controls, text="Auto-Snap", variable=self.auto_snap,
            bg=ACCENT, fg="white", selectcolor=ACCENT,
            activebackground=ACCENT, activeforeground="white"
        ).pack(side=tk.LEFT, padx=(10, 2))

        self.snap_method = tk.StringVar(value="Adaptive Multi-Stage")
        self.snap_combo = ttk.Combobox(
            header_controls, textvariable=self.snap_method, 
            values=["Adaptive Multi-Stage", "Simple Otsu (Legacy)"],
            state="readonly", width=19, font=("Segoe UI", 8)
        )
        self.snap_combo.pack(side=tk.LEFT, padx=(2, 2))

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
        self.show_violet = tk.BooleanVar(value=True)
        
        self.chk_red = tk.Checkbutton(header_controls, text="Red (0)", variable=self.show_red, bg=ACCENT, fg="#EF4444", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#EF4444", font=("Segoe UI", 8, "bold"), command=self._redraw_boxes)
        self.chk_red.pack(side=tk.LEFT, padx=(6, 2))
        self.chk_orange = tk.Checkbutton(header_controls, text="Org (0)", variable=self.show_orange, bg=ACCENT, fg="#F97316", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#F97316", font=("Segoe UI", 8, "bold"), command=self._redraw_boxes)
        self.chk_orange.pack(side=tk.LEFT, padx=2)
        self.chk_yellow = tk.Checkbutton(header_controls, text="Yel (0)", variable=self.show_yellow, bg=ACCENT, fg="#FACC15", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#FACC15", font=("Segoe UI", 8, "bold"), command=self._redraw_boxes)
        self.chk_yellow.pack(side=tk.LEFT, padx=2)
        self.chk_green = tk.Checkbutton(header_controls, text="Grn (0)", variable=self.show_green, bg=ACCENT, fg="#22C55E", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#22C55E", font=("Segoe UI", 8, "bold"), command=self._redraw_boxes)
        self.chk_green.pack(side=tk.LEFT, padx=2)
        self.chk_violet = tk.Checkbutton(header_controls, text="Vio (0)", variable=self.show_violet, bg=ACCENT, fg="#8B5CF6", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#8B5CF6", font=("Segoe UI", 8, "bold"), command=self._redraw_boxes)
        self.chk_violet.pack(side=tk.LEFT, padx=2)

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
            sidebar, text="Dataset Selection", font=FONT_LABEL,
            bg=SIDEBAR_BG, fg=TEXT_COLOR
        ).pack(anchor=tk.W, padx=12, pady=(12, 4))
        
        combo_values = ["Train", "Validation", "Excluded"]
        if self.current_set not in combo_values:
            combo_values.append(self.current_set)
            
        self.dataset_combo = ttk.Combobox(sidebar, values=combo_values, state="readonly", font=("Segoe UI", 10))
        self.dataset_combo.set(self.current_set)
        self.dataset_combo.pack(fill=tk.X, padx=12, pady=(0, 8))
        self.dataset_combo.bind("<<ComboboxSelected>>", self._change_dataset)

        self.embed_annotations = tk.BooleanVar(value=False)
        self.chk_embed_annotations = tk.Checkbutton(
            sidebar, text="Embed Annotations", variable=self.embed_annotations,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, selectcolor=BG_COLOR,
            activebackground=SIDEBAR_BG, activeforeground=TEXT_COLOR
        )
        self.chk_embed_annotations.pack(pady=(0, 4), anchor=tk.W, padx=10)

        self.btn_export_dataset = tk.Button(
            sidebar, text="Export Dataset", bg=ACCENT, fg="white",
            activebackground="#6D28D9", font=("Segoe UI", 9, "bold"), cursor="hand2", bd=0, pady=4,
            command=self.export_dataset
        )
        self.btn_export_dataset.pack(fill=tk.X, padx=12, pady=(0, 8))

        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=4)
        
        # ── Sidebar: Search Image ──────────────────────────
        tk.Label(
            sidebar, text="Batch Tools", font=FONT_LABEL,
            bg=SIDEBAR_BG, fg=TEXT_COLOR
        ).pack(anchor=tk.W, padx=12, pady=(10, 2))
        
        search_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        search_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(
            search_frame, textvariable=self.search_var, width=15, font=("Segoe UI", 10),
            bg="#FFFFFF", fg="black", insertbackground="black", bd=0
        )
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.search_entry.bind("<Return>", lambda e: self._search_image())
        
        tk.Button(search_frame, text="Go", bg="#4B5563", fg="white", font=("Segoe UI", 8, "bold"), bd=0, cursor="hand2", command=self._search_image).pack(side=tk.RIGHT)
        
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
            nav_frame, text="< Prev", bg="#888888", fg="white",
            activebackground="#555555", command=self._prev_image, **nav_btn_style
        )
        self.prev_btn.pack(side=tk.LEFT, padx=2)
        
        self.next_btn = tk.Button(
            nav_frame, text="Next >", bg="#888888", fg="white",
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
        tk.Label(ar_frame, text="Conf:", font=("Consolas", 9), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.conf_entry = tk.Entry(ar_frame, width=4, font=("Consolas", 10), bg="#FFFFFF", fg="black", insertbackground="black", bd=0)
        self.conf_entry.insert(0, "0.15")
        self.conf_entry.pack(side=tk.LEFT, padx=(2, 4))
        
        tk.Label(ar_frame, text="IoU:", font=("Consolas", 9), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(side=tk.LEFT)
        self.iou_entry = tk.Entry(ar_frame, width=4, font=("Consolas", 10), bg="#FFFFFF", fg="black", insertbackground="black", bd=0)
        self.iou_entry.insert(0, "0.25")
        self.iou_entry.pack(side=tk.LEFT, padx=(2, 0))

        ar_btn_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        ar_btn_frame.pack(anchor=tk.W, padx=12, pady=(2, 4))
        
        self.recount_btn = tk.Button(
            ar_btn_frame, text="Find Missing", bg="#8B5CF6", fg="white",
            activebackground="#7C3AED", command=self._auto_recount, font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", padx=8, pady=2
        )
        self.recount_btn.pack(side=tk.LEFT)

        self.discard_recount_btn = tk.Button(
            ar_btn_frame, text="✖", bg="#EF4444", fg="white",
            activebackground="#DC2626", command=self._discard_recount, font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", padx=6, pady=2
        )
        self.discard_recount_btn.pack(side=tk.LEFT, padx=(4, 0))

        # Smart Regional Recount Toggles
        self.smart_recount_var = tk.BooleanVar(value=False)
        self.force_recount_var = tk.BooleanVar(value=False)
        self.use_sahi_var = tk.BooleanVar(value=True)
        
        regional_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        regional_frame.pack(anchor=tk.W, padx=12, pady=(4, 2))
        
        tk.Checkbutton(
            regional_frame, text="Use SAHI", variable=self.use_sahi_var,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, selectcolor=BG_COLOR,
            activebackground=SIDEBAR_BG, activeforeground=TEXT_COLOR, font=("Segoe UI", 9)
        ).pack(anchor=tk.W)
        
        tk.Checkbutton(
            regional_frame, text="Smart Regional Recount", variable=self.smart_recount_var,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, selectcolor=BG_COLOR,
            activebackground=SIDEBAR_BG, activeforeground=TEXT_COLOR, font=("Segoe UI", 9)
        ).pack(anchor=tk.W)
        
        tk.Checkbutton(
            regional_frame, text="Force Recount (Clear Existing)", variable=self.force_recount_var,
            bg=SIDEBAR_BG, fg=TEXT_COLOR, selectcolor=BG_COLOR,
            activebackground=SIDEBAR_BG, activeforeground=TEXT_COLOR, font=("Segoe UI", 9)
        ).pack(anchor=tk.W)

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

        # ── Sidebar: Viewport ────────────────────────────────────────
        tk.Label(
            sidebar, text="Quadrant View", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(4, 2))
        
        self.view_mode = tk.StringVar(value="Full")
        view_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        view_frame.pack(fill=tk.X, padx=12, pady=4)
        
        btn_opts = {"font": ("Segoe UI", 9, "bold"), "cursor": "hand2", "bd": 0, "pady": 4}
        def set_view(m):
            self.view_mode.set(m)
            self._render_image()
            
        self.btn_full = tk.Button(view_frame, text="Full", bg=ACCENT, fg="white", activebackground="#6D28D9", command=lambda: set_view("Full"), **btn_opts)
        self.btn_full.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_q1 = tk.Button(view_frame, text="Q1", bg="#888888", fg="white", activebackground="#666666", command=lambda: set_view("Q1"), **btn_opts)
        self.btn_q1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_q2 = tk.Button(view_frame, text="Q2", bg="#888888", fg="white", activebackground="#666666", command=lambda: set_view("Q2"), **btn_opts)
        self.btn_q2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_q3 = tk.Button(view_frame, text="Q3", bg="#888888", fg="white", activebackground="#666666", command=lambda: set_view("Q3"), **btn_opts)
        self.btn_q3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_q4 = tk.Button(view_frame, text="Q4", bg="#888888", fg="white", activebackground="#666666", command=lambda: set_view("Q4"), **btn_opts)
        self.btn_q4.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # ── Sidebar: buttons ────────────────────────────────────────
        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=12)

        btn_style = {"font": ("Segoe UI", 10, "bold"), "width": 22, "cursor": "hand2", "bd": 0, "pady": 6}

        self.snap_btn = tk.Button(
            sidebar, text=" Snap Boxes to Edges", bg="#059669", fg="white",
            activebackground="#047857", command=self._snap_boxes, **btn_style
        )
        self.snap_btn.pack(pady=2)

        self._snap_undo_states = None  # saved box coords before last snap
        self.undo_snap_btn = tk.Button(
            sidebar, text="↩ Undo Snap", bg="#6B7280", fg="white",
            activebackground="#4B5563", command=self._undo_snap,
            state=tk.DISABLED, **btn_style
        )
        self.undo_snap_btn.pack(pady=2)

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
            sidebar, text="Dataset Management", font=FONT_LABEL,
            bg=SIDEBAR_BG, fg=TEXT_COLOR
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
        self.root.bind("s", lambda e: self.auto_snap.set(not self.auto_snap.get()))
        self.root.bind("S", lambda e: self.auto_snap.set(not self.auto_snap.get()))
        self.root.bind("<Shift-Up>", lambda e: self._adjust_scale(0.1))
        self.root.bind("<Shift-Down>", lambda e: self._adjust_scale(-0.1))
        self.root.bind("<MouseWheel>", self._on_mousewheel)
        self.root.bind("<plus>", lambda e: self._zoom_in())
        self.root.bind("<equal>", lambda e: self._zoom_in())
        self.root.bind("<minus>", lambda e: self._zoom_out())
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Key>", self._on_key)

    def _update_auto_size(self, event=None):
        if not hasattr(self, 'full_w') or not self.full_w: return
        try:
            w_px = float(self.entry_w.get())
            h_px = float(self.entry_h.get())
            if w_px > 0 and h_px > 0:
                self.default_w = w_px / self.full_w
                self.default_h = h_px / self.full_h
        except ValueError:
            pass # Invalid input, ignore until valid
        
