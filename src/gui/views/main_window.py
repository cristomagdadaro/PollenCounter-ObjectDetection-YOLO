from __future__ import annotations
from src.paths import *
from src.bounding_box import BoundingBox, calculate_iou_corners
from src.settings import load_settings, save_settings
from src.model_utils import get_latest_weights, collect_images
from src.theme import *


from typing import *
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw


import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter import ttk

from src.theme import (
    BOX_COLOR, BOX_COLOR_HOVER, ACTIVE_BOX_COLOR,
    BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR, PROGRESS_DONE, PROGRESS_TODO,
    OVERLAP_80_HEX, OVERLAP_80_RGB, OVERLAP_50_HEX, OVERLAP_50_RGB,
    OVERLAP_0_HEX, OVERLAP_0_RGB, AUTO_BOX_HEX, AUTO_BOX_RGB, BOX_RGB,
    FONT_LABEL, FONT_INPUT, HUMAN_COLOR, MODEL_COLOR,
)

class MainWindow:
    """The main view orchestrator. Responsible only for layout and UI widgets."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(" Pollen Annotator  YOLOv11s")
        self.root.configure(bg=BG_COLOR)
        self.root.minsize(1100, 700)
        
        self._build_menu()
        self.topbar = self._build_topbar()
        self.main_area = tk.Frame(self.root, bg=BG_COLOR)
        self.main_area.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_frame = self._build_canvas(self.main_area)
        self.sidebar = self._build_sidebar(self.main_area)
        
        self.status = tk.Label(
            self.root, text="Ready", font=("Segoe UI", 9),
            bg="#E0E0E0", fg="#666666", anchor=tk.W, padx=8
        )
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_menu(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        self.tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=self.tools_menu)
        # Menu commands will be populated by the controller

    def _build_topbar(self):
        top = tk.Frame(self.root, bg=ACCENT, height=60)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Label(
            top, text="  Pollen Grain Annotator", font=("Segoe UI", 14, "bold"),
            bg=ACCENT, fg="white"
        ).pack(side=tk.LEFT, padx=16)

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
        
        # 2. Box Scale
        tk.Label(header_controls, text="Scale:", font=("Segoe UI", 9, "bold"), bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=(10, 2))
        self.scale_var = tk.StringVar(value="1.0")
        self.scale_entry = tk.Entry(
            header_controls, textvariable=self.scale_var, width=5, font=("Consolas", 10),
            bg="#FFFFFF", fg="black", insertbackground="black", bd=0
        )
        self.scale_entry.pack(side=tk.LEFT, padx=2)

        # 3. Box Opacity
        tk.Label(header_controls, text="Opacity:", font=("Segoe UI", 9, "bold"), bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=(10, 2))
        self.opacity_var = tk.StringVar(value="0.2")
        self.opacity_entry = tk.Entry(
            header_controls, textvariable=self.opacity_var, width=5, font=("Consolas", 10),
            bg="#FFFFFF", fg="black", insertbackground="black", bd=0
        )
        self.opacity_entry.pack(side=tk.LEFT, padx=2)
        
        # 4. Border Thickness
        tk.Label(header_controls, text="Thickness:", font=("Segoe UI", 9, "bold"), bg=ACCENT, fg="white").pack(side=tk.LEFT, padx=(10, 2))
        self.thickness_var = tk.StringVar(value="3")
        self.thickness_entry = tk.Entry(
            header_controls, textvariable=self.thickness_var, width=5, font=("Consolas", 10),
            bg="#FFFFFF", fg="black", insertbackground="black", bd=0
        )
        self.thickness_entry.pack(side=tk.LEFT, padx=2)
        
        # 5. Fit Options
        self.fit_mode = tk.StringVar(value="H")
        self.btn_fit_w = tk.Button(header_controls, text="Fit W", font=("Segoe UI", 8, "bold"), bg="#4B5563", fg="white", bd=0, cursor="hand2", padx=4)
        self.btn_fit_w.pack(side=tk.LEFT, padx=(10, 2))
        self.btn_fit_h = tk.Button(header_controls, text="Fit H", font=("Segoe UI", 8, "bold"), bg="#4B5563", fg="white", bd=0, cursor="hand2", padx=4)
        self.btn_fit_h.pack(side=tk.LEFT, padx=2)

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
        
        self.btn_clean = tk.Button(header_controls, text="Clean", font=("Segoe UI", 8, "bold"), bg="#EF4444", fg="white", bd=0, cursor="hand2", padx=4)
        self.btn_clean.pack(side=tk.LEFT, padx=2)
        self.btn_undo_clean = tk.Button(header_controls, text="Undo", font=("Segoe UI", 8, "bold"), bg="#F59E0B", fg="white", bd=0, cursor="hand2", padx=4)
        self.btn_undo_clean.pack(side=tk.LEFT, padx=2)
        
        # 8. Visibility Filters
        self.show_red = tk.BooleanVar(value=True)
        self.show_orange = tk.BooleanVar(value=True)
        self.show_yellow = tk.BooleanVar(value=True)
        self.show_green = tk.BooleanVar(value=True)
        self.show_violet = tk.BooleanVar(value=True)
        
        self.chk_red = tk.Checkbutton(header_controls, text="Red (0)", variable=self.show_red, bg=ACCENT, fg="#EF4444", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#EF4444", font=("Segoe UI", 8, "bold"))
        self.chk_red.pack(side=tk.LEFT, padx=(6, 2))
        self.chk_orange = tk.Checkbutton(header_controls, text="Org (0)", variable=self.show_orange, bg=ACCENT, fg="#F97316", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#F97316", font=("Segoe UI", 8, "bold"))
        self.chk_orange.pack(side=tk.LEFT, padx=2)
        self.chk_yellow = tk.Checkbutton(header_controls, text="Yel (0)", variable=self.show_yellow, bg=ACCENT, fg="#FACC15", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#FACC15", font=("Segoe UI", 8, "bold"))
        self.chk_yellow.pack(side=tk.LEFT, padx=2)
        self.chk_green = tk.Checkbutton(header_controls, text="Grn (0)", variable=self.show_green, bg=ACCENT, fg="#22C55E", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#22C55E", font=("Segoe UI", 8, "bold"))
        self.chk_green.pack(side=tk.LEFT, padx=2)
        self.chk_violet = tk.Checkbutton(header_controls, text="Vio (0)", variable=self.show_violet, bg=ACCENT, fg="#8B5CF6", selectcolor="#FFFFFF", activebackground=ACCENT, activeforeground="#8B5CF6", font=("Segoe UI", 8, "bold"))
        self.chk_violet.pack(side=tk.LEFT, padx=2)

        self.progress_label = tk.Label(top, text="", font=("Segoe UI", 11), bg=ACCENT, fg="#FFFFFF")
        self.progress_label.pack(side=tk.RIGHT, padx=16)
        return top

    def _build_canvas(self, parent):
        canvas_frame = tk.Frame(parent, bg="#CCCCCC", bd=2, relief=tk.SUNKEN)
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
        return canvas_frame

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=SIDEBAR_BG, width=260)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 8), pady=8)
        sidebar.pack_propagate(False)

        # ── Sidebar: Dataset Set Selection ──────────────────────────
        tk.Label(
            sidebar, text="Dataset Set", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(16, 2))
        
        self.dataset_combo = ttk.Combobox(sidebar, state="readonly", font=("Segoe UI", 10))
        self.dataset_combo.pack(fill=tk.X, padx=12, pady=(0, 8))

        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=4)

        # ── Sidebar: file info ──────────────────────────────────────
        tk.Label(
            sidebar, text="Current Image", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(8, 2))

        self.image_combo = ttk.Combobox(sidebar, state="readonly", font=("Consolas", 9))
        self.image_combo.pack(fill=tk.X, padx=12, pady=(0, 2))
        
        nav_frame = tk.Frame(sidebar, bg=SIDEBAR_BG)
        nav_frame.pack(fill=tk.X, padx=12, pady=(4, 2))
        
        nav_btn_style = {"font": ("Segoe UI", 9, "bold"), "cursor": "hand2", "bd": 0, "pady": 4}
        
        self.prev_btn = tk.Button(
            nav_frame, text="◀ Prev", bg="#888888", fg="white",
            activebackground="#666666", **nav_btn_style
        )
        self.prev_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.next_btn = tk.Button(
            nav_frame, text="Next ▶", bg=ACCENT, fg="white",
            activebackground="#6D28D9", **nav_btn_style
        )
        self.next_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self.size_label = tk.Label(sidebar, text="", font=("Consolas", 9), bg=SIDEBAR_BG, fg="#666666")
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
            activebackground=SIDEBAR_BG, activeforeground=TEXT_COLOR
        )
        self.chk_compare.pack(pady=(0, 12))

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
            activebackground="#7C3AED", font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", padx=8, pady=2
        )
        self.recount_btn.pack(side=tk.LEFT)

        self.discard_recount_btn = tk.Button(
            ar_btn_frame, text="✖", bg="#EF4444", fg="white",
            activebackground="#DC2626", font=("Segoe UI", 9, "bold"), bd=0, cursor="hand2", padx=6, pady=2
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
        
        self.btn_batch = tk.Button(
            sidebar, text="⚙ Batch Tools", font=("Segoe UI", 10, "bold"),
            bg="#2563EB", fg="white", activebackground="#1D4ED8",
            bd=0, cursor="hand2", pady=4
        )
        self.btn_batch.pack(fill=tk.X, padx=12, pady=(0, 4))

        tk.Label(
            sidebar, text="Model:", font=("Consolas", 8),
            bg=SIDEBAR_BG, fg="#888888"
        ).pack(anchor=tk.W, padx=12, pady=(4, 0))
        
        self.model_combo = ttk.Combobox(sidebar, state="readonly", width=30)
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
            
        self.btn_full = tk.Button(view_frame, text="Full", bg=ACCENT, fg="white", activebackground="#6D28D9", **btn_opts)
        self.btn_full.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_q1 = tk.Button(view_frame, text="Q1", bg="#888888", fg="white", activebackground="#666666", **btn_opts)
        self.btn_q1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_q2 = tk.Button(view_frame, text="Q2", bg="#888888", fg="white", activebackground="#666666", **btn_opts)
        self.btn_q2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_q3 = tk.Button(view_frame, text="Q3", bg="#888888", fg="white", activebackground="#666666", **btn_opts)
        self.btn_q3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        self.btn_q4 = tk.Button(view_frame, text="Q4", bg="#888888", fg="white", activebackground="#666666", **btn_opts)
        self.btn_q4.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

        # ── Sidebar: buttons ────────────────────────────────────────
        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=12)

        btn_style = {"font": ("Segoe UI", 10, "bold"), "width": 22, "cursor": "hand2", "bd": 0, "pady": 6}

        self.snap_btn = tk.Button(
            sidebar, text=" Snap Boxes to Edges", bg="#059669", fg="white",
            activebackground="#047857", **btn_style
        )
        self.snap_btn.pack(pady=2)

        self.undo_snap_btn = tk.Button(
            sidebar, text="↩ Undo Snap", bg="#6B7280", fg="white",
            activebackground="#4B5563", state=tk.DISABLED, **btn_style
        )
        self.undo_snap_btn.pack(pady=2)

        self.export_btn = tk.Button(
            sidebar, text=" Export to JPG", bg="#D97706", fg="white",
            activebackground="#B45309", **btn_style
        )
        self.export_btn.pack(pady=2)

        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=8)

        self.clear_btn = tk.Button(
            sidebar, text="  Clear All Boxes", bg="#DC2626", fg="white",
            activebackground="#B91C1C", **btn_style
        )
        self.clear_btn.pack(pady=2)

        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=8)

        tk.Label(
            sidebar, text="Dataset Management", font=("Segoe UI", 11, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=12, pady=(4, 2))

        self.move_train_btn = tk.Button(
            sidebar, text="Move to Train", bg="#059669", fg="white",
            activebackground="#047857", **btn_style
        )
        self.move_train_btn.pack(pady=2)

        self.move_val_btn = tk.Button(
            sidebar, text="Move to Validation", bg="#2563EB", fg="white",
            activebackground="#1D4ED8", **btn_style
        )
        self.move_val_btn.pack(pady=2)

        self.exclude_btn = tk.Button(
            sidebar, text="Exclude Image", bg="#9CA3AF", fg="black",
            activebackground="#6B7280", **btn_style
        )
        self.exclude_btn.pack(pady=2)

        tk.Frame(sidebar, bg="#CCCCCC", height=1).pack(fill=tk.X, padx=12, pady=4)

        self.delete_btn = tk.Button(
            sidebar, text=" Delete Permanently", bg="#7F1D1D", fg="white",
            activebackground="#450A0A", **btn_style
        )
        self.delete_btn.pack(pady=2)

        return sidebar
