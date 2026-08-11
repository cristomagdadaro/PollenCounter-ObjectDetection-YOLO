#!/usr/bin/env python3
"""PollenCounter Master Launcher.

Serves as the main menu and entry point for the compiled PyInstaller application.
Uses a self-calling architecture to spawn independent processes for each tool.
"""

import sys
import subprocess
import multiprocessing
import tkinter as tk
from tkinter import ttk
from pathlib import Path

import os
import traceback
from tkinter import filedialog, messagebox
from datetime import datetime

# Fix for PyInstaller multiprocessing
multiprocessing.freeze_support()

# Global Error Logger
def global_exception_handler(exc_type, exc_value, exc_traceback):
    workspace = os.environ.get("POLLEN_WORKSPACE", str(Path(__file__).resolve().parent))
    log_file = Path(workspace) / "error_log.txt"
    try:
        with open(log_file, "a") as f:
            f.write(f"\n--- Crash Report: {datetime.now().isoformat()} ---\n")
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        
        # Try to show a popup if tkinter can still run
        temp_root = tk.Tk()
        temp_root.withdraw()
        messagebox.showerror("Fatal Error", f"The application crashed!\n\nError: {exc_value}\n\nA detailed crash report was saved to:\n{log_file}")
        temp_root.destroy()
    except:
        pass
        
sys.excepthook = global_exception_handler

# Add parent directory to path so imports work identically in source and compiled mode
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle
    MEIPASS_ROOT = Path(sys._MEIPASS).resolve()
    sys.path.insert(0, str(MEIPASS_ROOT))
else:
    # Running in normal Python environment
    MEIPASS_ROOT = Path(__file__).resolve().parent
    sys.path.insert(0, str(MEIPASS_ROOT))

def ensure_workspace():
    """Verify that we are in a valid workspace, otherwise ask the user to select one."""
    if "POLLEN_WORKSPACE" in os.environ:
        return True
        
    if getattr(sys, 'frozen', False):
        workspace = Path(sys.executable).parent.resolve()
    else:
        workspace = Path(__file__).resolve().parent
        
    # Check if this looks like our project folder
    if not ((workspace / "datasets").exists() or (workspace / "config").exists() or (workspace / "datasets_sliced").exists()):
        # We need a hidden root to spawn the dialog
        temp_root = tk.Tk()
        temp_root.withdraw()
        messagebox.showinfo("Workspace Required", "PollenCounter Studio needs to know where your data is stored.\n\nPlease select your primary workspace folder (the folder containing 'datasets' or 'config').")
        
        selected = filedialog.askdirectory(title="Select PollenCounter Workspace Folder")
        temp_root.destroy()
        
        if not selected:
            sys.exit(0)
            
        os.environ["POLLEN_WORKSPACE"] = selected
        
ensure_workspace()

# Router logic for launching tools in separate processes
if len(sys.argv) > 1:
    tool = sys.argv[1]
    try:
        import runpy
        # Override sys.argv so argparse in the scripts doesn't trip over the "tool" argument
        sys.argv = [f"scripts/{tool}.py"]
        
        if tool == "annotate":
            runpy.run_module("scripts.annotate", run_name="__main__")
        elif tool == "inference":
            runpy.run_module("scripts.inference", run_name="__main__")
        elif tool == "monitor":
            runpy.run_module("scripts.monitor", run_name="__main__")
        elif tool == "visualize":
            runpy.run_module("scripts.visualize", run_name="__main__")
        elif tool == "augment":
            runpy.run_module("scripts.augment_preview", run_name="__main__")
        elif tool == "export":
            runpy.run_module("scripts.export", run_name="__main__")
        else:
            print(f"Unknown tool: {tool}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error launching {tool}: {e}")
        import traceback
        traceback.print_exc()
        # Keep window open briefly on crash
        import time
        time.sleep(10)
        
    sys.exit(0)

# If no arguments, show the Main Menu Launcher
from src.theme import BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR

class MainLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("PollenCounter Suite")
        self.root.geometry("600x700")
        self.root.configure(bg=BG_COLOR)
        
        self._build_ui()
        
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=SIDEBAR_BG)
        header.pack(fill=tk.X, pady=(0, 20))
        tk.Label(header, text="PollenCounter Studio", font=("Segoe UI", 24, "bold"), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(pady=20)

        # Content
        content = tk.Frame(self.root, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True, padx=40)
        
        tools = [
            ("Smart Annotator", "Draw bounding boxes and manage dataset", "annotate"),
            ("Batch Inference", "Run models on large folders of images", "inference"),
            ("Export Model", "Convert models to ONNX or TensorRT for massive speedups", "export"),
            ("Augment Previewer", "Visually tune training.yaml augmentations", "augment"),
            ("Neural Net Visualizer", "Step-by-step visualizer of YOLO layer activations", "visualize"),
            ("Training Monitor", "Live real-time Matplotlib graphs of YOLO training", "monitor"),
        ]
        
        for name, desc, cmd in tools:
            # Card Frame
            frame = tk.Frame(content, bg="#1E293B", bd=0)
            frame.pack(fill=tk.X, pady=6)
            
            # Left side (Icon + Text)
            info_frame = tk.Frame(frame, bg="#1E293B")
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=12)
            
            tk.Label(info_frame, text=name, bg="#1E293B", fg="#F8FAFC", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
            tk.Label(info_frame, text=desc, bg="#1E293B", fg="#94A3B8", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(2, 0))
            
            # Right side (Button)
            btn = tk.Button(frame, text="Launch", bg="#3B82F6", fg="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", width=12,
                            activebackground="#2563EB", activeforeground="white",
                            command=lambda c=cmd: self._launch_tool(c))
            btn.pack(side=tk.RIGHT, padx=20, pady=12, ipady=4)
            
            # Hover effects
            btn.bind("<Enter>", lambda e, b=btn: b.config(bg="#2563EB"))
            btn.bind("<Leave>", lambda e, b=btn: b.config(bg="#3B82F6"))

    def _launch_tool(self, tool_arg):
        # Launch independent process so they don't block the launcher
        # Use sys.executable (which is the .exe in compiled mode, or python in source mode)
        # Pass the launcher script itself, followed by the tool argument
        
        if getattr(sys, 'frozen', False):
            # In compiled mode, sys.executable is launcher.exe
            subprocess.Popen([sys.executable, tool_arg])
        else:
            # In source mode, sys.executable is python, and we need to pass launcher.py
            subprocess.Popen([sys.executable, __file__, tool_arg])

if __name__ == "__main__":
    root = tk.Tk()
    app = MainLauncher(root)
    root.mainloop()
