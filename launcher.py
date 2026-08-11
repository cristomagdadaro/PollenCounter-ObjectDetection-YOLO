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
    if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
        
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
        elif tool == "view_logs":
            workspace = os.environ.get("POLLEN_WORKSPACE", str(Path(__file__).resolve().parent))
            log_file = Path(workspace) / "error_log.txt"
            if not log_file.exists():
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write("--- PollenCounter Studio Error Logs ---\n")
            os.startfile(str(log_file))
            sys.exit(0)
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
from src.theme import BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR, FONT_MAIN, FONT_HEADER, FONT_LABEL

class MainLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("PollenCounter Suite")
        self.root.geometry("600x700")
        self.root.configure(bg=BG_COLOR)
        
        self._build_ui()
        
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg=BG_COLOR)
        header.pack(fill=tk.X, pady=(20, 20))
        tk.Label(header, text="PollenCounter Studio", font=FONT_HEADER, bg=BG_COLOR, fg=TEXT_COLOR).pack()

        # Content
        content = tk.Frame(self.root, bg=BG_COLOR)
        content.pack(fill=tk.BOTH, expand=True, padx=40)
        
        tools = [
            ("Smart Annotator", "Draw bounding boxes and manage dataset", "annotate"),
            ("Batch Inference", "Run models on large folders of images", "inference"),
            ("Neural Net Visualizer", "Step-by-step visualizer of YOLO layer activations", "visualize"),
            ("Error Logs View", "View application crash and error logs", "view_logs"),
        ]
        
        # Add developer tools only if not compiled (source mode)
        if not getattr(sys, 'frozen', False):
            tools.extend([
                ("Export Model", "Convert models to ONNX or TensorRT for massive speedups", "export"),
                ("Augment Previewer", "Visually tune training.yaml augmentations", "augment"),
                ("Training Monitor", "Live real-time Matplotlib graphs of YOLO training", "monitor"),
            ])
        
        for name, desc, cmd in tools:
            # Card Frame (White Surface)
            frame = tk.Frame(content, bg=SIDEBAR_BG, bd=1, relief="solid")
            # We use a subtle border for cards. A frame inside a frame can simulate border color,
            # but setting highlightbackground and highlightthickness works too.
            frame.config(highlightbackground="#E2E8F0", highlightthickness=1, bd=0)
            frame.pack(fill=tk.X, pady=6)
            
            # Left side (Text)
            info_frame = tk.Frame(frame, bg=SIDEBAR_BG)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=12)
            
            tk.Label(info_frame, text=name, bg=SIDEBAR_BG, fg=TEXT_COLOR, font=FONT_LABEL).pack(anchor=tk.W)
            tk.Label(info_frame, text=desc, bg=SIDEBAR_BG, fg="#64748B", font=FONT_MAIN).pack(anchor=tk.W, pady=(2, 0))
            
            # Right side (Button)
            btn = tk.Button(
                frame, text="Launch", bg=ACCENT, fg="white",
                font=FONT_LABEL, bd=0, padx=20, pady=8,
                activebackground="#2563EB", cursor="hand2",
                command=lambda c=cmd: self._launch_tool(c)
            )
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
