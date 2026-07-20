import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import subprocess
import sys
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

plt.style.use('dark_background')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Theme constants
BG_COLOR = "#1E1E2E"
SIDEBAR_BG = "#181825"
ACCENT = "#8B5CF6"
TEXT_COLOR = "#FFFFFF"

class TrainingMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Pollen YOLOv11s — Training Monitor")
        self.root.geometry("1400x800")
        self.root.configure(bg=BG_COLOR)

        self.process = None
        self.is_training = False
        self.current_csv_path = None
        self.last_epoch_plotted = -1

        self._build_ui()
        self._poll_metrics()

    def _build_ui(self):
        # ── Top bar ─────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=ACCENT, height=48)
        top.pack(fill=tk.X)
        top.pack_propagate(False)

        tk.Label(
            top, text="📈 Training Dashboard", font=("Segoe UI", 14, "bold"),
            bg=ACCENT, fg="white"
        ).pack(side=tk.LEFT, padx=16)

        # ── Main area ───────────────────────────────────────────────
        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        sidebar = tk.Frame(main, bg=SIDEBAR_BG, width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="Configuration", font=("Segoe UI", 12, "bold"),
            bg=SIDEBAR_BG, fg=ACCENT
        ).pack(anchor=tk.W, padx=16, pady=(20, 10))

        # Model
        tk.Label(sidebar, text="Model", font=("Segoe UI", 10), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(anchor=tk.W, padx=16, pady=(10, 0))
        self.model_var = tk.StringVar(value="yolo11s.pt")
        tk.Entry(sidebar, textvariable=self.model_var, font=("Consolas", 11), bg="#313244", fg="white", bd=0, insertbackground="white").pack(fill=tk.X, padx=16, pady=4, ipady=4)

        # Device
        tk.Label(sidebar, text="Device", font=("Segoe UI", 10), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(anchor=tk.W, padx=16, pady=(10, 0))
        self.device_var = tk.StringVar(value="0")
        tk.Entry(sidebar, textvariable=self.device_var, font=("Consolas", 11), bg="#313244", fg="white", bd=0, insertbackground="white").pack(fill=tk.X, padx=16, pady=4, ipady=4)

        # Epochs
        tk.Label(sidebar, text="Epochs", font=("Segoe UI", 10), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(anchor=tk.W, padx=16, pady=(10, 0))
        self.epochs_var = tk.StringVar(value="150")
        tk.Entry(sidebar, textvariable=self.epochs_var, font=("Consolas", 11), bg="#313244", fg="white", bd=0, insertbackground="white").pack(fill=tk.X, padx=16, pady=4, ipady=4)

        # Batch Size
        tk.Label(sidebar, text="Batch Size", font=("Segoe UI", 10), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(anchor=tk.W, padx=16, pady=(10, 0))
        self.batch_var = tk.StringVar(value="4")
        tk.Entry(sidebar, textvariable=self.batch_var, font=("Consolas", 11), bg="#313244", fg="white", bd=0, insertbackground="white").pack(fill=tk.X, padx=16, pady=4, ipady=4)
        
        # Image Size
        tk.Label(sidebar, text="Image Size", font=("Segoe UI", 10), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(anchor=tk.W, padx=16, pady=(10, 0))
        self.imgsz_var = tk.StringVar(value="1024")
        tk.Entry(sidebar, textvariable=self.imgsz_var, font=("Consolas", 11), bg="#313244", fg="white", bd=0, insertbackground="white").pack(fill=tk.X, padx=16, pady=4, ipady=4)

        # K-Folds
        tk.Label(sidebar, text="K-Folds (1 = none)", font=("Segoe UI", 10), bg=SIDEBAR_BG, fg=TEXT_COLOR).pack(anchor=tk.W, padx=16, pady=(10, 0))
        self.kfolds_var = tk.StringVar(value="1")
        tk.Entry(sidebar, textvariable=self.kfolds_var, font=("Consolas", 11), bg="#313244", fg="white", bd=0, insertbackground="white").pack(fill=tk.X, padx=16, pady=4, ipady=4)

        tk.Frame(sidebar, bg="#45475A", height=1).pack(fill=tk.X, padx=16, pady=20)

        # Buttons
        self.start_btn = tk.Button(
            sidebar, text="▶ Start Training", font=("Segoe UI", 11, "bold"),
            bg="#059669", fg="white", activebackground="#047857",
            cursor="hand2", bd=0, pady=8, command=self._start_training
        )
        self.start_btn.pack(fill=tk.X, padx=16, pady=10)

        self.stop_btn = tk.Button(
            sidebar, text="⏹ Stop Training", font=("Segoe UI", 11, "bold"),
            bg="#DC2626", fg="white", activebackground="#B91C1C",
            cursor="hand2", bd=0, pady=8, command=self._stop_training, state=tk.DISABLED
        )
        self.stop_btn.pack(fill=tk.X, padx=16)

        # Content area (Right)
        content = tk.Frame(main, bg=BG_COLOR)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=16, pady=16)

        # ── Graphs ──
        graph_frame = tk.Frame(content, bg="#11111B", bd=1, relief=tk.SOLID)
        graph_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 16))

        self.fig = Figure(figsize=(10, 4), dpi=100)
        self.fig.patch.set_facecolor('#11111B')
        
        # Subplots
        self.ax_loss = self.fig.add_subplot(121)
        self.ax_loss.set_title("Training Loss", color="white")
        self.ax_loss.tick_params(colors="white")
        self.ax_loss.set_facecolor('#1E1E2E')

        self.ax_map = self.fig.add_subplot(122)
        self.ax_map.set_title("Validation Accuracy (mAP50)", color="white")
        self.ax_map.tick_params(colors="white")
        self.ax_map.set_facecolor('#1E1E2E')

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Terminal ──
        term_frame = tk.Frame(content, bg="#11111B", height=250)
        term_frame.pack(fill=tk.X)
        term_frame.pack_propagate(False)
        
        tk.Label(term_frame, text="Terminal Output", font=("Segoe UI", 10, "bold"), bg="#11111B", fg=ACCENT).pack(anchor=tk.W, padx=8, pady=4)
        
        self.terminal = scrolledtext.ScrolledText(
            term_frame, bg="#000000", fg="#00FF00", font=("Consolas", 10),
            bd=0, padx=8, pady=8, state=tk.DISABLED
        )
        self.terminal.pack(fill=tk.BOTH, expand=True)

    def _start_training(self):
        if self.is_training: return
        
        model = self.model_var.get()
        device = self.device_var.get()
        epochs = self.epochs_var.get()
        batch = self.batch_var.get()
        imgsz = self.imgsz_var.get()
        kfolds = self.kfolds_var.get()

        self.terminal.config(state=tk.NORMAL)
        self.terminal.delete(1.0, tk.END)
        self.terminal.insert(tk.END, f"> Starting training:\n  Model: {model}\n  Device: {device}\n  Epochs: {epochs}\n  Batch: {batch}\n  ImgSz: {imgsz}\n  K-Folds: {kfolds}\n\n")
        self.terminal.config(state=tk.DISABLED)

        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # lock it in ui
        self.model_var.set(model)
        self.device_var.set(device)
        self.epochs_var.set(epochs)
        self.batch_var.set(batch)
        self.imgsz_var.set(imgsz)
        self.kfolds_var.set(kfolds)
        self.is_training = True
        self.last_epoch_plotted = -1

        # Clear graphs
        self.ax_loss.clear()
        self.ax_map.clear()
        self.ax_loss.set_title("Training Loss", color="white")
        self.ax_map.set_title("Validation Accuracy (mAP50)", color="white")
        self.canvas.draw()

        # We will dynamically find the newest CSV in _poll_metrics
        self.current_csv_path = None
        self.start_time = pd.Timestamp.now().timestamp() # to ensure we only pick up NEW runs
        
        # Start subprocess
        run_name = "train"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train.py"),
            "--model", model,
            "--device", device,
            "--epochs", epochs,
            "--batch", batch,
            "--imgsz", imgsz,
            "--kfold", kfolds,
            "--name", run_name
        ]
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            cwd=str(PROJECT_ROOT)
        )
        
        # Thread to read stdout without blocking GUI
        threading.Thread(target=self._read_output, daemon=True).start()

    def _stop_training(self):
        if not self.is_training or not self.process: return
        
        if messagebox.askyesno("Stop Training", "Are you sure you want to abort training?"):
            self.process.terminate()
            self._log_terminal("\n\n> [WARNING] Training aborted by user.\n")
            self._finalize_training()

    def _read_output(self):
        for line in iter(self.process.stdout.readline, ''):
            self.root.after(0, self._log_terminal, line)
        
        self.process.wait()
        
        if self.process.returncode == 0:
            self.root.after(0, self._log_terminal, "\n\n> ✅ Training completed successfully!\n")
        else:
            self.root.after(0, self._log_terminal, f"\n\n> ❌ Training crashed (Exit code {self.process.returncode}).\n")
            
        self.root.after(0, self._finalize_training)

    def _log_terminal(self, text):
        self.terminal.config(state=tk.NORMAL)
        self.terminal.insert(tk.END, text)
        self.terminal.see(tk.END)
        self.terminal.config(state=tk.DISABLED)

    def _finalize_training(self):
        self.is_training = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def _find_newest_csv(self):
        project_dir = PROJECT_ROOT / "runs" / "detect"
        if not project_dir.exists():
            return None
        csv_files = list(project_dir.rglob("results.csv"))
        if not csv_files:
            return None
        # Only consider CSVs modified AFTER we clicked Start Training
        valid_csvs = [p for p in csv_files if p.stat().st_mtime > self.start_time - 5.0]
        if not valid_csvs:
            return None
        return max(valid_csvs, key=lambda p: p.stat().st_mtime)

    def _poll_metrics(self):
        if self.is_training:
            if not self.current_csv_path or not self.current_csv_path.exists():
                self.current_csv_path = self._find_newest_csv()
                
            if self.current_csv_path and self.current_csv_path.exists():
                try:
                    # YOLO writes CSV with leading spaces in headers sometimes
                    df = pd.read_csv(self.current_csv_path)
                    df.columns = df.columns.str.strip()
                    
                    if not df.empty and len(df) > self.last_epoch_plotted:
                        self.last_epoch_plotted = len(df)
                    
                    epochs = df['epoch']
                    box_loss = df['train/box_loss']
                    cls_loss = df['train/cls_loss']
                    map50 = df['metrics/mAP50(B)']
                    
                    self.ax_loss.clear()
                    self.ax_loss.set_title("Training Loss", color="white")
                    self.ax_loss.plot(epochs, box_loss, label='Box Loss', color='#F59E0B', linewidth=2)
                    self.ax_loss.plot(epochs, cls_loss, label='Class Loss', color='#EF4444', linewidth=2)
                    self.ax_loss.legend(facecolor='#1E1E2E', edgecolor='#45475A', labelcolor='white')
                    self.ax_loss.set_facecolor('#1E1E2E')
                    
                    self.ax_map.clear()
                    self.ax_map.set_title("Validation Accuracy (mAP50)", color="white")
                    self.ax_map.plot(epochs, map50, label='mAP50', color='#10B981', linewidth=2)
                    self.ax_map.fill_between(epochs, map50, alpha=0.2, color='#10B981')
                    self.ax_map.set_facecolor('#1E1E2E')
                    self.ax_map.set_ylim(0, 1.0)
                    
                    self.fig.tight_layout()
                    self.canvas.draw()
                except Exception as e:
                    pass # File might be locked mid-write by YOLO
                
        # Poll again in 2 seconds
        self.root.after(2000, self._poll_metrics)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingMonitorApp(root)
    root.mainloop()
