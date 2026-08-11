#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import yaml
import os
import sys
import psutil
import subprocess
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "detect"

class TrainingMonitor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Training Monitor V2")
        self.geometry("1050x850")
        self.configure(bg="#F8FAFC")
        
        # Style
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.style.configure("TFrame", background="#F8FAFC")
        self.style.configure("TLabel", background="#F8FAFC", foreground="#1E293B", font=("Roboto", 11))
        self.style.configure("Header.TLabel", font=("Roboto", 16, "bold"), foreground="#0F172A", background="#F8FAFC")
        self.style.configure("Danger.TButton", font=("Roboto", 12, "bold"), background="#EF4444", foreground="#FFFFFF")
        self.style.map("Danger.TButton", background=[("active", "#DC2626")])
        
        self.style.configure("Treeview", background="#FFFFFF", foreground="#1E293B", fieldbackground="#FFFFFF", rowheight=25, font=("Roboto", 10))
        self.style.configure("Treeview.Heading", background="#E2E8F0", foreground="#0F172A", font=("Roboto", 11, "bold"))
        self.style.map("Treeview", background=[("selected", "#3B82F6")])
        
        self.active_dir = None
        self.countdown = 10
        self.refresh_interval = 10
        
        # Historical bests for visual cues
        self.best_hist_p = 0.0
        self.best_hist_r = 0.0
        self.best_hist_map50 = 0.0
        self.best_hist_map5095 = 0.0
        
        self.max_epochs = 0
        self.patience = 50
        
        self.fix_xaxis = tk.BooleanVar(value=True)
        
        self._build_ui()
        self._find_active_run()
        self._read_config()
        self._refresh_data()
        self._tick()
        
    def _build_ui(self):
        # Sidebar
        sidebar = ttk.Frame(self, width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=20, pady=20)
        
        # --- Launch Section ---
        ttk.Label(sidebar, text="Launch Training", style="Header.TLabel").pack(anchor="w", pady=(0, 10))
        
        form = tk.Frame(sidebar, bg="#F8FAFC")
        form.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(form, text="Epochs:", bg="#F8FAFC", fg="#1E293B", font=("Roboto", 10)).grid(row=0, column=0, sticky="w", pady=2)
        self.ent_epochs = ttk.Entry(form, width=8)
        self.ent_epochs.insert(0, "100")
        self.ent_epochs.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        tk.Label(form, text="Batch Size:", bg="#F8FAFC", fg="#1E293B", font=("Roboto", 10)).grid(row=1, column=0, sticky="w", pady=2)
        self.ent_batch = ttk.Entry(form, width=8)
        self.ent_batch.insert(0, "4")
        self.ent_batch.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        
        tk.Label(form, text="Img Size:", bg="#F8FAFC", fg="#1E293B", font=("Roboto", 10)).grid(row=2, column=0, sticky="w", pady=2)
        self.ent_imgsz = ttk.Entry(form, width=8)
        self.ent_imgsz.insert(0, "512")
        self.ent_imgsz.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        
        tk.Label(form, text="Patience:", bg="#F8FAFC", fg="#1E293B", font=("Roboto", 10)).grid(row=3, column=0, sticky="w", pady=2)
        self.ent_patience = ttk.Entry(form, width=8)
        self.ent_patience.insert(0, "50")
        self.ent_patience.grid(row=3, column=1, sticky="w", padx=5, pady=2)
        
        self.btn_start = tk.Button(sidebar, text="Start Training", bg="#10B981", fg="#FFFFFF", font=("Roboto", 11, "bold"), bd=0, cursor="hand2", command=self._start_training)
        self.btn_start.pack(fill=tk.X, pady=(0, 20), ipady=5)
        
        # --- Config Section ---
        ttk.Label(sidebar, text="Current Config", style="Header.TLabel").pack(anchor="w", pady=(0, 10))
        self.lbl_folder = ttk.Label(sidebar, text="Folder: N/A")
        self.lbl_folder.pack(anchor="w", pady=2)
        self.lbl_model = ttk.Label(sidebar, text="Model: N/A")
        self.lbl_model.pack(anchor="w", pady=2)
        self.lbl_imgsz = ttk.Label(sidebar, text="Image Size: N/A")
        self.lbl_imgsz.pack(anchor="w", pady=2)
        self.lbl_batch = ttk.Label(sidebar, text="Batch Size: N/A")
        self.lbl_batch.pack(anchor="w", pady=2)
        self.lbl_train_img = ttk.Label(sidebar, text="Train Images: N/A")
        self.lbl_train_img.pack(anchor="w", pady=2)
        self.lbl_val_img = ttk.Label(sidebar, text="Val Images: N/A")
        self.lbl_val_img.pack(anchor="w", pady=(2, 20))
        
        # --- Status Section ---
        ttk.Label(sidebar, text="Run Status", style="Header.TLabel").pack(anchor="w", pady=(0, 10))
        self.lbl_epoch = ttk.Label(sidebar, text="Epoch: 0 / 0")
        self.lbl_epoch.pack(anchor="w", pady=2)
        
        self.lbl_time_left = ttk.Label(sidebar, text="Max ETA: N/A", foreground="#3B82F6")
        self.lbl_time_left.pack(anchor="w", pady=2)
        
        self.lbl_early_stop = ttk.Label(sidebar, text="Stop ETA: N/A", foreground="#F59E0B")
        self.lbl_early_stop.pack(anchor="w", pady=(2, 10))
        
        self.lbl_best_epoch = ttk.Label(sidebar, text="Best Epoch: 0", font=("Roboto", 12, "bold"), foreground="#059669")
        self.lbl_best_epoch.pack(anchor="w", pady=(10, 2))
        
        self.lbl_best_p = ttk.Label(sidebar, text="Precision: 0.00%")
        self.lbl_best_p.pack(anchor="w", pady=2)
        self.lbl_best_r = ttk.Label(sidebar, text="Recall: 0.00%")
        self.lbl_best_r.pack(anchor="w", pady=2)
        
        self.lbl_best_map50 = ttk.Label(sidebar, text="mAP50: 0.00%")
        self.lbl_best_map50.pack(anchor="w", pady=2)
        self.lbl_best_map5095 = ttk.Label(sidebar, text="mAP50-95: 0.00%")
        self.lbl_best_map5095.pack(anchor="w", pady=2)
        
        # --- Graph Options ---
        ttk.Label(sidebar, text="Graph Options", style="Header.TLabel").pack(anchor="w", pady=(20, 5))
        self.chk_xaxis = ttk.Checkbutton(sidebar, text="Lock X-Axis to Max Epochs", variable=self.fix_xaxis, command=self._manual_refresh)
        self.chk_xaxis.pack(anchor="w", pady=(0, 10))
        
        # --- Refresh Section ---
        self.lbl_refresh = ttk.Label(sidebar, text="Next Refresh: 10s", foreground="#10B981", font=("Roboto", 12, "bold"))
        self.lbl_refresh.pack(anchor="w", pady=(30, 10))
        
        self.btn_refresh = ttk.Button(sidebar, text="🔄 Refresh Now", command=self._manual_refresh)
        self.btn_refresh.pack(anchor="w", pady=(0, 20), fill=tk.X)
        
        # Stop Button
        self.btn_stop = ttk.Button(sidebar, text="⏹ Stop Training", style="Danger.TButton", command=self._stop_training)
        self.btn_stop.pack(anchor="w", pady=20, fill=tk.X)
        
        # Right Panel Split
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Main Graph Area
        graph_frame = ttk.Frame(right_panel)
        graph_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.fig = Figure(figsize=(8, 7), facecolor='#F8FAFC')
        self.ax1 = self.fig.add_subplot(3, 1, 1)
        self.ax2 = self.fig.add_subplot(3, 1, 2)
        self.ax3_train = self.fig.add_subplot(3, 2, 5)
        self.ax3_val = self.fig.add_subplot(3, 2, 6)
        
        self.fig.subplots_adjust(hspace=0.5, wspace=0.3, left=0.1, right=0.95, top=0.93, bottom=0.07)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # History Table Area
        table_frame = ttk.Frame(right_panel, height=130)
        table_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        columns = ("run", "epoch", "precision", "recall", "map50", "map5095")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=4)
        
        self.tree.heading("run", text="Model Run")
        self.tree.heading("epoch", text="Best Epoch")
        self.tree.heading("precision", text="Precision (%)")
        self.tree.heading("recall", text="Recall (%)")
        self.tree.heading("map50", text="mAP50 (%)")
        self.tree.heading("map5095", text="mAP50-95 (%)")
        
        self.tree.column("run", width=350, anchor="w")
        self.tree.column("epoch", width=80, anchor="center")
        self.tree.column("precision", width=100, anchor="center")
        self.tree.column("recall", width=100, anchor="center")
        self.tree.column("map50", width=100, anchor="center")
        self.tree.column("map5095", width=100, anchor="center")
        
        self.tree.tag_configure('live_better', background='#FEF3C7', foreground='#065F46', font=("Roboto", 10, "bold"))
        self.tree.tag_configure('live_worse', background='#FEF3C7', foreground='#991B1B', font=("Roboto", 10, "bold"))
        self.tree.tag_configure('live_neutral', background='#FEF3C7', foreground='#1E40AF', font=("Roboto", 10, "bold"))
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        
    def _find_active_run(self):
        if not RUNS_DIR.exists():
            return
            
        dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir()]
        if not dirs:
            return
            
        latest_dir = max(dirs, key=os.path.getmtime)
        if self.active_dir != latest_dir:
            self.active_dir = latest_dir
            self.lbl_folder.config(text=f"Folder: {self.active_dir.name}")
            self._read_config()
            self._fetch_history()

    def _fetch_history(self):
        if not RUNS_DIR.exists():
            return
            
        dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("i")]
        
        def get_i_num(d):
            try: return int(d.name.split('_')[0][1:])
            except: return -1
                
        dirs.sort(key=get_i_num)
        history_dirs = [d for d in dirs if d != self.active_dir][-3:]
        history_dirs.reverse()  # Sort newest history first
        
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Reset historical bests
        self.best_hist_p = 0.0
        self.best_hist_r = 0.0
        self.best_hist_map50 = 0.0
        self.best_hist_map5095 = 0.0
            
        for d in history_dirs:
            csv_path = d / "results.csv"
            if not csv_path.exists(): continue
            try:
                df = pd.read_csv(csv_path)
                df.columns = df.columns.str.strip()
                if df.empty or 'metrics/mAP50(B)' not in df.columns: continue
                
                best_idx = df['metrics/mAP50(B)'].idxmax()
                map50 = df['metrics/mAP50(B)'].iloc[best_idx] * 100
                map50_95 = df['metrics/mAP50-95(B)'].iloc[best_idx] * 100
                precision = df['metrics/precision(B)'].iloc[best_idx] * 100
                recall = df['metrics/recall(B)'].iloc[best_idx] * 100
                epochs = df.get('epoch', df.index + 1)
                best_epoch = epochs.iloc[best_idx]
                
                # Track historical bests
                self.best_hist_p = max(self.best_hist_p, precision)
                self.best_hist_r = max(self.best_hist_r, recall)
                self.best_hist_map50 = max(self.best_hist_map50, map50)
                self.best_hist_map5095 = max(self.best_hist_map5095, map50_95)
                
                self.tree.insert("", "end", values=(d.name, f"{best_epoch:.0f}", f"{precision:.2f}%", f"{recall:.2f}%", f"{map50:.2f}%", f"{map50_95:.2f}%"))
            except Exception as e:
                pass

    def _read_config(self):
        if not self.active_dir:
            return
            
        args_path = self.active_dir / "args.yaml"
        if not args_path.exists():
            return
            
        try:
            with open(args_path, 'r') as f:
                args = yaml.safe_load(f)
                
            self.max_epochs = args.get('epochs', 0)
            self.patience = args.get('patience', 100)
                
            self.lbl_model.config(text=f"Model: {args.get('model', 'N/A')}")
            self.lbl_imgsz.config(text=f"Image Size: {args.get('imgsz', 'N/A')}")
            self.lbl_batch.config(text=f"Batch Size: {args.get('batch', 'N/A')}")
            
            # Try to count images
            data_yaml_path = args.get('data')
            if data_yaml_path:
                data_path = Path(data_yaml_path)
                if not data_path.is_absolute():
                    data_path = PROJECT_ROOT / data_path
                    
                if data_path.exists():
                    with open(data_path, 'r') as df:
                        data_cfg = yaml.safe_load(df)
                        
                    dataset_root = Path(data_cfg.get('path', ''))
                    if not dataset_root.is_absolute():
                        dataset_root = PROJECT_ROOT / dataset_root
                        
                    # YOLO and our custom train.py often use train.txt
                    train_txt = dataset_root / "train.txt"
                    val_txt = dataset_root / "val.txt"
                    
                    if train_txt.exists():
                        count = sum(1 for _ in open(train_txt))
                        self.lbl_train_img.config(text=f"Train Images: {count}")
                    elif (dataset_root / data_cfg.get('train', '')).is_dir():
                        count = len(list((dataset_root / data_cfg.get('train', '')).glob('*.*')))
                        self.lbl_train_img.config(text=f"Train Images: {count}")
                        
                    if val_txt.exists():
                        count = sum(1 for _ in open(val_txt))
                        self.lbl_val_img.config(text=f"Val Images: {count}")
                    elif (dataset_root / data_cfg.get('val', '')).is_dir():
                        count = len(list((dataset_root / data_cfg.get('val', '')).glob('*.*')))
                        self.lbl_val_img.config(text=f"Val Images: {count}")
        except Exception as e:
            print(f"Error reading args.yaml: {e}")

    def _tick(self):
        self.countdown -= 1
        
        if self.countdown <= 0:
            self._find_active_run()
            self._refresh_data()
            self.countdown = self.refresh_interval
            
        self.lbl_refresh.config(text=f"Next Refresh: {self.countdown}s")
        self.after(1000, self._tick)

    def _manual_refresh(self):
        self._find_active_run()
        self._refresh_data()
        self.countdown = self.refresh_interval

    def _refresh_data(self):
        if not self.active_dir:
            return
            
        csv_path = self.active_dir / "results.csv"
        if not csv_path.exists():
            return
            
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            if df.empty:
                return
                
            epochs = df.get('epoch', df.index + 1)
            map50 = df['metrics/mAP50(B)'] * 100
            map50_95 = df['metrics/mAP50-95(B)'] * 100
            precision = df['metrics/precision(B)'] * 100
            recall = df['metrics/recall(B)'] * 100
            
            # Find the best epoch based on YOLO's actual fitness metric (0.1 * mAP50 + 0.9 * mAP50-95)
            fitness = (0.1 * map50) + (0.9 * map50_95)
            best_idx = fitness.idxmax()
            best_epoch_num = epochs.iloc[best_idx]
            
            # Compute dynamic refresh rate & time left
            if 'time' in df.columns and len(df) > 1:
                last_epoch_time = df['time'].iloc[-1] - df['time'].iloc[-2]
                if pd.notna(last_epoch_time) and last_epoch_time > 0:
                    self.refresh_interval = max(5, int(last_epoch_time))
                    
                    # Max Epoch ETA
                    epochs_left = self.max_epochs - epochs.iloc[-1]
                    if epochs_left > 0:
                        secs_left = epochs_left * last_epoch_time
                        m, s = divmod(int(secs_left), 60)
                        h, m = divmod(m, 60)
                        self.lbl_time_left.config(text=f"Max ETA: ~{h}h {m}m")
                    else:
                        self.lbl_time_left.config(text="Max ETA: Finishing...")
                        
                    # Early Stop ETA
                    if self.patience > 0:
                        pred_stop = best_epoch_num + self.patience
                        if pred_stop <= self.max_epochs:
                            early_epochs_left = pred_stop - epochs.iloc[-1]
                            if early_epochs_left > 0:
                                early_secs = early_epochs_left * last_epoch_time
                                em, es = divmod(int(early_secs), 60)
                                eh, em = divmod(em, 60)
                                self.lbl_early_stop.config(text=f"Stop ETA: ~{eh}h {em}m (@ Ep {pred_stop:.0f})")
                            else:
                                self.lbl_early_stop.config(text=f"Stop ETA: Imminent (@ Ep {pred_stop:.0f})")
                        else:
                            self.lbl_early_stop.config(text="Stop ETA: Unlikely (Past Max)")
                    else:
                        self.lbl_early_stop.config(text="Stop ETA: Disabled")
            else:
                self.refresh_interval = 10
                self.lbl_time_left.config(text="Max ETA: Calculating...")
                if self.patience > 0:
                    self.lbl_early_stop.config(text="Stop ETA: Calculating...")
                else:
                    self.lbl_early_stop.config(text="Stop ETA: Disabled")
            
            # Update Sidebar
            self.lbl_epoch.config(text=f"Epoch: {epochs.iloc[-1] + 1:.0f} / {self.max_epochs}")
            
            self.lbl_best_epoch.config(text=f"Best Epoch: {best_epoch_num:.0f}")
            self.lbl_best_p.config(text=f"Precision: {precision.iloc[best_idx]:.2f}%")
            self.lbl_best_r.config(text=f"Recall: {recall.iloc[best_idx]:.2f}%")
            self.lbl_best_map50.config(text=f"mAP50: {map50.iloc[best_idx]:.2f}%")
            self.lbl_best_map5095.config(text=f"mAP50-95: {map50_95.iloc[best_idx]:.2f}%")
            
            # Update Graphs
            self.ax1.clear()
            self.ax2.clear()
            self.ax3_train.clear()
            self.ax3_val.clear()
            
            # Style Axes
            for ax in [self.ax1, self.ax2, self.ax3_train, self.ax3_val]:
                ax.set_facecolor('#F8FAFC')
                ax.tick_params(colors='#1E293B')
                if self.fix_xaxis.get() and self.max_epochs > 0:
                    ax.set_xlim(left=1, right=self.max_epochs)
                for spine in ax.spines.values():
                    spine.set_color('#CBD5E1')
            
            # Chart 1: mAP
            self.ax1.plot(epochs, map50, color='#10B981', label='mAP50', linewidth=2)
            self.ax1.plot(epochs, map50_95, color='#3B82F6', label='mAP50-95', linewidth=2)
            self.ax1.set_title("Mean Average Precision", color='#0F172A', pad=10, fontdict={'family': 'Roboto', 'weight': 'bold'})
            self.ax1.legend(facecolor='#FFFFFF', labelcolor='#1E293B')
            self.ax1.grid(True, color='#E2E8F0', linestyle='--')
            
            # Chart 2: Precision & Recall
            self.ax2.plot(epochs, precision, color='#D946EF', label='Precision', linewidth=2)
            self.ax2.plot(epochs, recall, color='#EAB308', label='Recall', linewidth=2)
            self.ax2.set_title("Precision & Recall", color='#0F172A', pad=10, fontdict={'family': 'Roboto', 'weight': 'bold'})
            self.ax2.legend(facecolor='#FFFFFF', labelcolor='#1E293B')
            self.ax2.grid(True, color='#E2E8F0', linestyle='--')
            
            # Chart 3a: Train Losses
            self.ax3_train.plot(epochs, df['train/box_loss'], color='#EF4444', label='Box', linewidth=2)
            if 'train/cls_loss' in df.columns:
                self.ax3_train.plot(epochs, df['train/cls_loss'], color='#D946EF', label='Cls', linewidth=2)
            if 'train/dfl_loss' in df.columns:
                self.ax3_train.plot(epochs, df['train/dfl_loss'], color='#10B981', label='DFL', linewidth=2)
                
            self.ax3_train.set_title("Training Losses", color='#0F172A', pad=10, fontdict={'family': 'Roboto', 'weight': 'bold'})
            self.ax3_train.legend(facecolor='#FFFFFF', labelcolor='#1E293B', ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.0))
            self.ax3_train.grid(True, color='#E2E8F0', linestyle='--')
            
            # Chart 3b: Val Losses
            if 'val/box_loss' in df.columns:
                self.ax3_val.plot(epochs, df['val/box_loss'], color='#FFA726', label='Box', linewidth=2, linestyle='--')
                self.ax3_val.plot(epochs, df['val/cls_loss'], color='#B388FF', label='Cls', linewidth=2, linestyle='--')
                self.ax3_val.plot(epochs, df['val/dfl_loss'], color='#69F0AE', label='DFL', linewidth=2, linestyle='--')
                
            self.ax3_val.set_title("Validation Losses", color='white', pad=10)
            self.ax3_val.legend(facecolor='#2E2E2E', labelcolor='white', ncol=3, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, 1.0))
            self.ax3_val.grid(True, color='#333333', linestyle='--')
            
            # Update Treeview Current Row with Visual Cues
            curr_p = precision.iloc[best_idx]
            curr_r = recall.iloc[best_idx]
            curr_map50 = map50.iloc[best_idx]
            curr_map5095 = map50_95.iloc[best_idx]
            
            def cue(current, hist_best):
                if hist_best == 0: return f"{current:.2f}%"
                if current > hist_best: return f"{current:.2f}% ▲"
                elif current < hist_best: return f"{current:.2f}% ▼"
                return f"{current:.2f}%"
                
            current_values = (
                f"{self.active_dir.name} (Live)", 
                f"{epochs.iloc[best_idx]:.0f}", 
                cue(curr_p, self.best_hist_p), 
                cue(curr_r, self.best_hist_r), 
                cue(curr_map50, self.best_hist_map50), 
                cue(curr_map5095, self.best_hist_map5095)
            )
            
            children = self.tree.get_children()
            tag = 'live_better' if curr_map5095 > self.best_hist_map5095 else ('live_worse' if curr_map5095 < self.best_hist_map5095 else 'live_neutral')
            
            if children and str(self.tree.item(children[0])['values'][0]).endswith("(Live)"):
                self.tree.item(children[0], values=current_values, tags=(tag,))
            else:
                self.tree.insert("", 0, values=current_values, tags=(tag,))
                
            self.canvas.draw()
        except Exception as e:
            print(f"Error reading CSV: {e}")

    def _stop_training(self):
        if messagebox.askyesno("Confirm Stop", "Are you sure you want to forcibly stop training? (YOLO will still save the latest completed epoch as last.pt)"):
            killed = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and 'python' in proc.info['name'].lower() and 'train.py' in ' '.join(cmdline):
                        proc.kill()
                        killed = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if killed:
                messagebox.showinfo("Stopped", "Training process has been terminated.")
            else:
                messagebox.showwarning("Not Found", "Could not find a running train.py process.")

    def _start_training(self):
        # Validate inputs
        try:
            epochs = int(self.ent_epochs.get())
            batch = int(self.ent_batch.get())
            imgsz = int(self.ent_imgsz.get())
            patience = int(self.ent_patience.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for epochs, batch size, image size, and patience.")
            return
            
        # Check if already running
        for proc in psutil.process_iter(['name', 'cmdline']):
            try:
                cmd = proc.info['cmdline']
                if cmd and 'python' in proc.info['name'].lower() and 'train.py' in ' '.join(cmd):
                    messagebox.showerror("Error", "A training process is already running!")
                    return
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
                
        # Launch process
        script_path = PROJECT_ROOT / "scripts" / "train.py"
        cmd = [
            sys.executable, str(script_path),
            "--epochs", str(epochs),
            "--batch", str(batch),
            "--imgsz", str(imgsz),
            "--patience", str(patience)
        ]
        
        try:
            subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
            messagebox.showinfo("Started", "Training has started in the background! The graph will update automatically in a few moments.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")

if __name__ == "__main__":
    app = TrainingMonitor()
    app.mainloop()
