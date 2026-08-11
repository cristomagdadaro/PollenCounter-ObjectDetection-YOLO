import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import glob

# Add parent to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.theme import BG_COLOR, SIDEBAR_BG, ACCENT, TEXT_COLOR
from src.paths import PROJECT_ROOT

class ExportGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Model Export & Acceleration")
        self.geometry("600x500")
        self.configure(bg=BG_COLOR)

        self._build_ui()
        self._load_models()

    def _build_ui(self):
        main = tk.Frame(self, bg=BG_COLOR)
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        tk.Label(main, text="Export YOLOv11 Model", font=("Segoe UI", 16, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(anchor=tk.W, pady=(0, 15))

        # 1. Model Selection
        model_frame = tk.Frame(main, bg=BG_COLOR)
        model_frame.pack(fill=tk.X, pady=5)
        tk.Label(model_frame, text="Select Model:", font=("Segoe UI", 10, "bold"), bg=BG_COLOR, fg=TEXT_COLOR, width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(model_frame, state="readonly", width=40)
        self.model_combo.pack(side=tk.LEFT, padx=5)

        # 2. Format Selection
        format_frame = tk.Frame(main, bg=BG_COLOR)
        format_frame.pack(fill=tk.X, pady=5)
        tk.Label(format_frame, text="Export Format:", font=("Segoe UI", 10, "bold"), bg=BG_COLOR, fg=TEXT_COLOR, width=15, anchor=tk.W).pack(side=tk.LEFT)
        self.format_var = tk.StringVar(value="onnx")
        self.format_combo = ttk.Combobox(format_frame, textvariable=self.format_var, state="readonly", width=40, values=["onnx", "engine", "torchscript", "openvino"])
        self.format_combo.pack(side=tk.LEFT, padx=5)
        
        format_desc = tk.Label(main, text="ONNX is universally compatible. 'engine' is TensorRT (requires NVIDIA SDK).", font=("Segoe UI", 9, "italic"), bg=BG_COLOR, fg="#9CA3AF")
        format_desc.pack(anchor=tk.W, padx=115)

        # 3. Optimizations
        opt_frame = tk.LabelFrame(main, text="Optimizations", bg=BG_COLOR, fg=TEXT_COLOR, font=("Segoe UI", 10, "bold"), bd=1)
        opt_frame.pack(fill=tk.X, pady=15, ipady=5)

        self.fp16_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Half Precision (FP16) - 2x faster, 50% memory", variable=self.fp16_var, bg=BG_COLOR, fg=TEXT_COLOR, selectcolor=SIDEBAR_BG, activebackground=BG_COLOR, activeforeground=TEXT_COLOR, font=("Segoe UI", 9)).pack(anchor=tk.W, padx=10, pady=2)

        self.dynamic_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opt_frame, text="Dynamic Input Size (Support varying image resolutions)", variable=self.dynamic_var, bg=BG_COLOR, fg=TEXT_COLOR, selectcolor=SIDEBAR_BG, activebackground=BG_COLOR, activeforeground=TEXT_COLOR, font=("Segoe UI", 9)).pack(anchor=tk.W, padx=10, pady=2)

        # Export Button
        self.btn_export = tk.Button(main, text="Export Model", font=("Segoe UI", 11, "bold"), bg=ACCENT, fg="white", bd=0, cursor="hand2", command=self._start_export)
        self.btn_export.pack(fill=tk.X, pady=20, ipady=5)

        # Progress Output
        self.log_text = tk.Text(main, height=8, bg="#11111B", fg="#A6E3A1", font=("Consolas", 9), bd=0, padx=10, pady=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.insert(tk.END, "Ready to export.\n")
        self.log_text.config(state=tk.DISABLED)

    def _load_models(self):
        models = []
        
        # Pretrained
        for f in (PROJECT_ROOT / "pretrained_models").glob("*.pt"):
            models.append(f"pretrained_models/{f.name}")
            
        # Runs
        detect_dir = PROJECT_ROOT / "runs" / "detect"
        if detect_dir.exists():
            for d in detect_dir.iterdir():
                if d.is_dir() and (d / "weights" / "best.pt").exists():
                    models.append(f"runs/detect/{d.name}/weights/best.pt")
                    
        self.model_combo['values'] = models
        if models:
            self.model_combo.current(len(models)-1)

    def _log(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.update_idletasks()

    def _start_export(self):
        selection = self.model_combo.get()
        if not selection:
            messagebox.showerror("Error", "Please select a model.")
            return

        model_path = PROJECT_ROOT / selection
        fmt = self.format_var.get()
        fp16 = self.fp16_var.get()
        dynamic = self.dynamic_var.get()

        self.btn_export.config(state=tk.DISABLED, text="Exporting...")
        self._log(f"\n--- Starting Export ---")
        self._log(f"Model: {model_path.name}")
        self._log(f"Format: {fmt.upper()}")
        self._log(f"FP16: {fp16} | Dynamic: {dynamic}")

        threading.Thread(target=self._export_worker, args=(model_path, fmt, fp16, dynamic), daemon=True).start()

    def _export_worker(self, model_path, fmt, fp16, dynamic):
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            
            # Export triggers ultralytics internal routines
            out_file = model.export(
                format=fmt,
                half=fp16,
                dynamic=dynamic
            )
            
            self.after(0, lambda: self._log(f"\n[SUCCESS] Export complete!"))
            self.after(0, lambda: self._log(f"Saved to: {out_file}"))
            self.after(0, lambda: messagebox.showinfo("Success", f"Model exported successfully!\n\nFile: {out_file}"))
        except Exception as e:
            self.after(0, lambda m=str(e): self._log(f"\n[ERROR] {m}"))
            self.after(0, lambda m=str(e): messagebox.showerror("Export Error", f"Failed to export model:\n\n{m}"))
        finally:
            self.after(0, lambda: self.btn_export.config(state=tk.NORMAL, text="Export Model"))

def main():
    app = ExportGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
