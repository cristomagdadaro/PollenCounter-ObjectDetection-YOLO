import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(description="Run YOLO on a live video feed.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO .pt weights file")
    parser.add_argument("--source", type=str, default="0", help="Video source (e.g., '0' for webcam, or path to .mp4)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    
    args = parser.parse_args()
    
    if not Path(args.model).exists():
        print(f"[ERROR] Model not found at: {args.model}")
        return

    print(f"[INFO] Loading model {args.model}")
    model = YOLO(args.model)
    
    print(f"[INFO] Starting live inference on source: {args.source}")
    print("[INFO] Press 'q' in the video window to stop.")
    
    # Run continuous inference
    model.predict(
        source=args.source,
        conf=args.conf,
        show=True,       # Display the video feed
        stream=True      # Essential for live streams/webcams to avoid memory leaks
    )
    
    # The stream generator must be consumed for it to run
    for _ in model.predict(source=args.source, conf=args.conf, show=True, stream=True):
        pass

if __name__ == "__main__":
    main()
