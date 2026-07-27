import cv2
import shutil
import glob
from pathlib import Path

def apply_clahe():
    images_dir = Path("datasets/images")
    backup_dir = Path("datasets/images_original")
    
    if not backup_dir.exists():
        print("[INFO] Creating backup of datasets/images to datasets/images_original...")
        shutil.copytree(images_dir, backup_dir)
    else:
        print("[INFO] Backup datasets/images_original already exists. Proceeding...")

    # Find all image files
    extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(images_dir.rglob(ext))

    if not image_paths:
        print("[WARN] No images found in datasets/images!")
        return

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    
    count = 0
    for path in image_paths:
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            continue
            
        # Convert to LAB color space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to the lightness channel
        l_clahe = clahe.apply(l_channel)
        
        # Merge back and convert to BGR
        lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        # Overwrite the original image
        cv2.imwrite(str(path), img_clahe)
        count += 1
        
    print(f"[INFO] Successfully applied CLAHE to {count} images in {images_dir}")

if __name__ == "__main__":
    apply_clahe()
