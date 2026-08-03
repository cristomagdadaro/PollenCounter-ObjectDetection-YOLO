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


class BoxSnappingMixin:
    def _snap_single_box(self, box, img_bgr=None, method_name=None):
        """Snap a bounding box tightly to pollen edges using multi-stage adaptive segmentation.

        Pipeline: CLAHE → Adaptive Threshold → Morphological Cleanup →
        Distance-Transform Watershed (for touching grains) → Center-nearest
        contour selection → boundingRect.

        Falls back through simpler methods if the advanced pipeline fails.
        """
        if not hasattr(self, 'orig_pil_img') or not self.orig_pil_img: return False
        if img_bgr is None:
            try:
                img_bgr = cv2.cvtColor(np.array(self.orig_pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                return False

        img_h, img_w = img_bgr.shape[:2]
        x1 = int((box.x_center - box.w / 2) * img_w)
        y1 = int((box.y_center - box.h / 2) * img_h)
        x2 = int((box.x_center + box.w / 2) * img_w)
        y2 = int((box.y_center + box.h / 2) * img_h)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            return False

        # Expand the ROI by 15% on each side to capture edges that the user
        # may have clipped when drawing the box quickly.
        roi_w, roi_h = x2 - x1, y2 - y1
        pad_x = int(roi_w * 0.15)
        pad_y = int(roi_h * 0.15)
        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)
        rx2 = min(img_w, x2 + pad_x)
        ry2 = min(img_h, y2 + pad_y)
        roi = img_bgr[ry1:ry2, rx1:rx2]

        if roi.size == 0:
            return False

        roi_center_x = (x1 + x2) / 2.0 - rx1  # center of user box within the ROI
        roi_center_y = (y1 + y2) / 2.0 - ry1

        def _apply_box(contour):
            """Given a contour (in ROI coords), update the box and return True."""
            cx, cy, cw, ch = cv2.boundingRect(contour)
            new_x1 = max(0, rx1 + cx)
            new_y1 = max(0, ry1 + cy)
            new_x2 = min(img_w, rx1 + cx + cw)
            new_y2 = min(img_h, ry1 + cy + ch)
            if new_x2 <= new_x1 or new_y2 <= new_y1:
                return False
            box.w = (new_x2 - new_x1) / img_w
            box.h = (new_y2 - new_y1) / img_h
            box.x_center = (new_x1 + new_x2) / 2.0 / img_w
            box.y_center = (new_y1 + new_y2) / 2.0 / img_h
            return True

        def _pick_nearest(contours, min_area=50):
            """From a list of contours, pick the one whose centroid is closest
            to the center of the user's originally-drawn box.
            Rejects contours that are too large (background noise) or too small."""
            roi_area = roi.shape[0] * roi.shape[1]
            max_area = roi_area * 0.50  # contours >50% of ROI are background noise
            best, best_dist = None, float('inf')
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                ccx = M["m10"] / M["m00"]
                ccy = M["m01"] / M["m00"]
                dist = (ccx - roi_center_x) ** 2 + (ccy - roi_center_y) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best = cnt
            return best

        # Scale morphological kernel sizes proportionally to ROI dimensions.
        # Small ROIs (tight draws) get small kernels; large ROIs (sloppy draws)
        # get larger kernels so noise cleanup is proportional.
        roi_short = min(roi.shape[:2])
        close_k = max(5, min(15, roi_short // 20)) | 1   # odd, 5-15
        open_k  = max(3, min(11, roi_short // 30)) | 1   # odd, 3-11

        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            if method_name is None:
                method = getattr(self, 'snap_method', None)
                method_name = method.get() if method else "Adaptive Multi-Stage"

            if method_name == "Simple Otsu (Legacy)":
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                _, thresh = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    best = _pick_nearest(contours) or max(contours, key=cv2.contourArea)
                    return _apply_box(best)
                return False

            # ── Stage 1: CLAHE + Adaptive Threshold + Watershed ──────────
            try:
                # 1a. CLAHE: normalize uneven illumination locally
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)

                # 1b. Adaptive threshold: handles varying stain intensity
                #     blockSize capped at 51 to stay sensitive to small grains
                #     in large ROIs
                block = max(11, min(51, (min(enhanced.shape) // 8) | 1))
                thresh = cv2.adaptiveThreshold(
                    enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, block, 4
                )

                # 1c. Morphological close: fill small holes inside grains
                kernel_close = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (close_k, close_k))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

                # 1d. Morphological open: remove small noise specks
                kernel_open = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (open_k, open_k))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2)

                # 1e. Distance transform + Watershed to separate touching grains
                dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
                dist_max = dist_transform.max()
                if dist_max > 0:
                    _, sure_fg = cv2.threshold(
                        dist_transform, 0.45 * dist_max, 255, 0)
                    sure_fg = np.uint8(sure_fg)
                    # Dilate background away from edges
                    kernel_bg = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE, (3, 3))
                    sure_bg = cv2.dilate(thresh, kernel_bg, iterations=2)
                    unknown = cv2.subtract(sure_bg, sure_fg)

                    _, markers = cv2.connectedComponents(sure_fg)
                    markers = markers + 1  # watershed treats 0 as unknown
                    markers[unknown == 255] = 0

                    roi_3ch = roi.copy()
                    cv2.watershed(roi_3ch, markers)

                    # Extract contours from each watershed label
                    ws_contours = []
                    for label_id in np.unique(markers):
                        if label_id <= 1:  # skip background and boundary
                            continue
                        mask = np.zeros(gray.shape, dtype="uint8")
                        mask[markers == label_id] = 255
                        cnts, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        ws_contours.extend(cnts)

                    best = _pick_nearest(ws_contours)
                    if best is not None:
                        return _apply_box(best)

                # Watershed didn't produce good results; use the
                # CLAHE+adaptive threshold contours directly
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best = _pick_nearest(contours)
                if best is not None:
                    return _apply_box(best)

            except Exception:
                pass  # fall through to next strategy

            # ── Stage 2 (Fallback): CLAHE + Otsu ─────────────────────────
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
                blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
                _, thresh = cv2.threshold(
                    blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best = _pick_nearest(contours)
                if best is not None:
                    return _apply_box(best)
            except Exception:
                pass  # fall through to final fallback

            # ── Stage 3 (Final Fallback): Simple Otsu (original logic) ───
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                best = _pick_nearest(contours) or max(contours, key=cv2.contourArea)
                return _apply_box(best)

        except Exception:
            pass
        return False

    def _snap_boxes(self):
        """Recompute bounding boxes to snap perfectly to pollen edges using OpenCV."""
        if not hasattr(self, 'orig_pil_img') or not self.boxes: return
        import concurrent.futures

        # Save current box positions so user can undo
        self._snap_undo_states = [
            (b.x_center, b.y_center, b.w, b.h) for b in self.boxes
        ]

        try:
            img_bgr = cv2.cvtColor(np.array(self.orig_pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return

        # Fetch method_name safely on the main thread before starting workers
        method = getattr(self, 'snap_method', None)
        method_name = method.get() if method else "Adaptive Multi-Stage"

        def snap(box):
            return self._snap_single_box(box, img_bgr=img_bgr, method_name=method_name)

        updated = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(snap, self.boxes))
            updated = sum(1 for r in results if r)

        self._redraw_boxes()
        self._save_labels()
        self.status.config(text=f" Snapped {updated} boxes to edges")

        # Enable undo button
        if hasattr(self, 'undo_snap_btn'):
            self.undo_snap_btn.config(state=tk.NORMAL, bg="#DC2626")

    def _undo_snap(self):
        """Restore all boxes to their positions from before the last snap."""
        if not self._snap_undo_states or not self.boxes:
            return
        # Restore each box's saved coordinates
        for box, (xc, yc, w, h) in zip(self.boxes, self._snap_undo_states):
            box.x_center = xc
            box.y_center = yc
            box.w = w
            box.h = h
        self._snap_undo_states = None
        self._redraw_boxes()
        self._save_labels()
        self.status.config(text=" ↩ Snap undone — boxes restored")
        # Disable undo button
        if hasattr(self, 'undo_snap_btn'):
            self.undo_snap_btn.config(state=tk.DISABLED, bg="#6B7280")

