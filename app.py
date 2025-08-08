#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV1 – Real-Time Edge Detection with AI Background Replacement
Single-script MVP implementing Phases 1–5 with small extras for Phase 6.

Features:
- Live webcam capture (Windows friendly).
- Split-screen original vs. Canny (Phase 2).
- MediaPipe Selfie Segmentation to isolate person (Phase 3).
- Background replacement from /backgrounds (Phase 3).
- Combined pipeline: Canny edges applied ONLY to the person; clean background (Phase 4).
- Real-time controls via OpenCV trackbars; background toggle & view modes via keys (Phase 5).
- Saves screenshots with 's' to /outputs (tiny Phase 6 QoL).
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

# Try importing mediapipe with a useful error if missing
try:
    import mediapipe as mp
except Exception as e:
    print("ERROR: mediapipe is not installed. Run: pip install mediapipe")
    raise

# -----------------------------
# Utilities & Setup
# -----------------------------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def make_sample_backgrounds(bg_dir: str, w: int = 1280, h: int = 720):
    """
    Create a simple gradient and a checkerboard if backgrounds/ is empty.
    """
    ensure_dir(bg_dir)

    gradient_path = os.path.join(bg_dir, "sample_gradient.png")
    checker_path = os.path.join(bg_dir, "sample_checker.png")

    if not os.path.isfile(gradient_path):
        # Horizontal gradient BGR
        x = np.linspace(0, 1, w, dtype=np.float32)
        grad = np.tile(x, (h, 1))
        img = np.stack([
            (1.0 - grad) * 255.0,  # Blue channel fades
            grad * 255.0,          # Green channel rises
            np.full_like(grad, 60) # Slight red tint
        ], axis=2).astype(np.uint8)
        cv2.imwrite(gradient_path, img)

    if not os.path.isfile(checker_path):
        # Checkerboard
        tile = 40
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(0, h, tile):
            for x_ in range(0, w, tile):
                if ((x_ // tile) + (y // tile)) % 2 == 0:
                    img[y:y+tile, x_:x_+tile] = (40, 40, 40)
                else:
                    img[y:y+tile, x_:x_+tile] = (180, 180, 180)
        cv2.imwrite(checker_path, img)

def list_backgrounds(bg_dir: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(bg_dir, ext)))
    files.sort()
    return files

def resize_and_letterbox(img, target_w, target_h):
    """
    Resize while preserving aspect; letterbox to exact target size.
    """
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - nh) // 2
    left = (target_w - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def odd_from_slider(v: int, min_odd: int = 1):
    """
    Map trackbar integer to an odd number >= min_odd.
    Example: v in [0..10] -> odd kernel in [1..21]
    """
    k = max(min_odd, 2 * v + 1)
    if k % 2 == 0:
        k += 1
    return k

# -----------------------------
# Segmentation
# -----------------------------

class PersonSegmenter:
    def __init__(self, model_selection: int = 1):
        self.mp_selfie = mp.solutions.selfie_segmentation
        # model_selection=0: landscape / close; =1: general
        self.segmenter = self.mp_selfie.SelfieSegmentation(model_selection=model_selection)

    def mask(self, bgr_frame: np.ndarray, thresh: float = 0.5) -> np.ndarray:
        """
        Returns a binary mask (uint8 0/255) where 255 = person/foreground.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)
        seg = result.segmentation_mask  # float32 [0..1]
        if seg is None:
            return np.zeros(bgr_frame.shape[:2], dtype=np.uint8)
        m = (seg > float(thresh)).astype(np.uint8) * 255
        return m

    def close(self):
        self.segmenter.close()

# -----------------------------
# Canny Pipeline
# -----------------------------

def canny_edges(gray, blur_k: int, lo: int, hi: int):
    if blur_k > 1:
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    edges = cv2.Canny(gray, lo, hi)
    return edges

def overlay_edges_on_foreground(frame_bgr, edges_gray, person_mask):
    """
    Paint edges (white) only on the person region.
    """
    edges_on_person = cv2.bitwise_and(edges_gray, edges_gray, mask=person_mask)
    out = frame_bgr.copy()
    out[edges_on_person > 0] = (255, 255, 255)
    return out

# -----------------------------
# UI: Trackbars
# -----------------------------

class Controls:
    def __init__(self, window_name="Controls"):
        self.window = window_name
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 360, 220)
        # Trackbars
        # Set more sensitive defaults for edge detection and blur
        cv2.createTrackbar("Canny Low", self.window, 50, 255, lambda v: None)
        cv2.createTrackbar("Canny High", self.window, 150, 255, lambda v: None)
        # Start with a smaller blur kernel (slider value maps to odd kernel size)
        cv2.createTrackbar("Blur (0..10)", self.window, 1, 10, lambda v: None)
        cv2.createTrackbar("Mask Thresh", self.window, 128, 255, lambda v: None)

    def read(self):
        lo = cv2.getTrackbarPos("Canny Low", self.window)
        hi = cv2.getTrackbarPos("Canny High", self.window)
        blur_slider = cv2.getTrackbarPos("Blur (0..10)", self.window)
        blur_k = odd_from_slider(blur_slider, 1)
        mask_thresh = cv2.getTrackbarPos("Mask Thresh", self.window) / 255.0
        return lo, hi, blur_k, mask_thresh

# -----------------------------
# Main Loop
# -----------------------------

def run(camera_index: int, target_width: int, bg_dir: str, start_bg: str = None):
    # Ensure folders
    ensure_dir(bg_dir)
    ensure_dir("outputs")

    # Seed backgrounds if missing
    if len(list_backgrounds(bg_dir)) == 0:
        make_sample_backgrounds(bg_dir)

    backgrounds = list_backgrounds(bg_dir)
    if not backgrounds:
        print(f"WARNING: No backgrounds found in {bg_dir}. Using generated black canvas.")
        backgrounds = []

    # Background index
    bg_idx = 0
    if start_bg:
        start_bg_abs = os.path.abspath(start_bg)
        if os.path.isfile(start_bg_abs):
            backgrounds = [start_bg_abs] + [b for b in backgrounds if os.path.abspath(b) != start_bg_abs]
            bg_idx = 0
        else:
            print(f"WARNING: --bg '{start_bg}' not found. Ignoring.")

    # Camera capture (CAP_DSHOW helps on Windows)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        # Fallback: try default
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    # Pre-read a frame to establish size
    ok, frame = cap.read()
    if not ok:
        print("ERROR: Failed to read from webcam.")
        cap.release()
        sys.exit(1)

    # Establish target size (keep aspect, enforce width)
    h0, w0 = frame.shape[:2]
    if target_width <= 0:
        target_width = w0
    scale = target_width / float(w0)
    target_height = int(h0 * scale)

    # Init helpers
    segmenter = PersonSegmenter(model_selection=1)
    controls = Controls()
    main_win = "CV1 - Real-Time Edge + AI Background"
    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)

    # View modes: 'composite' (default), 'split', 'original', 'edges'
    view_modes = ["composite", "split", "original", "edges"]
    view_idx = 0

    print("Keybinds: [v] view mode  |  [b] background  |  [s] screenshot  |  [q] quit")

    try:
        last_fps_time = time.time()
        frame_count = 0
        fps = 0.0
        # Initialize variables for temporal smoothing of the segmentation mask and morphology operations
        mask_accum = None
        # A smoothing factor between 0 and 1; lower values produce smoother, more stable masks
        smooth_alpha = 0.5
        # Kernel used for morphological closing/opening to clean up the segmentation mask
        kernel = np.ones((5, 5), np.uint8)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Resize to target width
            h, w = frame.shape[:2]
            scale = target_width / float(w)
            target_height = int(h * scale)
            frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

            # Controls
            lo, hi, blur_k, mask_thresh = controls.read()

            # Prep background for this frame
            if backgrounds:
                bg_img = cv2.imread(backgrounds[bg_idx], cv2.IMREAD_COLOR)
                if bg_img is None:
                    # If unreadable, replace with black
                    bg_canvas = np.zeros_like(frame_resized)
                else:
                    bg_canvas = resize_and_letterbox(bg_img, target_width, target_height)
            else:
                bg_canvas = np.zeros_like(frame_resized)

            # Phase 2: Canny edges
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            edges = canny_edges(gray, blur_k=blur_k, lo=lo, hi=hi)

            # Phase 3: AI segmentation (binary mask) with temporal smoothing and morphology
            raw_mask = segmenter.mask(frame_resized, thresh=mask_thresh).astype(np.uint8)
            # Initialize the accumulator on the first iteration
            if mask_accum is None:
                mask_accum = raw_mask.astype("float32")
            # Smooth the mask over time using exponential moving average
            cv2.accumulateWeighted(raw_mask, mask_accum, smooth_alpha)
            smoothed_mask = cv2.convertScaleAbs(mask_accum)
            # Apply the threshold from the trackbar to produce a binary person mask
            _, person_mask = cv2.threshold(smoothed_mask, int(mask_thresh * 255), 255, cv2.THRESH_BINARY)
            # Morphological cleaning to remove specks and fill holes
            person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
            person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
            inv_mask = cv2.bitwise_not(person_mask)

            # Phase 4: Apply edges only on person; keep background clean
            # Paint edges on the person region in white
            frame_with_edges = overlay_edges_on_foreground(frame_resized, edges, person_mask)
            # Foreground = person region from frame_with_edges
            mask3 = cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR)
            inv_mask3 = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
            foreground = cv2.bitwise_and(frame_with_edges, mask3)
            background_clean = cv2.bitwise_and(bg_canvas, inv_mask3)
            composite = cv2.add(foreground, background_clean)

            # Phase 2 (split-screen original vs edges-only)
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            split = np.hstack([frame_resized, edges_bgr])

            # FPS counter (lightweight)
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 0.5:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            # HUD overlay
            def put_hud(img, text, y=28):
                cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            hud = f"FPS {fps:0.1f} | Canny[{lo},{hi}] Blur={blur_k} Mask={mask_thresh:0.2f} | BG {bg_idx+1}/{max(1,len(backgrounds))} | View={view_modes[view_idx]}"
            view = view_modes[view_idx]
            if view == "composite":
                out = composite
                put_hud(out, hud)
            elif view == "split":
                out = split
                put_hud(out, hud)
            elif view == "original":
                out = frame_resized.copy()
                put_hud(out, hud)
            else:  # "edges"
                out = edges_bgr.copy()
                put_hud(out, hud)

            cv2.imshow(main_win, out)

            # Key events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                view_idx = (view_idx + 1) % len(view_modes)
            elif key == ord('b'):
                if backgrounds:
                    bg_idx = (bg_idx + 1) % len(backgrounds)
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = os.path.join("outputs", f"cv1_frame_{ts}.png")
                cv2.imwrite(fname, out)
                print(f"Saved screenshot: {fname}")

    finally:
        segmenter.close()
        cap.release()
        cv2.destroyAllWindows()

# -----------------------------
# Entrypoint / CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CV1 – Real-Time Edge + AI Background (single script)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--width", type=int, default=1280, help="Target display width (default: 1280)")
    p.add_argument("--bg", type=str, default=None, help="Path to a specific background image to start with")
    p.add_argument("--backgrounds", type=str, default="backgrounds", help="Directory of background images")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        camera_index=args.camera,
        target_width=args.width,
        bg_dir=args.backgrounds,
        start_bg=args.bg
    )