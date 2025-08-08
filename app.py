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

# Attempt to import combo fusion helpers; fallback stub if missing
try:
    from segmentation_fusion import ComboFusion, ComboFusionConfig
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class ComboFusionConfig:
        edge_bias: float = 0.5
        frame_window: int = 3
        sharpness_weight: float = 0.5

    class ComboFusion:
        def __init__(self, cfg: ComboFusionConfig):
            self.cfg = cfg
        def update_config(self, cfg: ComboFusionConfig):
            self.cfg = cfg
        def process(self, frame_bgr):
            h, w = frame_bgr.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            prob = np.zeros((h, w), dtype=np.float32)
            return mask, prob

# Utilities & Setup

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
        x = np.linspace(0, 1, w, dtype=np.float32)
        grad = np.tile(x, (h, 1))
        img = np.stack([
            (1.0 - grad) * 255.0,
            grad * 255.0,
            np.full_like(grad, 60)
        ], axis=2).astype(np.uint8)
        cv2.imwrite(gradient_path, img)
    if not os.path.isfile(checker_path):
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

# Segmentation

class PersonSegmenter:
    def __init__(self, model_selection: int = 1):
        self.mp_selfie = mp.solutions.selfie_segmentation
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

# Canny Pipeline

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

# UI: Trackbars

class Controls:
    def __init__(self, window_name="Controls"):
        self.window = window_name
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 360, 220)
        # Trackbars for Canny and mask threshold
        cv2.createTrackbar("Canny Low", self.window, 50, 255, lambda v: None)
        cv2.createTrackbar("Canny High", self.window, 150, 255, lambda v: None)
        cv2.createTrackbar("Blur (0..10)", self.window, 1, 10, lambda v: None)
        cv2.createTrackbar("Mask Thresh", self.window, 128, 255, lambda v: None)
        # Combo-specific trackbars
        cv2.createTrackbar("Combo Edge Bias", self.window, 50, 100, lambda v: None)
        cv2.createTrackbar("Combo Frame Win", self.window, 3, 5, lambda v: None)
        cv2.createTrackbar("Combo Sharpness", self.window, 50, 100, lambda v: None)

    def read(self):
        lo = cv2.getTrackbarPos("Canny Low", self.window)
        hi = cv2.getTrackbarPos("Canny High", self.window)
        blur_slider = cv2.getTrackbarPos("Blur (0..10)", self.window)
        blur_k = odd_from_slider(blur_slider, 1)
        mask_thresh = cv2.getTrackbarPos("Mask Thresh", self.window) / 255.0
        return lo, hi, blur_k, mask_thresh

    def read_combo(self):
        eb = cv2.getTrackbarPos("Combo Edge Bias", self.window) / 100.0
        fw = cv2.getTrackbarPos("Combo Frame Win", self.window)
        fw = max(1, min(5, fw))
        sw = cv2.getTrackbarPos("Combo Sharpness", self.window) / 100.0
        return eb, fw, sw

# Main Loop

def run(camera_index: int, target_width: int, bg_dir: str, start_bg: str = None,
        mode: str = "combo", edge_bias: float = 0.5, frame_window: int = 3,
        sharpness_weight: float = 0.5):
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

    # Camera capture
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)

    # Pre-read a frame
    ok, frame = cap.read()
    if not ok:
        print("ERROR: Failed to read from webcam.")
        cap.release()
        sys.exit(1)

    h0, w0 = frame.shape[:2]
    if target_width <= 0:
        target_width = w0
    scale_init = target_width / float(w0)
    target_height = int(h0 * scale_init)

    # Initialize helpers
    segmenter = PersonSegmenter(model_selection=1)
    controls = Controls()
    main_win = "CV1 - Real-Time Edge + AI Background"
    cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)

    # View modes including combo
    view_modes = ["combo", "composite", "split", "original", "edges"]
    if mode not in view_modes:
        mode = "combo"
    view_idx = view_modes.index(mode)

    print("Keybinds: [v] view mode  |  [b] background  |  [s] screenshot  |  [q] quit")

    # Initialize combo fuser
    combo_fuser = ComboFusion(ComboFusionConfig(edge_bias=edge_bias,
                                                frame_window=max(1, min(5, frame_window)),
                                                sharpness_weight=sharpness_weight))

    try:
        last_fps_time = time.time()
        frame_count = 0
        fps = 0.0
        mask_accum = None
        smooth_alpha = 0.5
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

            # Prep background
            if backgrounds:
                bg_img = cv2.imread(backgrounds[bg_idx], cv2.IMREAD_COLOR)
                if bg_img is None:
                    bg_canvas = np.zeros_like(frame_resized)
                else:
                    bg_canvas = resize_and_letterbox(bg_img, target_width, target_height)
            else:
                bg_canvas = np.zeros_like(frame_resized)

            # Determine view
            view = view_modes[view_idx]

            if view == "combo":
                # Update combo parameters from trackbars
                eb, fw, sw = controls.read_combo()
                combo_fuser.update_config(ComboFusionConfig(edge_bias=eb, frame_window=fw, sharpness_weight=sw))
                # Run combo fusion on current frame
                m_final_255, _ = combo_fuser.process(frame_resized)
                person_mask_combo = m_final_255
                inv_mask_combo = cv2.bitwise_not(person_mask_combo)
                mask3_combo = cv2.cvtColor(person_mask_combo, cv2.COLOR_GRAY2BGR)
                inv_mask3_combo = cv2.cvtColor(inv_mask_combo, cv2.COLOR_GRAY2BGR)
                foreground_combo = cv2.bitwise_and(frame_resized, mask3_combo)
                background_combo = cv2.bitwise_and(bg_canvas, inv_mask3_combo)
                out = cv2.add(foreground_combo, background_combo)

                # FPS update
                frame_count += 1
                now = time.time()
                if now - last_fps_time >= 0.5:
                    fps = frame_count / (now - last_fps_time)
                    frame_count = 0
                    last_fps_time = now
                # HUD
                hud = f"FPS {fps:0.1f} | Combo[edge_bias={eb:0.2f}, fw={fw}, sw={sw:0.2f}] | BG {bg_idx+1}/{max(1,len(backgrounds))} | View=combo"
                cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

            else:
                # Controls for Canny and mask threshold
                lo, hi, blur_k, mask_thresh = controls.read()

                # Phase 2: Canny edges
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                edges = canny_edges(gray, blur_k=blur_k, lo=lo, hi=hi)

                # Phase 3: AI segmentation (binary mask) with temporal smoothing and morphology
                raw_mask = segmenter.mask(frame_resized, thresh=mask_thresh).astype(np.uint8)
                if mask_accum is None:
                    mask_accum = raw_mask.astype("float32")
                cv2.accumulateWeighted(raw_mask, mask_accum, smooth_alpha)
                smoothed_mask = cv2.convertScaleAbs(mask_accum)
                _, person_mask = cv2.threshold(smoothed_mask, int(mask_thresh * 255), 255, cv2.THRESH_BINARY)
                person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_CLOSE, kernel)
                person_mask = cv2.morphologyEx(person_mask, cv2.MORPH_OPEN, kernel)
                inv_mask = cv2.bitwise_not(person_mask)

                # Phase 4: Apply edges only on person; keep background clean
                frame_with_edges = overlay_edges_on_foreground(frame_resized, edges, person_mask)
                mask3 = cv2.cvtColor(person_mask, cv2.COLOR_GRAY2BGR)
                inv_mask3 = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
                foreground = cv2.bitwise_and(frame_with_edges, mask3)
                background_clean = cv2.bitwise_and(bg_canvas, inv_mask3)
                composite = cv2.add(foreground, background_clean)

                # Split-screen original vs edges-only
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                split = np.hstack([frame_resized, edges_bgr])

                # FPS update
                frame_count += 1
                now = time.time()
                if now - last_fps_time >= 0.5:
                    fps = frame_count / (now - last_fps_time)
                    frame_count = 0
                    last_fps_time = now

                hud = f"FPS {fps:0.1f} | Canny[{lo},{hi}] Blur={blur_k} Mask={mask_thresh:0.2f} | BG {bg_idx+1}/{max(1,len(backgrounds))} | View={view}"
                if view == "composite":
                    out = composite
                elif view == "split":
                    out = split
                elif view == "original":
                    out = frame_resized.copy()
                else:  # "edges"
                    out = edges_bgr.copy()
                # Put HUD
                cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

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

# Entrypoint / CLI

def parse_args():
    p = argparse.ArgumentParser(description="CV1 – Real-Time Edge + AI Background (single script)")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--width", type=int, default=1280, help="Target display width (default: 1280)")
    p.add_argument("--bg", type=str, default=None, help="Path to a specific background image to start with")
    p.add_argument("--backgrounds", type=str, default="backgrounds", help="Directory of background images")
    # New mode and combo options
    p.add_argument("--mode", type=str, default="combo", choices=["combo", "composite", "split", "original", "edges"],
                   help="Initial view mode (default: combo)")
    p.add_argument("--edge-bias", type=float, default=0.5, help="Edge vs AI blending weight for combo mode (0.0-1.0)")
    p.add_argument("--frame-window", type=int, default=3, help="Rolling buffer size for combo mode (1-5)")
    p.add_argument("--sharpness-weight", type=float, default=0.5, help="Quality weight for sharpness vs seg confidence (0-1)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        camera_index=args.camera,
        target_width=args.width,
        bg_dir=args.backgrounds,
        start_bg=args.bg,
        mode=args.mode,
        edge_bias=args.edge_bias,
        frame_window=args.frame_window,
        sharpness_weight=args.sharpness_weight
    )
