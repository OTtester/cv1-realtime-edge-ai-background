import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class ComboFusionConfig:
    """Configuration for combo fusion."""
    edge_bias: float = 0.5
    frame_window: int = 3
    sharpness_weight: float = 0.5


class ComboFusion:
    """
    Fuse AI segmentation probabilities with edge-based refinement.

    Maintains a rolling buffer of recent frames and their AI probability maps,
    selects the best frame based on sharpness and segmentation confidence,
    computes a soft edge prior, and fuses it with the AI probabilities.
    """

    def __init__(self, cfg: ComboFusionConfig):
        self.cfg = cfg
        # Buffer of (frame_bgr, ai_prob) tuples
        self.buffer: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=cfg.frame_window)

    def update_config(self, cfg: ComboFusionConfig) -> None:
        """Update configuration and resize buffer if window size changed."""
        # Resize buffer if frame_window changed
        if cfg.frame_window != self.cfg.frame_window:
            # Create new buffer with updated size and keep existing entries
            new_buffer = deque(list(self.buffer), maxlen=cfg.frame_window)
            self.buffer = new_buffer
        self.cfg = cfg

    def _compute_sharpness(self, frame: np.ndarray) -> float:
        """Compute sharpness metric using variance of Laplacian on grayscale."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _compute_confidence(self, prob: np.ndarray) -> float:
        """Compute segmentation confidence as mean of probability map."""
        return float(prob.mean())

    def process(self, frame_bgr: np.ndarray, ai_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Process the given frame and AI probability map to produce a refined mask.

        Args:
            frame_bgr: Current frame in BGR format.
            ai_prob: AI-derived person probabilities with shape (H, W) in [0,1].

        Returns:
            mask: Binary mask as uint8 (0 or 255).
            fused: Fused probability map in [0,1].
        """
        # Append current frame and AI probabilities to the buffer
        self.buffer.append((frame_bgr.copy(), ai_prob.copy()))

        # Collect sharpness and confidence metrics for normalization
        sharpness_list = []
        confidence_list = []
        for buf_frame, buf_prob in self.buffer:
            sharpness_list.append(self._compute_sharpness(buf_frame))
            confidence_list.append(self._compute_confidence(buf_prob))

        # Normalize metric lists to [0,1]
        def _normalize(lst):
            min_v = min(lst)
            max_v = max(lst)
            if max_v - min_v < 1e-9:
                return [0.0 for _ in lst]
            return [(v - min_v) / (max_v - min_v) for v in lst]

        sharp_norm = _normalize(sharpness_list)
        conf_norm = _normalize(confidence_list)

        # Select the frame with maximum weighted quality
        best_idx = 0
        best_quality = -1.0
        for idx, (s_norm, c_norm) in enumerate(zip(sharp_norm, conf_norm)):
            quality = self.cfg.sharpness_weight * s_norm + (1.0 - self.cfg.sharpness_weight) * c_norm
            if quality > best_quality:
                best_quality = quality
                best_idx = idx

        # Retrieve selected frame and probability map
        sel_frame, sel_prob = self.buffer[best_idx]

        # AI probabilities and binary mask
        P_ai = sel_prob.astype(np.float32)
        M_ai = (P_ai >= 0.5).astype(np.uint8)

        # Compute edges using Canny on grayscale
        gray_sel = cv2.cvtColor(sel_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_sel, 100, 200).astype(np.float32) / 255.0

        # Create a band around the AI mask using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(M_ai, kernel, iterations=3)
        eroded = cv2.erode(M_ai, kernel, iterations=3)
        band = cv2.subtract(dilated, eroded)

        # Restrict edges to the band and smooth
        E_band = edges * band
        E_band = cv2.GaussianBlur(E_band, (5, 5), 0)

        # Compute distance transform to edges: invert edge map for distance transform
        inv_edges = np.uint8(E_band < 0.1)
        dt = cv2.distanceTransform(inv_edges, cv2.DIST_L2, 5)

        # Convert distance transform to a soft edge prior
        sigma = 3.0
        P_edge = np.exp(-dt / sigma)
        P_edge *= M_ai  # apply mask
        # Normalize P_edge to [0,1]
        max_edge = P_edge.max()
        if max_edge > 0:
            P_edge /= max_edge

        # Fuse AI probabilities with edge prior
        fused = (1.0 - self.cfg.edge_bias) * P_ai + self.cfg.edge_bias * P_edge
        fused = np.clip(fused, 0.0, 1.0)

        # Threshold fused probabilities to get final mask
        M_final = (fused >= 0.5).astype(np.uint8)
        # Morphological closing to remove small holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        M_final = cv2.morphologyEx(M_final, cv2.MORPH_CLOSE, kernel_close)

        return (M_final * 255).astype(np.uint8), fused
