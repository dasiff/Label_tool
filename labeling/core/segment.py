"""Segmentation pipeline scaffold.

Responsibility:
- Provide functions to build features, call segmentation routines (e.g., felzenszwalb),
  and post-process into a consistent segments ndarray.

Inputs:
- clean_image (numpy array), boundary polygon, target_segments, options
Outputs:
- segments ndarray, n_segments, debug metadata
"""
from typing import Any, Dict
import numpy as np
import cv2


def _build_segment_features(self, rgb_small: np.ndarray) -> np.ndarray:
    """Build the feature image used for segmentation from an RGB image (small/rescaled).
    Returns an uint8 image suitable for felzenszwalb (H,W,3).
    Supports shadow-robust mode when `self.shadow_robust_var` is True.
    """
    # Ensure input is RGB uint8
    if rgb_small.dtype != np.uint8:
        rgb_small = np.clip(rgb_small, 0, 255).astype(np.uint8)

    # Convert to BGR for OpenCV LAB conversion
    bgr_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
    l, a, b_channel = cv2.split(lab)

    # Apply CLAHE only moderately to reduce shadow impact
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Normalize color channels and center
    l_normalized = l_enhanced.astype(float) / 255.0
    a_normalized = (a.astype(float) - 128) / 128.0
    b_normalized = (b_channel.astype(float) - 128) / 128.0

    # Default (legacy) feature composition: mix l,a,b with reduced lightness weight
    feature_img_default = np.stack([
        l_normalized * 0.3,
        a_normalized * 1.2,
        b_normalized * 1.2
    ], axis=-1)

    # Shadow-robust composition (chromaticity + gradient + lab colors)
    if getattr(self, 'shadow_robust_var', None) and self.shadow_robust_var.get():
        arr = rgb_small.astype(np.float32)
        denom = arr.sum(axis=2, keepdims=True) + 1e-6
        chroma_r = (arr[:, :, 0:1] / denom).squeeze()
        chroma_g = (arr[:, :, 1:2] / denom).squeeze()

        gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        grad_norm = grad / max(grad.max(), 1e-6)

        # Use a_normalized, b_normalized and gradient magnitude as three channels
        feature_img = np.stack([
            a_normalized,
            b_normalized,
            grad_norm
        ], axis=-1)
    else:
        feature_img = feature_img_default

    # Normalize to 0..1 and convert to uint8
    # Guard against constant images
    minv = feature_img.min()
    maxv = feature_img.max()
    if maxv - minv < 1e-6:
        feature_img = np.zeros_like(feature_img)
    else:
        feature_img = (feature_img - minv) / (maxv - minv)
    rgb_enhanced = (feature_img * 255).astype(np.uint8)
    return rgb_enhanced


def generate_segments(clean_image: Any, boundary: Any, *, target_segments: int = 50, **options) -> Dict[str, Any]:
    """Generate segments for the ROI defined by boundary.

    Returns a dict containing at least: {'segments': ndarray, 'n_segments': int}
    """
    raise NotImplementedError
