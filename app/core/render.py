from __future__ import annotations
from typing import List, Tuple, Optional
import cv2
import numpy as np

Point = Tuple[float, float]


def draw_polygon(bgr: np.ndarray, polygon: List[Point], color=(0, 255, 255), thickness=2) -> np.ndarray:
    out = bgr.copy()
    if not polygon:
        return out
    pts = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out
