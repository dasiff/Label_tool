import cv2
import numpy as np
from pathlib import Path
from app.core.boundary import estimate_boundary_from_overlay


def test_laquinta_returns_polygon():
    img_path = list(Path('data/raw_images').glob('*La_Quinta*.png'))
    assert len(img_path) > 0, "La Quinta test image not found in data/raw_images"
    img = cv2.imread(str(img_path[0]))
    res = estimate_boundary_from_overlay(img)
    assert res.polygon_px is not None, f"Expected polygon, got None. Warnings: {res.warnings}"
    # crude area check: compute filled polygon area using contourArea
    poly = res.polygon_px
    # remove closing duplicate if present
    if len(poly) > 1 and poly[0] == poly[-1]:
        poly = poly[:-1]
    pts = np.array(poly, dtype="float32")
    if pts.ndim == 2 and pts.shape[0] >= 3:
        area = abs(cv2.contourArea(pts.reshape(-1, 1, 2)))
        assert area > 100000, f"Polygon area too small: {area}"
    else:
        # if polygon is weird shape, still assert length > 3
        assert len(poly) >= 4
