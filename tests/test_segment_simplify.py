import numpy as np
import cv2
from labeling.core.segment import _simplify_component_mask


def make_jagged_rectangle(h=200, w=300):
    img = np.zeros((h, w), dtype=np.uint8)
    # Draw a rectangle
    cv2.rectangle(img, (50, 40), (w - 50, h - 40), 255, -1)
    # Add jagged teeth
    for x in range(60, w - 60, 10):
        cv2.rectangle(img, (x, 40), (x + 5, 55), 0, -1)
    return img > 0


def test_simplify_reduces_vertices():
    mask = make_jagged_rectangle()
    # Convert to boolean
    mask_bool = mask.astype(bool)
    # Create a minimal dummy self object with required attributes
    class Dummy:
        SMOOTHING_EPS = {'off': 0.0, 'low': 0.002, 'med': 0.01, 'high': 0.02}
    dummy = Dummy()
    simp = _simplify_component_mask(dummy, mask_bool, level='med')
    # Ensure simplified mask is not empty
    assert simp.sum() > 0
    # Check that simplified contour has fewer points than original
    contours_orig, _ = cv2.findContours((mask_bool.astype('uint8')*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_simp, _ = cv2.findContours((simp.astype('uint8')*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_orig:
        orig_pts = len(contours_orig[0])
        simp_pts = len(contours_simp[0]) if contours_simp else 0
        assert simp_pts <= orig_pts
