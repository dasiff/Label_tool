import numpy as np
from scripts.labeling_tool import LabelingTool


def make_jagged_square(h=100, w=100):
    mask = np.zeros((h, w), dtype=bool)
    # create a square with sawtooth edges
    for y in range(20, 80):
        for x in range(20, 80):
            if (x + y) % 3 == 0:
                mask[y, x] = True
    return mask


def perim_of_mask(m):
    import cv2
    c, _ = cv2.findContours(m.astype('uint8') * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not c:
        return 0, 0
    cnt = max(c, key=cv2.contourArea)
    return cv2.arcLength(cnt, True), len(cnt)


def test_smoothing_reduces_spikes():
    lt = LabelingTool()
    mask = make_jagged_square()
    perim_before, n_pts_before = perim_of_mask(mask)
    simp_med = lt._simplify_component_mask(mask, level='med')
    perim_med, n_pts_med = perim_of_mask(simp_med)
    # Perimeter should decrease
    assert perim_med <= perim_before
    # Number of contour points should shrink significantly for med
    assert n_pts_med < n_pts_before
    # Area should still be preserved by existing area checks
    assert simp_med.sum() >= mask.sum() * 0.5