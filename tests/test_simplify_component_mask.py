import numpy as np
from scripts.labeling_tool import LabelingTool


def make_jagged_square(h=100,w=100):
    mask = np.zeros((h,w), dtype=bool)
    # create a square with sawtooth edges
    for y in range(20,80):
        for x in range(20,80):
            if (x+y) % 3 == 0:
                mask[y,x] = True
    return mask


def test_simplify_reduces_vertices_and_preserves_area():
    lt = LabelingTool()
    mask = make_jagged_square()
    area_before = mask.sum()
    simp_low = lt._simplify_component_mask(mask, level='low')
    simp_med = lt._simplify_component_mask(mask, level='med')
    simp_high = lt._simplify_component_mask(mask, level='high')
    # All simplified masks should have fewer or equal True pixels than before (since simplification polygon approximates)
    assert simp_low.sum() <= area_before
    assert simp_med.sum() <= area_before
    assert simp_high.sum() <= area_before
    # Ensure area not catastrophically reduced (e.g., not less than 50%)
    assert simp_med.sum() >= area_before * 0.5
    # Ensure polygon simplification actually reduces perimeter complexity for med/high
    # Compare perimeters via contours
    import cv2
    def perim_of_mask(m):
        c,_ = cv2.findContours(m.astype('uint8')*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not c:
            return 0
        return cv2.arcLength(max(c, key=cv2.contourArea), True)
    perim_before = perim_of_mask(mask)
    perim_med = perim_of_mask(simp_med)
    assert perim_med <= perim_before
