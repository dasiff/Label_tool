import numpy as np
from labeling.core.split_core import build_combined_line_mask, constrained_subtract_and_label, closed_polygon_handler


def test_vertical_line_splits_rectangle():
    h, w = 200, 300
    segs = np.zeros((h, w), dtype=np.int32)
    segs[0:h, 0:w] = 1
    seg_mask = (segs == 1).astype(np.uint8)

    # Emulate snapping: vertical spanning line snapped to top/bottom edges
    poly = [(150, 0), (150, 199)]
    lm = build_combined_line_mask([poly], seg_mask, (h, w), thickness=1, endpoint_radius=2)
    labeled, num, region_sizes = constrained_subtract_and_label(seg_mask, lm)
    # Expect two regions when cut reaches both boundaries
    assert num >= 2
    assert region_sizes[0][0] + (region_sizes[1][0] if len(region_sizes) > 1 else 0) <= seg_mask.sum()


def test_closed_polygon_inside_creates_mask():
    h, w = 200, 200
    segs = np.zeros((h, w), dtype=np.int32)
    segs[0:h, 0:w] = 1
    seg_mask = (segs == 1).astype(np.uint8)
    poly = [(50,50),(150,50),(150,150),(50,150)]
    applied, mask, area = closed_polygon_handler(poly, seg_mask, min_side_px=100)
    assert applied is True
    assert area == np.sum(mask)
    assert area >= 100


def test_closed_polygon_too_small_is_rejected():
    h, w = 100, 100
    segs = np.zeros((h, w), dtype=np.int32)
    segs[0:h, 0:w] = 1
    seg_mask = (segs == 1).astype(np.uint8)
    poly = [(1,1),(3,1),(3,3),(1,3)]
    applied, mask, area = closed_polygon_handler(poly, seg_mask, min_side_px=100)
    assert applied is False
    assert area == np.sum(mask)
    assert area < 100


def test_line_disconnects_rectangle_internal_cut():
    from labeling.core.split_core import line_disconnects
    h, w = 200, 300
    segs = np.zeros((h, w), dtype=np.int32)
    segs[40:160, 40:260] = 1
    seg_mask = (segs == 1).astype(np.uint8)
    # vertical line that spans the full segment height (touches edges)
    poly = [(120, 40), (120, 160)]
    from labeling.core.split_core import build_combined_line_mask
    lm = build_combined_line_mask([[poly[0], poly[1]]], seg_mask, (h, w), thickness=1, endpoint_radius=2)
    assert lm.sum() > 0
    # The line should disconnect left-right
    assert line_disconnects(seg_mask, lm) is True


def test_line_disconnects_donut_single_cut_doesnt_disconnect():
    from labeling.core.split_core import line_disconnects
    h, w = 200, 200
    segs = np.zeros((h, w), dtype=np.int32)
    segs[10:190, 10:190] = 1
    # create hole
    segs[70:130, 70:130] = 0
    seg_mask = (segs == 1).astype(np.uint8)
    # left-side vertical cut that should not yet disconnect the annulus
    poly = [(40, 20), (40, 180)]
    from labeling.core.split_core import build_combined_line_mask
    lm = build_combined_line_mask([[poly[0], poly[1]]], seg_mask, (h, w), thickness=1, endpoint_radius=2)
    # The line alone should not disconnect the annulus
    assert line_disconnects(seg_mask, lm) is False
