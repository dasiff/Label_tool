import numpy as np
import cv2
from types import SimpleNamespace
from labeling.core.manual_split import _apply_split_core


def make_dummy_self(segments, boundary=None, clean_image=None):
    s = SimpleNamespace()
    s.segments = segments
    s.current_boundary = boundary
    s.clean_image = clean_image if clean_image is not None else np.zeros((segments.shape[0], segments.shape[1], 3), dtype=np.uint8)
    s._running_in_background = False
    return s


def test_apply_split_core_rectangle_split():
    h, w = 100, 100
    segs = np.ones((h, w), dtype=np.int32)
    seg_mask = (segs == 1)
    # vertical line spanning the full segment height (touches edges)
    line_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(line_mask, (50, 0), (50, h-1), color=1, thickness=1)
    full_line_mask = line_mask.copy()
    boundary = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=float)
    dummy = make_dummy_self(segs.copy(), boundary=boundary)
    applied_segments, applied, info = _apply_split_core(dummy, seg_mask, line_mask, full_line_mask, 1, [(50,0),(50,h-1)], debug_img=np.zeros((h,w,3),dtype=np.uint8), seg_area=seg_mask.sum())
    assert applied is True
    assert 'new_seg_id' in info and info['new_seg_id'] is not None
    # ensure some pixels set to new id
    new_id = info['new_seg_id']
    assert (applied_segments == new_id).sum() > 0


def test_apply_split_core_donut_single_cut_no_split():
    h, w = 200, 200
    segs = np.zeros((h, w), dtype=np.int32)
    segs[20:180, 20:180] = 1
    # create hole
    segs[80:120, 80:120] = 0
    seg_mask = (segs == 1)
    # left side vertical cut that should not disconnect the annulus
    line_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(line_mask, (40, 30), (40, 170), color=1, thickness=1)
    full_line_mask = line_mask.copy()
    boundary = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=float)
    dummy = make_dummy_self(segs.copy(), boundary=boundary)
    applied_segments, applied, info = _apply_split_core(dummy, seg_mask, line_mask, full_line_mask, 1, [(40,30),(40,170)], debug_img=np.zeros((h,w,3),dtype=np.uint8), seg_area=seg_mask.sum())
    assert applied is False
