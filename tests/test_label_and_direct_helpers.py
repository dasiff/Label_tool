import numpy as np
from types import SimpleNamespace
from labeling.core.manual_split import _label_regions_with_retry, _attempt_direct_split


def make_dummy(h=20, w=40, background=False):
    d = SimpleNamespace()
    d._running_in_background = background
    d.root = SimpleNamespace()
    d.root.after_called = False
    def after(delay, func):
        d.root.after_called = True
    d.root.after = after
    d.current_boundary = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    d.clean_image = np.zeros((h, w, 3), dtype=np.uint8)
    return d


def test_label_regions_with_retry_no_retry():
    d = make_dummy(background=False)
    h, w = 30, 60
    seg_mask = np.ones((h,w), dtype=bool)
    # create a full vertical cut to produce two regions
    line_mask = np.zeros((h,w), dtype=np.uint8)
    x = w//2
    line_mask[:, x] = 255

    labeled, num, final_line, retry = _label_regions_with_retry(d, seg_mask.astype(np.uint8), line_mask, force_global=False)
    assert retry is False
    assert num >= 2


def test_label_regions_with_retry_background_retry():
    d = make_dummy(background=True)
    h, w = 30, 60
    seg_mask = np.ones((h,w), dtype=bool)
    # small localized line that doesn't split the segment
    line_mask = np.zeros((h,w), dtype=np.uint8)
    line_mask[10, 10] = 255

    labeled, num, final_line, retry = _label_regions_with_retry(d, seg_mask.astype(np.uint8), line_mask, force_global=False)
    # Under strict snapping policy, we do not schedule a background retry
    assert retry is False
    assert d.root.after_called is False


def test_attempt_direct_split_succeeds():
    h, w = 30, 60
    segs = np.ones((h, w), dtype=np.int32)
    labeled = np.zeros((h, w), dtype=np.int32)
    # left half region 1, right half region 2
    xcut = w//2
    labeled[:, :xcut] = 1
    labeled[:, xcut:] = 2
    region_sizes = [(int(np.sum(labeled == 1)), 1), (int(np.sum(labeled == 2)), 2)]

    new_segs, applied, info = _attempt_direct_split(None, segs.copy(), 1, labeled, region_sizes, min_side_px=10)
    assert applied is True
    assert 'new_seg_id' in info and info['new_seg_id'] is not None
    assert np.sum(new_segs == int(info['new_seg_id'])) > 0
