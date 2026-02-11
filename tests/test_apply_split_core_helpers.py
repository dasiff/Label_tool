import numpy as np
from labeling.core.manual_split import _widen_line_and_label, _attempt_seed_split, _attempt_touched_splits, _attempt_global_cut, _compute_line_intersections


def test_widen_line_and_label_vertical_split():
    h, w = 50, 200
    seg_mask = np.ones((h, w), dtype=bool)
    # full vertical 1-px line at center
    line_mask = np.zeros((h, w), dtype=np.uint8)
    x = w // 2
    line_mask[:, x] = 255

    labeled, num, final_line = _widen_line_and_label(seg_mask.astype(np.uint8), line_mask)
    assert num >= 2
    assert final_line.sum() > 0


def test_attempt_seed_split_basic():
    h, w = 50, 200
    seg_mask = np.ones((h, w), dtype=bool)
    # create a vertical cut so the segment is disconnected for seed flood
    xcut = w // 2
    seg_mask_with_cut = seg_mask.copy()
    seg_mask_with_cut[:, xcut] = False
    # line points vertical across middle so orthogonal seeds are left/right
    points = [(xcut, 10), (xcut, h-10)]
    new_segments = np.ones((h, w), dtype=np.int32)

    new_segments_after, applied, info = _attempt_seed_split(seg_mask_with_cut.astype(np.uint8), np.zeros((h,w), dtype=np.uint8), points, new_segments)
    assert applied is True
    assert 'new_seg_id' in info and info['new_seg_id'] is not None
    new_id = int(info['new_seg_id'])
    assert np.sum(new_segments_after == new_id) > 0


def test_attempt_touched_splits_succeeds():
    h, w = 50, 200
    segs = np.zeros((h, w), dtype=np.int32)
    # left area = seg 1 columns 0..99, right area = seg 2 columns 100..199
    segs[:, 0:100] = 1
    segs[:, 100:] = 2

    # full_line_mask touches seg 2 in the middle column 150
    full_line_mask = np.zeros((h, w), dtype=np.uint8)
    full_line_mask[:, 150] = 255

    new_segs, applied, info = _attempt_touched_splits(segs.copy(), full_line_mask, seg_id=1)
    assert applied is True
    assert 'new_seg_id' in info and info['new_seg_id'] is not None
    new_id = int(info['new_seg_id'])
    assert np.sum(new_segs == new_id) > 0


def test_compute_line_intersections_and_global_cut():
    h, w = 50, 200
    # build a dummy self with required attributes
    class Dummy:
        pass
    d = Dummy()
    d.segments = np.ones((h, w), dtype=np.int32)
    d.clean_image = np.zeros((h, w, 3), dtype=np.uint8)
    d.current_boundary = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])

    # a line that crosses the image horizontally - should give at least 2 intersections
    p0f = (-10.0, h/2.0)
    p1f = (w+10.0, h/2.0)
    inters = _compute_line_intersections(d, np.array(p0f), np.array(p1f))
    assert len(inters) >= 2

    # now call global cut to exercise using those intersections
    points = [p0f, p1f]
    new_segs, applied, info = _attempt_global_cut(d, d.segments.copy(), points)
    assert applied is True
    assert 'new_seg_id' in info and info['new_seg_id'] is not None
    new_id = int(info['new_seg_id'])
    assert np.sum(new_segs == new_id) > 0
