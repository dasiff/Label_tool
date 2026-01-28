import numpy as np
from types import SimpleNamespace
from labeling.core.manual_split import _apply_split_core


def make_dummy_self(h=40, w=80):
    s = SimpleNamespace()
    s.segments = np.ones((h, w), dtype=np.int32)
    s.split_history = []
    s.segment_labels = {1: 'A'}
    s.current_boundary = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    s.clean_image = np.zeros((h, w, 3), dtype=np.uint8)
    return s


def test_split_requires_snapping_success():
    self = make_dummy_self()
    h, w = self.segments.shape
    seg_mask = (self.segments == 1)
    xcut = w // 2
    line_mask = np.zeros((h, w), dtype=np.uint8)
    line_mask[:, xcut] = 255
    full_line_mask = line_mask.copy()
    points = [(xcut, 0), (xcut, h-1)]

    new_segs, applied, info = _apply_split_core(self, seg_mask, line_mask, full_line_mask, seg_id=1, points=points, debug_img=self.clean_image, seg_area=seg_mask.sum(), require_snapped=True, snapped_endpoints=True)
    assert applied is True
    assert 'new_seg_id' in info and info['new_seg_id'] is not None
    new_id = int(info['new_seg_id'])
    assert np.sum(new_segs == new_id) > 0


def test_split_fails_without_snapping():
    self = make_dummy_self()
    h, w = self.segments.shape
    seg_mask = (self.segments == 1)
    xcut = w // 2
    line_mask = np.zeros((h, w), dtype=np.uint8)
    line_mask[:, xcut] = 255
    full_line_mask = line_mask.copy()
    points = [(xcut, 0), (xcut, h-1)]

    new_segs, applied, info = _apply_split_core(self, seg_mask, line_mask, full_line_mask, seg_id=1, points=points, debug_img=self.clean_image, seg_area=seg_mask.sum(), require_snapped=True, snapped_endpoints=False)
    assert applied is False
    assert info.get('reason') == 'no_snap'
    # ensure no segments changed
    assert np.array_equal(new_segs, self.segments)
