import numpy as np
from types import SimpleNamespace
from labeling.core.manual_split import _apply_split_state, SplitState


def make_dummy_self(h=20, w=40):
    segs = np.ones((h, w), dtype=np.int32)
    dummy = SimpleNamespace()
    dummy.segments = segs.copy()
    dummy.split_history = []
    dummy.segment_labels = {1: 'parking'}
    dummy.n_segments = int(dummy.segments.max())
    dummy.splitting_segment_id = None
    dummy.clean_image = np.zeros((h, w, 3), dtype=np.uint8)
    dummy.current_boundary = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    dummy._running_in_background = False
    return dummy


def test_apply_split_state_success():
    self = make_dummy_self()
    state = SplitState(segments=self.segments.copy(), manual_polylines=[], segment_labels=self.segment_labels.copy(), split_history=[], n_segments=self.n_segments, splitting_segment_id=None)

    h, w = self.segments.shape
    seg_mask = (self.segments == 1)

    # Build a vertical cut down the middle that will split the segment into two large regions
    line_mask = np.zeros_like(self.segments, dtype=np.uint8)
    xcut = w // 2
    line_mask[:, xcut] = 255
    full_line_mask = line_mask.copy()

    points = [(xcut, 0), (xcut, h-1)]  # (x,y) points

    state_before = state.segments.copy()
    state, applied, info = _apply_split_state(self, state, seg_mask, line_mask, full_line_mask, seg_id=1, points=points, debug_img=self.clean_image, seg_area=seg_mask.sum())

    assert applied is True
    assert 'new_seg_id' in info and info['new_seg_id'] is not None
    new_id = int(info['new_seg_id'])

    # New id should be present in state and self
    assert state.splitting_segment_id == new_id
    assert self.splitting_segment_id == new_id

    # Segment labels should be inherited
    assert state.segment_labels.get(new_id) == 'parking'
    assert self.segment_labels.get(new_id) == 'parking'

    # History should have been appended
    assert len(state.split_history) == 1
    assert np.array_equal(state.split_history[0], state_before)

    # The new segments array should contain pixels assigned to the new id
    assert np.sum(state.segments == new_id) > 0


def test_apply_split_state_noop():
    self = make_dummy_self()
    state = SplitState(segments=self.segments.copy(), manual_polylines=[], segment_labels=self.segment_labels.copy(), split_history=[], n_segments=self.n_segments, splitting_segment_id=None)

    h, w = self.segments.shape
    seg_mask = (self.segments == 1)

    # Empty line mask -> should not apply
    line_mask = np.zeros_like(self.segments, dtype=np.uint8)
    full_line_mask = line_mask.copy()
    points = [(0,0),(1,1)]

    state_before = state.segments.copy()
    state, applied, info = _apply_split_state(self, state, seg_mask, line_mask, full_line_mask, seg_id=1, points=points, debug_img=self.clean_image, seg_area=seg_mask.sum())

    assert applied is False
    assert info == {} or 'reason' in info
    # history not appended
    assert len(state.split_history) == 0
    # segments unchanged
    assert np.array_equal(state.segments, state_before)
