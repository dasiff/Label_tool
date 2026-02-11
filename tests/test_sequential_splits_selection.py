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


def test_sequential_splits_keep_selection_and_history():
    self = make_dummy_self()
    state = SplitState(segments=self.segments.copy(), manual_polylines=[], segment_labels=self.segment_labels.copy(), split_history=[], n_segments=self.n_segments, splitting_segment_id=None)

    h, w = self.segments.shape

    # First split: vertical in middle
    xcut1 = w // 2
    line_mask1 = np.zeros_like(self.segments, dtype=np.uint8)
    line_mask1[:, xcut1] = 255
    full_line_mask1 = line_mask1.copy()
    points1 = [(xcut1, 0), (xcut1, h-1)]

    state_before = state.segments.copy()
    state, applied1, info1 = _apply_split_state(self, state, (state.segments == 1), line_mask1, full_line_mask1, seg_id=1, points=points1, debug_img=self.clean_image, seg_area=(state.segments == 1).sum())
    assert applied1 is True
    new1 = int(info1['new_seg_id'])
    assert state.splitting_segment_id == new1
    assert self.splitting_segment_id == new1
    assert state.segment_labels.get(new1) == 'parking'
    assert self.segment_labels.get(new1) == 'parking'
    assert len(state.split_history) == 1

    # Second split: split the newly created segment further (vertical in its half)
    seg_id2 = new1
    seg_mask2 = (state.segments == seg_id2)
    assert seg_mask2.sum() > 0

    # choose a vertical cut inside that half: at 3/4 width of full image
    xcut2 = (3 * w) // 4
    line_mask2 = np.zeros_like(self.segments, dtype=np.uint8)
    line_mask2[:, xcut2] = 255
    full_line_mask2 = line_mask2.copy()
    points2 = [(xcut2, 0), (xcut2, h-1)]

    state, applied2, info2 = _apply_split_state(self, state, seg_mask2, line_mask2, full_line_mask2, seg_id=seg_id2, points=points2, debug_img=self.clean_image, seg_area=seg_mask2.sum())
    assert applied2 is True
    new2 = int(info2['new_seg_id'])

    # selection updated
    assert state.splitting_segment_id == new2
    assert self.splitting_segment_id == new2

    # history appended again
    assert len(state.split_history) == 2

    # new id has pixels
    assert np.sum(state.segments == new2) > 0
