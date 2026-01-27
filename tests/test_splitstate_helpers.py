import numpy as np
from types import SimpleNamespace
from labeling.core.manual_split import _push_split_history, _apply_new_segments_counts, _inherit_label_and_select, SplitState


def make_dummy(h=10, w=10):
    s = SimpleNamespace()
    s.segments = np.ones((h, w), dtype=np.int32)
    s.split_history = []
    s.n_segments = int(s.segments.max())
    s.splitting_segment_id = None
    s.segment_labels = {1: 'label1'}
    return s


def test_push_split_history_trims():
    state = make_dummy()
    self = make_dummy()
    # push > maxlen items
    for i in range(12):
        state.segments = (np.ones((5,5), dtype=np.int32) * (i+1))
        self.segments = (np.ones((5,5), dtype=np.int32) * (i+1))
        _push_split_history(state, self, maxlen=10)

    assert len(state.split_history) == 10
    assert len(self.split_history) == 10
    # earliest entry should correspond to value 3
    assert np.all(state.split_history[0] == 3)
    assert np.all(self.split_history[0] == 3)


def test_apply_new_segments_counts_updates():
    state = make_dummy()
    self = make_dummy()
    new_segments = np.zeros((5,5), dtype=np.int32)
    new_segments[0:2, :] = 1
    new_segments[2:, :] = 2
    _apply_new_segments_counts(state, self, new_segments)
    assert np.array_equal(state.segments, new_segments)
    assert np.array_equal(self.segments, new_segments)
    assert state.n_segments == 2
    assert self.n_segments == 2


def test_inherit_label_and_select():
    state = make_dummy()
    self = make_dummy()
    info = {'new_seg_id': 5}
    res = _inherit_label_and_select(state, self, 1, info)
    assert res == 5
    assert state.splitting_segment_id == 5
    assert self.splitting_segment_id == 5
    assert state.segment_labels[5] == 'label1'
    assert self.segment_labels[5] == 'label1'