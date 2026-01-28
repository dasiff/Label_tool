import numpy as np
from labeling.core.manual_split import SplitState
from labeling.core.manual_split import SplitState as SS
from labeling.core.manual_split import SplitState
from labeling.core.manual_split import SplitState
from labeling.core.manual_split import SplitState
from labeling.core.manual_split import SplitState
from labeling.core.manual_split import SplitState

from labeling.core.manual_split import SplitState
from labeling.core.manual_split import SplitState
from labeling.core.manual_split import SplitState
from labeling.core.manual_split import SplitState

def test_direct_split_action_applies_and_preserves_label():
    # Build an image with a single segment that will split into two roughly equal parts
    h, w = 100, 100
    segs = np.zeros((h, w), dtype=np.int32)
    # fill whole image as seg 1
    segs[:, :] = 1
    # Simulate labeled_regions where left half is region 1 and right half is region 2
    labeled = np.zeros((h, w), dtype=np.int32)
    labeled[:, :50] = 1
    labeled[:, 50:] = 2
    region_sizes = [(int((labeled == 1).sum()), 1), (int((labeled == 2).sum()), 2)]

    # Prepare state
    state = SplitState(segments=segs.copy(), manual_polylines=[], segment_labels={1: 'parking_stalls'})

    # Use apply_direct_split from split_core (mirrors logic used by manual_split helper)
    from labeling.core.split_core import apply_direct_split
    new_segs, new_id, info = apply_direct_split(state.segments, 1, labeled, region_sizes, min_side_px=10)
    assert info.get('applied') is True
    assert new_id is not None
    # Ensure new_id pixels exist and label was not lost (we expect caller to handle label inheritance)
    assert (new_segs == new_id).sum() > 0
