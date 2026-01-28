import numpy as np
from scripts.labeling_tool import LabelingTool


def test_mask_segments_to_roi_compacts_and_masks():
    lt = LabelingTool()
    lt._tk_available = False
    h, w = 200, 200
    lt.clean_image = np.ones((h, w, 3), dtype=np.uint8) * 120
    lt.current_boundary = np.array([[40,40],[160,40],[160,160],[40,160]], dtype=float)
    # Use a small buffer to ensure outside areas remain zeroed for this unit test
    lt.road_buffer_px = 10
    lt._compute_buffer_mask()
    assert getattr(lt, 'buffer_mask', None) is not None

    segs = np.zeros((h, w), dtype=int)
    segs[50:80, 50:80] = 5
    segs[10:20, 10:20] = 6
    segs[30:60, 30:60] = 7
    lt.segments = segs
    lt._mask_segments_to_roi()

    # Outside area should be 0
    assert not (lt.segments[10:20, 10:20] != 0).any()

    # Positive IDs should start at 1
    uniques = np.unique(lt.segments)
    positives = uniques[uniques > 0]
    if len(positives) > 0:
        assert positives.min() == 1
