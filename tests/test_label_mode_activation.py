from scripts.labeling_tool import LabelingTool
import numpy as np


def test_enter_label_mode_from_segments_off_and_generate_segments():
    lt = LabelingTool()
    # simulate a previously-saved boundary that is approved but segments missing
    lt.current_boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=float)
    lt.boundary_approved = True
    lt.segments = None
    # Simulate an active splitting selection
    lt.splitting_segment_id = 5

    lt._set_mode('label')

    # After switching, transient splitting selection should be cleared and segments should be generated
    assert lt.splitting_segment_id is None
    assert lt.segments is not None
    assert lt.n_segments >= 1
