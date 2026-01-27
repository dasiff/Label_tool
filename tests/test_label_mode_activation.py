from scripts.labeling_tool import LabelingTool
import numpy as np


def test_enter_label_mode_from_segments_off_and_generate_segments():
    lt = LabelingTool()
    # simulate a previously-saved boundary that is approved but segments missing
    lt.current_boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=float)
    lt.boundary_approved = True
    lt.segments = None
    # Simulate being in manual mode (on)
    lt.manual_mode = True

    lt._set_mode('label')

    # After switching, manual_mode should be off and segments should be generated
    assert lt.manual_mode is False
    assert lt.segments is not None
    assert lt.n_segments >= 1
