from scripts.labeling_tool import LabelingTool
import numpy as np


def test_auto_approve_generates_segments_on_label_mode():
    lt = LabelingTool()
    # Set up boundary and a simple clean image to allow segmentation
    lt.current_boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=float)
    lt.clean_image = (np.ones((100,100,3), dtype='uint8') * 255)
    lt.boundary_approved = False
    lt.segments = None

    lt._set_mode('label')

    # After switching to label mode, boundary should be approved and segments generated
    assert lt.boundary_approved is True
    assert lt.segments is not None
    assert lt.n_segments >= 1
