from scripts.labeling_tool import LabelingTool
import numpy as np


def test_high_target_yields_more_than_three_segments():
    lt = LabelingTool()
    lt.clean_image = np.ones((200,200,3), dtype=np.uint8)*255
    # Create some artificial texture so segmentation can split area
    lt.clean_image[50:150, 50:150] = np.random.randint(0,255,(100,100,3), dtype=np.uint8)
    lt.current_boundary = np.array([[40,40],[160,40],[160,160],[40,160]], dtype=float)
    lt.boundary_approved = True
    lt.target_segments = 378
    # Allow small segments in test so high target produces many segments
    lt.min_segment_px = 10
    lt._generate_segments()
    # Expect more than 3 segments when target is high
    assert lt.n_segments > 3
    # And expect at least some smaller segments preserved
    assert any((lt.segments == i).sum() >= 30 for i in range(1, lt.n_segments+1))