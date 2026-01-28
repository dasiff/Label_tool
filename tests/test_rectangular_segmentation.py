import pytest
import numpy as np
from scripts.labeling_tool import LabelingTool

@pytest.mark.skip("WIP: rectangular-area segmentation algorithm to be implemented")
def test_rectangular_segmentation_hint():
    """Placeholder test: expect segmentation to consider larger rectangular polygonal areas.

    This test will be implemented once the rectangular-area seeding algorithm is added.
    """
    lt = LabelingTool()
    # Setup a synthetic image and ensure boundary approved
    lt.clean_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    lt.current_boundary = np.array([[10,10],[190,10],[190,190],[10,190]], dtype=float)
    lt.boundary_approved = True
    # TODO: create synthetic rectangular seeds and assert segmentation merges accordingly
    assert True
