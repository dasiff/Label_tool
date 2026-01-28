import numpy as np
from scripts.labeling_tool import LabelingTool


def test_default_full_no_access_not_drawn():
    lt = LabelingTool()
    lt.current_boundary = np.array([[10,10],[110,10],[110,110],[10,110]], dtype=float)
    lt.boundary_approved = True
    # Default full-boundary no_access
    lt.boundary_access_segments = [{'start_frac':0.0,'end_frac':1.0,'label':'no_access'}]

    # Ensure drawing doesn't create heavy access overlay artists for the default
    lt._draw_access_segments()
    # access_artists should be empty because default full no_access is not rendered
    assert hasattr(lt, 'access_artists')
    assert len(lt.access_artists) == 0
