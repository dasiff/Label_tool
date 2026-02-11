import numpy as np
from scripts.labeling_tool import LabelingTool


def test_add_access_segment_two_clicks():
    lt = LabelingTool()
    lt.current_boundary = np.array([[10,10],[110,10],[110,110],[10,110]], dtype=float)
    lt.boundary_approved = True
    lt.boundary_access_segments = [{'start_frac':0.0,'end_frac':1.0,'label':'no_access'}]

    proj, frac = lt._closest_point_on_boundary(60, 10)
    lt._add_boundary_access_click(proj[0], proj[1], click_button=1)
    assert lt.access_click_start is not None

    proj2, frac2 = lt._closest_point_on_boundary(110, 60)
    lt._add_boundary_access_click(proj2[0], proj2[1], click_button=1)

    # Expect a new access_allowed segment appended (length should increase)
    assert len(lt.boundary_access_segments) > 1
    assert any(seg.get('label') == 'access_allowed' and not (seg['start_frac']==0.0 and seg['end_frac']==1.0) for seg in lt.boundary_access_segments)


def test_right_click_cancels_provisional():
    lt = LabelingTool()
    lt.current_boundary = np.array([[10,10],[110,10],[110,110],[10,110]], dtype=float)
    lt.boundary_approved = True
    lt.boundary_access_segments = [{'start_frac':0.0,'end_frac':1.0,'label':'no_access'}]

    proj, frac = lt._closest_point_on_boundary(60, 10)
    lt._add_boundary_access_click(proj[0], proj[1], click_button=1)
    assert lt.access_click_start is not None

    # Right-click to cancel
    lt._add_boundary_access_click(0, 0, click_button=3)
    assert lt.access_click_start is None
