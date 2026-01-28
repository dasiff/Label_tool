import numpy as np
from labeling.core.manual_split import snap_endpoints


def test_snap_endpoints_prefers_line_when_closer():
    # Edge coords are far; other line pixel is close
    edge_coords = np.array([[0, 0], [0, 100], [100, 0], [100, 100]])  # (y,x)
    p0 = (50, 50)
    p_last = (52, 51)
    # other line pixel very close to p0
    other_line_coords = np.array([[50, 50], [52, 51]])

    ep0, ep_last, source0, sourcelast = snap_endpoints(p0, p_last, edge_coords, other_line_coords, max_snap_dist=100)
    # Expect endpoints to snap to nearest line pixels
    assert source0 == 'line'
    assert sourcelast == 'line'


def test_snap_endpoints_prefers_edge_when_closer():
    # Edge coords very close to both endpoints; other line pixel is far
    edge_coords = np.array([[51, 50], [50, 51]])  # (y,x) near endpoints
    p0 = (50, 50)
    p_last = (51, 52)
    other_line_coords = np.array([[200, 200]])

    ep0, ep_last, source0, sourcelast = snap_endpoints(p0, p_last, edge_coords, other_line_coords, max_snap_dist=100)
    assert source0 == 'edge'
    assert sourcelast == 'edge' or sourcelast == 'edge'
