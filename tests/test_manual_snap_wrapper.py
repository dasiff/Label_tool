import numpy as np
from labeling.core.manual_split import _snap_endpoints_for_polyline


def test_snap_wrapper_prefers_line_when_closer():
    h, w = 200, 200
    # Edge coords: rectangle boundary far
    edge_coords = np.array([[0,0],[0,199],[199,0],[199,199]])  # (y,x)
    p0 = (50, 50)
    p_last = (150, 150)
    # other polyline very close to p0
    other_polylines = [[(50,50),(51,51)]]

    seg_uint8 = np.zeros((h, w), dtype=np.uint8)
    seg_uint8[10:190, 10:190] = 1

    s0, s1, src0, src1 = _snap_endpoints_for_polyline(p0, p_last, edge_coords, other_polylines, seg_uint8)
    assert src0 == 'line'


def test_snap_wrapper_prefers_edge_when_closer():
    h, w = 200, 200
    # Edge coords near endpoints
    edge_coords = np.array([[49,50],[151,150]])  # (y,x)
    p0 = (50, 50)
    p_last = (150, 150)
    other_polylines = []

    seg_uint8 = np.zeros((h, w), dtype=np.uint8)
    seg_uint8[0:h, 0:w] = 1

    s0, s1, src0, src1 = _snap_endpoints_for_polyline(p0, p_last, edge_coords, other_polylines, seg_uint8)
    assert src0 == 'edge'
    assert src1 == 'edge' or src1 == 'none'