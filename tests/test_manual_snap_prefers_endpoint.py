import numpy as np
from scripts.labeling_tool import LabelingTool


def test_manual_snap_prefers_nearby_endpoint():
    lt = LabelingTool()
    # small image
    h, w = 200, 200
    img = np.zeros((h, w, 3), dtype=np.uint8) + 120
    lt.clean_image = img.copy()
    rect = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=float)
    lt.current_boundary = rect
    lt.original_boundary = rect.copy()
    lt.boundary_approved = True
    lt._segment_myself()
    assert lt.n_segments == 1

    lt.splitting_segment_id = 1
    # Draw two short collinear lines very close to each other so endpoints are closer to each other
    lt.manual_polylines = [ [(80, 60), (80, 140)], [(82, 62), (82, 142)] ]
    lt._apply_manual_split()

    # Confirm that snapping info exists and that we preferred endpoint snapping for at least one side
    assert hasattr(lt, '_last_snap_info')
    si = lt._last_snap_info
    assert si['start']['source'] in ('line', 'edge')
    assert si['end']['source'] in ('line', 'edge')
    # Ensure at least one of the snap sources used 'line' (preferred behavior)
    assert si['start']['source'] == 'line' or si['end']['source'] == 'line'