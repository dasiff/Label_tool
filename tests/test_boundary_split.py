import numpy as np
from scripts.labeling_tool import LabelingTool


def test_enforce_boundary_split_splits_crossing_segment():
    lt = LabelingTool()
    # Create a simple canvas with one segment ID (1) that crosses the boundary line
    h, w = 40, 40
    segs = np.zeros((h, w), dtype=np.int32)
    # Fill left block and right block connected by a thin bridge crossing x=19
    segs[5:35, 5:18] = 1
    segs[5:35, 22:35] = 1
    segs[18:22, 18:22] = 1  # bridge across center (this crosses the boundary we'll draw)
    lt.segments = segs.copy()
    # Boundary as vertical line through x=19..20
    lt.current_boundary = np.array([[20,0],[20,39],[0,39],[0,0]], dtype=float)

    lt._enforce_boundary_split(thickness=1)
    # After enforcing, segments touching left vs right sides should not be the same id
    left_ids = set(np.unique(lt.segments[:, :18])) - {0}
    right_ids = set(np.unique(lt.segments[:, 22:])) - {0}
    assert left_ids
    assert right_ids
    # Ensure they are different segment ids (split occurred)
    assert left_ids.isdisjoint(right_ids)
