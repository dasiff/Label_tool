import numpy as np
from scripts.labeling_tool import LabelingTool


def test_single_line_across_irregular_boundary_splits():
    t = LabelingTool()
    # Simple image
    t.clean_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    # Make boundary as an irregular polygon; use a rectangle for simplicity
    t.current_boundary = np.array([[20, 20], [180, 20], [180, 180], [20, 180]], dtype=float)
    t.boundary_approved = True

    # Create a segment that touches the top boundary (segment inside boundary)
    segs = np.zeros((200, 200), dtype=np.int32)
    # segment id 1 is central but touches the top side via irregularity
    segs[25:180, 50:150] = 1
    t.segments = segs
    t.n_segments = 1

    # Simulate user drawing a single polyline whose endpoints are along the irregular boundary
    # Points are roughly on the top edge; endpoints may snap to edge pixels
    t.splitting_segment_id = 1
    t.manual_polylines = [[(60, 20), (100, 60)]]  # start near top boundary, end slightly inside

    # Apply manual split (headless synchronous)
    t._apply_manual_split()

    # Expect that a split was applied: number of segments should increase
    assert t.n_segments >= 2, f"Expected split to create additional segment, got {t.n_segments}"
    # Confirm segment ids changed in mask
    unique_ids = np.unique(t.segments)
    assert len(unique_ids[unique_ids > 0]) >= 2