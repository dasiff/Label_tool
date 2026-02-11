import numpy as np
from scripts.labeling_tool import LabelingTool


def make_rect_image(h=200, w=300):
    img = np.zeros((h, w, 3), dtype=np.uint8) + 120
    # draw white background and a darker rectangle region
    cv2 = __import__('cv2')
    cv2.rectangle(img, (40, 40), (260, 160), (80, 80, 80), -1)
    return img


def test_sequential_splits_keep_selection():
    lt = LabelingTool()
    img = make_rect_image()
    lt.clean_image = img.copy()
    h, w = img.shape[:2]
    # Set boundary to the image rect and approve
    rect = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=float)
    lt.current_boundary = rect
    lt.original_boundary = rect.copy()
    lt.boundary_approved = True
    # Create a single segment corresponding to the darker rectangle drawn in make_rect_image
    segs = np.zeros((h, w), dtype=np.int32)
    segs[40:160, 40:260] = 1
    lt.segments = segs
    lt.n_segments = 1

    # Select the single large segment and make a vertical split (two cuts)
    # Simulate selecting segment by clicking (we set splitting_segment_id)
    seg_id = 1
    lt.splitting_segment_id = seg_id
    # Draw first split: vertical line at x=120 (endpoints on segment edges to ensure snapping)
    lt.manual_polylines = [[(120, 40), (120, 160)]]
    lt._apply_manual_split()
    # Verify a split happened and selection remains set to one of the parts
    assert lt.n_segments >= 2
    assert lt.splitting_segment_id is not None

    # Now perform a second split on the currently selected segment (horizontal cut with endpoints on edges)
    current = lt.splitting_segment_id
    lt.manual_polylines = [[(60, 100), (240, 100)]]
    # Ensure we're splitting an existing segment
    assert np.any(lt.segments == current)
    lt._apply_manual_split()
    # After second split, ensure there are at least 3 segments and selection preserved
    assert lt.n_segments >= 3
    assert lt.splitting_segment_id is not None


def test_closed_loop_creates_inner_segment():
    lt = LabelingTool()
    img = make_rect_image()
    lt.clean_image = img.copy()
    h, w = img.shape[:2]
    rect = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=float)
    lt.current_boundary = rect
    lt.original_boundary = rect.copy()
    lt.boundary_approved = True
    # Create a single segment corresponding to the darker rectangle drawn in make_rect_image
    segs = np.zeros((h, w), dtype=np.int32)
    segs[40:160, 40:260] = 1
    lt.segments = segs
    lt.n_segments = 1

    # Select segment and draw a small square inside
    lt.splitting_segment_id = 1
    square = [(100,80),(140,80),(140,120),(100,120),(100,80)]
    lt.manual_polylines = [square]
    lt._apply_manual_split()
    # Expect a new segment created for the square
    assert lt.n_segments >= 2
    # Find newly created small segment area roughly equals square area (~1600 px)
    areas = [int(np.sum(lt.segments==i)) for i in range(1, lt.n_segments+1)]
    assert any(a >= 1500 and a <= 2000 for a in areas)
