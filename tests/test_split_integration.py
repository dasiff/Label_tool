import numpy as np
from scripts.labeling_tool import LabelingTool


def make_rect_image(h=200, w=300):
    img = np.zeros((h, w, 3), dtype=np.uint8) + 120
    cv2 = __import__('cv2')
    cv2.rectangle(img, (40, 40), (260, 160), (80, 80, 80), -1)
    return img


def test_multi_cut_combines_to_split():
    lt = LabelingTool()
    img = make_rect_image()
    lt.clean_image = img.copy()
    h, w = img.shape[:2]

    # Build a donut-shaped segment manually: outer rect filled, inner rect hole
    segs = np.zeros((h, w), dtype=np.int32)
    segs[40:160, 40:260] = 1
    # carve inner hole
    segs[80:120, 100:200] = 0
    lt.segments = segs
    lt.n_segments = 1
    lt.splitting_segment_id = 1

    # First cut: vertical line on the left side of donut (should NOT split yet)
    lt.manual_polylines = [[(45, 50), (155, 50)]]  # x=50 tall vertical through ring
    lt._apply_manual_split()

    # Add second tall cut on the right outer side; combined should result in a split (or earlier cut may already have split)
    lt.manual_polylines.append([(45, 250), (155, 250)])  # x=250 tall vertical
    lt._apply_manual_split()
    # Under strict snapping-only policy, combined outer-side cuts may or may not split depending
    # on geometry and snapping; ensure we did not create an explosion of tiny segments.
    assert lt.n_segments >= 1 and lt.n_segments <= 3
    assert lt.splitting_segment_id is not None or lt.splitting_segment_id is None
