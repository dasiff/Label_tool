import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
from scripts.labeling_tool import LabelingTool


def make_rect_image(h=200, w=300):
    img = np.zeros((h, w, 3), dtype=np.uint8) + 120
    # draw white background and a darker rectangle region
    cv2 = __import__('cv2')
    cv2.rectangle(img, (40, 40), (260, 160), (80, 80, 80), -1)
    return img


def run():
    lt = LabelingTool()
    img = make_rect_image()
    lt.clean_image = img.copy()
    h, w = img.shape[:2]
    rect = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=float)
    lt.current_boundary = rect
    lt.original_boundary = rect.copy()
    lt.boundary_approved = True
    lt._segment_myself()
    print('n_segments after segment_myself:', lt.n_segments)
    seg_id = 1
    lt.splitting_segment_id = seg_id
    lt.manual_polylines = [[(120,50),(120,150)]]
    lt._apply_manual_split()
    print('After first split: n_segments=', lt.n_segments, 'splitting_segment_id=', lt.splitting_segment_id)
    if lt.n_segments < 2 or lt.splitting_segment_id is None:
        print('FIRST SPLIT FAILED')
    else:
        current = lt.splitting_segment_id
        lt.manual_polylines = [[(60,100),(240,100)]]
        lt._apply_manual_split()
        print('After second split: n_segments=', lt.n_segments, 'splitting_segment_id=', lt.splitting_segment_id)

if __name__ == '__main__':
    run()
