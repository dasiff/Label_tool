# Quick test to validate _mask_segments_to_roi behavior
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import cv2
from scripts.labeling_tool import LabelingTool


def run():
    lt = LabelingTool()
    # Make headless environment explicit
    lt._tk_available = False
    # Create simple clean image
    h, w = 200, 200
    img = np.ones((h, w, 3), dtype=np.uint8) * 120
    lt.clean_image = img
    # Create boundary - rectangle 40:160
    boundary = np.array([[40,40],[160,40],[160,160],[40,160]], dtype=float)
    lt.current_boundary = boundary
    # Compute buffer (default 96px) -> will make buffer cover outside region
    lt._compute_buffer_mask()
    buf = getattr(lt, 'buffer_mask', None)
    print('buffer_mask exists:', buf is not None, 'buffer pixels:', None if buf is None else int(np.sum(buf)))

    # Create a segments array with some labels outside the roi and inside
    segs = np.zeros((h, w), dtype=np.int32)
    # inside ROI: set a region to id 5
    segs[50:80, 50:80] = 5
    # outside ROI: set some region to id 6
    segs[10:20, 10:20] = 6
    # partially overlapping region which should be masked partially
    segs[30:60, 30:60] = 7
    lt.segments = segs
    lt.n_segments = int(lt.segments.max())
    print('Before mask: unique segments', np.unique(lt.segments))

    lt._mask_segments_to_roi()
    print('After mask: unique segments', np.unique(lt.segments))
    # Assert outside area (10:20) zeroed
    if np.any(lt.segments[10:20, 10:20] != 0):
        print('FAIL: outside region not zeroed')
    else:
        print('PASS: outside region zeroed')
    # Ensure segments now compacted and positive ids start at 1
    uniques = np.unique(lt.segments)
    positives = uniques[uniques > 0]
    if len(positives) > 0 and positives.min() == 1:
        print('PASS: positive IDs compacted and start at 1')
    else:
        print('FAIL: IDs not compacted as expected')

if __name__ == '__main__':
    run()
