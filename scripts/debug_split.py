import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import cv2
from labeling.core.split_core import mask_from_polylines
from scipy import ndimage

def make_donut(h=200,w=300):
    segs = np.zeros((h,w), dtype=np.int32)
    segs[40:160, 40:260] = 1
    segs[80:120, 100:200] = 0
    return segs

segs = make_donut()
h,w = segs.shape
p1 = [(50,45),(50,155)]
p2 = [(250,45),(250,155)]
line1 = mask_from_polylines([p1], (h,w), thickness=3)
line2 = mask_from_polylines([p2], (h,w), thickness=3)
combined = ((line1 + line2) > 0).astype(np.uint8)
seg_mask = (segs == 1).astype(np.uint8)
line_in_seg = combined & seg_mask
print('seg_mask sum', seg_mask.sum())
print('line1 sum', line1.sum(), 'line2 sum', line2.sum(), 'combined sum', combined.sum(), 'line_in_seg sum', line_in_seg.sum())
labeled, num = ndimage.label(seg_mask & (~line_in_seg))
print('num regions after cut', num)
ys, xs = np.where(line_in_seg)
print('line coords sample', list(zip(xs[:10], ys[:10])))
