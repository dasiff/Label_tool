import numpy as np
import cv2
from scipy import ndimage
h,w = 200,300
seg_uint8 = np.ones((h,w), dtype=np.uint8)
kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(seg_uint8, kernel, iterations=1)
edge_mask = seg_uint8 - eroded
# build line mask
line_mask = np.zeros((h,w), dtype=np.uint8)
pts = np.array([[120,0],[120,199]], dtype=np.int32)
cv2.polylines(line_mask, [pts], isClosed=False, color=255, thickness=1)
print('line sum', line_mask.sum(), 'edge_sum', edge_mask.sum())
line_cc, n_line_cc = ndimage.label(line_mask>0)
print('n_line_cc', n_line_cc)
for cc_id in range(1, n_line_cc+1):
    cc_mask = (line_cc == cc_id)
    print('cc_id', cc_id, 'intersects edge?', np.any(cc_mask & edge_mask))
