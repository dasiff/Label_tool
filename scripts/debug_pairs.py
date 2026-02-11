import sys
sys.path.append(r'G:/My Drive/parking_spaces')
from scripts.labeling_tool import LabelingTool
import numpy as np
import cv2

def make_img():
    img = np.zeros((200,300,3), dtype=np.uint8) + 120
    cv2.rectangle(img, (40,40),(260,160),(80,80,80), -1)
    return img

pairs = [ (50,250), (80,220), (90,210), (100,200), (80,200), (120,180), (100,220) ]
for x1,x2 in pairs:
    lt = LabelingTool()
    lt.clean_image = make_img()
    h,w = 200,300
    seg = np.zeros((h,w), dtype=np.int32)
    seg[40:160,40:260]=1
    seg[80:120,100:200]=0
    lt.segments = seg
    lt.n_segments = 1
    lt.splitting_segment_id = 1
    lt.manual_polylines = [[(45, x1), (155, x1)]]
    lt._apply_manual_split()
    print(f'after first cut: n_segments={lt.n_segments}, snap={getattr(lt, "_last_snap_info", None)}')
    lt.manual_polylines.append([(45, x2), (155, x2)])
    lt._apply_manual_split()
    print(f'pair {x1},{x2} -> n_segments {lt.n_segments}, splitting_segment_id {lt.splitting_segment_id}, last_snap {getattr(lt, "_last_snap_info", None)}')
