import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from labeling.core.split_core import mask_from_polylines, label_regions_after_cut, apply_direct_split
import numpy as np

# make donut
segs = np.zeros((200,300), dtype=int)
segs[40:160,40:260] = 1
segs[80:120,100:200] = 0
h,w = segs.shape
p1 = [(100,40),(100,160)]
line1 = mask_from_polylines([p1], (h,w), thickness=2)
seg_mask = (segs==1).astype(np.uint8)
labeled,num,region_sizes = label_regions_after_cut(seg_mask, line1)
print('num regions', num, 'region_sizes', region_sizes)
new_segs,new_id,info = apply_direct_split(segs,1,labeled,region_sizes,min_side_px=1000)
print('apply_direct_split info:', info)
