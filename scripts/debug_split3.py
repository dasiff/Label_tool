from labeling.core.split_core import build_combined_line_mask, constrained_subtract_and_label
import numpy as np
from scipy import ndimage

h,w=200,300
segs=np.zeros((h,w),dtype=int)
segs[:,:]=1
seg_mask=(segs==1).astype(np.uint8)
poly=[(150,10),(150,190)]
lm=build_combined_line_mask([poly],seg_mask,(h,w),thickness=1,endpoint_radius=2)
print('line sum',lm.sum())
seg_without = seg_mask.copy(); seg_without[lm>0]=0
labeled,num=ndimage.label(seg_without)
print('num regions after cut',num)
# compute connectivity via flood from left side
from collections import deque
visited=np.zeros_like(seg_without,dtype=bool)
q=deque()
start=(h//2,10)
if seg_without[start]:
    q.append(start); visited[start]=True
while q:
    y,x=q.popleft()
    for dy,dx in [(1,0),(-1,0),(0,1),(0,-1)]:
        ny,nx=y+dy,x+dx
        if 0<=ny<h and 0<=nx<w and seg_without[ny,nx] and not visited[ny,nx]:
            visited[ny,nx]=True; q.append((ny,nx))
reach_right=visited[:, -10:].any()
print('reach_right?',reach_right)
