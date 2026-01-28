from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import cv2
import numpy as np
from app.core.boundary import _red_mask_hsv

p = list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*Drey*'))
if not p:
    raise SystemExit('No Drey image found')
img_path = p[0]
img = cv2.imread(str(img_path))
print(f'Using image: {img_path}')

# Compute cleaned mask same as diagnostic
red_mask = _red_mask_hsv(img)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
mask_cleaned = np.zeros_like(red_mask)
if num_labels > 1:
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    largest_idx = np.argmax(areas) + 1
    largest_area = areas[largest_idx - 1]
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if i == largest_idx or (area >= largest_area * 0.1 and area >= 50):
            mask_cleaned[labels == i] = 255

# Try skimage skeletonize
use_skimage = True
try:
    from skimage.morphology import skeletonize
except Exception:
    use_skimage = False
    print('skimage not available, falling back to simple thinning')

if use_skimage:
    skel = skeletonize(mask_cleaned > 0)
    skel = (skel.astype(np.uint8) * 255)
else:
    # fallback: iterative hit-or-miss style thinning (simple)
    skel = mask_cleaned.copy()
    prev = np.zeros_like(skel)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    for i in range(1000):
        eroded = cv2.erode(skel, kernel, iterations=1)
        temp = cv2.dilate(eroded, kernel, iterations=1)
        skel_next = skel - (skel & (skel - temp))
        if np.array_equal(skel_next, skel):
            break
        skel = skel_next
    skel = (skel > 0).astype(np.uint8) * 255

cv2.imwrite('data/debug/drey_skeleton.png', skel)
print('Saved skeleton: data/debug/drey_skeleton.png')

# Find connected components of skeleton
nl, labs, stats, _ = cv2.connectedComponentsWithStats((skel>0).astype(np.uint8), connectivity=8)
print(f'Skeleton components: {nl-1}')

# Find endpoints on skeleton (8-connectivity neighbors count ==1)
kernel8 = np.ones((3,3), np.uint8); kernel8[1,1]=0
neighbor_count = cv2.filter2D((skel>0).astype(np.uint8), -1, kernel8)
endpoints = np.column_stack(np.where((skel>0) & (neighbor_count==1)))
print(f'Found {len(endpoints)} skeleton endpoints')

# Group endpoints by component
endpoint_groups = {}
for (r,c) in endpoints:
    lbl = labs[r,c]
    endpoint_groups.setdefault(lbl, []).append((r,c))

# Prepare visualization: color each skeleton component
h, w = skel.shape
vis = np.zeros((h,w,3), dtype=np.uint8)
colors = {}
for lbl in range(1, nl):
    cnt = (labs==lbl).astype(np.uint8)*255
    pts = np.column_stack(np.where(cnt>0))
    if pts.size==0: continue
    hue = int(180.0 * (lbl%8) / 8)
    col = cv2.cvtColor(np.uint8([[[hue,200,200]]]), cv2.COLOR_HSV2BGR)[0,0].tolist()
    colors[lbl] = tuple(int(x) for x in col)
    # draw skeleton points
    ys, xs = pts[:,0], pts[:,1]
    vis[ys, xs] = colors[lbl]

# draw endpoints with numbers
ordered_pts = []
num=1
for lbl, group in endpoint_groups.items():
    for (r,c) in group:
        cv2.circle(vis, (int(c), int(r)), 6, (255,255,255), -1)
        cv2.putText(vis, str(num), (int(c)+8,int(r)+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        ordered_pts.append(((r,c), lbl))
        num+=1

cv2.imwrite('data/debug/drey_skeleton_vis.png', vis)
print('Saved skeleton visualization: data/debug/drey_skeleton_vis.png')

# Now iteratively connect nearest endpoint pairs across different components until skeleton becomes one component
pairs = []
for i, (p0,l0) in enumerate(ordered_pts):
    for j, (p1,l1) in enumerate(ordered_pts):
        if j<=i: continue
        if l0==l1: continue
        r0,c0 = p0; r1,c1 = p1
        d = np.hypot(r0-r1,c0-c1)
        pairs.append((d, (r0,c0),(r1,c1), l0, l1))
pairs.sort(key=lambda x: x[0])

mask_join = mask_cleaned.copy()
joined = []
for k,(d,p0,p1,l0,l1) in enumerate(pairs[:200]):
    if d>500: break
    pt1=(int(p0[1]),int(p0[0])); pt2=(int(p1[1]),int(p1[0]))
    cv2.line(mask_join, pt1, pt2, 255, 1)
    joined.append((pt1,pt2,d))
    # check interior
    filled = mask_join.copy()
    cv2.floodFill(filled, None, (0,0), 128)
    interior = (filled==0).astype(np.uint8)*255
    interior_area = np.count_nonzero(interior)
    print(f'Applied {len(joined)} joins, interior area={interior_area}')
    if interior_area>1000:
        print('Interior achieved')
        break

cv2.imwrite('data/debug/drey_skeleton_joins.png', mask_join)
print('Saved skeleton-join mask: data/debug/drey_skeleton_joins.png')

if interior_area>1000:
    # save border
    contours_i, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_i:
        largest = max(contours_i, key=cv2.contourArea)
        eps = max(1.0, 0.002*cv2.arcLength(largest, True))
        approx = cv2.approxPolyDP(largest, eps, True).reshape(-1,2)
        out_vis = cv2.imread(str(img_path))
        cv2.polylines(out_vis, [approx], True, (0,255,255), 3)
        cv2.imwrite('data/debug/drey_skeleton_joined_border.png', out_vis)
        print('Saved skeleton-joined border: data/debug/drey_skeleton_joined_border.png')
else:
    print('No interior formed by skeleton-based joins')