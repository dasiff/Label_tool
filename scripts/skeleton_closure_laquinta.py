from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import cv2
import numpy as np
from app.core.boundary import _red_mask_hsv

p = list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*La_Quinta*'))
if not p:
    raise SystemExit('No La Quinta image found')
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
cv2.imwrite('data/debug/laquinta_mask_cleaned.png', mask_cleaned)
print('Saved cleaned mask: data/debug/laquinta_mask_cleaned.png')

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
    # fallback: iterative thinning (simple)
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

cv2.imwrite('data/debug/laquinta_skeleton.png', skel)
print('Saved skeleton: data/debug/laquinta_skeleton.png')

# Find skeleton endpoints
kernel8 = np.ones((3,3), np.uint8); kernel8[1,1]=0
neighbor_count = cv2.filter2D((skel>0).astype(np.uint8), -1, kernel8)
endpoints = np.column_stack(np.where((skel>0) & (neighbor_count==1)))
print(f'Found {len(endpoints)} skeleton endpoints')

# Prepare visualization
h, w = skel.shape
vis = np.zeros((h,w,3), dtype=np.uint8)
ys, xs = np.where(skel>0)
vis[ys, xs] = (255,255,255)
for idx, (r,c) in enumerate(endpoints, start=1):
    cv2.circle(vis, (int(c), int(r)), 8, (0,0,255), -1)
    cv2.putText(vis, str(idx), (int(c)+10,int(r)+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.imwrite('data/debug/laquinta_skeleton_vis.png', vis)
print('Saved skeleton visualization: data/debug/laquinta_skeleton_vis.png')

# Connect nearest endpoint pairs iteratively until interior appears
pairs = []
for i,(r0,c0) in enumerate(endpoints):
    for j,(r1,c1) in enumerate(endpoints):
        if j<=i: continue
        d = np.hypot(r0-r1, c0-c1)
        pairs.append((d,(r0,c0),(r1,c1)))
pairs.sort(key=lambda x: x[0])

mask_join = mask_cleaned.copy()
joined = []
interior_area = 0
def _is_connected(binary_mask, pt1, pt2):
    # Check for a path of non-zero pixels between pt1 and pt2 using BFS within bbox
    h, w = binary_mask.shape
    x1, y1 = pt1; x2, y2 = pt2
    xmin = max(0, min(x1, x2) - 5); xmax = min(w-1, max(x1, x2) + 5)
    ymin = max(0, min(y1, y2) - 5); ymax = min(h-1, max(y1, y2) + 5)
    sub = (binary_mask[ymin:ymax+1, xmin:xmax+1] > 0).astype(np.uint8)
    start = (y1 - ymin, x1 - xmin)
    goal = (y2 - ymin, x2 - xmin)
    from collections import deque
    q = deque([start])
    seen = set([start])
    while q:
        y,x = q.popleft()
        if (y,x) == goal:
            return True
        for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = y+dy, x+dx
            if 0 <= ny < sub.shape[0] and 0 <= nx < sub.shape[1] and sub[ny,nx] and (ny,nx) not in seen:
                seen.add((ny,nx)); q.append((ny,nx))
    return False

for k,(d,p0,p1) in enumerate(pairs):
    if d>1000: break
    pt1=(int(p0[1]),int(p0[0])); pt2=(int(p1[1]),int(p1[0]))

    # Try drawing a join, ensure it creates a continuous connection; increase thickness if needed
    thickness = 1
    max_thickness = 11
    connected = False
    while thickness <= max_thickness:
        # draw on a temporary mask
        temp = mask_join.copy()
        cv2.line(temp, pt1, pt2, 255, thickness)
        # also mark small circles at endpoints to be safe
        cv2.circle(temp, pt1, max(1, thickness//2), 255, -1)
        cv2.circle(temp, pt2, max(1, thickness//2), 255, -1)
        if _is_connected(temp, pt1, pt2):
            # accept this join
            cv2.line(mask_join, pt1, pt2, 255, thickness)
            cv2.circle(mask_join, pt1, max(1, thickness//2), 255, -1)
            cv2.circle(mask_join, pt2, max(1, thickness//2), 255, -1)
            joined.append((pt1,pt2,d,thickness))
            connected = True
            print(f"DEBUG: Drew join with thickness={thickness} connecting {pt1} <-> {pt2} (dist {d:.1f}px)")
            break
        thickness += 2
    if not connected:
        print(f"DEBUG: Failed to create continuous join between {pt1} and {pt2} up to thickness {max_thickness}")

    filled = mask_join.copy()
    cv2.floodFill(filled, None, (0,0), 128)
    interior = (filled==0).astype(np.uint8)*255
    interior_area = np.count_nonzero(interior)
    print(f'Applied {len(joined)} joins, interior area={interior_area}')
    cv2.imwrite(f'data/debug/laquinta_mask_joined_{len(joined)}.png', mask_join)
    if interior_area>1000:
        print('Interior achieved')
        break

cv2.imwrite('data/debug/laquinta_skeleton_joins.png', mask_join)
print('Saved skeleton-join mask: data/debug/laquinta_skeleton_joins.png')

if interior_area>1000:
    contours_i, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours_i:
        largest = max(contours_i, key=cv2.contourArea)
        eps = max(1.0, 0.002*cv2.arcLength(largest, True))
        approx = cv2.approxPolyDP(largest, eps, True).reshape(-1,2)
        out_vis = img.copy()
        cv2.polylines(out_vis, [approx], True, (0,255,255), 3)
        cv2.imwrite('data/debug/laquinta_skeleton_joined_border.png', out_vis)
        print('Saved skeleton-joined border: data/debug/laquinta_skeleton_joined_border.png')
else:
    print('No interior formed by skeleton-based joins')