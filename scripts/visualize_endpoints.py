from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import cv2
import numpy as np
from app.core.boundary import _red_mask_hsv

# Find Drey image
p = list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*Drey*'))
if not p:
    raise SystemExit('No Drey image found')
img_path = p[0]
img = cv2.imread(str(img_path))
print(f'Using image: {img_path}')

# Compute red mask and clean small components (same logic as boundary)
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

# Find endpoints (4-connectivity)
kernel_4 = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.uint8)
neighbor_count = cv2.filter2D((mask_cleaned > 0).astype(np.uint8), -1, kernel_4)
neighbor_count = neighbor_count * (mask_cleaned > 0)
endpoints = np.column_stack(np.where(neighbor_count == 1))  # rows (y), cols (x)
print(f'Found {len(endpoints)} endpoints')

# Map endpoints to nearest contour points to get contour-based ordering
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f'DEBUG: Found {len(contours)} contours in cleaned mask')

endpoint_info = []  # list of (endpoint_idx, y, x, contour_id, contour_idx)
for ei, (r, c) in enumerate(endpoints):
    best_cid, best_idx, best_d = None, None, float('inf')
    for cid, cnt in enumerate(contours):
        cnt_pts = cnt.reshape(-1, 2)
        # compute distances to all points - sample every nth point if contour is huge
        step = max(1, len(cnt_pts) // 3000)
        sample = cnt_pts[::step]
        dists = np.linalg.norm(sample - np.array([c, r]), axis=1)
        minpos = np.argmin(dists)
        d = dists[minpos]
        # recover true index in original cnt_pts
        true_idx = minpos * step
        if d < best_d:
            best_d = d
            best_cid = cid
            best_idx = true_idx
    endpoint_info.append((ei, int(r), int(c), best_cid, int(best_idx)))

# Group endpoints by contour and sort by their index along the contour
from collections import defaultdict
grouped = defaultdict(list)
for info in endpoint_info:
    ei, r, c, cid, idx = info
    grouped[cid].append((idx, ei, r, c))
for cid in grouped:
    grouped[cid].sort(key=lambda x: x[0])
    print(f'DEBUG: Contour {cid} has {len(grouped[cid])} endpoints')

# Build overall ordering: traverse contour endpoints in order; between contours pick nearest unvisited endpoint
ordered = []
visited = set()
if len(endpoints) > 0:
    # pick starting endpoint: top-most (min y) among all endpoints
    start = min([(r, c, ei) for (ei, r, c, cid, idx) in endpoint_info], key=lambda x: (x[0], x[1]))
    cur_r, cur_c, cur_ei = start
    # find its contour and index in the group
    cur_cid = None
    for cid, lst in grouped.items():
        for j, (idx, ei, r, c) in enumerate(lst):
            if ei == cur_ei:
                cur_cid = cid
                cur_pos_in_group = j
                break
        if cur_cid is not None:
            break
    if cur_cid is None:
        # fallback ordering
        endpoints_ordered = endpoints[np.lexsort((endpoints[:,1], endpoints[:,0]))]
    else:
        # traverse
        # process current contour from cur_pos to end
        def process_contour_from(cid, start_pos):
            lst = grouped[cid]
            for k in range(start_pos, len(lst)):
                idx, ei, r, c = lst[k]
                if ei not in visited:
                    ordered.append((ei, r, c))
                    visited.add(ei)
        process_contour_from(cur_cid, cur_pos_in_group)
        # process remaining by nearest neighbor selection
        while len(visited) < len(endpoints):
            last_r, last_c = ordered[-1][1], ordered[-1][2]
            # find nearest unvisited endpoint
            min_d = float('inf')
            next_e = None
            for info in endpoint_info:
                ei, r, c, cid, idx = info
                if ei in visited:
                    continue
                d = np.hypot(r - last_r, c - last_c)
                if d < min_d:
                    min_d = d
                    next_e = (ei, r, c, cid, idx)
            if next_e is None:
                break
            # process its contour from that endpoint forward
            ei, r, c, cid, idx = next_e
            # find position in grouped[cid]
            lst = grouped[cid]
            pos = None
            for j, (ii, eii, rr, cc) in enumerate(lst):
                if eii == ei:
                    pos = j
                    break
            if pos is None:
                # just add it
                ordered.append((ei, r, c))
                visited.add(ei)
            else:
                process_contour_from(cid, pos)

        # convert ordered to coordinates
        endpoints_ordered = [ (r,c) for (ei,r,c) in ordered ]

# Color different contour segments differently and label endpoints
vis = img.copy()
mask_vis = cv2.cvtColor(mask_cleaned, cv2.COLOR_GRAY2BGR)

# Generate distinct colors using HSV wheel
num_contours = len(contours)
colors = []
for i in range(num_contours):
    hue = int(180.0 * i / max(1, num_contours))
    col = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0,0].tolist()
    colors.append((int(col[0]), int(col[1]), int(col[2])))

# Draw each contour in its color
for cid, cnt in enumerate(contours):
    color = colors[cid]
    cv2.drawContours(vis, contours, cid, color, 3)
    cv2.drawContours(mask_vis, contours, cid, color, 2)

# Draw numbered endpoints with the color of their contour
ordered_with_info = []
for num, (r,c) in enumerate(endpoints_ordered, start=1):
    ordered_with_info.append((num, r, c))

for num, r, c in ordered_with_info:
    # find which contour owns this endpoint (closest)
    best_cid, best_d = None, float('inf')
    for cid, cnt in enumerate(contours):
        pts = cnt.reshape(-1,2)
        d = np.min(np.hypot(pts[:,1]-r, pts[:,0]-c))
        if d < best_d:
            best_d = d
            best_cid = cid
    color = colors[best_cid] if best_cid is not None else (0,0,255)
    x, y = int(c), int(r)
    # outer ring
    cv2.circle(vis, (x, y), 18, (0,0,0), -1)
    cv2.circle(vis, (x, y), 14, color, -1)
    # number with outline
    text = str(num)
    text_pos = (x + 20, y + 8)
    cv2.putText(vis, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(vis, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    # small marker on mask_vis
    cv2.circle(mask_vis, (x, y), 8, color, -1)
    cv2.putText(mask_vis, str(num), (x+10, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

out_dir = Path('data/debug')
cv2.imwrite(str(out_dir / 'drey_endpoints_numbered_colored.png'), vis)
cv2.imwrite(str(out_dir / 'drey_mask_with_endpoints_numbered_colored.png'), mask_vis)
print('Saved: data/debug/drey_endpoints_numbered_colored.png and data/debug/drey_mask_with_endpoints_numbered_colored.png')
out_dir = Path('data/debug')
out_dir.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_dir / 'drey_endpoints_numbered.png'), vis)
cv2.imwrite(str(out_dir / 'drey_mask_with_endpoints_numbered.png'), mask_vis)
print('Saved: data/debug/drey_endpoints_numbered.png and dre y_mask_with_endpoints_numbered.png')
