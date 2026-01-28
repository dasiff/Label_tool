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
cv2.imwrite('data/debug/drey_mask_cleaned_for_join.png', mask_cleaned)

contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f'Found {len(contours)} contours')
if len(contours) < 2:
    print('Less than 2 contours; nothing to join')
    raise SystemExit()

# Compute endpoints (4-connectivity)
kernel_4 = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.uint8)
neighbor_count = cv2.filter2D((mask_cleaned > 0).astype(np.uint8), -1, kernel_4)
neighbor_count = neighbor_count * (mask_cleaned > 0)
endpoints = np.column_stack(np.where(neighbor_count == 1))  # rows (y), cols (x)

cnt_pts = [c.reshape(-1,2) for c in contours[:2]]
# assign endpoints to closest contour
assign = {0: [], 1: []}
for (r, c) in endpoints:
    d0 = np.min(np.linalg.norm(cnt_pts[0] - np.array([c, r]), axis=1))
    d1 = np.min(np.linalg.norm(cnt_pts[1] - np.array([c, r]), axis=1))
    if d0 <= d1:
        assign[0].append((int(r), int(c)))
    else:
        assign[1].append((int(r), int(c)))

# Fallback to sampling points if missing
if len(assign[0]) == 0:
    s = cnt_pts[0][::max(1, len(cnt_pts[0])//100)]
    assign[0] = [(int(pt[1]), int(pt[0])) for pt in s[:10]]
if len(assign[1]) == 0:
    s = cnt_pts[1][::max(1, len(cnt_pts[1])//100)]
    assign[1] = [(int(pt[1]), int(pt[0])) for pt in s[:10]]

pairs = []
for p0 in assign[0]:
    for p1 in assign[1]:
        d = np.hypot(p0[0]-p1[0], p0[1]-p1[1])
        pairs.append((d, p0, p1))
pairs.sort(key=lambda x: x[0])

# Draw candidate joins (top 10)
cand_vis = img.copy()
for i, (d, p0, p1) in enumerate(pairs[:10], start=1):
    pt1 = (int(p0[1]), int(p0[0]))
    pt2 = (int(p1[1]), int(p1[0]))
    color = tuple(int(c) for c in np.random.randint(50,255,size=3).tolist())
    cv2.line(cand_vis, pt1, pt2, color, 2)
    cv2.circle(cand_vis, pt1, 6, color, -1)
    cv2.circle(cand_vis, pt2, 6, color, -1)
    cv2.putText(cand_vis, str(i), (pt1[0]+8, pt1[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

cv2.imwrite('data/debug/drey_candidate_joins.png', cand_vis)
print('Saved candidate joins: data/debug/drey_candidate_joins.png')

# Iteratively apply closest joins until we get a valid interior or run out of reasonable pairs
mask_joined = mask_cleaned.copy()
interior_found = False
applied = []
max_attempts = min(50, len(pairs))
for idx in range(max_attempts):
    d, p0, p1 = pairs[idx]
    pt1 = (int(p0[1]), int(p0[0]))
    pt2 = (int(p1[1]), int(p1[0]))
    cv2.line(mask_joined, pt1, pt2, 255, 1)
    applied.append((pt1, pt2, d))
    # check connectivity / interior
    filled = mask_joined.copy()
    h, w = filled.shape
    cv2.floodFill(filled, None, (0,0), 128)
    interior = (filled == 0).astype(np.uint8) * 255
    interior_area = np.count_nonzero(interior)
    print(f"DEBUG: After applying {len(applied)} joins, interior area={interior_area}")
    cv2.imwrite(f'data/debug/drey_mask_joined_{len(applied)}.png', mask_joined)
    cv2.imwrite(f'data/debug/drey_interior_after_join_{len(applied)}.png', interior)
    if interior_area > 1000:
        interior_found = True
        print(f"DEBUG: Interior found after {len(applied)} joins")
        break

# Save the final joined mask
cv2.imwrite('data/debug/drey_mask_joined.png', mask_joined)
print(f'Saved final joined mask with {len(applied)} joins: data/debug/drey_mask_joined.png')

# If interior found, extract polygon
if interior_found:
    contours_i, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f'Interior contours after joins: {len(contours_i)}')
    if contours_i:
        largest = max(contours_i, key=cv2.contourArea)
        # Simplify slightly
        eps = max(1.0, 0.002 * cv2.arcLength(largest, True))
        approx = cv2.approxPolyDP(largest, eps, True).reshape(-1,2)
        out_vis = img.copy()
        cv2.polylines(out_vis, [approx], True, (0,255,255), 3)
        cv2.imwrite('data/debug/drey_joined_border.png', out_vis)
        print('Saved joined border: data/debug/drey_joined_border.png')
else:
    print('No interior found after candidate joins; consider skeleton/graph closure or manual review')
