from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import cv2
import numpy as np
from app.core.boundary import estimate_boundary_from_overlay

# Find Drey Hotel image
p = list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*Drey*'))
if not p:
    p = list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*Drey_Hotel*'))
if not p:
    raise SystemExit('No Drey image found')
img_path = p[0]
print(f'Using image: {img_path}')

img = cv2.imread(str(img_path))
result = estimate_boundary_from_overlay(img)
poly = result.polygon_px
if poly:
    poly_np = np.array(poly, dtype=np.int32)
    print(f'Polygon vertices: {len(poly_np)}')

    # Save border-only visualization
    orig_vis = img.copy()
    cv2.polylines(orig_vis, [poly_np], True, (0, 255, 255), 3)
    out_path = Path('data/debug/drey_border_only.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), orig_vis)
    print(f'Saved border-only: {out_path}')

    # Also save filled visualization
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [poly_np], 255)
    vis_filled = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis_filled, [poly_np], True, (0, 255, 255), 2)
    cv2.imwrite('data/debug/drey_filled_polygon.png', vis_filled)
    print('Saved filled visualization: data/debug/drey_filled_polygon.png')

    # Print area
    filled_area = np.count_nonzero(mask)
    total_area = mask.shape[0] * mask.shape[1]
    print(f'Filled area: {filled_area} px ({filled_area/total_area:.1%})')
else:
    print('No polygon returned; saving diagnostic overlays')
    # Compute raw red mask and diagnostics
    from app.core.boundary import _red_mask_hsv
    red_mask = _red_mask_hsv(img)
    cv2.imwrite('data/debug/drey_red_mask.png', red_mask)

    # Remove tiny components like boundary module
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
    cv2.imwrite('data/debug/drey_mask_cleaned.png', mask_cleaned)

    # Find endpoints (4-connectivity)
    kernel_4 = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.uint8)
    neighbor_count = cv2.filter2D((mask_cleaned > 0).astype(np.uint8), -1, kernel_4)
    neighbor_count = neighbor_count * (mask_cleaned > 0)
    endpoints = np.column_stack(np.where(neighbor_count == 1))
    print(f'Endpoints found: {len(endpoints)}')

    # Draw endpoints on image
    diag_vis = img.copy()
    for (r, c) in endpoints:
        cv2.circle(diag_vis, (int(c), int(r)), 5, (0, 0, 255), -1)
    cv2.imwrite('data/debug/drey_endpoints.png', diag_vis)
    print('Saved diagnostics: dre y_red_mask.png, dre y_mask_cleaned.png, dre y_endpoints.png')
