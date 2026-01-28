import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.boundary import estimate_boundary_from_overlay
import cv2
import numpy as np

img_path = list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*La_Quinta*.png'))[0]
img = cv2.imread(str(img_path))
result = estimate_boundary_from_overlay(img)
poly = np.array(result.polygon_px, dtype=np.int32)

print(f'Polygon: {len(poly)} vertices')
print(f'Vertices:\n{poly}')

# Create a mask and fill the polygon
mask = np.zeros(img.shape[:2], dtype=np.uint8)

# Check polygon area (should be positive for counter-clockwise)
area = cv2.contourArea(poly)
print(f'Polygon area: {area}')
if area < 0:
    poly = poly[::-1]  # Reverse winding
    print('Reversed polygon winding')

cv2.fillPoly(mask, [poly], 255)

# Alternative: try drawing contours
# mask = np.zeros(img.shape[:2], dtype=np.uint8)
# cv2.drawContours(mask, [poly], 0, 255, -1)  # -1 means filled

filled_area = np.count_nonzero(mask)
total_area = mask.shape[0] * mask.shape[1]

print(f'\nFilled polygon area: {filled_area} pixels ({filled_area/total_area:.1%} of image)')
print(f'Expected ~45% of image = {total_area * 0.45:.0f} pixels')

# Save mask for inspection (fallback to interior mask)
# estimate_boundary_from_overlay does not return metadata in BoundaryOutput; write interior mask saved in boundary module instead if available
try:
    # If boundary module wrote debug masks, preserve them
    import os
    if os.path.exists('data/debug/laquinta_interior_mask.png'):
        cv2.imwrite('data/debug/laquinta_mask_after_morphology.png', cv2.imread('data/debug/laquinta_interior_mask.png'))
    else:
        # Save blank mask to indicate missing debug mask
        cv2.imwrite('data/debug/laquinta_mask_after_morphology.png', np.zeros_like(img[:,:,0]))
except Exception:
    cv2.imwrite('data/debug/laquinta_mask_after_morphology.png', np.zeros_like(img[:,:,0]))

# Save visualization (filled mask with polygon overlay)
vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
cv2.polylines(vis, [poly], True, (0, 255, 255), 2)
cv2.imwrite('data/debug/laquinta_filled_polygon.png', vis)
print(f'\nFilled visualization saved to: data/debug/laquinta_filled_polygon.png')

# Also save original image with only the polygon border (no fill)
orig_vis = img.copy()
cv2.polylines(orig_vis, [poly], True, (0, 255, 255), 3)
cv2.imwrite('data/debug/laquinta_border_only.png', orig_vis)
print(f'Border-only visualization saved to: data/debug/laquinta_border_only.png')
