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

contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(f'Found {len(contours)} contours')

h, w = mask_cleaned.shape
out = np.zeros((h, w, 3), dtype=np.uint8)

# Generate colors
for i, cnt in enumerate(contours):
    hue = int(180.0 * i / max(1, len(contours)))
    col = cv2.cvtColor(np.uint8([[[hue, 200, 200]]]), cv2.COLOR_HSV2BGR)[0,0].tolist()
    color = (int(col[0]), int(col[1]), int(col[2]))
    # draw contour
    cv2.drawContours(out, contours, i, color, 2)
    # label at centroid
    M = cv2.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.putText(out, str(i+1), (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

out_path = Path('data/debug/drey_contour_segments.png')
out_path.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_path), out)
print(f'Saved contour segments to {out_path}')
