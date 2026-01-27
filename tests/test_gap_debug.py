import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

# Load La Quinta image
img_path = list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*La_Quinta*.png'))[0]
img = cv2.imread(str(img_path))

# Red detection
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Basic cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Save original mask
cv2.imwrite('data/debug/step1_original_mask.png', mask)
print(f"Step 1: Original mask saved")

# Find contour and check for gap
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
largest = max(contours, key=cv2.contourArea)
peri = cv2.arcLength(largest, True)
approx = cv2.approxPolyDP(largest, 0.001 * peri, True).reshape(-1, 2)

# Find longest edge (gap)
max_edge_len = 0
max_edge_idx = -1
for i in range(len(approx) - 1):
    edge_len = np.linalg.norm(approx[i + 1] - approx[i])
    if edge_len > max_edge_len:
        max_edge_len = edge_len
        max_edge_idx = i

print(f"Longest edge: {max_edge_idx}→{max_edge_idx+1}, length: {max_edge_len:.1f}px")
print(f"Gap endpoints: {approx[max_edge_idx]} → {approx[max_edge_idx+1]}")

# Close the gap
mask_closed = mask.copy()
pt1, pt2 = approx[max_edge_idx], approx[max_edge_idx + 1]
cv2.line(mask_closed, tuple(pt1), tuple(pt2), 255, 20)
cv2.imwrite('data/debug/step2_gap_closed.png', mask_closed)
print(f"Step 2: Gap closed with 20px line")

# Thicken boundary
kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_thick = cv2.dilate(mask_closed, kernel_thick, iterations=1)
cv2.imwrite('data/debug/step3_thickened.png', mask_thick)
print(f"Step 3: Boundary thickened")

# Try flood fill
filled = mask_thick.copy()
x, y, w, h = cv2.boundingRect(largest)
seed = (x + w // 2, y + h // 2)
cv2.floodFill(filled, None, seed, 255)

filled_pixels = np.count_nonzero(filled)
total_pixels = filled.shape[0] * filled.shape[1]
print(f"Flood fill from {seed}: {filled_pixels} pixels ({filled_pixels/total_pixels:.1%})")

cv2.imwrite('data/debug/step4_flood_filled.png', filled)
print(f"Step 4: Flood fill result saved")

print(f"\nAll debug images saved to data/debug/")
