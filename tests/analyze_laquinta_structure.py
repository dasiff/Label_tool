"""Check if La Quinta actually has double-line boundary"""
import cv2
import numpy as np
from pathlib import Path

img_path = Path("data/raw_images/La_Quinta_Inn_by_Wyndham_Dallas_Uptown_4440_N_Central_Expy_Dallas_TX_75206_United_States.png")
img = cv2.imread(str(img_path))

# Find red pixels
b, g, r = cv2.split(img)
red_mask = (r > 200) & (g < 100) & (b < 100)

# Check the red mask structure
print(f"Total red pixels: {np.sum(red_mask)}")

# Use findContours with RETR_TREE to see hierarchy
contours, hierarchy = cv2.findContours(red_mask.astype(np.uint8) * 255, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nNumber of contours found: {len(contours)}")

for i, (contour, hier) in enumerate(zip(contours, hierarchy[0])):
    area = cv2.contourArea(contour)
    if area > 100:  # Only significant contours
        print(f"Contour {i}: area={area:.0f}, hierarchy={hier}")
        print(f"  - Next: {hier[0]}, Previous: {hier[1]}, First child: {hier[2]}, Parent: {hier[3]}")

# Visualize the contours with different colors
vis = img.copy()
if len(contours) > 0:
    # Draw outer contours in cyan
    for i, hier in enumerate(hierarchy[0]):
        if hier[3] == -1:  # No parent = outer contour
            cv2.drawContours(vis, contours, i, (255, 255, 0), 3)
            print(f"\nOuter contour {i}: area={cv2.contourArea(contours[i]):.0f}")
    
    # Draw inner contours in green
    for i, hier in enumerate(hierarchy[0]):
        if hier[3] != -1:  # Has parent = inner contour (hole)
            cv2.drawContours(vis, contours, i, (0, 255, 0), 3)
            print(f"Inner contour {i}: area={cv2.contourArea(contours[i]):.0f}, parent={hier[3]}")

cv2.imwrite("data/debug/laquinta_contour_hierarchy.png", vis)
print(f"\nVisualization saved")

# Also check: is the red boundary just a thin line, or a filled region?
# Sample a few points along the boundary
y_coords, x_coords = np.where(red_mask)
y_min, y_max = y_coords.min(), y_coords.max()
x_min, x_max = x_coords.min(), x_coords.max()

# Check horizontal cross-section at middle
y_mid = (y_min + y_max) // 2
horizontal_slice = red_mask[y_mid, x_min:x_max+1]
red_runs = []
in_run = False
run_start = 0
for i, val in enumerate(horizontal_slice):
    if val and not in_run:
        run_start = i
        in_run = True
    elif not val and in_run:
        red_runs.append(i - run_start)
        in_run = False
if in_run:
    red_runs.append(len(horizontal_slice) - run_start)

print(f"\nHorizontal cross-section at y={y_mid}:")
print(f"Red pixel runs: {red_runs}")
print(f"This tells us: {len(red_runs)} separate red segments across the width")
if len(red_runs) == 2:
    print(f"CONFIRMED: Double-line boundary (gap between lines)")
elif len(red_runs) == 1:
    print(f"Single filled boundary region")
