"""Check if the two contours form inner/outer boundary of same region"""
import cv2
import numpy as np
from pathlib import Path

img_path = Path("data/raw_images/La_Quinta_Inn_by_Wyndham_Dallas_Uptown_4440_N_Central_Expy_Dallas_TX_75206_United_States.png")
img = cv2.imread(str(img_path))

# Find red pixels
b, g, r = cv2.split(img)
red_mask = (r > 200) & (g < 100) & (b < 100)

# Get contours
contours, _ = cv2.findContours(red_mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get bounding boxes
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    print(f"Contour {i}: bbox=({x},{y},{w},{h}), area={area:.0f}")
    
    # Check if this is a thin outline or filled region
    bbox_area = w * h
    fill_ratio = area / bbox_area if bbox_area > 0 else 0
    print(f"  Fill ratio: {fill_ratio:.3f} (1.0 = filled rectangle, <0.1 = thin outline)")

# Combine both contours and measure enclosed area
combined_mask = np.zeros_like(red_mask, dtype=np.uint8)
cv2.drawContours(combined_mask, contours, -1, 255, thickness=cv2.FILLED)
enclosed_area = np.sum(combined_mask > 0)
print(f"\nTotal area enclosed by red contours: {enclosed_area} ({enclosed_area/(img.shape[0]*img.shape[1])*100:.1f}%)")

# Save visualization
vis = img.copy()
cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)
cv2.fillPoly(vis, contours, (0, 255, 255), lineType=cv2.LINE_AA)
cv2.addWeighted(img, 0.7, vis, 0.3, 0, vis)

cv2.imwrite("data/debug/laquinta_filled_analysis.png", vis)
print("Saved filled visualization")
