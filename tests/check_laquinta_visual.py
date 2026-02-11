"""Visualize La Quinta red boundary to understand the issue"""
import cv2
import numpy as np
from pathlib import Path

img_path = Path("data/raw_images/La_Quinta_Inn_by_Wyndham_Dallas_Uptown_4440_N_Central_Expy_Dallas_TX_75206_United_States.png")
img = cv2.imread(str(img_path))

# Find red pixels
b, g, r = cv2.split(img)
red_mask = (r > 200) & (g < 100) & (b < 100)

# Get bounds
y_coords, x_coords = np.where(red_mask)
print(f"Image size: {img.shape[1]} x {img.shape[0]}")
print(f"Red pixels: {len(x_coords)}")
print(f"Red bounds: x=[{x_coords.min()}, {x_coords.max()}] y=[{y_coords.min()}, {y_coords.max()}]")
print(f"Red region: {x_coords.max() - x_coords.min()} x {y_coords.max() - y_coords.min()}")
print(f"Red region covers: {(x_coords.max() - x_coords.min())/img.shape[1]:.1%} width, {(y_coords.max() - y_coords.min())/img.shape[0]:.1%} height")

# Create visualization showing red overlay in context
vis = img.copy()
# Highlight red pixels in bright cyan
vis[red_mask] = [255, 255, 0]

# Draw bounding box around red region
cv2.rectangle(vis, (x_coords.min(), y_coords.min()), (x_coords.max(), y_coords.max()), (0, 255, 0), 3)

output_path = Path("data/debug/laquinta_red_context.png")
cv2.imwrite(str(output_path), vis)
print(f"\nVisualization saved to: {output_path}")

# Also create a zoomed-in view of just the red region
margin = 50
x1 = max(0, x_coords.min() - margin)
x2 = min(img.shape[1], x_coords.max() + margin)
y1 = max(0, y_coords.min() - margin)
y2 = min(img.shape[0], y_coords.max() + margin)

zoomed = img[y1:y2, x1:x2]
zoomed_vis = zoomed.copy()
zoomed_red_mask = red_mask[y1:y2, x1:x2]
zoomed_vis[zoomed_red_mask] = [255, 255, 0]

zoom_output = Path("data/debug/laquinta_red_zoomed.png")
cv2.imwrite(str(zoom_output), zoomed_vis)
print(f"Zoomed view saved to: {zoom_output}")
