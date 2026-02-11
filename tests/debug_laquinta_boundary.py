"""Debug La Quinta boundary detection"""
import cv2
import numpy as np
from pathlib import Path

# Load images
raw_path = Path("data/raw_images/La_Quinta_Inn_by_Wyndham_Dallas_Uptown_4440_N_Central_Expy_Dallas_TX_75206_United_States.png")
adjusted_path = Path("data/adjusted_images/La_Quinta_Inn_by_Wyndham_Dallas_Uptown_4440_N_Central_Expy_Dallas_TX_75206_United_States.png")

print("=== RAW IMAGE ===")
if raw_path.exists():
    img_raw = cv2.imread(str(raw_path))
    print(f"Shape: {img_raw.shape}")
    
    # Check for red pixels
    b, g, r = cv2.split(img_raw)
    red_mask = (r > 200) & (g < 100) & (b < 100)
    red_count = np.sum(red_mask)
    print(f"Red pixels: {red_count} ({red_count/(img_raw.shape[0]*img_raw.shape[1])*100:.2f}%)")
    
    # Show where red pixels are
    if red_count > 0:
        y_coords, x_coords = np.where(red_mask)
        print(f"Red pixel bounds: x=[{x_coords.min()}, {x_coords.max()}], y=[{y_coords.min()}, {y_coords.max()}]")
        print(f"Red region size: {x_coords.max() - x_coords.min()} x {y_coords.max() - y_coords.min()}")
else:
    print("Raw image not found")

print("\n=== ADJUSTED IMAGE ===")
if adjusted_path.exists():
    img_adj = cv2.imread(str(adjusted_path))
    print(f"Shape: {img_adj.shape}")
    
    # Check for red pixels
    b, g, r = cv2.split(img_adj)
    red_mask = (r > 200) & (g < 100) & (b < 100)
    red_count = np.sum(red_mask)
    print(f"Red pixels: {red_count} ({red_count/(img_adj.shape[0]*img_adj.shape[1])*100:.2f}%)")
    
    # Show where red pixels are
    if red_count > 0:
        y_coords, x_coords = np.where(red_mask)
        print(f"Red pixel bounds: x=[{x_coords.min()}, {x_coords.max()}], y=[{y_coords.min()}, {y_coords.max()}]")
        print(f"Red region size: {x_coords.max() - x_coords.min()} x {y_coords.max() - y_coords.min()}")
        
        # Visualize red mask
        vis = img_adj.copy()
        vis[red_mask] = [0, 255, 255]  # Highlight red in yellow
        cv2.imwrite("data/debug/laquinta_red_pixels.png", vis)
        print("Red pixel visualization saved to data/debug/laquinta_red_pixels.png")
else:
    print("Adjusted image not found")

print("\n=== CHECKING DEBUG IMAGE ===")
debug_path = Path("data/debug/La_Quinta_Inn_by_Wyndham_Dallas_Uptown_4440_N_Central_Expy_Dallas_TX_75206_United_States_boundary_debug.png")
if debug_path.exists():
    print(f"Debug image exists: {debug_path}")
else:
    print("No debug image found")
