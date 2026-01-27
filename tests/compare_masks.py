import cv2
import numpy as np

mask1 = cv2.imread('data/debug/step3_thickened.png', 0)
mask2 = cv2.imread('data/debug/mask_before_flood.png', 0)

print(f"Standalone script mask: {np.count_nonzero(mask1)} white pixels")
print(f"boundary.py mask: {np.count_nonzero(mask2)} white pixels")
print(f"Difference: {np.count_nonzero(mask1) - np.count_nonzero(mask2)} pixels")

# Check shapes
print(f"\nShapes: {mask1.shape} vs {mask2.shape}")

# XOR to find differences
diff = cv2.bitwise_xor(mask1, mask2)
print(f"Different pixels: {np.count_nonzero(diff)}")

if np.count_nonzero(diff) > 0:
    cv2.imwrite('data/debug/mask_xor_diff.png', diff)
    print("Saved XOR difference to mask_xor_diff.png")
    
    # Find where differences are
    y_coords, x_coords = np.where(diff > 0)
    print(f"Difference locations: x=[{x_coords.min()}..{x_coords.max()}], y=[{y_coords.min()}..{y_coords.max()}]")
