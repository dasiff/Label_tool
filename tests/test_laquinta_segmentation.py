"""Test La Quinta segmentation to debug issues"""
import cv2
import numpy as np
from pathlib import Path
from app.core.boundary import estimate_boundary_from_overlay
from skimage.segmentation import felzenszwalb

# Load La Quinta image
img_path = Path("data/raw_images/La_Quinta_Inn_by_Wyndham_Dallas_Uptown_4440_N_Central_Expy_Dallas_TX_75206_United_States.png")
img = cv2.imread(str(img_path))
print(f"Image shape: {img.shape}")
print(f"Image dtype: {img.dtype}")

# Test boundary detection
result = estimate_boundary_from_overlay(img)
print(f"\nBoundary detected: {result.polygon_px is not None}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Warnings: {result.warnings}")
if result.polygon_px:
    print(f"Vertices: {len(result.polygon_px)}")
    print(f"Polygon area: {cv2.contourArea(np.array(result.polygon_px, dtype=np.float32)):.0f} pixels")
    boundary = np.array(result.polygon_px)
    
    # Create mask
    h, w = img.shape[:2]
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [boundary.astype(np.int32)], 255)
    roi_area = np.sum(roi_mask > 0)
    print(f"ROI area: {roi_area} pixels ({roi_area/(h*w)*100:.1f}% of image)")
    
    # Test segmentation parameters
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Downsample
    scale_factor = 0.5
    small_h, small_w = int(h * scale_factor), int(w * scale_factor)
    rgb_small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    
    # Enhance
    bgr_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
    l, a, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    l_normalized = l_enhanced.astype(float) / 255.0
    a_normalized = (a.astype(float) - 128) / 128.0
    b_normalized = (b_channel.astype(float) - 128) / 128.0
    
    feature_img = np.stack([
        l_normalized * 0.3,
        a_normalized * 1.2,
        b_normalized * 1.2
    ], axis=-1)
    
    feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
    rgb_enhanced = (feature_img * 255).astype(np.uint8)
    
    # ROI mask
    boundary_small = boundary * scale_factor
    roi_mask_small = np.zeros((small_h, small_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask_small, [boundary_small.astype(np.int32)], 255)
    roi_area_small = np.sum(roi_mask_small > 0)
    
    target_segments = 50
    scale = max(int(roi_area_small / target_segments * 2), 50)
    min_size = max(1000, int(roi_area_small * 0.01))
    
    print(f"\nSegmentation parameters:")
    print(f"  scale: {scale}")
    print(f"  min_size: {min_size}")
    print(f"  roi_area_small: {roi_area_small}")
    print(f"  rgb_enhanced shape: {rgb_enhanced.shape}")
    print(f"  rgb_enhanced dtype: {rgb_enhanced.dtype}")
    print(f"  rgb_enhanced min/max: {rgb_enhanced.min()}/{rgb_enhanced.max()}")
    
    # Run segmentation
    print("\nRunning felzenszwalb segmentation...")
    segments_small = felzenszwalb(rgb_enhanced, scale=scale, sigma=0, min_size=min_size)
    print(f"Initial segments: {segments_small.max()}")
    
    # Count segments inside ROI
    segments_in_roi = segments_small * (roi_mask_small > 0)
    unique_in_roi = np.unique(segments_in_roi)
    unique_in_roi = unique_in_roi[unique_in_roi > 0]  # Remove background
    print(f"Segments in ROI: {len(unique_in_roi)}")
    
    # Upscale
    segments_full = cv2.resize(segments_small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Filter and renumber
    segments = np.zeros_like(segments_full)
    new_id = 1
    for old_id in np.unique(segments_full):
        seg_mask = (segments_full == old_id) & (roi_mask > 0)
        if seg_mask.sum() >= 100:
            segments[seg_mask] = new_id
            new_id += 1
    
    print(f"Final segments: {segments.max()}")
    
    # Visualize
    output_path = Path("data/debug/laquinta_segmentation_test.png")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Create visualization
    vis = img.copy()
    
    # Draw segments with random colors
    np.random.seed(42)
    for seg_id in range(1, segments.max() + 1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        mask = segments == seg_id
        vis[mask] = vis[mask] * 0.5 + np.array(color) * 0.5
    
    # Draw boundary
    cv2.polylines(vis, [boundary.astype(np.int32)], True, (0, 255, 255), 3)
    
    cv2.imwrite(str(output_path), vis)
    print(f"\nVisualization saved to: {output_path}")
else:
    print("\nNo boundary detected - cannot segment!")
