"""
Debug script to visualize boundary detection and compare with red overlay.
Shows side-by-side comparison of:
1. Original image with red overlay
2. Detected red mask
3. Detected boundary overlaid on original
4. All three for comparison

Usage:
    python scripts/debug_boundary_detection.py "path/to/image.png"
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.boundary import estimate_boundary_from_overlay, _red_mask_hsv


def debug_boundary_detection(image_path: Path, save_output: bool = True):
    """
    Visualize boundary detection process step by step.
    
    Args:
        image_path: Path to image file with red overlay
        save_output: If True, save debug visualization to disk
    """
    # Read image
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    
    print(f"Image: {image_path.name}")
    print(f"Size: {w}x{h} ({w*h:,} pixels)")
    
    # Step 1: Generate red mask
    red_mask = _red_mask_hsv(bgr)
    red_pixel_count = np.sum(red_mask > 0)
    red_density = red_pixel_count / (h * w)
    
    print(f"\nRed pixel detection:")
    print(f"  Total red pixels: {red_pixel_count:,}")
    print(f"  Density: {red_density:.4f} ({red_density*100:.2f}%)")
    
    # Step 2: Detect boundary
    boundary_result = estimate_boundary_from_overlay(bgr)
    
    print(f"\nBoundary detection:")
    print(f"  Method: {boundary_result.method}")
    print(f"  Confidence: {boundary_result.confidence:.3f}")
    if boundary_result.polygon_px:
        print(f"  Vertices: {len(boundary_result.polygon_px)}")
        print(f"  Polygon vertices:")
        for i, (x, y) in enumerate(boundary_result.polygon_px):
            print(f"    {i}: ({x:.1f}, {y:.1f})")
    else:
        print(f"  Status: NOT DETECTED")
    
    if boundary_result.warnings:
        print(f"  Warnings: {boundary_result.warnings}")
    
    # Morphological processing to show what convex hull will see
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask_closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Original image with red overlay
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"Original Image\n{image_path.name}", fontsize=10)
    axes[0, 0].axis('off')
    
    # 2. Red mask (raw)
    axes[0, 1].imshow(red_mask, cmap='gray')
    axes[0, 1].set_title(f"Red Pixel Mask (Raw)\n{red_pixel_count:,} pixels ({red_density*100:.2f}%)", fontsize=10)
    axes[0, 1].axis('off')
    
    # 3. Red mask after morphological closing
    axes[1, 0].imshow(red_mask_closed, cmap='gray')
    axes[1, 0].set_title("Red Mask (After Morphological Close)\nUsed for convex hull", fontsize=10)
    axes[1, 0].axis('off')
    
    # 4. Detected boundary on original
    axes[1, 1].imshow(rgb)
    if boundary_result.polygon_px:
        polygon = np.array(boundary_result.polygon_px)
        poly_patch = mpatches.Polygon(polygon, fill=False, edgecolor='yellow', linewidth=1, label='Detected Boundary')
        axes[1, 1].add_patch(poly_patch)
        
        # Also show the red pixels overlay
        red_overlay = rgb.copy()
        red_overlay[red_mask > 0] = [255, 0, 0]
        axes[1, 1].imshow(red_overlay, alpha=0.3)
        
        axes[1, 1].set_title(f"Detected Boundary (Yellow) + Red Pixels Overlay\n{len(boundary_result.polygon_px)} vertices, confidence={boundary_result.confidence:.2f}", fontsize=10)
    else:
        axes[1, 1].set_title("No Boundary Detected", fontsize=10, color='red')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save output to dedicated debug folder
    if save_output:
        debug_folder = Path(__file__).parent.parent / "data" / "debug"
        debug_folder.mkdir(parents=True, exist_ok=True)
        output_path = debug_folder / f"{image_path.stem}_boundary_debug.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved debug visualization: {output_path}")
    
    plt.show()
    
    # Additional diagnostics: show RGB values of a sample of red pixels
    if red_pixel_count > 0:
        red_ys, red_xs = np.where(red_mask > 0)
        sample_size = min(10, len(red_xs))
        sample_indices = np.random.choice(len(red_xs), sample_size, replace=False)
        
        print(f"\nSample of {sample_size} red pixel RGB values:")
        print("  Idx    X      Y      R    G    B")
        for idx in sample_indices:
            x, y = red_xs[idx], red_ys[idx]
            b, g, r = bgr[y, x]
            print(f"  {idx:3d}  {x:4d}  {y:4d}   {r:3d}  {g:3d}  {b:3d}")
        
        # Check if red pixels form a boundary (should be scattered, not clustered)
        red_coords = np.column_stack([red_xs, red_ys])
        centroid = red_coords.mean(axis=0)
        distances_from_center = np.linalg.norm(red_coords - centroid, axis=1)
        
        print(f"\nRed pixel distribution:")
        print(f"  Centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})")
        print(f"  Avg distance from centroid: {distances_from_center.mean():.1f} px")
        print(f"  Std distance from centroid: {distances_from_center.std():.1f} px")
        
        # Check if pixels are mostly on the perimeter (good boundary) or scattered (noise)
        # A good boundary should have high std (pixels far from center on all sides)
        if distances_from_center.std() < 50:
            print(f"  ⚠ WARNING: Red pixels are clustered (low spread). May not be a boundary.")
        else:
            print(f"  ✓ Red pixels are well-distributed (good boundary candidate)")


def main():
    parser = argparse.ArgumentParser(
        description="Debug boundary detection by visualizing red mask and detected polygon."
    )
    parser.add_argument("image", type=Path, help="Path to image file")
    parser.add_argument("--no-save", action="store_true", help="Don't save output image")
    
    args = parser.parse_args()
    
    # Verify image exists
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    debug_boundary_detection(args.image, save_output=not args.no_save)


if __name__ == "__main__":
    main()
