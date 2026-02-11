"""
Generate boundary debug images for all images in raw_images folder.
Shows side-by-side comparison of red mask and detected polygon.
"""
from pathlib import Path
import cv2
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.boundary import estimate_boundary_from_overlay, save_boundary_debug_image


def main():
    # Setup paths
    raw_images_folder = Path(__file__).parent.parent / "data" / "raw_images"
    debug_folder = Path(__file__).parent.parent / "data" / "debug"
    debug_folder.mkdir(parents=True, exist_ok=True)
    
    # Get all images
    image_files = sorted(list(raw_images_folder.glob("*.png")) + 
                        list(raw_images_folder.glob("*.jpg")))
    
    print(f"Found {len(image_files)} images in {raw_images_folder}")
    print(f"Saving debug images to {debug_folder}")
    print("-" * 80)
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        # Load image
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"  ERROR: Could not load image")
            continue
        
        # Detect boundary
        result = estimate_boundary_from_overlay(bgr)
        
        if result.polygon_px:
            print(f"  ✓ Detected {len(result.polygon_px)} vertices (confidence: {result.confidence:.2f})")
            if result.warnings:
                for warning in result.warnings:
                    print(f"    ⚠ {warning}")
        else:
            print(f"  ✗ No boundary detected")
            if result.warnings:
                for warning in result.warnings:
                    print(f"    ⚠ {warning}")
        
        # Save debug image
        debug_path = debug_folder / f"{image_path.stem}_boundary_debug.png"
        save_boundary_debug_image(bgr, result.polygon_px, str(debug_path))
    
    print("\n" + "=" * 80)
    print(f"✓ Done! Debug images saved to: {debug_folder}")
    print("=" * 80)


if __name__ == "__main__":
    main()
