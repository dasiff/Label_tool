"""
Example workflow for processing parking space images with Milestone 1.

This script demonstrates:
1. Loading images with metadata
2. Analyzing via the pipeline
3. Storing results
4. Preparing for Label Studio annotation
"""

from pathlib import Path
import json
from scripts.image_manager import ImageMetadata, save_image_with_metadata, list_images
from scripts.test_analyze import analyze_image


def workflow_example():
    """
    Complete example: save an image → analyze it → store results.
    """
    
    print("=== Milestone 1 Workflow Example ===\n")
    
    # Step 1: Create metadata for a sample image
    print("1. Creating image metadata...")
    metadata = ImageMetadata(
        address="123 Main Street, New York, NY 10001",
        lat=40.7128,
        lon=-74.0060,
        zoom=19,
        source="google_maps",
        notes="Sample parking lot with red boundary overlay"
    )
    print(f"   Address: {metadata.address}")
    print(f"   Lat/Lon: {metadata.lat}, {metadata.lon}")
    print(f"   Zoom: {metadata.zoom}")
    
    # Step 2: In a real scenario, you would:
    # - Download image from Google Maps API using `google_maps_static_url()`
    # - Or accept upload from UI
    # - Or read from local file
    
    # For now, we'll assume image_bytes is loaded
    # image_bytes = open("sample_image.png", "rb").read()
    # save_image_with_metadata(image_bytes, metadata)
    
    print("\n2. Simulating image download (in real workflow)...")
    print("   # image_bytes = requests.get(google_maps_static_url(...)).content")
    print("   # save_image_with_metadata(image_bytes, metadata)")
    
    # Step 3: Analyze all stored images
    print("\n3. Analyzing stored images...")
    all_images = list_images(Path("images"))
    
    if all_images:
        for image_path, meta in all_images:
            print(f"\n   Processing: {meta.address}")
            
            # Analyze using Milestone 1 pipeline
            result = analyze_image(
                image_path=image_path,
                lat=meta.lat,
                zoom=meta.zoom,
                output_dir=Path("results"),
            )
            
            # Store result
            result_path = Path("results") / f"{image_path.stem}_analysis.json"
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"   Result saved: {result_path}")
            
            # Print summary
            boundary = result["boundary"]
            scale = result["scale"]
            print(f"   Boundary: {len(boundary['polygon_px']) if boundary['polygon_px'] else 0} vertices, "
                  f"confidence {boundary['confidence']:.2f}")
            print(f"   Scale: {scale['scale_m_per_px']:.4f} m/px, confidence {scale['confidence']:.2f}")
    else:
        print("   No images found in images/ directory")
    
    # Step 4: Prepare for Label Studio
    print("\n4. Label Studio Preparation:")
    print("   - Export all images from images/ directory")
    print("   - Upload to Label Studio project")
    print("   - Annotate parking boundaries")
    print("   - Export as JSON from Label Studio")
    
    # Step 5: Convert annotations to masks for training
    print("\n5. Convert annotations to masks:")
    print("   python scripts/labelstudio/export_to_masks.py \\")
    print("     --input label_studio_export.json \\")
    print("     --output masks/ \\")
    print("     --image_dir images/")


def quick_test_on_sample():
    """
    Quick test on a single sample image (if you have one).
    """
    sample_image = Path("sample_image.png")
    
    if not sample_image.exists():
        print(f"Create a {sample_image} file with a red boundary overlay to test.")
        return
    
    print("Quick test on sample image:")
    result = analyze_image(
        image_path=sample_image,
        lat=40.7128,
        zoom=19,
        output_dir=Path("results"),
    )
    
    print(json.dumps(result, indent=2))


def batch_process_directory(image_dir: Path):
    """
    Process all images in a directory.
    
    Usage:
        batch_process_directory(Path("my_images"))
    """
    print(f"Batch processing images in {image_dir}...\n")
    
    for image_file in image_dir.glob("*.png"):
        print(f"Processing {image_file.name}...")
        
        # For demonstration, use arbitrary lat/zoom
        result = analyze_image(
            image_path=image_file,
            lat=40.7128,
            zoom=19,
            output_dir=Path("batch_results"),
        )
        
        if result["boundary"]["polygon_px"]:
            print(f"  ✓ Boundary detected: {len(result['boundary']['polygon_px'])} vertices")
        else:
            print(f"  ✗ No boundary detected")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Batch process a directory
        image_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "images")
        batch_process_directory(image_dir)
    elif len(sys.argv) > 1 and sys.argv[1] == "--sample":
        # Quick test on sample
        quick_test_on_sample()
    else:
        # Full workflow example
        workflow_example()
