"""
Simple script to test the /analyze endpoint on a local image file.
Saves the annotated output to disk for inspection.

Usage:
    python scripts/test_analyze.py path/to/image.png --lat 40.7128 --zoom 19
"""

from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import sys

from PIL import Image
import io

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.utils import decode_image_bytes_to_bgr
from app.core.boundary import estimate_boundary_from_overlay
from app.core.scale import choose_scale
from app.core.render import draw_polygon
from app.core.utils import encode_bgr_to_base64_png


def analyze_image(
    image_path: Path,
    scale_m_per_px: float | None = None,
    lat: float | None = None,
    zoom: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Analyze an image using the Milestone 1 pipeline.
    
    Args:
        image_path: Path to image file
        scale_m_per_px: Optional known scale
        lat: Optional latitude for Web Mercator calculation
        zoom: Optional zoom level
        output_dir: Optional directory to save annotated image
    
    Returns:
        Dictionary with boundary, scale, and metadata
    """
    # Read image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    bgr = decode_image_bytes_to_bgr(image_bytes)
    
    # Detect boundary
    boundary_out = estimate_boundary_from_overlay(bgr)
    
    # Estimate scale
    scale_out = choose_scale(scale_m_per_px, lat, zoom)
    
    # Render annotation
    annotated = bgr
    if boundary_out.polygon_px:
        annotated = draw_polygon(annotated, boundary_out.polygon_px)
    
    # Save annotated image if output_dir specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_annotated.png"
        
        # Convert BGR to RGB and save
        rgb = annotated[:, :, ::-1]
        img = Image.fromarray(rgb)
        img.save(output_path)
        print(f"Saved annotated image: {output_path}")
    
    result = {
        "image_path": str(image_path),
        "image_shape": list(bgr.shape),
        "boundary": {
            "polygon_px": boundary_out.polygon_px,
            "confidence": boundary_out.confidence,
            "method": boundary_out.method,
            "warnings": boundary_out.warnings,
        },
        "scale": {
            "scale_m_per_px": scale_out.scale_m_per_px,
            "confidence": scale_out.confidence,
            "method": scale_out.method,
            "warnings": scale_out.warnings,
        },
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test Milestone 1 boundary and scale extraction on a sample image."
    )
    parser.add_argument("image", type=Path, help="Path to image file")
    parser.add_argument("--scale", type=float, default=None, help="Known scale (meters per pixel)")
    parser.add_argument("--lat", type=float, default=None, help="Latitude (degrees)")
    parser.add_argument("--zoom", type=int, default=None, help="Web Mercator zoom level")
    parser.add_argument("--output", type=Path, default=None, help="Directory to save annotated image")
    
    args = parser.parse_args()
    
    # Verify image exists
    if not args.image.exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)
    
    # Analyze
    result = analyze_image(
        image_path=args.image,
        scale_m_per_px=args.scale,
        lat=args.lat,
        zoom=args.zoom,
        output_dir=args.output,
    )
    
    # Print results
    print("\n=== Milestone 1 Analysis Results ===\n")
    print(json.dumps(result, indent=2))
    
    # Summary
    print("\n=== Summary ===")
    if result["boundary"]["polygon_px"]:
        vertices = len(result["boundary"]["polygon_px"])
        print(f"✓ Boundary detected: {vertices} vertices, confidence {result['boundary']['confidence']:.2f}")
    else:
        print(f"✗ Boundary not detected: {result['boundary']['warnings']}")
    
    if result["scale"]["scale_m_per_px"] is not None:
        print(f"✓ Scale estimated: {result['scale']['scale_m_per_px']:.4f} m/px (confidence {result['scale']['confidence']:.2f})")
    else:
        print(f"✗ Scale not available: {result['scale']['warnings']}")


if __name__ == "__main__":
    main()
