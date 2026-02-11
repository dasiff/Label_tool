"""
Convert Label Studio JSON export to class masks for training.

Milestone 1: Placeholder implementation.
Reads Label Studio JSON and writes binary masks (parking boundary).
Later: Add polygon simplification, confidence scoring, augmentation.
"""

from __future__ import annotations
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw


def load_label_studio_export(json_path: Path) -> List[Dict]:
    """Load Label Studio JSON export."""
    with open(json_path) as f:
        return json.load(f)


def extract_polygons_from_annotation(annotation: Dict) -> List[List[Tuple[int, int]]]:
    """Extract polygon coordinates from a Label Studio annotation result."""
    polygons = []
    if "result" not in annotation:
        return polygons
    
    for result in annotation["result"]:
        if result.get("type") != "polygonlabels":
            continue
        value = result.get("value", {})
        points = value.get("points", [])
        if points:
            polygons.append([(int(p[0]), int(p[1])) for p in points])
    
    return polygons


def create_mask_from_polygons(image_shape: Tuple[int, int], polygons: List[List[Tuple[int, int]]]) -> np.ndarray:
    """
    Create a binary mask from a list of polygons.
    
    Args:
        image_shape: (height, width) of the image
        polygons: List of polygon coordinate lists
    
    Returns:
        Binary mask (255 for polygon, 0 for background)
    """
    mask = Image.new("L", (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    
    for poly in polygons:
        draw.polygon(poly, fill=255)
    
    return np.array(mask, dtype=np.uint8)


def export_to_masks(
    label_studio_json: Path,
    output_dir: Path,
    image_dir: Optional[Path] = None,
):
    """
    Convert Label Studio export to binary masks.
    
    Args:
        label_studio_json: Path to Label Studio export JSON
        output_dir: Directory to write masks
        image_dir: Optional directory containing original images (for validation)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_label_studio_export(label_studio_json)
    
    for task in data:
        task_id = task.get("id", "unknown")
        image_name = task.get("image", f"image_{task_id}.png")
        
        # Get image dimensions
        if image_dir:
            image_path = image_dir / image_name
            if image_path.exists():
                img = Image.open(image_path)
                height, width = img.size[1], img.size[0]
            else:
                print(f"Warning: image not found {image_path}, skipping")
                continue
        else:
            # Assume images are 1024x1024 if not provided
            height, width = 1024, 1024
        
        # Extract polygons from annotations
        polygons = []
        for annotation in task.get("annotations", []):
            polygons.extend(extract_polygons_from_annotation(annotation))
        
        # Create and save mask
        if polygons:
            mask = create_mask_from_polygons((height, width), polygons)
            mask_path = output_dir / f"{Path(image_name).stem}_mask.png"
            Image.fromarray(mask).save(mask_path)
            
            # Save metadata
            meta_path = output_dir / f"{Path(image_name).stem}_meta.json"
            meta = {
                "task_id": task_id,
                "image_name": image_name,
                "polygon_count": len(polygons),
                "height": height,
                "width": width,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            
            print(f"Exported {mask_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Label Studio JSON export to class masks."
    )
    parser.add_argument(
        "--input", required=True, help="Path to Label Studio export JSON"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for masks"
    )
    parser.add_argument(
        "--image_dir", help="Optional directory containing original images"
    )
    
    args = parser.parse_args()
    
    export_to_masks(
        label_studio_json=Path(args.input),
        output_dir=Path(args.output),
        image_dir=Path(args.image_dir) if args.image_dir else None,
    )


if __name__ == "__main__":
    main()
