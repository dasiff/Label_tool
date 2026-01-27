"""
Image downloader and metadata manager for parking space analysis.

Stores images with associated metadata (lat, zoom, address) for scale and georeference.
Supports both manual image upload and automated downloads from Google Maps Static API.

Structure:
    images/
      {address_or_id}/
        image.png
        metadata.json          # {"lat": 40.7128, "lon": -74.0060, "zoom": 19, "address": "...", ...}
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime


@dataclass
class ImageMetadata:
    """Metadata for a parking space analysis image."""
    address: str
    lat: float
    lon: float
    zoom: int = 19  # Default to high resolution
    timestamp: str = ""  # ISO format
    source: str = "unknown"  # "google_maps", "manual_upload", etc.
    notes: Optional[str] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"


def save_image_with_metadata(
    image_bytes: bytes,
    metadata: ImageMetadata,
    output_dir: Path = Path("images"),
) -> Path:
    """
    Save an image and its metadata to disk.
    
    Args:
        image_bytes: Raw image file bytes
        metadata: ImageMetadata with lat, lon, address, etc.
        output_dir: Root directory for image storage
    
    Returns:
        Path to saved image file
    
    Structure:
        images/{sanitized_address}/image.png
        images/{sanitized_address}/metadata.json
    """
    # Sanitize address for directory name
    safe_name = (
        metadata.address
        .lower()
        .replace(" ", "_")
        .replace(",", "")
        .replace("/", "-")
    )
    image_subdir = output_dir / safe_name
    image_subdir.mkdir(parents=True, exist_ok=True)
    
    # Save image
    image_path = image_subdir / "image.png"
    with open(image_path, "wb") as f:
        f.write(image_bytes)
    
    # Save metadata
    meta_path = image_subdir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)
    
    print(f"Saved: {image_path}")
    print(f"Metadata: {meta_path}")
    
    return image_path


def load_image_with_metadata(image_path: Path) -> tuple[bytes, ImageMetadata]:
    """
    Load image and associated metadata.
    
    Args:
        image_path: Path to image.png
    
    Returns:
        (image_bytes, ImageMetadata)
    """
    # Load image
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    
    # Load metadata
    meta_path = image_path.parent / "metadata.json"
    with open(meta_path) as f:
        meta_dict = json.load(f)
    
    metadata = ImageMetadata(**meta_dict)
    return image_bytes, metadata


def list_images(image_dir: Path = Path("images")) -> list[tuple[Path, ImageMetadata]]:
    """
    List all images in the image directory.
    
    Returns:
        List of (image_path, metadata) tuples
    """
    if not image_dir.exists():
        return []
    
    results = []
    for meta_path in image_dir.glob("*/metadata.json"):
        image_path = meta_path.parent / "image.png"
        if image_path.exists():
            with open(meta_path) as f:
                meta_dict = json.load(f)
            metadata = ImageMetadata(**meta_dict)
            results.append((image_path, metadata))
    
    return sorted(results, key=lambda x: x[1].timestamp)


def google_maps_static_url(
    lat: float,
    lon: float,
    zoom: int = 19,
    size: tuple[int, int] = (512, 512),
    api_key: Optional[str] = None,
) -> str:
    """
    Generate Google Maps Static API URL for a location.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        zoom: Web Mercator zoom level (default 19 for parcel detail)
        size: Output image size (width, height)
        api_key: Google Maps API key (or set GOOGLE_MAPS_API_KEY env var)
    
    Returns:
        URL string for downloading the image
    
    Usage:
        url = google_maps_static_url(40.7128, -74.0060, zoom=19)
        # Then download with requests: response = requests.get(url)
    """
    if api_key is None:
        import os
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    
    if not api_key:
        raise ValueError(
            "Google Maps API key required. "
            "Set GOOGLE_MAPS_API_KEY environment variable or pass api_key parameter."
        )
    
    width, height = size
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}"
        f"&zoom={zoom}"
        f"&size={width}x{height}"
        f"&maptype=satellite"
        f"&key={api_key}"
    )
    return url


if __name__ == "__main__":
    # Example usage
    print("Image metadata manager for parking space analysis.")
    print("\nExample:")
    print("  from scripts.image_manager import ImageMetadata, save_image_with_metadata")
    print("  meta = ImageMetadata(address='123 Main St', lat=40.7128, lon=-74.0060, zoom=19)")
    print("  save_image_with_metadata(image_bytes, meta)")
