from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ScaleOutput:
    scale_m_per_px: Optional[float]
    confidence: float
    method: str
    warnings: List[str]


def scale_from_lat_zoom(lat_deg: float, zoom: int) -> float:
    """
    Compute meters-per-pixel using Web Mercator projection.
    
    Formula: m/px = cos(lat) * 2π*R / (256 * 2^zoom)
    where R = Earth radius (6378137 m)
    
    Args:
        lat_deg: Latitude in degrees (e.g., 40.7128 for NYC)
        zoom: Web Mercator zoom level (typically 18-20 for parcel-level)
    
    Returns:
        Meters per pixel
    """
    R = 6378137.0  # Earth radius in meters
    lat_rad = math.radians(lat_deg)
    cos_lat = math.cos(lat_rad)
    
    m_per_px = (cos_lat * 2 * math.pi * R) / (256 * (2 ** zoom))
    return m_per_px


def choose_scale(
    user_scale_m_per_px: Optional[float],
    lat: Optional[float],
    zoom: Optional[int]
) -> ScaleOutput:
    """
    Prioritize scale sources in order:
    1. User-provided scale (highest confidence)
    2. Lat + zoom via Web Mercator formula (high confidence)
    3. None (placeholder, very low confidence)
    
    Args:
        user_scale_m_per_px: Manually provided scale (meters per pixel)
        lat: Latitude in degrees (from image metadata or geocoding)
        zoom: Web Mercator zoom level (from image metadata)
    
    Returns:
        ScaleOutput with scale_m_per_px, confidence, method, warnings
    """
    # Priority 1: User-provided scale (most reliable)
    if user_scale_m_per_px is not None:
        return ScaleOutput(
            scale_m_per_px=float(user_scale_m_per_px),
            confidence=1.0,
            method="user_provided",
            warnings=[],
        )
    
    # Priority 2: Lat + zoom Web Mercator
    if lat is not None and zoom is not None:
        try:
            m_per_px = scale_from_lat_zoom(lat, int(zoom))
            return ScaleOutput(
                scale_m_per_px=m_per_px,
                confidence=0.95,
                method="lat_zoom_webmercator",
                warnings=[],
            )
        except Exception as e:
            return ScaleOutput(
                scale_m_per_px=None,
                confidence=0.0,
                method="lat_zoom_webmercator",
                warnings=[f"Failed to compute scale from lat/zoom: {str(e)}"],
            )
    
    # Fallback: No scale available
    return ScaleOutput(
        scale_m_per_px=None,
        confidence=0.0,
        method="unknown",
        warnings=["No scale provided; pass scale_m_per_px or both (lat, zoom)."],
    )
