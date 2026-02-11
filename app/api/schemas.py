from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


Point = Tuple[float, float]  # pixel coords (x, y)
Polygon = List[Point]


class AnalyzeRequest(BaseModel):
    # Optional geographic boundary (lat/lon). If provided later we can georeference.
    parcel_coords_latlon: Optional[List[Tuple[float, float]]] = Field(
        default=None,
        description="Parcel boundary in (lat, lon) pairs (optional).",
    )
    # Optional known scale (meters per pixel). If provided, skip estimating.
    scale_m_per_px: Optional[float] = Field(
        default=None, description="Known meters-per-pixel scale (optional)."
    )


class BoundaryResult(BaseModel):
    polygon_px: Optional[Polygon] = None
    confidence: float = 0.0
    method: str = "none"
    warnings: List[str] = Field(default_factory=list)


class ScaleResult(BaseModel):
    scale_m_per_px: Optional[float] = None
    confidence: float = 0.0
    method: str = "none"
    warnings: List[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    boundary: BoundaryResult
    scale: ScaleResult
    annotated_image_base64_png: str
    debug: Dict[str, Any] = Field(default_factory=dict)
