from __future__ import annotations
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import AnalyzeRequest, AnalyzeResponse, BoundaryResult, ScaleResult
from app.core.utils import decode_image_bytes_to_bgr, encode_bgr_to_base64_png
from app.core.boundary import estimate_boundary_from_overlay
from app.core.scale import choose_scale
from app.core.render import draw_polygon

app = FastAPI(title="Parking Spaces Feasibility API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: UploadFile = File(...),
    scale_m_per_px: float | None = None,
    lat: float | None = None,
    zoom: int | None = None,
):
    """
    Analyze parking space feasibility from aerial image.
    
    Args:
        image: Aerial image file (PNG, JPEG, etc.)
        scale_m_per_px: Optional known scale (meters per pixel)
        lat: Optional latitude in degrees (for Web Mercator scale calculation)
        zoom: Optional Web Mercator zoom level (required with lat)
    
    Returns:
        AnalyzeResponse with boundary polygon, scale, annotated image (base64 PNG), and debug metadata
    """
    image_bytes = await image.read()
    bgr = decode_image_bytes_to_bgr(image_bytes)

    # Detect boundary from red overlay
    boundary_out = estimate_boundary_from_overlay(bgr)
    
    # Estimate scale (user_provided > lat/zoom > none)
    scale_out = choose_scale(scale_m_per_px, lat, zoom)

    # Render boundary polygon on annotated image
    annotated = bgr
    if boundary_out.polygon_px:
        annotated = draw_polygon(annotated, boundary_out.polygon_px)

    resp = AnalyzeResponse(
        boundary=BoundaryResult(
            polygon_px=boundary_out.polygon_px,
            confidence=boundary_out.confidence,
            method=boundary_out.method,
            warnings=boundary_out.warnings,
        ),
        scale=ScaleResult(
            scale_m_per_px=scale_out.scale_m_per_px,
            confidence=scale_out.confidence,
            method=scale_out.method,
            warnings=scale_out.warnings,
        ),
        annotated_image_base64_png=encode_bgr_to_base64_png(annotated),
        debug={"image_shape": list(bgr.shape)},
    )
    return resp


@app.get("/healthz")
def healthz():
    return {"ok": True}
