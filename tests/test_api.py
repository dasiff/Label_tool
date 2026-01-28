import io
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def create_test_image(width: int = 200, height: int = 200, with_red_boundary: bool = False) -> bytes:
    """
    Generate a test image as PNG bytes.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        with_red_boundary: If True, draw a red rectangle boundary
    
    Returns:
        PNG bytes
    """
    img = Image.new("RGB", (width, height), color=(73, 109, 137))  # Gray background
    
    if with_red_boundary:
        draw = ImageDraw.Draw(img)
        # Draw a red rectangle boundary with thicker lines and some inset
        inset = 20
        # Draw multiple passes to ensure enough red pixels for detection
        for offset in range(-2, 3):  # Draw thicker line
            draw.rectangle(
                [(inset + offset, inset + offset), (width - inset + offset, height - inset + offset)],
                outline=(255, 0, 0),  # Red
                width=5
            )
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_analyze_basic_no_red_boundary():
    """Test /analyze with image that has no red boundary (should return low confidence)."""
    image_bytes = create_test_image(with_red_boundary=False)
    response = client.post(
        "/analyze",
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Check response schema
    assert "boundary" in data
    assert "scale" in data
    assert "annotated_image_base64_png" in data
    assert "debug" in data
    
    # No red pixels, so boundary should be None with low confidence
    assert data["boundary"]["polygon_px"] is None
    assert data["boundary"]["confidence"] == 0.0
    assert data["boundary"]["method"] == "red_overlay_contour"
    assert len(data["boundary"]["warnings"]) > 0
    
    # Scale should be None (no scale provided)
    assert data["scale"]["scale_m_per_px"] is None
    assert data["scale"]["confidence"] == 0.0


def test_analyze_with_red_boundary():
    """Test /analyze with red boundary overlay."""
    image_bytes = create_test_image(width=200, height=200, with_red_boundary=True)
    response = client.post(
        "/analyze",
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should detect red boundary
    assert data["boundary"]["polygon_px"] is not None
    assert data["boundary"]["confidence"] > 0.0
    assert data["boundary"]["method"] == "red_overlay_morphological"
    
    # Polygon should have reasonable vertices (at least 3)
    polygon = data["boundary"]["polygon_px"]
    assert len(polygon) >= 3


def test_analyze_with_scale_m_per_px():
    """Test /analyze with user-provided scale."""
    image_bytes = create_test_image(with_red_boundary=True)
    response = client.post(
        "/analyze",
        files={"image": ("test.png", image_bytes, "image/png")},
        params={"scale_m_per_px": 0.5},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Scale should be accepted with high confidence
    assert data["scale"]["scale_m_per_px"] == 0.5
    assert data["scale"]["confidence"] == 1.0
    assert data["scale"]["method"] == "user_provided"
    assert len(data["scale"]["warnings"]) == 0


def test_analyze_with_lat_zoom():
    """Test /analyze with lat and zoom for Web Mercator scale calculation."""
    image_bytes = create_test_image(with_red_boundary=True)
    response = client.post(
        "/analyze",
        files={"image": ("test.png", image_bytes, "image/png")},
        params={"lat": 40.7128, "zoom": 19},  # NYC at high zoom
    )
    assert response.status_code == 200
    data = response.json()
    
    # Scale should be computed from lat/zoom
    assert data["scale"]["scale_m_per_px"] is not None
    assert data["scale"]["scale_m_per_px"] > 0
    assert data["scale"]["confidence"] == 0.95
    assert data["scale"]["method"] == "lat_zoom_webmercator"
    assert len(data["scale"]["warnings"]) == 0
    
    # Scale should be reasonable for zoom 19 (tight pixels)
    # At zoom 19, NYC is roughly 0.19 m/px
    assert 0.1 < data["scale"]["scale_m_per_px"] < 0.3


def test_analyze_scale_priority_user_over_lat_zoom():
    """Test that user-provided scale takes priority over lat/zoom."""
    image_bytes = create_test_image(with_red_boundary=True)
    response = client.post(
        "/analyze",
        files={"image": ("test.png", image_bytes, "image/png")},
        params={"scale_m_per_px": 1.0, "lat": 40.7128, "zoom": 19},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should use user-provided scale, not lat/zoom
    assert data["scale"]["scale_m_per_px"] == 1.0
    assert data["scale"]["method"] == "user_provided"
    assert data["scale"]["confidence"] == 1.0


def test_analyze_annotated_image_is_base64():
    """Test that annotated image is valid base64 PNG."""
    image_bytes = create_test_image(with_red_boundary=True)
    response = client.post(
        "/analyze",
        files={"image": ("test.png", image_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    
    # Annotated image should be base64-decodable PNG
    import base64
    decoded = base64.b64decode(data["annotated_image_base64_png"])
    # Verify it's a valid PNG (starts with PNG magic bytes)
    assert decoded[:8] == b'\x89PNG\r\n\x1a\n'


def test_analyze_missing_zoom_with_lat():
    """Test that scale fails gracefully when lat is provided but zoom is missing."""
    image_bytes = create_test_image(with_red_boundary=True)
    response = client.post(
        "/analyze",
        files={"image": ("test.png", image_bytes, "image/png")},
        params={"lat": 40.7128},  # Missing zoom
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should fall back to unknown (no scale)
    assert data["scale"]["scale_m_per_px"] is None
    assert data["scale"]["confidence"] == 0.0
    assert "No scale provided" in data["scale"]["warnings"][0]
