# Parking Spaces Feasibility API – Copilot Instructions

## Project Overview
This is a FastAPI-based backend for analyzing parking space feasibility from aerial imagery. The system detects parking space boundaries and estimates scale from image overlays. Development is structured in **milestones**:
- **Milestone 1** ✅ (Active): Red overlay boundary detection (HSV thresholding + convex hull) & Web Mercator scale estimation
- **Milestone 2+**: ML-based detection, model training infrastructure

## Architecture & Key Components

### API Layer (`app/api/schemas.py`)
- **AnalyzeRequest**: Accepts optional `parcel_coords_latlon` (lat/lon boundary) and `scale_m_per_px` (known scale override)
- **BoundaryResult** / **ScaleResult**: Structured outputs with `confidence`, `method`, `warnings`
- **AnalyzeResponse**: Returns detected boundary polygon, scale, annotated image (base64 PNG), and debug metadata
- **Type aliases**: `Point = Tuple[float, float]` (pixel coordinates), `Polygon = List[Point]`

### Core Detection (`app/core/`)

#### **`boundary.py`** – Milestone 1: Red Overlay Detection
- **`estimate_boundary_from_overlay()`**: HSV-based red detection → morphology close → convex hull → Douglas-Peucker simplification
- **Process**:
  1. Convert BGR to HSV (handles red hue wrapping at 0/180)
  2. Threshold two red ranges: [0-10] + [170-180]
  3. Morphological close (5×5 kernel, 2 iterations) to connect line gaps
  4. Extract convex hull from red pixels
  5. Simplify hull using Douglas-Peucker (epsilon = 0.5% perimeter)
  6. Score confidence based on pixel density (heuristic: expect ~1% of image to be red)
- **Returns**: Polygon, confidence 0.0–1.0, method name, warnings
- **Robustness**: Requires ≥500 red pixels to avoid false positives

#### **`scale.py`** – Milestone 1: Web Mercator + User Scale
- **`choose_scale()`**: Priority system for scale sources
  - Priority 1: User-provided `scale_m_per_px` (confidence 1.0)
  - Priority 2: Latitude + zoom via Web Mercator formula (confidence 0.95)
  - Fallback: None with warning
- **`scale_from_lat_zoom(lat, zoom)`**: Computes meters/pixel using standard formula:
  ```
  m/px = cos(lat_rad) * 2π*R / (256 * 2^zoom)
  ```
  - R = 6378137 m (Earth radius)
  - Assumes image is a Web Mercator tile (standard for Google Maps)

#### **`render.py`** – Polygon Visualization
- `draw_polygon()`: OpenCV overlay (yellow 0,255,255 in BGR by default)

#### **`utils.py`** – Image Codecs
- `decode_image_bytes_to_bgr()`: PIL → BGR numpy array
- `encode_bgr_to_base64_png()`: BGR → base64 PNG for JSON

### FastAPI Entrypoint (`app/main.py`)
- **POST /analyze**: Accepts image + optional `scale_m_per_px`, `lat`, `zoom`
  - Returns AnalyzeResponse with boundary, scale, annotated image, debug metadata
- **GET /healthz**: Health check

## Data Flow (Milestone 1)
```
Image File (uploaded)
    ↓
[decode_image_bytes_to_bgr] → BGR numpy array
    ↓
[_red_mask_hsv] → Binary mask (red pixels = 255)
    ↓
[Morph close] → Connected red components
    ↓
[cv2.convexHull] → Outer polygon
    ↓
[cv2.approxPolyDP] → Simplified polygon (5-20 vertices typical)
    ↓
[Confidence score] → 0.0–1.0 based on pixel density
    ↓
[choose_scale] → Priority: user > lat/zoom > None
    ↓
[draw_polygon] → Annotated BGR image (yellow overlay)
    ↓
[encode_bgr_to_base64_png] → Base64 PNG string
    ↓
AnalyzeResponse (JSON)
```

## Developer Workflows

### Local Development
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/bin/activate             # Linux/Mac

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Testing
```bash
pytest tests/ -v                          # Run all tests
pytest tests/test_api.py::test_analyze_with_red_boundary -v  # Specific test
pytest tests/ --cov=app --cov-report=html # Coverage report
```

### Test Specific Image
```bash
python scripts/test_analyze.py /path/to/image.png --lat 40.7128 --zoom 19 --output ./output
```

### Docker (Optional)
```bash
docker-compose up --build
# API at http://localhost:8000
```

## Key Conventions & Patterns

### Image Handling
- **Input**: Upload as multipart form-data (MIME: image/png, image/jpeg)
- **Internal**: OpenCV BGR uint8 arrays (H×W×3)
- **Output**: Base64-encoded PNG for JSON serialization
- **No in-place mutations**: `bgr.copy()` used in rendering to avoid side effects

### Error Handling & Confidence
- All detection results include `confidence: float` (0.0–1.0) and `warnings: List[str]`
- Missing data → return None + low confidence + descriptive warning
- No exceptions on detection failures; always degrade gracefully

### Polygon Representation
- **Pixel coordinates**: (x, y) tuples, origin top-left
- **Open format**: List of Points; rendering auto-closes with `cv2.polylines(..., isClosed=True)`
- **From convex hull**: Already ordered CCW, typically 4-8 vertices for rectilinear parcels

### Scale Estimation (Milestone 1)
- **Web Mercator**: Standard formula for Google Maps tiles
  - Requires both lat AND zoom; neither alone is sufficient
  - Zoom 19 typical for parcel-level (≈0.2 m/px in continental US)
  - Zoom 18 ≈0.4 m/px, Zoom 20 ≈0.1 m/px
- **Fallback** (future): Car-based scale (detect vehicles, infer length)

### Dependencies
- **Computer vision**: OpenCV-headless (headless for server deployments)
- **Geometry**: Shapely 2.0.5 (for future snap-to-grid, area validation)
- **ML (deferred)**: PyTorch + TorchVision kept in requirements.txt for consistency

## Image Storage & Metadata (`scripts/image_manager.py`)
- **Purpose**: Manage aerial images with associated metadata (lat, lon, zoom, address)
- **Structure**: `images/{sanitized_address}/image.png` + `metadata.json`
- **Key classes**:
  - `ImageMetadata`: Dataclass with lat, lon, zoom, address, source, timestamp
  - `save_image_with_metadata()`: Store image + JSON metadata
  - `load_image_with_metadata()`: Retrieve with metadata
  - `list_images()`: Enumerate all stored images
- **Google Maps Static API helper**: `google_maps_static_url()` generates download links (requires API key)

## Label Studio Integration (`scripts/labelstudio/`)
- **export_to_masks.py**: Converts Label Studio JSON export → class masks
- **README.md**: Setup instructions, annotation format, training workflow
- Not integrated into main API; used offline for training data prep

## File Structure Reference
```
app/
  main.py                    # FastAPI app + POST /analyze route
  api/
    schemas.py               # Pydantic models
  core/
    boundary.py              # HSV red detection + convex hull (Milestone 1)
    scale.py                 # Web Mercator + user scale (Milestone 1)
    render.py                # draw_polygon()
    utils.py                 # Image codecs
scripts/
  test_analyze.py            # Standalone test script for sample images
  image_manager.py           # Image storage + metadata management
  labelstudio/
    export_to_masks.py
    README.md
models/                      # Trained model weights (Milestone 2+)
tests/
  test_api.py                # Integration tests (includes red boundary detection)
Dockerfile
docker-compose.yml
requirements.txt
```

## Testing & Validation (Milestone 1)

### Unit Tests
- `test_healthz`: Health check
- `test_analyze_with_red_boundary`: Detects red rectangle boundary
- `test_analyze_with_scale_m_per_px`: User-provided scale
- `test_analyze_with_lat_zoom`: Web Mercator calculation
- `test_analyze_scale_priority_user_over_lat_zoom`: Priority check
- `test_analyze_annotated_image_is_base64`: Valid PNG encoding

### Integration
- `scripts/test_analyze.py`: Test on real sample images with metadata
- Swagger UI at `/docs`: Interactive endpoint testing

## Common Debugging (Milestone 1)

- **Boundary not detected (polygon_px = None)**: Check that red overlay is visible in image; threshold may need tuning if red is different shade
- **Confidence too low**: Adjust pixel density heuristic in `boundary.py` if red coverage differs from assumption (~1%)
- **Scale is None**: Verify either `scale_m_per_px` OR both `lat` + `zoom` passed to `/analyze`
- **Scale seems wrong**: Check zoom level (should be 18-20 for parcel detail); verify latitude is correct
- **Polygon too simplified/rough**: Increase `epsilon` parameter in `approxPolyDP()` call (currently 0.5% of perimeter)
- **CORS errors**: Update `allow_origins` in main.py if testing from different domain

## Next Steps (Milestone 2)
- Train ML model on Label Studio annotations (convex hull → UNet segmentation)
- Implement car-based scale fallback
- Add address → latitude geocoding if pulling images manually
- Optimize inference pipeline (batch processing, model quantization)

