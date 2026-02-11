Labeling Tool — Boundary Access & Road Painting

Overview

This document describes the new Boundary Access (segment-level) and Road (pixel-level) annotation tools available in `scripts/labeling_tool.py`.

Boundary Access (segment-level)

- Purpose: Mark where vehicles are allowed to enter/exit the property.
- Interaction: Activate **Boundary Access Mode**. Click two points on the property boundary (they will snap to the nearest boundary). The shorter arc is chosen by default. Left-click toggles `access_allowed` ↔ `no_access`. Right-click deletes a segment.
- Storage: Saved in the JSON as `boundary_access.segments` with both `start_frac`/`end_frac` (fractional along boundary) and exported `start_px`/`end_px` pixel coordinates for convenience.

Road Painting (pixel-level)

- Purpose: Annotate public road adjacency in a buffered zone outside the property.
- Interaction: Activate **Road Mode**. Use brush (left paint, right erase). Painting is restricted to a buffer outside the property and cannot modify interior pixels.
- Buffer: Configurable in pixels (e.g. 64–128 px) or percentage of image size (5–10%). Buffer is enforced during painting.
- Storage: Road mask is exported as `{image_name}_road_mask.png` and referenced in `{image_name}_labels.json` as `road_mask_file`; `road_mask_info` includes buffer and brush settings.

Example JSON (excerpt)

{
  "image_id": "image1",
  "boundary_access": {
    "segments": [
      {
        "start_frac": 0.123,
        "end_frac": 0.204,
        "label": "access_allowed",
        "start_px": [123.4, 45.6],
        "end_px": [234.5, 67.8]
      }
    ]
  },
  "road_mask_file": "image1_road_mask.png",
  "road_mask_info": {
    "buffer_mode": "px",
    "buffer_px": 96,
    "buffer_pct": 0.05,
    "brush_px": 16
  }
}

Notes

- The tool keeps segment-level access annotations (semantically concise) and pixel-level derived masks for segmentation training (practical for models).
- For training, use the road mask and an ignore mask to limit loss computation to the buffered region.
