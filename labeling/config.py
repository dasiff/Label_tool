"""Configuration and constants for simplified labeling tool.

Responsibility:
- Expose CLASSES, DISPLAY_NAMES, CLASS_COLORS, and default parameters used across
  the simplified code path. Keep it small and easily overrideable for tests.
"""

CLASSES = [
    "parking_stalls",
    "driveway_road",
    "building",
    "vegetation",
    "sidewalk",
    "pool_equipment",
]

DISPLAY_NAMES = {cls: cls.replace("_", " ").title() for cls in CLASSES}

CLASS_COLORS = {
    "parking_stalls": [0.39, 0.58, 0.93],
    "driveway_road": [1.0, 0.84, 0.0],
    "building": [0.85, 0.3, 0.3],
    "vegetation": [0.13, 0.55, 0.13],
    "sidewalk": [0.78, 0.78, 0.78],
    "pool_equipment": [0.6, 0.4, 0.8],
}

DEFAULTS = {
    'min_segment_px': 1000,
    'target_segments': 50,
    'fast_preview': True,
}
