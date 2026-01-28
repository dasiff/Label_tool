# Milestone 1 – Parameter Tuning Guide

This document describes the parameters in `app/core/boundary.py` that can be adjusted for different boundary overlay colors and image conditions.

## Red Boundary Detection – HSV Thresholds

### Current Settings (for pure red overlays)

**Location:** `app/core/boundary.py`, function `_red_mask_hsv()`

```python
# Red hue wraps around 0/180 in HSV
lower1 = np.array([0, 120, 80])      # Lower red: H [0-10]
upper1 = np.array([10, 255, 255])
lower2 = np.array([170, 120, 80])    # Upper red: H [170-180]
upper2 = np.array([180, 255, 255])
```

**Meaning:**
- `H (Hue)` [0-180]: Color (0=red, 60=green, 120=blue, 180=red wraps)
- `S (Saturation)` [0-255]: Color intensity (0=white, 255=pure)
- `V (Value)` [0-255]: Brightness (0=black, 255=bright)

**Default Range:**
- Red hue: 0-10 OR 170-180 (wraps around)
- Saturation: 120-255 (avoid washed-out red)
- Value: 80-255 (avoid too dark)

### Adjusting for Different Colors

**Yellow Boundary:**
```python
lower1 = np.array([10, 100, 100])
upper1 = np.array([25, 255, 255])
```

**White Boundary:**
```python
lower1 = np.array([0, 0, 200])
upper1 = np.array([180, 100, 255])
```

**Blue Boundary:**
```python
lower1 = np.array([100, 120, 80])
upper1 = np.array([130, 255, 255])
```

**Green Boundary:**
```python
lower1 = np.array([40, 100, 80])
upper1 = np.array([80, 255, 255])
```

## Morphological Close Parameters

**Location:** `app/core/boundary.py`, function `estimate_boundary_from_overlay()`

```python
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
```

**Parameters:**
- **Kernel size** `(5, 5)`: Gap size to fill. Larger = fills bigger gaps
- **Iterations** `2`: Number of close operations. More = stronger closure

**Adjustments:**
- **Very thin boundary, lots of gaps:** Increase to `(7, 7)` kernel, 3 iterations
- **Thick boundary, unnecessary smoothing:** Decrease to `(3, 3)` kernel, 1 iteration
- **Dashed boundary:** Use larger kernel like `(11, 11)` and 4 iterations

## Convex Hull Simplification

**Location:** `app/core/boundary.py`, function `estimate_boundary_from_overlay()`

```python
peri = cv2.arcLength(hull, True)
epsilon = 0.005 * peri
approx = cv2.approxPolyDP(hull, epsilon, True)
```

**Parameters:**
- **Epsilon** (default 0.5% of perimeter): Tolerance for Douglas-Peucker algorithm
  - Smaller epsilon = more vertices, fits boundary more tightly
  - Larger epsilon = fewer vertices, cleaner polygon

**Adjustments:**
- **Too many vertices (>20):** Increase to `0.01 * peri` (1%)
- **Too coarse (missing detail):** Decrease to `0.002 * peri` (0.2%)
- **Very detailed boundary:** Use `0.001 * peri` (0.1%)

## Red Pixel Density Thresholds

**Location:** `app/core/boundary.py`, function `estimate_boundary_from_overlay()`

```python
MIN_PIXELS = 500
expected_density = 0.01  # 1% of image
```

**Parameters:**
- **MIN_PIXELS** (default 500): Minimum red pixels required to detect boundary
  - Small images or thin boundaries: Lower to 200
  - Large images or noisy conditions: Raise to 1000
- **expected_density** (default 0.01): Expected fraction of red pixels
  - Thin line boundary: Use 0.005 (0.5%)
  - Thick boundary: Use 0.02 (2%)

## Confidence Scoring

**Location:** `app/core/boundary.py`, function `estimate_boundary_from_overlay()`

```python
if pixel_density < 0.001:
    confidence = 0.0
elif pixel_density > 0.05:
    confidence = 0.5
else:
    confidence = 1.0 - abs(pixel_density - expected_density) / (2 * expected_density)
```

**Adjustments:**
- **Too strict** (confidence always low): Widen the ranges or adjust `expected_density`
- **Too lenient** (confidence always high): Narrow the ranges or penalize more

## Testing Your Changes

After modifying parameters:

```bash
python scripts/test_analyze.py /path/to/image.png --output ./results
```

Check the console output:
- If `confidence` is 0.0: Boundary not detected, need to adjust HSV thresholds
- If `polygon_px` is None: Not enough pixels, lower `MIN_PIXELS` or adjust saturation threshold
- If polygon has too many vertices: Increase `epsilon` in Douglas-Peucker

## Multi-Color Detection (Future)

To support multiple boundary colors simultaneously:

```python
def _multi_color_mask(bgr: np.ndarray, colors=['red', 'yellow', 'white']):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    if 'red' in colors:
        # Add red ranges
        m = cv2.inRange(hsv, [0, 120, 80], [10, 255, 255])
        mask = cv2.bitwise_or(mask, m)
        # ... etc
    
    if 'yellow' in colors:
        m = cv2.inRange(hsv, [10, 100, 100], [25, 255, 255])
        mask = cv2.bitwise_or(mask, m)
    
    return mask
```

Then call `_multi_color_mask(bgr, colors=['red', 'yellow'])` to detect both.

## Quick Troubleshooting

| Symptom | Check | Solution |
|---------|-------|----------|
| Boundary never detected | HSV thresholds | Print actual HSV range of boundary (see section below) |
| Boundary too coarse | epsilon parameter | Decrease from 0.005 to 0.002 |
| Boundary too detailed | epsilon parameter | Increase from 0.005 to 0.01 |
| Too many gaps unfilled | Morphology kernel | Increase from (5,5) to (7,7) or (11,11) |
| False positives (noise) | MIN_PIXELS | Increase threshold |
| Missing thin boundaries | Morphology iterations | Increase from 2 to 4 iterations |

## Debugging: Find Your Boundary's HSV Range

```python
import cv2
import numpy as np

# Read your image
bgr = cv2.imread("image.png")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# Click on boundary pixels and print their HSV values
print(f"Sample pixel HSV: H={hsv[100, 100, 0]}, S={hsv[100, 100, 1]}, V={hsv[100, 100, 2]}")

# Or create a simple range detector:
# Replace [0, 120, 80] and [10, 255, 255] with values slightly smaller/larger
# than your sample pixel's HSV
```

Then adjust the thresholds in `_red_mask_hsv()` accordingly.
