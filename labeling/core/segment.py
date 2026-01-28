"""Segmentation pipeline scaffold.

Responsibility:
- Provide functions to build features, call segmentation routines (e.g., felzenszwalb),
  and post-process into a consistent segments ndarray.

Inputs:
- clean_image (numpy array), boundary polygon, target_segments, options
Outputs:
- segments ndarray, n_segments, debug metadata
"""
from typing import Any, Dict
import numpy as np
import cv2
from scipy import ndimage
from skimage.segmentation import felzenszwalb


def _build_segment_features(self, rgb_small: np.ndarray) -> np.ndarray:
    """Build the feature image used for segmentation from an RGB image (small/rescaled).
    Returns an uint8 image suitable for felzenszwalb (H,W,3).
    Supports shadow-robust mode when `self.shadow_robust_var` is True.
    """
    # Ensure input is RGB uint8
    if rgb_small.dtype != np.uint8:
        rgb_small = np.clip(rgb_small, 0, 255).astype(np.uint8)

    # Convert to BGR for OpenCV LAB conversion
    bgr_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
    l, a, b_channel = cv2.split(lab)

    # Apply CLAHE only moderately to reduce shadow impact
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Normalize color channels and center
    l_normalized = l_enhanced.astype(float) / 255.0
    a_normalized = (a.astype(float) - 128) / 128.0
    b_normalized = (b_channel.astype(float) - 128) / 128.0

    # Default (legacy) feature composition: mix l,a,b with reduced lightness weight
    feature_img_default = np.stack([
        l_normalized * 0.3,
        a_normalized * 1.2,
        b_normalized * 1.2
    ], axis=-1)

    # Shadow-robust composition (chromaticity + gradient + lab colors)
    if getattr(self, 'shadow_robust_var', None) and self.shadow_robust_var.get():
        arr = rgb_small.astype(np.float32)
        denom = arr.sum(axis=2, keepdims=True) + 1e-6
        chroma_r = (arr[:, :, 0:1] / denom).squeeze()
        chroma_g = (arr[:, :, 1:2] / denom).squeeze()

        gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        grad_norm = grad / max(grad.max(), 1e-6)

        # Use a_normalized, b_normalized and gradient magnitude as three channels
        feature_img = np.stack([
            a_normalized,
            b_normalized,
            grad_norm
        ], axis=-1)
    else:
        feature_img = feature_img_default

    # Normalize to 0..1 and convert to uint8
    # Guard against constant images
    minv = feature_img.min()
    maxv = feature_img.max()
    if maxv - minv < 1e-6:
        feature_img = np.zeros_like(feature_img)
    else:
        feature_img = (feature_img - minv) / (maxv - minv)
    rgb_enhanced = (feature_img * 255).astype(np.uint8)
    return rgb_enhanced


def _generate_segments(self):
    """Generate segments that follow edges with straight boundaries."""
    if self.current_boundary is None:
        return

    print(f"DEBUG _generate_segments: Using boundary with {len(self.current_boundary)} vertices")
    print(f"DEBUG _generate_segments: Boundary corners: {self.current_boundary[:2]}")

    # Use clean image for segmentation
    rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Fast preview mode uses a smaller scale for quick interactive feedback
    if getattr(self, 'fast_preview_var', None) and self.fast_preview_var.get():
        scale_factor = 0.25  # very small for fast previews
        max_attempts_local = 2
    else:
        scale_factor = 0.5
        max_attempts_local = 6
    # If the user requested a very high target_segments, favor a higher-resolution
    # feature scale so that small texture details are preserved and segmentation
    # can produce many regions (avoids flaky low-count outputs on small previews).
    if getattr(self, 'target_segments', 0) > 200:
        scale_factor = max(scale_factor, 0.5)
        max_attempts_local = max(max_attempts_local, 6)
    small_h, small_w = int(h * scale_factor), int(w * scale_factor)
    rgb_small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)

    # Enhance contrast using CLAHE for better edge detection
    bgr_small = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
    l, a, b_channel = cv2.split(lab)

    # Apply CLAHE only moderately to reduce shadow impact
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Normalize color channels to reduce shadow impact and emphasize material differences
    # Boost color channels (a, b) relative to lightness to prioritize material over shadow
    l_normalized = l_enhanced.astype(float) / 255.0
    a_normalized = (a.astype(float) - 128) / 128.0  # Center and normalize
    b_normalized = (b_channel.astype(float) - 128) / 128.0

    # Default feature image (legacy): emphasize color over brightness
    feature_img_default = np.stack([
        l_normalized * 0.3,  # Reduce lightness weight (shadows)
        a_normalized * 1.2,  # Boost green-red
        b_normalized * 1.2   # Boost blue-yellow
    ], axis=-1)

    # Shadow-robust feature composition: chromaticity + gradient magnitude + color axes
    if getattr(self, 'shadow_robust_var', None) and self.shadow_robust_var.get():
        # Compute chromaticity channels (r/(r+g+b), g/(r+g+b)) to reduce brightness effect
        arr = rgb_small.astype(np.float32)
        denom = arr.sum(axis=2, keepdims=True) + 1e-6
        chroma_r = (arr[:, :, 0:1] / denom).squeeze()
        chroma_g = (arr[:, :, 1:2] / denom).squeeze()
        # Use a and b channels from LAB for material color
        # Compute gradient magnitude from grayscale (lightness) to emphasize edges
        gray = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        # Normalize gradients to 0..1
        if grad.max() > 0:
            grad_norm = grad / grad.max()
        else:
            grad_norm = grad
        # Compose feature image with a, b, and gradient (higher priority) - stack into 3 channels
        feature_img = np.stack([
            a_normalized,     # material color a
            b_normalized,     # material color b
            grad_norm         # edges/texture
        ], axis=-1)
    else:
        feature_img = feature_img_default

    # Scale back to 0-1 range and convert to uint8
    feature_img = (feature_img - feature_img.min()) / (feature_img.max() - feature_img.min())
    rgb_enhanced = (feature_img * 255).astype(np.uint8)

    # Create downsampled ROI mask from boundary
    boundary_small = self.current_boundary * scale_factor
    roi_mask_small = np.zeros((small_h, small_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask_small, [boundary_small.astype(np.int32)], 255)
    roi_area = np.sum(roi_mask_small > 0)
    # Compute buffer mask (in full-resolution coordinates) and map to small scale
    self._compute_buffer_mask()  # ensures self.buffer_mask exists
    if self.buffer_mask is not None:
        buffer_small = cv2.resize(self.buffer_mask.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        buffer_small = np.zeros_like(roi_mask_small, dtype=bool)
    # Build expanded ROI including interior + outside buffer area
    roi_expanded_small = (roi_mask_small > 0) | buffer_small
    # Create an edge/boundary mask (1-pixel) in the small scale and exclude it to make the boundary a hard split
    edge_mask_small = np.zeros((small_h, small_w), dtype=np.uint8)
    cv2.polylines(edge_mask_small, [boundary_small.astype(np.int32)], isClosed=True, color=255, thickness=1)
    edge_mask_small = edge_mask_small > 0
    # Exclude the boundary pixels from ROI to enforce split
    roi_expanded_small[edge_mask_small] = False
    # If shadow_robust was toggled, include its state in the debug prints
    print(f"DEBUG _generate_segments: shadow_robust={getattr(self, 'shadow_robust_var', False).get() if hasattr(self, 'shadow_robust_var') else False}")
    
    # Build the feature image used for Felzenszwalb segmentation
    # This abstracts the previous enhancement + gives us a shadow-robust path
    rgb_enhanced = self._build_segment_features(rgb_small)

    # Optionally pre-smooth to reduce texture/shadow noise
    if getattr(self, 'pre_smooth_var', None) and self.pre_smooth_var.get():
        try:
            # bilateral on BGR image (preserve edges)
            bgr_small = cv2.bilateralFilter(bgr_small, d=9, sigmaColor=75, sigmaSpace=75)
            rgb_small = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

    # Use Felzenszwalb segmentation - follows edges with straighter boundaries
    # Adjust scale based on ROI size and target segments. The 'scale' parameter controls coarseness; smaller -> more segments.
    estimated_scale = float(roi_area) / max(self.target_segments, 1)
    # Apply a moderate multiplier and enforce a small lower bound to allow fine segmentation
    init_scale = max(int(max(estimated_scale * 0.8, 10)), 10)
    # Adjust min_size based on ROI area - for fast preview use larger minimum to reduce compute
    if getattr(self, 'fast_preview_var', None) and self.fast_preview_var.get():
        min_size_base = max(50, int(max(roi_area * 0.001, 50)))
    else:
        min_size_base = max(20, int(max(roi_area * 0.0005, 20)))  # Allow fairly small segments (>=20 px)

    # Adaptive segmentation: try multiple attempts with decreasing scale/min_size if results are too coarse
    scale = init_scale
    min_size = min_size_base
    max_attempts = max_attempts_local
    attempt = 0
    target_threshold = max(5, int(self.target_segments // 20))  # aim for at least this many segments
    segments_small = None
    segments_full = None
    while attempt < max_attempts:
        segments_small = felzenszwalb(rgb_enhanced, scale=scale, sigma=0, min_size=min_size)
        # Upscale for evaluation
        segments_full = cv2.resize(segments_small, (w, h), interpolation=cv2.INTER_NEAREST)
        unique_full = len(np.unique(segments_full))
        print(f"DEBUG _generate_segments attempt={attempt}, scale={scale}, min_size={min_size}, unique={unique_full}")
        # If segmentation meets threshold, accept
        if unique_full >= target_threshold:
            break
        # Otherwise, make segmentation finer and retry (more aggressive reductions)
        attempt += 1
        old_scale, old_min = scale, min_size
        # Reduce scale by 40% (more gradual) and allow min_size down to 1
        scale = max(1, int(max(1, scale * 0.6)))
        min_size = max(1, int(max(1, min_size * 0.5)))
        print(f"DEBUG _generate_segments: retry={attempt}, scale {old_scale}->{scale}, min_size {old_min}->{min_size}, unique_full={unique_full}")
    # segments_small and segments_full set from last attempt

    # Final fallback: if segmentation is still trivially small, force a fine-grained attempt (scale=1,min_size=1)
    unique_full = len(np.unique(segments_full)) if segments_full is not None else 0
    if unique_full <= 3:
        try:
            print("DEBUG _generate_segments: final fine-grain fallback (scale=1,min_size=1)")
            segments_small = felzenszwalb(rgb_enhanced, scale=1, sigma=0, min_size=1)
            segments_full = cv2.resize(segments_small, (w, h), interpolation=cv2.INTER_NEAREST)
            unique_full = len(np.unique(segments_full))
            print(f"DEBUG _generate_segments: fallback unique_full={unique_full}")
        except Exception:
            pass

    
    # Mask out segments outside expanded ROI (interior + buffer) and renumber
    # Compute full-resolution ROI mask (interior) and buffer mask (precomputed by _compute_buffer_mask)
    roi_mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask_full, [self.current_boundary.astype(np.int32)], 255)
    roi_mask_bool = (roi_mask_full > 0)
    # Ensure buffer mask computed
    self._compute_buffer_mask()
    if getattr(self, 'buffer_mask', None) is not None:
        buffer_mask_full = self.buffer_mask.astype(bool)
    else:
        buffer_mask_full = np.zeros((h, w), dtype=bool)
    # Build expanded ROI and exclude boundary edge pixels to enforce hard split
    roi_expanded = roi_mask_bool | buffer_mask_full
    edge_mask_full = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(edge_mask_full, [self.current_boundary.astype(np.int32)], isClosed=True, color=255, thickness=1)
    edge_mask_full = edge_mask_full > 0
    roi_expanded[edge_mask_full] = False

    segments = np.zeros_like(segments_full)
    segment_map = {}  # Map old IDs to new IDs
    new_id = 1
    
    # Kernels for morphological operations
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smooth edges
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # Break thin connections

    # Preallocate gradient image for boundary strength checks (used in merging)
    gray_full = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray_full, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_full, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_full = np.sqrt(gx * gx + gy * gy)
    
    for old_id in np.unique(segments_full):
        seg_mask = (segments_full == old_id) & (roi_expanded)
        # Allow much smaller segments to survive when target is high
        if seg_mask.sum() < getattr(self, 'min_segment_px', 1000):  # Skip tiny segments
            continue
        
        seg_mask_uint8 = seg_mask.astype(np.uint8) * 255
        
        # First: Apply morphological closing to smooth squiggly edges (fill indentations from parking lines)
        closed = cv2.morphologyEx(seg_mask_uint8, cv2.MORPH_CLOSE, kernel_close)
        
        # Second: Apply morphological opening to break thin connections
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
        
        # Find separate connected components after smoothing and breaking connections
        labeled_components, num_components = ndimage.label(opened > 0)
        
        # Assign each disconnected chunk its own segment ID
        for component_id in range(1, num_components + 1):
            component_mask = labeled_components == component_id
            if component_mask.sum() >= getattr(self, 'min_segment_px', 1000):  # Only keep chunks >= min_segment_px
                # Optionally simplify the component contour to reduce squiggles
                simp_mask = component_mask
                try:
                    simp_mask = self._simplify_component_mask(component_mask, level=self.segment_smoothing_level)
                except Exception:
                    pass
                if simp_mask.sum() >= getattr(self, 'min_segment_px', 1000):
                    segments[simp_mask] = new_id
                    new_id += 1

    # After initial region construction, grad_mag_full is available for merging tests
    self._grad_mag_full = grad_mag_full
    
    self.segments = segments
    self.n_segments = segments.max()  # Actual segment count

    # Enforce boundary as hard split to ensure no segment crosses the boundary
    try:
        self._enforce_boundary_split()
    except Exception:
        pass

    # Optional auto-merge step to reduce spurious small regions (run after boundary split)
    if getattr(self, 'auto_merge_var', None) and self.auto_merge_var.get():
        try:
            self._postprocess_merge(target=self.target_segments)
        except Exception as e:
            print('Warning: postprocess merge failed', e)


    # Update smoothing level from current UI selection (if any)
    try:
        self.segment_smoothing_level = self.segment_smoothing_var.get()
    except Exception:
        pass

    # Display segments
    self._update_display()
    
    try:
        self.set_seg_status(f"✓ {self.n_segments} segments (target: {self.target_segments})")
    except Exception:
        pass
    self._update_progress()


def _simplify_component_mask(self, mask, level='med'):
    """Simplify the polygon of a boolean component mask.
    level: 'off'|'low'|'med'|'high'
    Returns a boolean mask of the simplified polygon (same shape as mask).
    """
    eps_frac = self.SMOOTHING_EPS.get(level, 0.01)
    if eps_frac <= 0.0:
        return mask
    mask_uint8 = mask.astype('uint8') * 255
    # Apply a small morphological closing to fill jagged teeth before contour extraction
    kernel_size_map = {'off':1, 'low':3, 'med':5, 'high':7}
    k = kernel_size_map.get(level, 5)
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed0 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    except Exception:
        closed0 = mask_uint8
    contours, _ = cv2.findContours(closed0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    c = max(contours, key=lambda x: cv2.contourArea(x))
    perim = cv2.arcLength(c, True)
    eps = max(1.0, perim * eps_frac)

    # Optional contour smoothing (Chaikin corner cutting) to remove spike extremes
    smoothing_iters_map = {'off': 0, 'low': 1, 'med': 2, 'high': 3}
    iters = smoothing_iters_map.get(level, 2)

    def chaikin(points, iterations=1):
        pts = points.copy()
        for _ in range(max(0, iterations)):
            if len(pts) < 2:
                break
            new_pts = []
            n = len(pts)
            for i in range(n):
                p0 = pts[i]
                p1 = pts[(i + 1) % n]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_pts.append(q)
                new_pts.append(r)
            pts = np.array(new_pts)
        return pts

    try:
        pts = c.reshape(-1, 2).astype(float)
        if iters > 0 and len(pts) >= 3:
            smoothed = chaikin(pts, iterations=iters)
            # Remove consecutive duplicates and ensure at least 3 points
            # Round to integer and remove duplicates preserving order
            rounded = np.round(smoothed).astype(int)
            # Remove duplicates by checking consecutive equal rows
            keep_idx = [0]
            for i in range(1, len(rounded)):
                if not np.array_equal(rounded[i], rounded[i - 1]):
                    keep_idx.append(i)
            rounded = rounded[keep_idx]
            if len(rounded) >= 3:
                smooth_cnt = rounded.reshape(-1, 1, 2).astype(np.int32)
                approx = cv2.approxPolyDP(smooth_cnt, eps, True)
            else:
                approx = cv2.approxPolyDP(c, eps, True)
        else:
            approx = cv2.approxPolyDP(c, eps, True)
    except Exception:
        approx = cv2.approxPolyDP(c, eps, True)

    poly_mask = np.zeros_like(mask_uint8)
    try:
        cv2.fillPoly(poly_mask, [approx], 255)
        poly_bool = poly_mask.astype(bool)
        # Do not allow simplification to increase area beyond original - intersect with original mask
        poly_bool = poly_bool & mask
        # Ensure we didn't dramatically shrink area; otherwise fallback
        area_thresholds = {'off': 1.0, 'low': 0.2, 'med': 0.5, 'high': 0.3}
        frac = area_thresholds.get(level, 0.3)
        if poly_bool.sum() >= max( int(mask.sum() * frac), 1 ):
            return poly_bool
        else:
            return mask
    except Exception:
        return mask


def _postprocess_merge(self, target:int=None, min_size:int=None, boundary_grad_thresh:float=20.0):
    if min_size is None:
        min_size = getattr(self, 'min_segment_px', 100)
    """Merge small regions towards neighbors based on color similarity and weak boundary strength.
    - target: desired approximate number of segments. Merge small regions until reaching target or no merges possible.
    - min_size: size below which a region is considered 'small' and eligible to be merged.
    - boundary_grad_thresh: mean gradient magnitude threshold; don't merge across strong boundaries.
    """
    if self.segments is None:
        return
    seg = self.segments.copy()
    h, w = seg.shape
    unique_ids = [int(x) for x in np.unique(seg) if x != 0]
    if len(unique_ids) <= 1:
        return
    # compute region areas and mean color
    rgb = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB).astype(float)
    areas = {}
    means = {}
    for uid in unique_ids:
        mask = (seg == uid)
        areas[uid] = int(mask.sum())
        if areas[uid] > 0:
            means[uid] = rgb[mask].mean(axis=0)
        else:
            means[uid] = np.array([0.0, 0.0, 0.0])
    # precompute adjacency and boundary gradients using edge pairs (right and down neighbors)
    from collections import defaultdict
    boundary_pixels = defaultdict(list)  # (a,b) -> list of coords
    for dy, dx in [(0,1),(1,0)]:
        a = seg[:, :-dx or None]
        b = seg[:, dx or None:]
        if dx == 1:
            coords = np.argwhere(a != b)
            for y,x in coords:
                id_a = int(seg[y,x])
                id_b = int(seg[y,x+1])
                if id_a == id_b or id_a == 0 or id_b == 0:
                    continue
                key = tuple(sorted((id_a, id_b)))
                boundary_pixels[key].append((y,x))
        else:
            coords = np.argwhere(a != b)
            for y,x in coords:
                id_a = int(seg[y,x])
                id_b = int(seg[y+1,x])
                if id_a == id_b or id_a == 0 or id_b == 0:
                    continue
                key = tuple(sorted((id_a, id_b)))
                boundary_pixels[key].append((y,x))
    # iterative small-region merging (merge small slivers even if current count <= target)
    current_unique = set(unique_ids)
    target = target or self.target_segments
    iters = 0
    while iters < 1000:
        iters += 1
        # find smallest region
        small_id = min(current_unique, key=lambda u: areas.get(u, 0))
        # Stop condition: smallest region large enough and we are at or below target
        if areas.get(small_id, 0) >= min_size and len(current_unique) <= max(1, int(target)):
            break
        # find neighbors
        neighbor_candidates = []
        for pair, coords in boundary_pixels.items():
            if small_id in pair:
                other = pair[0] if pair[1] == small_id else pair[1]
                neighbor_candidates.append((other, coords))
        if not neighbor_candidates:
            # isolated small region; remove it by assigning to largest region
            largest = max(current_unique, key=lambda u: areas.get(u,0))
            if largest == small_id:
                break
            seg[seg==small_id] = largest
            current_unique.remove(small_id)
        else:
            # evaluate best neighbor by color distance and boundary gradient
            best_n = None
            best_score = float('inf')
            for other, coords in neighbor_candidates:
                grads = [self._grad_mag_full[y,x] for (y,x) in coords]
                mean_grad = float(np.mean(grads)) if grads else 0.0
                if mean_grad > boundary_grad_thresh:
                    continue
                # color distance (Euclidean in RGB)
                dist = np.linalg.norm(means.get(small_id, np.zeros(3)) - means.get(other, np.zeros(3)))
                # prefer neighbor with small distance and reasonable area
                score = dist / (1 + areas.get(other,1))
                if score < best_score:
                    best_score = score
                    best_n = other
            if best_n is None:
                # no neighbor suitable (strong borders) - stop
                break
            seg[seg==small_id] = best_n
            current_unique.remove(small_id)
        # recompute adjacency/areas/means for next iteration
        boundary_pixels = defaultdict(list)
        unique_ids2 = set([int(x) for x in np.unique(seg) if x != 0])
        for dy, dx in [(0,1),(1,0)]:
            a = seg[:, :-dx or None]
            b = seg[:, dx or None:]
            if dx == 1:
                coords2 = np.argwhere(a != b)
                for y,x in coords2:
                    id_a = int(seg[y,x])
                    id_b = int(seg[y,x+1])
                    if id_a == id_b or id_a == 0 or id_b == 0:
                        continue
                    key = tuple(sorted((id_a, id_b)))
                    boundary_pixels[key].append((y,x))
            else:
                coords2 = np.argwhere(a != b)
                for y,x in coords2:
                    id_a = int(seg[y,x])
                    id_b = int(seg[y+1,x])
                    if id_a == id_b or id_a == 0 or id_b == 0:
                        continue
                    key = tuple(sorted((id_a, id_b)))
                    boundary_pixels[key].append((y,x))
        areas = {}
        means = {}
        for uid in unique_ids2:
            mask = (seg == uid)
            areas[uid] = int(mask.sum())
            if areas[uid] > 0:
                means[uid] = rgb[mask].mean(axis=0)
            else:
                means[uid] = np.array([0.0,0.0,0.0])
    # reassign compact ids
    new_seg = np.zeros_like(seg, dtype=np.int32)
    new_id = 1
    for uid in sorted([int(x) for x in np.unique(seg) if x != 0]):
        new_seg[seg == uid] = new_id
        new_id += 1
    self.segments = new_seg
    self.n_segments = int(self.segments.max())
    print(f"_postprocess_merge: reduced to {self.n_segments} segments (target {target}) after {iters} iterations")
    return



def generate_segments(clean_image: Any, boundary: Any, *, target_segments: int = 50, **options) -> Dict[str, Any]:
    """Generate segments for the ROI defined by boundary.

    Returns a dict containing at least: {'segments': ndarray, 'n_segments': int}
    """
    raise NotImplementedError
