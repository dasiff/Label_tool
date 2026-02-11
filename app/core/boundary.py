from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import sys
from scipy.ndimage import gaussian_filter

Point = Tuple[float, float]
Polygon = List[Point]


@dataclass
class BoundaryOutput:
    polygon_px: Optional[Polygon]
    confidence: float
    method: str
    warnings: List[str]


def _red_mask_hsv(bgr: np.ndarray) -> np.ndarray:
    """
    Create a binary mask for red-colored overlay lines.
    Detects pure/near-pure red pixels by looking for high red channel
    with low green and blue channels (artificial boundaries are drawn in pure red: 255,0,0 in BGR).
    """
    # Extract color channels (BGR format)
    b = bgr[:, :, 0].astype(np.float32)
    g = bgr[:, :, 1].astype(np.float32)
    r = bgr[:, :, 2].astype(np.float32)
    
    # Detect red pixels: high red, low green, low blue
    # Threshold: R > 200, G < 100, B < 100
    # This catches pure red (255,0,0) and near-red tones
    red_mask = (r > 200) & (g < 100) & (b < 100)
    
    result = (red_mask * 255).astype(np.uint8)
    red_count = np.sum(result > 0)
    print(f"DEBUG _red_mask_hsv: Found {red_count} red pixels (image size: {bgr.shape})")
    
    return result


def estimate_boundary_from_overlay(bgr: np.ndarray) -> BoundaryOutput:
    """
    Milestone 1: Detect red boundary overlay via HSV thresholding + morphology + contour extraction.
    
    CRITICAL: NEVER USE CONVEX HULL - parking lots are often non-convex (L-shaped, irregular)
    Convex hull destroys the actual shape. Always preserve the detected contour as-is.
    
    Process:
    1. Convert BGR to HSV
    2. Threshold red color (handles hue wrapping)
    3. Morphological close to connect line gaps
    4. Extract contour from red pixels (NOT convex hull)
    5. Simplify contour polygon with Douglas-Peucker
    6. Score confidence based on pixel density
    
    Returns polygon in pixel coordinates (x, y).
    """
    h, w = bgr.shape[:2]
    
    # Step 1: Generate red mask
    mask = _red_mask_hsv(bgr)
    red_pixel_count = np.sum(mask > 0)
    print(f"DEBUG _red_mask_hsv: Found {red_pixel_count} red pixels (image size: {bgr.shape})")
    
    # CRITICAL: Detect and merge disconnected boundary components
    # Find connected components in the raw mask
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    print(f"DEBUG: Found {num_components - 1} connected components in raw mask")
    
    if num_components > 2:  # More than background + main boundary
        # Detected multiple components; avoid morphological merging to preserve boundary accuracy
        print(f"DEBUG: Multiple components detected ({num_components - 1}) - proceeding with targeted endpoint/component connections (no morphological merge)")
    
    # After removing tiny outliers, check for disconnected components and small gaps
    # Keep only reasonably large components to filter noise
    num_components2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_components2 > 1:
        areas = [stats2[i, cv2.CC_STAT_AREA] for i in range(1, num_components2)]
        largest_area = max(areas)
        keep_idxs = [i for i in range(1, num_components2) if stats2[i, cv2.CC_STAT_AREA] >= max(50, 0.01 * largest_area)]
        filtered_mask = np.zeros_like(mask)
        for i in keep_idxs:
            filtered_mask[labels2 == i] = 255
        removed = np.sum(mask > 0) - np.sum(filtered_mask > 0)
        if removed > 0:
            print(f"DEBUG: Removed {removed} pixels of small outliers")
        mask = filtered_mask.copy()

    # If there are still multiple components, connect only those that are very close
    num_components3, labels3, stats3, centroids3 = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_components3 > 2:
        # For each pair of components, compute closest contour points and connect if within threshold
        comp_idxs = list(range(1, num_components3))
        comp_contours = {}
        for i in comp_idxs:
            comp_mask = (labels3 == i).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                comp_contours[i] = cnts[0].reshape(-1, 2)
        merge_threshold = 50.0  # only connect components closer than this (px)
        for i in comp_contours:
            for j in comp_contours:
                if j <= i:
                    continue
                pts1 = comp_contours[i]
                pts2 = comp_contours[j]
                # sample points for speed
                s1 = pts1[::10]
                s2 = pts2[::10]
                min_dist = float('inf')
                best_p1 = None
                best_p2 = None
                for p1 in s1:
                    for p2 in s2:
                        d = np.linalg.norm(p1 - p2)
                        if d < min_dist:
                            min_dist = d
                            best_p1 = tuple(p1.astype(int))
                            best_p2 = tuple(p2.astype(int))
                if min_dist <= merge_threshold:
                    cv2.line(mask, best_p1, best_p2, 255, 1)
                    print(f"DEBUG: Merged close components at {best_p1} and {best_p2} (dist {min_dist:.1f}px)")

    # Now find endpoints (4-connectivity) on the cleaned mask
    kernel_4_neighbors = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.uint8)
    neighbor_count = cv2.filter2D((mask > 0).astype(np.uint8), -1, kernel_4_neighbors)
    neighbor_count = neighbor_count * (mask > 0)
    endpoints = np.column_stack(np.where(neighbor_count == 1))  # row, col
    print(f"DEBUG: After cleaning, found {len(endpoints)} endpoint(s): {endpoints[:20] if len(endpoints)>0 else '[]'}")

    # Only connect small gaps: pair endpoints with distance <= small_gap_thresh
    small_gap_thresh = 30.0  # pixels
    max_auto_connect = 12  # guard against connecting too many pairs
    if 0 < len(endpoints) <= max_auto_connect:
        eps = list(endpoints)
        pairs_connected = 0
        # Greedy pairing by shortest distance
        while len(eps) >= 2 and pairs_connected < max_auto_connect:
            min_dist = float('inf')
            best_i, best_j = 0, 1
            for i in range(len(eps)):
                for j in range(i+1, len(eps)):
                    d = np.linalg.norm(eps[i] - eps[j])
                    if d < min_dist:
                        min_dist = d
                        best_i, best_j = i, j
            if min_dist <= small_gap_thresh:
                p1 = tuple(eps[best_i][::-1].astype(int))
                p2 = tuple(eps[best_j][::-1].astype(int))
                cv2.line(mask, p1, p2, 255, 1)
                print(f"DEBUG: Connected endpoint pair {p1} <-> {p2} (gap {min_dist:.1f}px)")
                pairs_connected += 1
                # remove paired endpoints
                eps = [ep for idx, ep in enumerate(eps) if idx not in (best_i, best_j)]
            else:
                # no more small gaps to auto-connect
                break
        if pairs_connected == 0 and len(endpoints) > 0:
            print(f"DEBUG: Found endpoints but none within {small_gap_thresh}px to auto-connect; leaving for manual review")
        # Recompute endpoints remaining after attempted connections
        neighbor_count_after = cv2.filter2D((mask > 0).astype(np.uint8), -1, kernel_4_neighbors)
        neighbor_count_after = neighbor_count_after * (mask > 0)
        endpoints_after = np.column_stack(np.where(neighbor_count_after == 1))
        print(f"DEBUG: Endpoints remaining after auto-connection: {len(endpoints_after)}: {endpoints_after[:20] if len(endpoints_after)>0 else '[]'}")
        # Second pass: try a slightly larger threshold for remaining small gaps
        if 0 < len(endpoints_after) <= max_auto_connect:
            second_thresh = 60.0
            eps = list(endpoints_after)
            pairs2 = 0
            while len(eps) >= 2:
                min_dist = float('inf')
                bi, bj = 0, 1
                for i in range(len(eps)):
                    for j in range(i+1, len(eps)):
                        d = np.linalg.norm(eps[i] - eps[j])
                        if d < min_dist:
                            min_dist = d
                            bi, bj = i, j
                if min_dist <= second_thresh:
                    p1 = tuple(eps[bi][::-1].astype(int))
                    p2 = tuple(eps[bj][::-1].astype(int))
                    cv2.line(mask, p1, p2, 255, 1)
                    print(f"DEBUG: [2nd pass] Connected endpoint pair {p1} <-> {p2} (gap {min_dist:.1f}px)")
                    pairs2 += 1
                    eps = [ep for idx, ep in enumerate(eps) if idx not in (bi, bj)]
                else:
                    break
            if pairs2 == 0 and len(eps) > 0:
                print(f"DEBUG: Remaining endpoints after 2nd pass: {len(eps)} - not auto-connecting")
    elif len(endpoints) > max_auto_connect:
        print(f"DEBUG: Too many endpoints ({len(endpoints)}) - skipping auto-connection to avoid errors")
    
    # For very irregular/broken boundaries, do a stronger CLOSE
    # Check if we have disconnected components
    num_labels_initial, labels_initial, stats_initial, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # We'll bridge gaps manually, so only use gentle close for slightly broken boundaries
    if num_labels_initial > 4:  # Many small components = very broken, needs strong close
        print(f"DEBUG: Detected very broken boundary with {num_labels_initial - 1} components - applying stronger CLOSE")
        # Use larger kernel to bridge small gaps
        strong_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, strong_close_kernel, iterations=2)
        print(f"DEBUG: After strong CLOSE (7x7, iter=2): {np.sum(mask > 0)} pixels")
        # Save original mask before component cleanup for gap detection
    mask_before_cleanup = mask.copy()
        # Step 2.5: Remove small isolated components (outliers)
    # Keep anything connected to the main boundary
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    print(f"DEBUG: Found {num_labels - 1} connected components")
    if num_labels > 1:  # More than just background
        # Find the largest component (the main boundary)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        largest_idx = np.argmax(areas) + 1  # +1 because we skip background (0)
        largest_area = areas[largest_idx - 1]
        
        # Keep largest component + any component at least 10% of its size AND at least 50 pixels
        # This filters out single-pixel or tiny outliers
        mask_cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if i == largest_idx or (area >= largest_area * 0.1 and area >= 50):
                mask_cleaned[labels == i] = 255
        mask = mask_cleaned
        print(f"DEBUG: After outlier removal: {np.sum(mask > 0)} pixels")
    
    # Step 3: Final gentle cleanup - do not use morphological close to bridge large gaps;
    # rely on explicit endpoint connections to preserve boundary accuracy
    print(f"DEBUG: Skipping final morphological close to preserve boundary accuracy; pixel count: {np.sum(mask > 0)}")
    
    # Re-check component count after cleanup for gap detection
    num_labels_final, labels_final, stats_final, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    print(f"DEBUG: After cleanup: {num_labels_final - 1} components remain")

    # Step 3.5: Find all red pixels (ignore pixels within 5px of image edge to reduce noise)
    edge_margin = 5
    mask_interior = mask.copy()
    mask_interior[:edge_margin, :] = 0
    mask_interior[-edge_margin:, :] = 0
    mask_interior[:, :edge_margin] = 0
    mask_interior[:, -edge_margin:] = 0
    
    ys, xs = np.where(mask_interior > 0)
    red_pixel_count = len(xs)
    
    # Minimum threshold: need at least 100 pixels for a valid boundary (was 500 for thick test lines)
    if red_pixel_count < 100:
        return BoundaryOutput(
            polygon_px=None,
            confidence=0.0,
            method="red_overlay_contour",
            warnings=[f"Could not find enough red boundary pixels (found {red_pixel_count}, need 100+)."],
        )

    # Step 3.5: For broken boundaries, find endpoints and connect them
    # This handles gaps in the red line without morphological operations
    if num_labels_final > 2:
        print(f"DEBUG: Multiple components detected ({num_labels_final - 1}) - finding endpoints to connect")
        
        # Find the two largest components
        areas = [(i, stats_final[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels_final)]
        areas.sort(key=lambda x: x[1], reverse=True)
        
        if len(areas) >= 2:
            # Get masks for the two largest components
            comp1_idx, comp2_idx = areas[0][0], areas[1][0]
            comp1_mask = (labels_final == comp1_idx).astype(np.uint8) * 255
            comp2_mask = (labels_final == comp2_idx).astype(np.uint8) * 255
            
            # Find endpoints of each component (pixels with only 1 or 2 neighbors)
            def find_endpoints(mask):
                # Get skeleton
                skeleton = cv2.ximgproc.thinning(mask)
                # Find pixels with only 1 neighbor (endpoints)
                kernel = np.ones((3, 3), np.uint8)
                neighbors = cv2.filter2D(skeleton // 255, -1, kernel) - 1  # Subtract self
                endpoints = np.where((skeleton > 0) & (neighbors <= 1))
                return list(zip(endpoints[1], endpoints[0]))  # (x, y) format
            
            endpoints1 = find_endpoints(comp1_mask)
            endpoints2 = find_endpoints(comp2_mask)
            
            if len(endpoints1) > 0 and len(endpoints2) > 0:
                # Find closest pair of endpoints between the two components
                min_dist = float('inf')
                best_p1, best_p2 = None, None
                for p1 in endpoints1:
                    for p2 in endpoints2:
                        dist = np.linalg.norm(np.array(p1) - np.array(p2))
                        if dist < min_dist:
                            min_dist = dist
                            best_p1, best_p2 = p1, p2
                
                if best_p1 and best_p2:
                    # Draw thick line connecting the endpoints
                    cv2.line(mask_interior, best_p1, best_p2, 255, 10)
                    print(f"DEBUG: Connected endpoints {best_p1} to {best_p2} (gap: {min_dist:.1f}px)")
                    
                    # Apply smoothing to the connected line
                    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    mask_interior = cv2.morphologyEx(mask_interior, cv2.MORPH_CLOSE, smooth_kernel, iterations=2)
            else:
                print(f"DEBUG: Could not find clear endpoints, using fallback connection")
                # Fallback: connect closest points between components
                pts1 = np.column_stack(np.where(comp1_mask > 0))[:, ::-1]
                pts2 = np.column_stack(np.where(comp2_mask > 0))[:, ::-1]
                min_dist = float('inf')
                best_p1, best_p2 = None, None
                for p1 in pts1[::10]:
                    for p2 in pts2[::10]:
                        dist = np.linalg.norm(p1 - p2)
                        if dist < min_dist:
                            min_dist = dist
                            best_p1, best_p2 = tuple(p1.astype(int)), tuple(p2.astype(int))
                
                if best_p1 and best_p2:
                    cv2.line(mask_interior, best_p1, best_p2, 255, 10)
                    print(f"DEBUG: Connected closest points (gap: {min_dist:.1f}px)")
    
    # Simple approach: Detect and close small gaps by connecting endpoints
    # After initial morphological operations, find endpoints and connect them
    
    # Find endpoints in the current mask (pixels with exactly 1 neighbor)
    kernel_neighbors = np.ones((3, 3), np.uint8)
    kernel_neighbors[1, 1] = 0
    neighbor_count = cv2.filter2D((mask_interior > 0).astype(np.uint8), -1, kernel_neighbors)
    neighbor_count = neighbor_count * (mask_interior > 0)
    
    endpoints = np.column_stack(np.where(neighbor_count == 1))  # row, col
    print(f"DEBUG: Found {len(endpoints)} endpoint(s) in mask after initial processing")
    
    # If we have exactly 2 endpoints, connect them (they're the gap)
    if len(endpoints) == 2:
        ep1 = tuple(endpoints[0][::-1].astype(int))  # (x, y)
        ep2 = tuple(endpoints[1][::-1].astype(int))
        gap_dist = np.linalg.norm(endpoints[0] - endpoints[1])
        cv2.line(mask_interior, ep1, ep2, 255, 1)
        print(f"DEBUG: Connected 2 endpoints with gap of {gap_dist:.1f}px")
    elif len(endpoints) > 2:
        # Multiple gaps - connect closest pairs
        eps_list = list(endpoints)
        while len(eps_list) >= 2:
            min_dist = float('inf')
            best_i, best_j = 0, 1
            for i in range(len(eps_list)):
                for j in range(i+1, len(eps_list)):
                    dist = np.linalg.norm(eps_list[i] - eps_list[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_i, best_j = i, j
            
            ep1 = tuple(eps_list[best_i][::-1].astype(int))
            ep2 = tuple(eps_list[best_j][::-1].astype(int))
            cv2.line(mask_interior, ep1, ep2, 255, 1)
            print(f"DEBUG: Connected endpoints with gap of {min_dist:.1f}px")
            
            eps_list = [ep for idx, ep in enumerate(eps_list) if idx not in [best_i, best_j]]
    else:
        print(f"DEBUG: Boundary already closed ({len(endpoints)} endpoints)")
    
    # Use the gap-closed mask for contour extraction
    mask_closed_large = mask_interior

    # If there are exactly two external contours, try joining them by nearest endpoints
    contours_tmp, _ = cv2.findContours(mask_closed_large, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours_tmp) == 2:
        print(f"DEBUG: Exactly two contours found - attempting nearest-endpoint joins")
        # Recompute 4-connectivity endpoints on the current mask
        kernel_4_neighbors = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=np.uint8)
        neighbor_count_tmp = cv2.filter2D((mask_closed_large > 0).astype(np.uint8), -1, kernel_4_neighbors)
        neighbor_count_tmp = neighbor_count_tmp * (mask_closed_large > 0)
        endpoints_tmp = np.column_stack(np.where(neighbor_count_tmp == 1))  # row, col

        cnt_pts = [c.reshape(-1, 2) for c in contours_tmp]

        # Assign endpoints to closest contour (if no endpoints found for a contour we'll fall back to contour points)
        assign = {0: [], 1: []}
        for (r, c) in endpoints_tmp:
            d0 = np.min(np.linalg.norm(cnt_pts[0] - np.array([c, r]), axis=1))
            d1 = np.min(np.linalg.norm(cnt_pts[1] - np.array([c, r]), axis=1))
            if d0 <= d1:
                assign[0].append((r, c))
            else:
                assign[1].append((r, c))

        # If a contour has no endpoints, use a few sampled contour points as candidates
        if len(assign[0]) == 0:
            s = cnt_pts[0][::max(1, len(cnt_pts[0])//50)]
            assign[0] = [(int(pt[1]), int(pt[0])) for pt in s[:10]]
        if len(assign[1]) == 0:
            s = cnt_pts[1][::max(1, len(cnt_pts[1])//50)]
            assign[1] = [(int(pt[1]), int(pt[0])) for pt in s[:10]]

        # Build all pairwise distances and sort
        pairs = []
        for p0 in assign[0]:
            for p1 in assign[1]:
                d = np.hypot(p0[0]-p1[0], p0[1]-p1[1])
                pairs.append((d, p0, p1))
        pairs.sort(key=lambda x: x[0])

        # Connect closest pairs until components reduce to 1 or we exhaust reasonable pairs
        max_pairs = 6
        pairs_used = 0
        for d, p0, p1 in pairs:
            if pairs_used >= max_pairs:
                break
            # Draw line between p0 (r,c) and p1 (r,c) on mask
            pt1 = (int(p0[1]), int(p0[0]))  # (x,y)
            pt2 = (int(p1[1]), int(p1[0]))
            cv2.line(mask_closed_large, pt1, pt2, 255, 1)
            print(f"DEBUG: Connected contours at {pt1} <-> {pt2} (dist {d:.1f}px)")
            pairs_used += 1
            # Recompute components
            nlab, labs, stats, _ = cv2.connectedComponentsWithStats(mask_closed_large, connectivity=8)
            if nlab <= 2:  # background + main
                print("DEBUG: Contours merged successfully")
                break
        else:
            print("DEBUG: Tried nearest-pair joins but contours may still be disconnected")

    # The mask contains the boundary line. To get the parking lot contour,
    # we need to find the interior boundary.
    # Flood fill from the corners to mark exterior, then find contours of interior
    filled_mask = mask_closed_large.copy()
    
    # Flood fill from corners to mark exterior
    h, w = filled_mask.shape
    cv2.floodFill(filled_mask, None, (0, 0), 128)  # Mark exterior
    cv2.floodFill(filled_mask, None, (w-1, 0), 128)
    cv2.floodFill(filled_mask, None, (0, h-1), 128)  
    cv2.floodFill(filled_mask, None, (w-1, h-1), 128)
    
    # Now the interior (parking lot) is still 255, exterior is 128, boundary is 255
    # Save debug images to help diagnose small-gap issues
    try:
        os.makedirs('data/debug', exist_ok=True)
        cv2.imwrite('data/debug/laquinta_mask_closed_large.png', mask_closed_large)
    except Exception:
        pass

    # Interior pixels are those not reachable from the corners (i.e., still 0 after marking exterior)
    interior_mask = (filled_mask == 0).astype(np.uint8) * 255
    print(f"DEBUG: Interior mask area: {np.count_nonzero(interior_mask)} (should be large if boundary closed)")
    try:
        cv2.imwrite('data/debug/laquinta_interior_mask.png', interior_mask)
    except Exception:
        pass

    contours, _ = cv2.findContours(interior_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    warnings = []
    if len(contours) == 0:
        print("DEBUG: No interior contours found - attempting skeleton-based gap closure", file=sys.stderr)
        # Skeleton-based fallback: compute 1-pixel skeleton and iteratively connect nearest endpoints
        try:
            # Prefer skimage's skeletonize when available
            from skimage.morphology import skeletonize
            sk = skeletonize((mask_interior > 0))
            sk = (sk.astype(np.uint8) * 255)
        except Exception:
            # Fallback simple thinning approach
            sk = mask_interior.copy()
            kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
            for _ in range(200):
                eroded = cv2.erode(sk, kernel, iterations=1)
                temp = cv2.dilate(eroded, kernel, iterations=1)
                sk_next = sk - (sk & (sk - temp))
                if np.array_equal(sk_next, sk):
                    break
                sk = sk_next
            sk = (sk > 0).astype(np.uint8) * 255

        # Find skeleton endpoints
        kernel8 = np.ones((3,3), np.uint8); kernel8[1,1]=0
        neighbor_count = cv2.filter2D((sk > 0).astype(np.uint8), -1, kernel8)
        endpoints = np.column_stack(np.where((sk > 0) & (neighbor_count == 1)))  # row,col
        print(f"DEBUG: Skeleton endpoints found: {len(endpoints)}", file=sys.stderr)

        # If endpoints are too few, sample points along contour as candidates
        candidate_pts = []
        if len(endpoints) >= 2:
            candidate_pts = [(int(r), int(c)) for (r, c) in endpoints]
        else:
            cnts_tmp, _ = cv2.findContours(mask_interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in cnts_tmp:
                pts = cnt.reshape(-1,2)
                step = max(1, len(pts)//200)
                sampled = pts[::step][:50]
                for pt in sampled:
                    candidate_pts.append((int(pt[1]), int(pt[0])))
            print(f"DEBUG: Using {len(candidate_pts)} contour-sampled candidates", file=sys.stderr)

        # Build candidate pairs from candidate_pts
        if len(candidate_pts) >= 2:
            pairs = []
            N = len(candidate_pts)
            for i in range(N):
                r0, c0 = candidate_pts[i]
                for j in range(i+1, N):
                    r1, c1 = candidate_pts[j]
                    d = np.hypot(r0 - r1, c0 - c1)
                    pairs.append((d, (r0, c0), (r1, c1)))
            pairs.sort(key=lambda x: x[0])

            mask_join = mask_interior.copy()
            interior_area = 0
            applied = 0

            def _is_connected(binary_mask, p1, p2):
                # BFS in small bbox around points to check for connectivity
                h_, w_ = binary_mask.shape
                r1, c1 = p1; r2, c2 = p2
                xmin = max(0, min(c1, c2) - 5); xmax = min(w_-1, max(c1, c2) + 5)
                ymin = max(0, min(r1, r2) - 5); ymax = min(h_-1, max(r1, r2) + 5)
                sub = (binary_mask[ymin:ymax+1, xmin:xmax+1] > 0).astype(np.uint8)
                start = (r1 - ymin, c1 - xmin); goal = (r2 - ymin, c2 - xmin)
                from collections import deque
                q = deque([start]); seen = {start}
                while q:
                    y, x = q.popleft()
                    if (y, x) == goal:
                        return True
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < sub.shape[0] and 0 <= nx < sub.shape[1] and sub[ny, nx] and (ny, nx) not in seen:
                            seen.add((ny, nx)); q.append((ny, nx))
                return False

            max_attempts = min(400, len(pairs))
            for idx in range(max_attempts):
                d, p0, p1 = pairs[idx]
                if d > max(mask_interior.shape) * 0.5:
                    # skip excessively large joins
                    continue
                # Try increasing thickness until connected locally
                thickness = 1
                connected = False
                while thickness <= 11:
                    tmp = mask_join.copy()
                    # draw line and endpoint caps
                    cv2.line(tmp, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), 255, thickness)
                    cv2.circle(tmp, (int(p0[1]), int(p0[0])), max(1, thickness//2), 255, -1)
                    cv2.circle(tmp, (int(p1[1]), int(p1[0])), max(1, thickness//2), 255, -1)
                    if _is_connected(tmp, p0, p1):
                        # accept join
                        cv2.line(mask_join, (int(p0[1]), int(p0[0])), (int(p1[1]), int(p1[0])), 255, thickness)
                        cv2.circle(mask_join, (int(p0[1]), int(p0[0])), max(1, thickness//2), 255, -1)
                        cv2.circle(mask_join, (int(p1[1]), int(p1[0])), max(1, thickness//2), 255, -1)
                        connected = True
                        applied += 1
                        print(f"DEBUG: Applied skeleton/contour join #{applied} (thickness={thickness}, dist={d:.1f})", file=sys.stderr)
                        break
                    thickness += 2
                if not connected:
                    print(f"DEBUG: Could not connect candidate points at dist {d:.1f}", file=sys.stderr)
                # Check interior after each applied join
                filled_tmp = mask_join.copy()
                cv2.floodFill(filled_tmp, None, (0, 0), 128)
                interior_tmp = (filled_tmp == 0).astype(np.uint8) * 255
                interior_area = np.count_nonzero(interior_tmp)
                print(f"DEBUG: After {applied} joins, interior area={interior_area}", file=sys.stderr)
                if interior_area > max(1000, 0.001 * h * w):
                    mask_interior = mask_join
                    contours, _ = cv2.findContours(interior_tmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    warnings.append("skeleton_gap_closure")
                    print("DEBUG: Skeleton/contour gap closure succeeded", file=sys.stderr)
                    break

        if len(contours) == 0:
            # still no contours after skeleton attempts
            return BoundaryOutput(
                polygon_px=None,
                confidence=0.0,
                method="red_overlay_morphological",
                warnings=["No interior contours found."] + warnings,
            )


    # Use the largest interior contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    image_area = h * w
    x, y, w_bbox, h_bbox = cv2.boundingRect(largest_contour)
    bbox_area = w_bbox * h_bbox
    
    print(f"DEBUG: Contour bbox: {w_bbox}×{h_bbox} ({bbox_area/image_area:.1%} of image)")
    
    # Step 5: Use the exact exterior contour of the filled interior as the polygon
    approx = largest_contour.reshape(-1, 2)

    # Remove consecutive duplicate/near-duplicate points (distance < 1px)
    cleaned_pts = [tuple(map(int, approx[0]))]
    for pt in approx[1:]:
        pt_t = tuple(map(int, pt))
        if np.linalg.norm(np.array(pt_t) - np.array(cleaned_pts[-1])) >= 1.0:
            cleaned_pts.append(pt_t)
    # Also remove the final point if it duplicates the first
    if len(cleaned_pts) > 1 and cleaned_pts[0] == cleaned_pts[-1]:
        cleaned_pts = cleaned_pts[:-1]

    print(f"DEBUG: Raw contour points: {len(cleaned_pts)} points before simplification")

    # Adaptive Douglas-Peucker: target a polygon of ~50-100 vertices
    pts_np = np.array(cleaned_pts, dtype=np.int32).reshape(-1, 1, 2)
    peri = cv2.arcLength(pts_np, True)
    target_min, target_max = 50, 100

    # Binary search epsilon between tiny and modest fraction of perimeter
    low = 0.0001 * peri
    high = 0.02 * peri
    best_approx = None
    best_score = float('inf')
    best_eps = None
    for _ in range(20):
        mid = (low + high) / 2.0
        approx_mid = cv2.approxPolyDP(pts_np, max(1.0, mid), True).reshape(-1, 2)
        n = len(approx_mid)
        # Score: distance from target range center
        center = (target_min + target_max) / 2.0
        score = abs(n - center)
        if score < best_score:
            best_score = score
            best_approx = approx_mid
            best_eps = mid
        if target_min <= n <= target_max:
            best_approx = approx_mid
            best_eps = mid
            break
        if n < target_min:
            # too few vertices -> decrease epsilon
            high = mid
        else:
            # too many vertices -> increase epsilon
            low = mid

    if best_approx is None:
        # fallback to light simplification
        eps_fb = max(1.0, 0.001 * peri)
        best_approx = cv2.approxPolyDP(pts_np, eps_fb, True).reshape(-1, 2)
        best_eps = eps_fb

    print(f"DEBUG: Adaptive simplified to {len(best_approx)} vertices (eps~{best_eps:.2f})")

    # Convert to list of (x, y) tuples (float for consistency)
    polygon = [(float(x), float(y)) for x, y in best_approx]

    # Ensure polygon is closed (first and last points should be the same for fillPoly)
    if len(polygon) > 0 and polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    # Step 6: Confidence scoring
    # Heuristic: density of red pixels relative to image area
    # Assume 0.5-1.5% of image should be red boundary pixels
    pixel_density = float(red_pixel_count) / float(h * w)
    expected_density = 0.01  # 1%
    
    # Confidence: 1.0 if density ~1%, 0.5 at edges (0.5% or 2%), 0.0 outside [0.1%, 5%]
    if pixel_density < 0.001:
        confidence = 0.0
    elif pixel_density > 0.05:
        confidence = 0.5  # too much red (noisy image)
    else:
        # Peak at expected_density, drop off towards edges
        confidence = 1.0 - abs(pixel_density - expected_density) / (2 * expected_density)
        confidence = float(np.clip(confidence, 0.0, 1.0))

    warnings = []
    if len(polygon) > 50:
        warnings.append(f"Boundary has {len(polygon)} vertices - consider reviewing for complexity.")
    
    return BoundaryOutput(
        polygon_px=polygon,
        confidence=confidence,
        method="red_overlay_morphological",
        warnings=warnings,
    )


def save_boundary_debug_image(bgr: np.ndarray, polygon: Optional[Polygon], output_path: str):
    """
    Save a debug image showing the detected red line and reconstructed polygon.
    Creates a side-by-side comparison.
    
    Args:
        bgr: Original image (BGR)
        polygon: Detected polygon (can be None)
        output_path: Where to save the debug image
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    h, w = bgr.shape[:2]
    
    # Left: Red mask after morphological operations
    mask = _red_mask_hsv(bgr)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    
    num_labels_initial, labels_initial, stats_initial, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels_initial > 3:
        strong_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, strong_close_kernel, iterations=3)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        largest_idx = np.argmax(areas) + 1
        largest_area = areas[largest_idx - 1]
        mask_cleaned = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if i == largest_idx or area >= largest_area * 0.1:
                mask_cleaned[labels == i] = 255
        mask = mask_cleaned
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    
    # Convert mask to RGB for display
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Right: Original image with polygon overlay
    original_with_polygon = bgr.copy()
    if polygon is not None and len(polygon) > 0:
        poly_np = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(original_with_polygon, [poly_np], isClosed=True, color=(0, 255, 255), thickness=2)
        # Draw vertices
        for pt in polygon:
            cv2.circle(original_with_polygon, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
    
    # Create side-by-side
    debug_img = np.hstack([mask_rgb, original_with_polygon])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug_img, "Red Mask (After Morphology)", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(debug_img, "Detected Polygon", (w + 10, 30), font, 1, (0, 255, 255), 2)
    
    if polygon is not None:
        cv2.putText(debug_img, f"{len(polygon)} vertices", (w + 10, 60), font, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(debug_img, "No polygon detected", (w + 10, 60), font, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, debug_img)
    print(f"  Saved debug image")


def refine_boundary_bayesian(bgr: np.ndarray, polygon: Polygon, search_radius: int = 50, debug: bool = False) -> Polygon:
    """
    Bayesian refinement using Generalized Hough Transform (GHT).
    
    Process:
    1. Remove red overlay by inpainting
    2. Create R-table from original polygon template (gradient angles → displacement vectors)
    3. Compute Canny edge map on inpainted image
    warnings = []
    if len(polygon) > 50:
        warnings.append(f"Boundary has {len(polygon)} vertices - consider reviewing for complexity.")
    
    return BoundaryOutput(
        polygon_px=polygon,
        confidence=confidence,
        method="red_overlay_morphological",
        warnings=warnings
        bgr: Input image (BGR)
        polygon: Initial polygon vertices as list of (x, y) tuples (the template)
        search_radius: Max offset to search (default 50 pixels)
        debug: If True, print debug information
    
    Returns:
        Refined polygon with uniform offset applied (template translated to detected location)
    """
    if not polygon or len(polygon) < 3:
        return polygon
    
    h, w = bgr.shape[:2]
    target_vertices = len(polygon)
    
    if debug:
        print(f"[REFINE] Starting GHT refinement: {target_vertices} vertices, search_radius={search_radius}")
    
    # Step 1: Remove red overlay by inpainting
    try:
        red_mask = _red_mask_hsv(bgr)
        if red_mask.dtype != np.uint8:
            red_mask = (red_mask > 0).astype(np.uint8) * 255
    except Exception:
        r_chan = bgr[:, :, 2]
        g_chan = bgr[:, :, 1]
        b_chan = bgr[:, :, 0]
        red_mask = ((r_chan > 200) & (g_chan < 100) & (b_chan < 100)).astype(np.uint8) * 255
    
    if red_mask.sum() > 0:
        img_for_edges = cv2.inpaint(bgr, red_mask, 3, cv2.INPAINT_TELEA)
        if debug:
            print(f"[REFINE] Inpainted {red_mask.sum()} red pixels")
    else:
        img_for_edges = bgr.copy()
        if debug:
            print(f"[REFINE] No red pixels found, using original image")
    
    # Step 2: Build R-table from template polygon (offline phase)
    # Reference point = centroid of template polygon
    centroid_x = sum(p[0] for p in polygon) / len(polygon)
    centroid_y = sum(p[1] for p in polygon) / len(polygon)
    
    # Sample points along polygon edges to create template edge points
    template_edge_points = []
    num_samples_per_edge = 20  # Sample density along each edge
    
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        
        # Sample points along this edge
        for t in np.linspace(0, 1, num_samples_per_edge, endpoint=False):
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            
            # Calculate gradient angle (perpendicular to edge direction)
            edge_dx = x2 - x1
            edge_dy = y2 - y1
            edge_angle = np.arctan2(edge_dy, edge_dx)
            gradient_angle = edge_angle + np.pi / 2  # Perpendicular to edge
            
            # Displacement vector from edge point to centroid
            r = np.sqrt((centroid_x - px)**2 + (centroid_y - py)**2)
            alpha = np.arctan2(centroid_y - py, centroid_x - px)
            
            template_edge_points.append({
                'gradient_angle': gradient_angle % (2 * np.pi),
                'r': r,
                'alpha': alpha
            })
    
    if debug:
        print(f"[REFINE] Built R-table with {len(template_edge_points)} template edge points")
        print(f"[REFINE] Template centroid: ({centroid_x:.1f}, {centroid_y:.1f})")
    
    # Step 3: Compute Canny edge map on target image (online phase)
    gray = cv2.cvtColor(img_for_edges, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Canny edge detection with auto thresholding
    sigma = 0.33
    v = np.median(blurred)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    
    canny = cv2.Canny(blurred, lower, upper)
    if debug:
        print(f"[REFINE] Canny thresholds: lower={lower}, upper={upper}, edge_pixels={canny.sum()}")
    
    # Compute gradient angles for all edge pixels
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    gradient_angles = np.arctan2(sobel_y, sobel_x) % (2 * np.pi)
    
    # Step 4: Voting procedure - build accumulator for possible centroid locations
    # Restrict to search_radius around original centroid
    accumulator = np.zeros((2 * search_radius + 1, 2 * search_radius + 1), dtype=np.float64)
    
    edge_ys, edge_xs = np.where(canny > 0)
    
    if debug:
        print(f"[REFINE] Voting with {len(edge_ys)} edge pixels in target image")
    
    # Angle tolerance for matching (radians) - allow some variation
    angle_tolerance = np.pi / 16  # 11.25 degrees (stricter than 22.5)
    
    vote_count = 0
    for edge_x, edge_y in zip(edge_xs, edge_ys):
        edge_gradient = gradient_angles[edge_y, edge_x]
        
        # Find matching template points with similar gradient angles
        for template_pt in template_edge_points:
            angle_diff = abs(template_pt['gradient_angle'] - edge_gradient)
            # Handle wraparound (0 and 2π are the same)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            if angle_diff < angle_tolerance:
                # Vote for potential centroid location
                r = template_pt['r']
                alpha = template_pt['alpha']
                
                # Predicted centroid location
                vote_x = edge_x + r * np.cos(alpha)
                vote_y = edge_y + r * np.sin(alpha)
                
                # Convert to accumulator coordinates (relative to original centroid)
                acc_x = int(round(vote_x - centroid_x)) + search_radius
                acc_y = int(round(vote_y - centroid_y)) + search_radius
                
                # Check bounds
                if 0 <= acc_x < accumulator.shape[1] and 0 <= acc_y < accumulator.shape[0]:
                    accumulator[acc_y, acc_x] += 1
                    vote_count += 1
    
    if debug:
        print(f"[REFINE] Total votes cast: {vote_count}")
        print(f"[REFINE] Accumulator max: {accumulator.max()}")
        
        # Show top vote locations (debug)
        top_indices = np.argsort(accumulator.flatten())[-5:][::-1]
        top_indices = np.unravel_index(top_indices, accumulator.shape)
        print(f"[REFINE] Top 5 vote locations:")
        for i in range(len(top_indices[0])):
            dy, dx = top_indices[0][i], top_indices[1][i]
            offset_x = dx - search_radius
            offset_y = dy - search_radius
            votes = accumulator[dy, dx]
            mag = np.sqrt(offset_x**2 + offset_y**2)
            print(f"         ({offset_x:+3d}, {offset_y:+3d}): {votes:6.0f} votes (mag={mag:.1f}px)")
    
    
    # Step 5: Find peak in accumulator - use PURE VOTE COUNTS (no Gaussian prior)
    # The Gaussian prior was suppressing larger offsets that actually had more votes
    # Let votes speak for themselves - if roof edge gets more votes than curb, use it
    
    if accumulator.max() == 0:
        # No votes, return original polygon
        if debug:
            print(f"[REFINE] No votes in accumulator, returning original polygon")
        return polygon
    
    # Find peak using ONLY vote counts (no prior weighting)
    peak_y, peak_x = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    
    # Convert accumulator coordinates back to image offset
    offset_x = peak_x - search_radius
    offset_y = peak_y - search_radius
    
    votes_at_peak = accumulator[peak_y, peak_x]
    offset_mag = np.sqrt(offset_x**2 + offset_y**2)
    
    if debug:
        print(f"[REFINE] Best offset (votes only, NO PRIOR): ({offset_x}, {offset_y}) with {votes_at_peak:.0f} votes (mag={offset_mag:.1f}px)")
    
    # Step 6: Apply the offset to template polygon
    if votes_at_peak >= 5:  # Require at least 5 votes for confidence
        refined_polygon = [(x + offset_x, y + offset_y) for x, y in polygon]
        if debug:
            print(f"[REFINE] Applying GHT offset: ({offset_x}, {offset_y})")
    else:
        refined_polygon = polygon
        if debug:
            print(f"[REFINE] Insufficient votes ({votes_at_peak:.0f}), returning original polygon")
    
    return refined_polygon

