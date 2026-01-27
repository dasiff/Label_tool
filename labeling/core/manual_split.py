"""Manual split logic scaffold.

Responsibility:
- Apply open polyline splits and closed-loop polygon splits to a segments ndarray,
  with fallback strategies (seed flood, global cut). Should be usable synchronously
  or from a background worker.

Inputs: segments ndarray, polylines, selected segment id, optional image masks
Outputs: updated segments ndarray, newly created segment ids, status/debug info
"""
from typing import Any, Dict, List, Tuple
import numpy as np
import cv2


def apply_manual_split(self):
    """Ported implementation of `_apply_manual_split` from `scripts/labeling_tool.py`.

    This function operates on attributes of `self` (the LabelingTool instance) and
    modifies `self.segments`, `self.split_history`, and related state as the original
    method did. Copied verbatim to preserve behavior during mechanical extraction.
    """
    # The implementation intentionally uses `self` attributes directly to avoid
    # complex argument lists; this mirrors the original method's behavior.
    if len(self.manual_polylines) == 0 or self.segments is None:
        return
    
    # Check if user selected a segment to split
    if not hasattr(self, 'splitting_segment_id') or self.splitting_segment_id is None:
        self.manual_status.config(text="No segment selected! Click a segment first")
        return
    
    # Save current state for undo
    self.split_history.append(self.segments.copy())
    if len(self.split_history) > 10:  # Limit history to last 10 splits
        self.split_history.pop(0)
    self.undo_split_btn.config(state=tk.NORMAL)
    
    h, w = self.segments.shape
    print(f"\n=== MANUAL SPLIT DEBUG ===")
    print(f"Image size: {h}x{w}")
    print(f"Target segment: {self.splitting_segment_id}")
    print(f"Number of polylines: {len(self.manual_polylines)}")
    
    # Create line mask from ALL polylines
    # Use 1-pixel line - just needs to create a gap for flood fill
    line_thickness = 1  # 1 pixel wide
    endpoint_radius = 2  # 2 pixel radius at endpoints to ensure connection
    
    # Only split the selected segment (not all segments the line touches)
    seg_id = self.splitting_segment_id
    new_seg_id = self.segments.max() + 1
    print(f"Splitting segment {seg_id} into multiple regions")
    
    seg_mask = (self.segments == seg_id)
    seg_area = np.sum(seg_mask)
    print(f"Processing segment {seg_id} (area: {seg_area} pixels)")
    
    # Create debug visualization
    debug_img = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2RGB).copy()
    # Dim everything except the segment
    mask_3ch = np.stack([seg_mask] * 3, axis=2)
    debug_img[~mask_3ch] = (debug_img[~mask_3ch] * 0.3).astype(np.uint8)
    
    # Find edge pixels of the segment (constrained to selected segment only)
    seg_uint8 = seg_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(seg_uint8, kernel, iterations=1)
    edge_mask = seg_uint8 - eroded
    edge_coords = np.argwhere(edge_mask > 0)  # (y, x) format
    
    # Fallback: if erosion didn't produce edge pixels (rare), extract contour points
    if len(edge_coords) == 0:
        try:
            contours, _ = cv2.findContours((seg_uint8 * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                c = max(contours, key=lambda x: cv2.contourArea(x))
                # contour points are (x,y); convert to (y,x)
                edge_coords = np.array([ [pt[0][1], pt[0][0]] for pt in c.reshape(-1,1,2) ])
        except Exception:
            edge_coords = np.empty((0,2), dtype=int)

    if len(edge_coords) == 0:
        print(f"  → No edges found - segment too small")
        self.manual_status.config(text="Segment too small to split!")
        return
    
    # Create combined line mask from all polylines
    line_mask = np.zeros((h, w), dtype=np.uint8)
    segments_added = 0  # Counter for how many new segments were created during this operation
    # Prepare vars used for debug visualization to avoid UnboundLocalError when closed-loops are handled
    edge_pt_0 = None
    edge_pt_last = None
    extended_points_array = None
    
    for line_idx, manual_line_points in enumerate(self.manual_polylines):
        print(f"  Processing line {line_idx + 1} with {len(manual_line_points)} points")
        points = np.array(manual_line_points, dtype=np.int32)
        
        # Check if this is a closed polyline (endpoints close to each other)
        p0 = points[0]
        p_last = points[-1]
        dist_endpoints = np.linalg.norm(np.array([p0[1], p0[0]]) - np.array([p_last[1], p_last[0]]))
        
        # Snap first point - consider both edge pixels AND the last point
        dist_to_edges_0 = np.linalg.norm(edge_coords - np.array([p0[1], p0[0]]), axis=1)
        nearest_edge_dist_0 = dist_to_edges_0.min()
        
        # Snap last point - consider both edge pixels AND the first point
        dist_to_edges_last = np.linalg.norm(edge_coords - np.array([p_last[1], p_last[0]]), axis=1)
        nearest_edge_dist_last = dist_to_edges_last.min()
        
        # Determine if this should be a closed loop
        is_closed_loop = (dist_endpoints < nearest_edge_dist_0 and dist_endpoints < nearest_edge_dist_last)
        
        if is_closed_loop:
            # Treat a closed polyline as a filled polygon: if the drawn polygon lies inside the
            # selected segment, create a new segment from that polygon rather than stretching to the boundary.
            poly_pts = points.astype(np.int32)
            poly_mask_local = np.zeros((h, w), dtype=np.uint8)
            try:
                cv2.fillPoly(poly_mask_local, [poly_pts], 255)
            except Exception:
                poly_mask_local = np.zeros((h, w), dtype=np.uint8)
            constrained_poly = (poly_mask_local > 0) & seg_mask
            area_poly = int(np.sum(constrained_poly))
            print(f"    Closed polygon: area inside segment = {area_poly} px")
            # Minimum meaningful size threshold (consistent with other splits)
            poly_min = max(100, int(seg_area * 0.01))
            if area_poly >= poly_min:
                new_seg_id_local = self.segments.max() + 1
                self.segments[constrained_poly] = new_seg_id_local
                segments_added += 1
                print(f"  → Closed-loop split: created new seg {new_seg_id_local} with area {area_poly}")
                # Visualize successful polygon assignment in debug image (green fill)
                try:
                    cv2.polylines(debug_img, [poly_pts], isClosed=True, color=(0,255,0), thickness=2)
                    mask_rgb = np.zeros_like(debug_img)
                    mask_rgb[constrained_poly] = (0,255,0)
                    alpha = 0.4
                    debug_img = (debug_img * (1-alpha) + mask_rgb * alpha).astype(np.uint8)
                except Exception:
                    pass
                # Keep the newly created segment selected so user can continue edits
                self.splitting_segment_id = new_seg_id_local
                try:
                    self._update_display_with_highlight(self.splitting_segment_id)
                except Exception:
                    pass
            else:
                print(f"    Closed-loop too small (min {poly_min}) - ignoring polygon")
                try:
                    cv2.polylines(debug_img, [poly_pts], isClosed=True, color=(255,0,0), thickness=2)
                except Exception:
                    pass
            # Move on to next manual polyline (closed-loop handled)
            continue
        else:
            # Smart snapping: prefer nearby drawn endpoints, but otherwise snap to nearest edge pixel
            # and avoid extremely long diagonals by limiting snap distance.
            max_snap_dist = max(20, int(min(h, w) * 0.07))  # 7% of smaller dim, at least 20px

            # Build list of other endpoints (x,y) from all other polylines
            other_endpoints = []
            for other_idx, ol in enumerate(self.manual_polylines):
                if other_idx == line_idx:
                    continue
                if len(ol) >= 1:
                    other_endpoints.append(tuple(ol[0]))
                    other_endpoints.append(tuple(ol[-1]))

            # Use shared helper to compute snapped endpoints
            edge_pt_0, edge_pt_last, chosen0_source, chosen_last_source = snap_endpoints(p0, p_last, edge_coords, other_endpoints, max_snap_dist)

            # Debug print what snapping source we chose and store for tests/debugging
            try:
                print(f"    Snap sources: start={chosen0_source}, end={chosen_last_source}, max_snap_dist={max_snap_dist}")
            except Exception:
                pass
            try:
                # Save last snap info for tests / debugging
                self._last_snap_info = {
                    'start': {'pt': tuple(edge_pt_0), 'source': chosen0_source},
                    'end': {'pt': tuple(edge_pt_last), 'source': chosen_last_source}
                }
            except Exception:
                pass

            # If snapping caused endpoints to collapse (very close together), try a contour-intersection or farthest-projection fallback


            edge_pt_0, edge_pt_last = stretch_endpoints_if_collapsed(edge_pt_0, edge_pt_last, seg_uint8, p0, p_last)
            
            print(f"    Snapping: {tuple(p0)} -> {edge_pt_0}, {tuple(p_last)} -> {edge_pt_last}")
        
        # Create extended points array
        extended_points = [edge_pt_0] + [tuple(pt) for pt in points[1:-1]] + [edge_pt_last]
        extended_points_array = np.array(extended_points, dtype=np.int32)
        
        # Add this polyline to the combined line mask
        cv2.polylines(line_mask, [extended_points_array], isClosed=False, color=255, thickness=line_thickness)
        for point in [extended_points_array[0], extended_points_array[-1]]:
            cv2.circle(line_mask, tuple(point), radius=endpoint_radius, color=255, thickness=-1)
        
        # Visualize this line in debug image (different colors for different lines)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        color = colors[line_idx % len(colors)]
        cv2.polylines(debug_img, [extended_points_array], isClosed=False, color=color, thickness=3)
    
    # Keep a copy of the unconstrained combined line mask so we can split multiple segments
    full_line_mask = line_mask.copy()
    # CRITICAL: Constrain line_mask to only pixels within the selected segment
    # This prevents the line from extending into neighboring segments
    line_mask = line_mask & seg_mask.astype(np.uint8) * 255
    
    print(f"Combined line mask pixels (within segment): {np.sum(line_mask > 0)}")
    
    # Add visualization to debug image
    # Show edge pixels in blue
    debug_img[edge_mask > 0] = [0, 0, 255]
    # Draw user's original points in yellow
    for pt in points:
        cv2.circle(debug_img, tuple(pt), 5, (255, 255, 0), -1)
    # Draw snapped endpoints in cyan (if available)
    if edge_pt_0 is not None:
        try:
            cv2.circle(debug_img, edge_pt_0, 8, (0, 255, 255), -1)
        except Exception:
            pass
    if edge_pt_last is not None:
        try:
            cv2.circle(debug_img, edge_pt_last, 8, (0, 255, 255), -1)
        except Exception:
            pass
    # Draw extended line in red (if available)
    if extended_points_array is not None:
        try:
            cv2.polylines(debug_img, [extended_points_array], isClosed=False, color=(255, 0, 0), thickness=3)
        except Exception:
            pass
    
    # If the constrained line is tiny (e.g., endpoints collapsed), try to compute true boundary intersections
    if line_mask.sum() < 20:
        try:
            # find contour of segment and intersect with infinite line through original endpoints
            contours, _ = cv2.findContours((seg_mask.astype(np.uint8)*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                c = max(contours, key=lambda x: cv2.contourArea(x)).reshape(-1, 2)
                p0f = np.array(points[0], dtype=float)
                p1f = np.array(points[-1], dtype=float)
                line_dir = p1f - p0f
                if np.linalg.norm(line_dir) > 1e-6:
                    inters = []
                    for i in range(len(c)):
                        a = c[i]
                        b = c[(i+1) % len(c)]
                        A = np.array([a[0], a[1]], dtype=float)
                        B = np.array([b[0], b[1]], dtype=float)
                        s = B - A
                        r = p1f - p0f
                        M = np.column_stack((s, -r))  # solves M * [t,u] = (p0 - A)
                        bvec = (p0f - A)
                        try:
                            # Solve linear system robustly
                            sol, *_ = np.linalg.lstsq(M, bvec, rcond=None)
                            t = float(sol[0]); u = float(sol[1])
                            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                                X = A + t * s
                                inters.append((float(X[0]), float(X[1])))
                        except Exception:
                            continue
                    # after checking all contour segments, if we found 2 or more intersections, pick farthest apart
                    if len(inters) >= 2:
                        # pick two that are farthest along the line
                        # project to line parameter u
                        us = [np.dot(np.array(p)-p0f, line_dir)/np.dot(line_dir,line_dir) for p in inters]
                        sorted_idx = np.argsort(us)
                        pA = inters[sorted_idx[0]]
                        pB = inters[sorted_idx[-1]]
                        edge_pt_0 = (int(round(pA[0])), int(round(pA[1])))
                        edge_pt_last = (int(round(pB[0])), int(round(pB[1])))
                        # rebuild extended points array between these two exact intersection endpoints
                        extended_points = [edge_pt_0, edge_pt_last]
                        extended_points_array = np.array(extended_points, dtype=np.int32)
                        # rebuild full and constrained masks
                        cv2.polylines(line_mask, [extended_points_array], isClosed=False, color=255, thickness=line_thickness)
                        for point in [extended_points_array[0], extended_points_array[-1]]:
                            cv2.circle(line_mask, tuple(point), radius=endpoint_radius, color=255, thickness=-1)
        except Exception:
            pass

        # Subtract the line from the segment to create a gap
        seg_without_line = seg_mask.copy()
        seg_without_line[line_mask > 0] = False
        
        # Find connected components on each side of the line
        labeled_regions, num_regions = ndimage.label(seg_without_line)
        
        print(f"  After removing line, found {num_regions} regions")
        
        # If constrained intersection was tiny, try widening the line to force a split
        if num_regions < 2 and line_mask.sum() > 0:
            for widen in (3,5,9):
                try:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (widen, widen))
                    dil = cv2.dilate(line_mask, k, iterations=1)
                    seg_without_line2 = seg_mask.copy()
                    seg_without_line2[dil > 0] = False
                    labeled2, num2 = ndimage.label(seg_without_line2)
                    print(f"  Retry widen={widen}: found {num2} regions")
                    if num2 >= 2:
                        labeled_regions = labeled2
                        num_regions = num2
                        line_mask = dil
                        break
                except Exception:
                    continue

        if num_regions < 2:
            # If this line didn't split the selected segment and we are running in the
            # background worker for UI responsiveness, prefer to fall back to the main-thread
            # full split routine (safer for interactions). If we are in the main thread,
            # continue with seed-split attempts as before.
            if getattr(self, '_running_in_background', False):
                # Schedule the full synchronous split to run in the main thread (fallback)
                try:
                    self.root.after(0, lambda: self._apply_manual_split())
                except Exception:
                    # As last resort, do nothing here to avoid blocking
                    pass
                return

            # 1) Sample along the drawn line and perform orthogonal seed floods to find two opposite seeds within the segment.
            p0f = np.array(points[0], dtype=float)
            p1f = np.array(points[-1], dtype=float)
            line_dir = p1f - p0f
            found_seed_split = False
            if np.linalg.norm(line_dir) > 1e-6:
                norm = np.array([-line_dir[1], line_dir[0]])
                for t_frac in (0.2, 0.35, 0.5, 0.65, 0.8):
                    p_mid = p0f + t_frac * line_dir
                    if np.linalg.norm(norm) < 1e-6:
                        continue
                    n = norm / np.linalg.norm(norm)
                    for delta in (8, 16, 32, 64):
                        sA = np.round(p_mid + n * delta).astype(int)
                        sB = np.round(p_mid - n * delta).astype(int)
                        h, w = seg_mask.shape
                        if not (0 <= sA[1] < h and 0 <= sA[0] < w and 0 <= sB[1] < h and 0 <= sB[0] < w):
                            continue
                        if not (seg_mask[sA[1], sA[0]] and seg_mask[sB[1], sB[0]]):
                            continue
[...] (truncated)


# --- Endpoint snapping helpers (extracted from scripts/labeling_tool.py) ---

def snap_endpoints(p0: Tuple[int,int], p_last: Tuple[int,int], edge_coords: np.ndarray,
                   other_endpoints: List[Tuple[int,int]], max_snap_dist: int) -> Tuple[Tuple[int,int], Tuple[int,int], str, str]:
    """Compute snapped endpoints for a drawn polyline.

    Returns: (edge_pt_0, edge_pt_last, chosen0_source, chosen_last_source)
    Sources are one of: 'edge', 'edge_distant', 'endpoint'.
    """
    # Convert inputs to numpy arrays for distance computations
    p0_arr = np.array([p0[1], p0[0]])  # (y,x) for comparison with edge_coords
    p1_arr = np.array([p_last[1], p_last[0]])

    # Distances to edge pixels
    if edge_coords.size == 0:
        dist_to_edges_0 = np.array([], dtype=float)
        dist_to_edges_last = np.array([], dtype=float)
    else:
        dist_to_edges_0 = np.linalg.norm(edge_coords - p0_arr, axis=1)
        dist_to_edges_last = np.linalg.norm(edge_coords - p1_arr, axis=1)

    # Prepare other endpoints array
    other_eps_arr = (np.array([[pt[1], pt[0]] for pt in other_endpoints])
                     if other_endpoints else np.empty((0,2)))

    # Start point snapping
    chosen0 = None
    chosen0_source = 'edge'
    if other_eps_arr.shape[0] > 0:
        d_eps0 = np.linalg.norm(other_eps_arr - p0_arr, axis=1)
        ep0_idx = int(np.argmin(d_eps0))
        ep0_dist = float(d_eps0[ep0_idx])
        if edge_coords.size > 0:
            nearest_edge_dist_0 = float(dist_to_edges_0[np.argmin(dist_to_edges_0)])
        else:
            nearest_edge_dist_0 = float('inf')
        if ep0_dist <= min(nearest_edge_dist_0, max_snap_dist):
            chosen0 = other_endpoints[ep0_idx]
            chosen0_source = 'endpoint'

    if chosen0 is None:
        if edge_coords.size > 0:
            nearest_edge_0 = edge_coords[np.argmin(dist_to_edges_0)]
            nearest_edge_dist_0 = float(dist_to_edges_0[np.argmin(dist_to_edges_0)])
            chosen0 = (int(nearest_edge_0[1]), int(nearest_edge_0[0]))
            chosen0_source = 'edge' if nearest_edge_dist_0 <= max_snap_dist else 'edge_distant'
        else:
            chosen0 = (int(p0[0]), int(p0[1]))
            chosen0_source = 'edge_distant'

    # Last point snapping
    chosen_last = None
    chosen_last_source = 'edge'
    if other_eps_arr.shape[0] > 0:
        d_eplast = np.linalg.norm(other_eps_arr - p1_arr, axis=1)
        ep_last_idx = int(np.argmin(d_eplast))
        ep_last_dist = float(d_eplast[ep_last_idx])
        if edge_coords.size > 0:
            nearest_edge_dist_last = float(dist_to_edges_last[np.argmin(dist_to_edges_last)])
        else:
            nearest_edge_dist_last = float('inf')
        if ep_last_dist <= min(nearest_edge_dist_last, max_snap_dist):
            chosen_last = other_endpoints[ep_last_idx]
            chosen_last_source = 'endpoint'

    if chosen_last is None:
        if edge_coords.size > 0:
            nearest_edge_last = edge_coords[np.argmin(dist_to_edges_last)]
            nearest_edge_dist_last = float(dist_to_edges_last[np.argmin(dist_to_edges_last)])
            # Avoid choosing same exact edge pixel as start - handled by caller if necessary
            chosen_last = (int(nearest_edge_last[1]), int(nearest_edge_last[0]))
            chosen_last_source = 'edge' if nearest_edge_dist_last <= max_snap_dist else 'edge_distant'
        else:
            chosen_last = (int(p_last[0]), int(p_last[1]))
            chosen_last_source = 'edge_distant'

    return chosen0, chosen_last, chosen0_source, chosen_last_source


def stretch_endpoints_if_collapsed(ep0: Tuple[int,int], ep1: Tuple[int,int], seg_uint8_local: np.ndarray,
                                    p0: Tuple[int,int], p_last: Tuple[int,int]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """Attempt to expand collapsed endpoints by intersecting with contour or projecting extremes.

    If expansion is unsuccessful, returns the original endpoints.
    """
    dx = ep0[0] - ep1[0]
    dy = ep0[1] - ep1[1]
    if dx*dx + dy*dy > 25:
        return ep0, ep1
    try:
        contours, _ = cv2.findContours(seg_uint8_local*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            c = max(contours, key=lambda x: cv2.contourArea(x)).reshape(-1,2)
            p0f = np.array(p0, dtype=float)
            p1f = np.array(p_last, dtype=float)
            line_dir = p1f - p0f
            if np.linalg.norm(line_dir) > 1e-6:
                inters = []
                for i in range(len(c)):
                    A = c[i]
                    B = c[(i+1) % len(c)]
                    A = np.array([float(A[0]), float(A[1])])
                    B = np.array([float(B[0]), float(B[1])])
                    s = B - A
                    r = line_dir
                    M = np.column_stack((s, -r))
                    bvec = (p0f - A)
                    try:
                        sol, *_ = np.linalg.lstsq(M, bvec, rcond=None)
                        t = float(sol[0]); u = float(sol[1])
                        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                            X = A + t * s
                            inters.append((float(X[0]), float(X[1])))
                    except Exception:
                        continue
                if len(inters) >= 2:
                    us = [np.dot(np.array(p)-p0f, line_dir)/np.dot(line_dir,line_dir) for p in inters]
                    sidx = np.argsort(us)
                    pA = inters[sidx[0]]
                    pB = inters[sidx[-1]]
                    return (int(round(pA[0])), int(round(pA[1]))), (int(round(pB[0])), int(round(pB[1])))
        # If contour intersection failed, fallback to projection extremes along the line within the segment
        pts = np.argwhere(seg_uint8_local > 0)  # (y,x)
        if pts.shape[0] > 0:
            pts_xy = np.vstack([pts[:,1], pts[:,0]]).T.astype(float)
            p0f = np.array(p0, dtype=float)
            p1f = np.array(p_last, dtype=float)
            r = p1f - p0f
            denom = np.dot(r,r)
            if denom > 1e-6:
                us = np.dot(pts_xy - p0f, r) / denom
                min_idx = np.argmin(us)
                max_idx = np.argmax(us)
                pA = pts_xy[min_idx]
                pB = pts_xy[max_idx]
                return (int(round(pA[0])), int(round(pA[1]))), (int(round(pB[0])), int(round(pB[1])))
    except Exception:
        pass
    return ep0, ep1
