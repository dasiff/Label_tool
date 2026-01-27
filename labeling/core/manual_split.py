"""Manual split logic scaffold.

Responsibility:
- Apply open polyline splits and closed-loop polygon splits to a segments ndarray,
  with fallback strategies (seed flood, global cut). Should be usable synchronously
  or from a background worker.

Inputs: segments ndarray, polylines, selected segment id, optional image masks
Outputs: updated segments ndarray, newly created segment ids, status/debug info
"""
from typing import Any, Dict, List, Tuple


def apply_manual_split(segments: Any, polylines: List[List[Tuple[int, int]]], selected_id: int) -> Dict[str, Any]:
    """Apply manual split polylines to `segments`.

    Returns a dict with keys: 'segments', 'new_segment_ids', 'status'.
    """
    raise NotImplementedError


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
