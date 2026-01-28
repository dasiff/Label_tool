"""Manual split logic scaffold.

Responsibility:
- Apply open polyline splits and closed-loop polygon splits to a segments ndarray,
  with fallback strategies (seed flood, global cut). Should be usable synchronously
  or from a background worker.

Inputs: segments ndarray, polylines, selected segment id, optional image masks
Outputs: updated segments ndarray, newly created segment ids, status/debug info
"""
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import cv2
from scipy import ndimage
from dataclasses import dataclass, field


@dataclass
class SplitState:
    """Container for local state used during a manual split operation."""
    segments: np.ndarray
    manual_polylines: List[List[Tuple[int,int]]]
    segment_labels: Dict[int, str]
    split_history: List[np.ndarray] = field(default_factory=list)
    n_segments: int = 0
    splitting_segment_id: Optional[int] = None


def apply_manual_split(self):
    """Ported implementation of `_apply_manual_split` from `scripts/labeling_tool.py`.

    This function operates on attributes of `self` (the LabelingTool instance) and
    modifies `self.segments`, `self.split_history`, and related state as the original
    method did. Copied verbatim to preserve behavior during mechanical extraction.
    """
    # The implementation intentionally uses `self` attributes directly to avoid
    # complex argument lists; this mirrors the original method's behavior.
    try:
        print(f"DEBUG manual_split.apply_manual_split called - splitting_segment_id={getattr(self,'splitting_segment_id',None)}, n_polylines={len(getattr(self,'manual_polylines',[]))}, segments_present={self.segments is not None}")
    except Exception:
        pass
    if len(self.manual_polylines) == 0 or self.segments is None:
        try:
            print("DEBUG manual_split.apply_manual_split: early return - no polylines or segments missing")
        except Exception:
            pass
        return
    
    # Check if user selected a segment to split
    if not hasattr(self, 'splitting_segment_id') or self.splitting_segment_id is None:
        self.set_manual_status("No segment selected! Click a segment first")
        try:
            print("DEBUG manual_split.apply_manual_split: No segment selected - aborting")
        except Exception:
            pass
        return
    
    # NOTE: defer saving to split_history until we are about to mutate segments
    # (we want undo entries only for successful mutations). Do not append here.
    pass
    
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
        self.set_manual_status("Segment too small to split!")
        return
    
    # Create combined line mask from all polylines
    segments_added = 0  # Counter for how many new segments were created during this operation
    # Prepare vars used for debug visualization to avoid UnboundLocalError when closed-loops are handled
    edge_pt_0 = None
    edge_pt_last = None
    extended_points_array = None

    # Build full constrained line mask using helper
    from labeling.core.split_core import build_combined_line_mask, closed_polygon_handler, constrained_subtract_and_label
    # Initialize a default (empty) line mask so it's always defined even if there are no open polylines
    line_mask = np.zeros((h, w), dtype=np.uint8)

    # Handle closed polygons individually first, collecting remaining open polylines
    open_polylines = []
    all_open_polylines_snapped = True
    for line_idx, manual_line_points in enumerate(self.manual_polylines):
        pts = np.array(manual_line_points, dtype=np.int32)
        p0 = pts[0]
        p_last = pts[-1]
        dist_endpoints = np.linalg.norm(np.array([p0[1], p0[0]]) - np.array([p_last[1], p_last[0]]))
        # Use a conservative closed-loop threshold: endpoints closer than 3 px => closed
        if dist_endpoints <= 3:
            applied, new_mask, area = closed_polygon_handler(pts.tolist(), seg_mask, min_side_px=max(100, int(seg_area*0.01)))
            if applied:
                # Create new segment from mask
                new_seg_id_local = int(self.segments.max()) + 1
                self.segments[new_mask] = new_seg_id_local
                segments_added += 1
                print(f"  → Closed-loop split: created new seg {new_seg_id_local} with area {area}")
                self.splitting_segment_id = new_seg_id_local
                try:
                    self._update_display_with_highlight(self.splitting_segment_id)
                except Exception:
                    pass
            continue
        else:
            # perform strict snapping of endpoints to either nearest edge or nearest other line pixel
            # Build other polylines excluding current line
            other_polylines = [ol for oi, ol in enumerate(self.manual_polylines) if oi != line_idx and len(ol) >= 2]

            # Snap endpoints using helper that centralizes mask building and stretching
            edge_pt_0, edge_pt_last, chosen0_source, chosen_last_source = _snap_endpoints_for_polyline(tuple(p0), tuple(p_last), edge_coords, other_polylines, seg_uint8, max_snap_dist=999999)
            try:
                self._last_snap_info = {
                    'start': {'pt': tuple(edge_pt_0), 'source': chosen0_source},
                    'end': {'pt': tuple(edge_pt_last), 'source': chosen_last_source}
                }
            except Exception:
                pass

            # determine if both endpoints were snapped to allowed sources ('edge' or 'line')
            try:
                poly_snapped = (chosen0_source in ('edge', 'line')) and (chosen_last_source in ('edge', 'line'))
            except Exception:
                poly_snapped = False
            if not poly_snapped:
                all_open_polylines_snapped = False

            # If snapped endpoints lie outside the selected segment (e.g., along irregular boundary),
            # nudge them slightly inward toward the segment centroid so the cut passes through the
            # interior rather than along the exterior edge.
            try:
                def _nudge_point_inside(seg_mask_uint8, pt_xy, max_nudge=10):
                    # seg_mask_uint8: 2D uint8 mask (1 inside segment), pt_xy: (x,y)
                    h, w = seg_mask_uint8.shape
                    x0, y0 = int(pt_xy[0]), int(pt_xy[1])
                    # If already inside but located on an edge pixel, treat it as outside to nudge inward
                    def _is_edge(xi, yi):
                        if not (0 <= yi < h and 0 <= xi < w):
                            return False
                        if not seg_mask_uint8[yi, xi]:
                            return False
                        # If any 8-neighborhood neighbor is outside, it's an edge
                        for ny in range(max(0, yi-1), min(h, yi+2)):
                            for nx in range(max(0, xi-1), min(w, xi+2)):
                                if not seg_mask_uint8[ny, nx]:
                                    return True
                        return False

                    if 0 <= y0 < h and 0 <= x0 < w and seg_mask_uint8[y0, x0] and not _is_edge(x0, y0):
                        return (x0, y0)
                    # Compute centroid of segment in (y,x) coordinates
                    ysxs = np.argwhere(seg_mask_uint8 > 0)
                    if ysxs.size == 0:
                        return (x0, y0)
                    centroid_y, centroid_x = ysxs.mean(axis=0)
                    # Direction from point toward centroid (in x,y)
                    dir_x = centroid_x - x0
                    dir_y = centroid_y - y0
                    norm = np.hypot(dir_x, dir_y)
                    if norm < 1e-6:
                        return (x0, y0)
                    ux, uy = dir_x / norm, dir_y / norm
                    for step in range(1, max_nudge + 1):
                        nx = int(round(x0 + ux * step))
                        ny = int(round(y0 + uy * step))
                        if 0 <= ny < h and 0 <= nx < w and seg_mask_uint8[ny, nx]:
                            return (nx, ny)
                    return (x0, y0)

                # Apply inward nudging to each snapped endpoint
                try:
                    edge_pt_0 = _nudge_point_inside(seg_uint8, edge_pt_0, max_nudge=10)
                except Exception:
                    pass
                try:
                    edge_pt_last = _nudge_point_inside(seg_uint8, edge_pt_last, max_nudge=10)
                except Exception:
                    pass
            except Exception:
                pass

            # Build extended polyline including snapped endpoints
            extended = [edge_pt_0] + [tuple(pt) for pt in pts[1:-1]] + [edge_pt_last]
            open_polylines.append(extended)

    # Build line mask from remaining open polylines
    if open_polylines:
        line_mask = build_combined_line_mask(open_polylines, seg_mask, (h, w), thickness=line_thickness, endpoint_radius=endpoint_radius)

        # If the constructed line_mask is unexpectedly small (e.g., endpoints on boundary
        # resulted in a very short intersection), try extending the endpoints slightly into
        # the interior to ensure the cut traverses the segment.
        try:
            if line_mask.sum() < max(10, int(seg_area * 0.002)):
                # Compute centroid of the segment (y,x)
                ysxs = np.argwhere(seg_mask)
                if ysxs.size > 0:
                    cy, cx = ysxs.mean(axis=0)
                    for poly in open_polylines:
                        # first and last points
                        p0 = tuple(poly[0])
                        p1 = tuple(poly[-1])
                        # Cast to integer tuples
                        p0i = (int(round(p0[0])), int(round(p0[1])))
                        p1i = (int(round(p1[0])), int(round(p1[1])))
                        # Create a short line from p0/p1 toward centroid
                        for px, py in (p0i, p1i):
                            dir_x = cx - px
                            dir_y = cy - py
                            norm = (dir_x**2 + dir_y**2) ** 0.5
                            if norm < 1e-6:
                                continue
                            ux, uy = dir_x / norm, dir_y / norm
                            # extend 5..15 pixels inward
                            for ext in (5, 10, 15):
                                nx = int(round(px + ux * ext))
                                ny = int(round(py + uy * ext))
                                if 0 <= ny < h and 0 <= nx < w and seg_mask[ny, nx]:
                                    # Draw a small connecting line
                                    try:
                                        cv2.line(line_mask, (px, py), (nx, ny), color=255, thickness=line_thickness)
                                        break
                                    except Exception:
                                        pass
        except Exception:
            pass


    
    # Keep a copy of the unconstrained combined line mask so we can split multiple segments
    full_line_mask = line_mask.copy()

    # CRITICAL: Constrain line_mask to only pixels within the selected segment
    # This prevents the line from extending into neighboring segments
    line_mask = line_mask & (seg_mask.astype(np.uint8) * 255)

    # QUICK ATTEMPT: Under strict-snapping, try a direct split here before running other fallbacks.
    # If both endpoints were snapped, attempt the direct split via the core helper and apply immediately.
    try:
        new_segments, applied, info = _apply_split_core(self, seg_mask, line_mask, full_line_mask, seg_id, np.array(self.manual_polylines[0], dtype=int) if len(self.manual_polylines) else None, debug_img, line_thickness=line_thickness, endpoint_radius=endpoint_radius, seg_area=seg_area, edge_mask=edge_mask, require_snapped=True, snapped_endpoints=all_open_polylines_snapped)
        if applied:
            # record history and apply result
            try:
                self.split_history.append(self.segments.copy())
                if len(self.split_history) > 10:
                    self.split_history.pop(0)
            except Exception:
                pass
            self.segments = new_segments
            try:
                self.n_segments = int(self.segments.max())
            except Exception:
                self.n_segments = int(self.segments.max()) if self.segments is not None else 0
            try:
                new_seg_id = int(info.get('new_seg_id')) if info.get('new_seg_id') is not None else None
                if new_seg_id is not None:
                    self.splitting_segment_id = new_seg_id
            except Exception:
                pass
            try:
                self.set_manual_status("Split applied! Draw another or toggle off")
            except Exception:
                pass
            try:
                self._update_display_with_highlight(self.splitting_segment_id)
            except Exception:
                pass
            return
    except Exception:
        pass

    # If we already created closed-loop segments, finalize UI state and return early
    if segments_added > 0:
        try:
            self.n_segments = int(self.segments.max())
        except Exception:
            self.n_segments = int(self.segments.max()) if self.segments is not None else 0
        try:
            self.set_manual_status("Split applied! Draw another or toggle off")
        except Exception:
            pass
        try:
            self.set_finalize_enabled(True)
        except Exception:
            pass
        # Keep newly created segment selected
        try:
            if getattr(self, 'splitting_segment_id', None) is not None:
                self._update_display_with_highlight(self.splitting_segment_id)
        except Exception:
            pass
        try:
            self._clear_manual_line()
        except Exception:
            pass
        try:
            self._reset_submit_button()
        except Exception:
            pass
        try:
            self._update_display()
        except Exception:
            pass
        try:
            self._update_progress()
        except Exception:
            pass
        return


# --- Extracted split orchestration helper ---
def _apply_split_core(self, seg_mask, line_mask, full_line_mask, seg_id, points, debug_img, line_thickness=1, endpoint_radius=2, seg_area=0, edge_mask=None, require_snapped: bool=False, snapped_endpoints: bool=False):
    """Core split orchestration (STRICT SNAPPING MODE).

    Behavior under strict snapping:
      - If `require_snapped` is True, the split is only attempted when
        `snapped_endpoints` is True (both endpoints snapped to an allowed source).
      - Only direct split via `apply_direct_split` is used. No widening, seed-splits,
        touched-splits, contour fallbacks, or global cuts are performed.

    Returns (new_segments_array, applied_bool, info_dict)
    """
    # Work on a copy so we are pure-ish
    new_segments = self.segments.copy()
    applied = False
    info = {}

    # Label connected components (strict policy: no background retry)
    labeled_regions, num_regions, line_mask, _ = _label_regions_with_retry(self, seg_mask, line_mask, force_global=False)

    # Enforce strict snapping if caller requested it
    if require_snapped and not snapped_endpoints:
        return new_segments, False, {'reason': 'no_snap'}

    # compute region sizes
    region_sizes = []
    for rid in range(1, max(1, num_regions) + 1):
        region_sizes.append((int((labeled_regions == rid).sum()), rid))
    region_sizes.sort(reverse=True)

    # Only attempt direct split
    # Use a more permissive floor so uneven splits (small piece + large piece) can succeed
    min_side_size = max(30, int(seg_area * 0.01))
    if num_regions >= 2:
        # Primary attempt: require both sides to meet full minimum
        if region_sizes[0][0] >= min_side_size and region_sizes[1][0] >= min_side_size:
            new_segments, applied, info = _attempt_direct_split(self, new_segments, seg_id, labeled_regions, region_sizes, min_side_px=min_side_size)
            if applied:
                return new_segments, True, info
        # Secondary attempt: allow an uneven split if the smaller side is at least a small floor
        small_floor = 20
        if region_sizes[1][0] >= small_floor:
            # Try a relaxed direct split with a lower min_side_px
            new_segments_relaxed, applied_relaxed, info_relaxed = _attempt_direct_split(self, new_segments, seg_id, labeled_regions, region_sizes, min_side_px=small_floor)
            if applied_relaxed:
                try:
                    info_relaxed['note'] = 'applied_relaxed_min'
                except Exception:
                    pass
                return new_segments_relaxed, True, info_relaxed

    return new_segments, False, {'reason': 'no_split'}


# --- Helper: label regions and optionally schedule retry when running in background ---
def _label_regions_with_retry(self, seg_mask, line_mask, force_global: bool = False):
    """Label connected components after subtracting `line_mask` from `seg_mask`.

    Strict mode: no widening, no background retries. Returns (labeled_regions, num_regions, final_line_mask, retry_flag)
    where retry_flag is always False under the strict snapping policy.
    """
    # Subtract the line from the segment to create a gap
    seg_without_line = seg_mask.copy()
    seg_without_line[line_mask > 0] = False

    # Label connected components
    labeled_regions, num_regions = ndimage.label(seg_without_line)

    return labeled_regions, num_regions, line_mask, False

# --- Helper: widening/dilation retry ---
def _widen_line_and_label(seg_mask, line_mask):
    """Try widening `line_mask` using morphological dilations and relabel.

    Returns: (labeled_regions, num_regions, final_line_mask)
    """
    for widen in (3, 5, 9):
        try:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (widen, widen))
            dil = cv2.dilate(line_mask, k, iterations=1)
            seg_without_line2 = seg_mask.copy()
            seg_without_line2[dil > 0] = False
            labeled2, num2 = ndimage.label(seg_without_line2)
            if num2 >= 2:
                return labeled2, num2, dil
        except Exception:
            continue
    return ndimage.label(seg_mask & (~(line_mask > 0))) + (line_mask, )



# --- Helper: wrapper for apply_direct_split to isolate IO and errors ---
def _attempt_direct_split(self, new_segments, seg_id, labeled_regions, region_sizes, min_side_px=100):
    """Attempt direct split via `apply_direct_split` and return standardized result."""
    try:
        from labeling.core.split_core import apply_direct_split
        new_seg_arr, new_seg_id, res_info = apply_direct_split(new_segments, seg_id, labeled_regions, region_sizes, min_side_px=min_side_px)
        if res_info.get('applied'):
            info = {'new_seg_id': int(new_seg_id), 'side1_count': res_info.get('side1_count'), 'side2_count': res_info.get('side2_count')}
            return new_seg_arr, True, info
        else:
            return new_segments, False, {'reason': res_info.get('reason')}
    except Exception:
        return new_segments, False, {}



# --- Helper: seed-split using orthogonal floods ---
def _attempt_seed_split(seg_mask, line_mask, points, new_segments):
    """Try seed-based split using orthogonal seeds along `points` line.

    Returns (new_segments, applied, info)
    """
    try:
        p0f = np.array(points[0], dtype=float)
        p1f = np.array(points[-1], dtype=float)
        line_dir = p1f - p0f
        if np.linalg.norm(line_dir) <= 1e-6:
            return new_segments, False, {}
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
                from collections import deque
                def flood(seed):
                    visited = np.zeros_like(seg_mask, dtype=bool)
                    q = deque()
                    q.append((seed[1], seed[0]))
                    visited[seed[1], seed[0]] = True
                    while q:
                        y,x = q.popleft()
                        for ny in range(max(0,y-1), min(seg_mask.shape[0], y+2)):
                            for nx in range(max(0,x-1), min(seg_mask.shape[1], x+2)):
                                if not visited[ny,nx] and seg_mask[ny,nx]:
                                    visited[ny,nx] = True
                                    q.append((ny,nx))
                    return visited
                maskA = flood(sA)
                maskB = flood(sB)
                inter = np.logical_and(maskA, maskB).sum()
                if inter == 0 and maskA.sum() > 0 and maskB.sum() > 0:
                    if maskA.sum() < maskB.sum():
                        new_id = int(new_segments.max()) + 1
                        new_segments[maskA] = new_id
                    else:
                        new_id = int(new_segments.max()) + 1
                        new_segments[maskB] = new_id
                    return new_segments, True, {'new_seg_id': int(new_id), 'method': 'seed_split'}
    except Exception:
        pass
    return new_segments, False, {}



# --- Helper: split segments touched by unconstrained full_line_mask ---
def _attempt_touched_splits(new_segments, full_line_mask, seg_id):
    try:
        touched_seg_ids = np.unique(new_segments[(full_line_mask > 0)])
        touched_seg_ids = [int(x) for x in touched_seg_ids if x != 0]
        for tid in touched_seg_ids:
            if tid == seg_id:
                continue
            t_mask = (new_segments == tid)
            t_seg_area = int(t_mask.sum())
            if t_seg_area == 0:
                continue
            t_line_mask = (full_line_mask & (t_mask.astype(np.uint8) * 255))
            if t_line_mask.sum() == 0:
                continue
            t_seg_without_line = t_mask.copy()
            t_seg_without_line[t_line_mask > 0] = False
            t_labeled, t_num = ndimage.label(t_seg_without_line)
            if t_num < 2:
                continue
            t_region_sizes = []
            for region_id in range(1, t_num + 1):
                t_region_sizes.append((int((t_labeled == region_id).sum()), region_id))
            t_region_sizes.sort(reverse=True)
            t_side1_count, t_side1_id = t_region_sizes[0]
            t_side2_count, t_side2_id = t_region_sizes[1]
            t_min_side = max(100, int(t_seg_area * 0.01))
            if t_side1_count >= t_min_side and t_side2_count >= t_min_side:
                new_id_local = int(new_segments.max()) + 1
                side2_mask_local = (t_labeled == t_side2_id)
                new_segments[side2_mask_local] = new_id_local
                return new_segments, True, {'new_seg_id': int(new_id_local), 'method': 'touched_split', 'tid': tid}
    except Exception:
        pass
    return new_segments, False, {}

    # If forced global cut requested: delegate to helper
    if force_global:
        new_segments, applied, info = _attempt_global_cut(self, new_segments, points, line_thickness=line_thickness, endpoint_radius=endpoint_radius)
        if applied:
            return new_segments, True, info


# --- Helper: global cut across boundary/box intersections ---
def _attempt_global_cut(self, new_segments, points, line_thickness=1, endpoint_radius=2):
    """Compute intersections with boundary or image box, create a global line,
    and attempt splits on segments the line touches. Returns (new_segments, applied, info)."""
    try:
        p0f = np.array(points[0], dtype=float)
        p1f = np.array(points[-1], dtype=float)
        inters = _compute_line_intersections(self, p0f, p1f)
        if len(inters) >= 2:
            line_dir = p1f - p0f
            us = [np.dot(np.array(p)-p0f, line_dir)/np.dot(line_dir,line_dir) for p in inters]
            sidx = np.argsort(us)
            gA = inters[sidx[0]]
            gB = inters[sidx[-1]]
            global_line = np.array([(int(round(gA[0])), int(round(gA[1]))), (int(round(gB[0])), int(round(gB[1])))], dtype=np.int32)
            # build global mask covering whole image
            global_mask = np.zeros((self.clean_image.shape[0], self.clean_image.shape[1]), dtype=np.uint8)
            cv2.polylines(global_mask, [global_line], isClosed=False, color=255, thickness=line_thickness)
            for point in [global_line[0], global_line[-1]]:
                cv2.circle(global_mask, tuple(point), radius=endpoint_radius, color=255, thickness=-1)
            touched_seg_ids2 = np.unique(new_segments[(global_mask > 0)])
            touched_seg_ids2 = [int(x) for x in touched_seg_ids2 if x != 0]
            for tid in touched_seg_ids2:
                t_mask = (new_segments == tid)
                t_seg_area = int(t_mask.sum())
                if t_seg_area == 0:
                    continue
                t_line_mask = (global_mask & (t_mask.astype(np.uint8) * 255))
                if t_line_mask.sum() == 0:
                    continue
                t_seg_without_line = t_mask.copy()
                t_seg_without_line[t_line_mask > 0] = False
                t_labeled, t_num = ndimage.label(t_seg_without_line)
                if t_num < 2:
                    continue
                t_region_sizes = []
                for region_id in range(1, t_num + 1):
                    t_region_sizes.append((int((t_labeled == region_id).sum()), region_id))
                t_region_sizes.sort(reverse=True)
                t_side1_count, t_side1_id = t_region_sizes[0]
                t_side2_count, t_side2_id = t_region_sizes[1]
                t_min_side = max(100, int(t_seg_area * 0.01))
                if t_side1_count >= t_min_side and t_side2_count >= t_min_side:
                    new_seg_id_local = int(new_segments.max()) + 1
                    side2_mask_local = (t_labeled == t_side2_id)
                    new_segments[side2_mask_local] = new_seg_id_local
                    return new_segments, True, {'new_seg_id': int(new_seg_id_local), 'method': 'global_cut'}
    except Exception:
        pass
    return new_segments, False, {}


# --- Helper: compute intersections with boundary and image box ---
def _compute_line_intersections(self, p0f, p1f):
    """Compute intersections between the line (p0f->p1f) and the current boundary
    and the image box. Returns a list of intersection (x,y) tuples. Prefers
    boundary intersections, falls back to box intersections when boundary has
    fewer than 2 intersections.
    """
    inters = []
    bpoly = self.current_boundary.astype(float)
    for i in range(len(bpoly)):
        A = bpoly[i]
        B = bpoly[(i+1) % len(bpoly)]
        s = B - A
        r = p1f - p0f
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

    box_inters = []
    W = self.clean_image.shape[1]
    H = self.clean_image.shape[0]
    box_edges = [ (np.array([0.0,0.0]), np.array([W-1.0,0.0])), (np.array([W-1.0,0.0]), np.array([W-1.0,H-1.0])), (np.array([W-1.0,H-1.0]), np.array([0.0,H-1.0])), (np.array([0.0,H-1.0]), np.array([0.0,0.0])) ]
    for (A,B) in box_edges:
        s = B - A
        r = p1f - p0f
        M = np.column_stack((s, -r))
        bvec = (p0f - A)
        try:
            sol, *_ = np.linalg.lstsq(M, bvec, rcond=None)
            t = float(sol[0]); u = float(sol[1])
            if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
                X = A + t * s
                box_inters.append((float(X[0]), float(X[1])))
        except Exception:
            continue

    if len(inters) < 2 and len(box_inters) >= 2:
        return box_inters
    return inters


# --- SplitState orchestration helper ---


def _push_split_history(state, self_obj, maxlen: int = 10):
    """Push current segments into both state and self split history lists with trimming."""
    try:
        state.split_history.append(state.segments.copy())
        if len(state.split_history) > maxlen:
            state.split_history.pop(0)
    except Exception:
        pass
    try:
        self_obj.split_history.append(self_obj.segments.copy())
        if len(self_obj.split_history) > maxlen:
            self_obj.split_history.pop(0)
    except Exception:
        pass


def _apply_new_segments_counts(state, self_obj, new_segments):
    """Apply new segments array and update n_segments on both state and self."""
    state.segments = new_segments
    self_obj.segments = new_segments
    try:
        state.n_segments = int(new_segments.max())
    except Exception:
        state.n_segments = int(new_segments.max()) if new_segments is not None else 0
    try:
        self_obj.n_segments = int(new_segments.max())
    except Exception:
        self_obj.n_segments = int(new_segments.max()) if new_segments is not None else 0


def _inherit_label_and_select(state, self_obj, seg_id, info):
    """Set splitting_segment_id on both state and self, and inherit label if present."""
    try:
        new_id = int(info.get('new_seg_id')) if info.get('new_seg_id') is not None else None
        if new_id is None:
            return None
        state.splitting_segment_id = new_id
        self_obj.splitting_segment_id = new_id
        orig_label = state.segment_labels.get(seg_id, None)
        if orig_label is not None:
            state.segment_labels[new_id] = orig_label
            self_obj.segment_labels[new_id] = orig_label
        return new_id
    except Exception:
        return None


def _apply_split_state(self, state, seg_mask, line_mask, full_line_mask, seg_id, points, debug_img, line_thickness=1, endpoint_radius=2, seg_area=0, edge_mask=None, force_global: bool=False):
    """Apply split result to SplitState and the LabelingTool `self`.

    - Calls _apply_split_core and, on success, updates both the provided `state`
      object and the live `self` state (segments, split_history, n_segments, selection, labels)
    - Returns (updated_state, applied, info)
    """
    # Note: _apply_split_core now enforces strict snapping behavior; when invoked via state helper
    # we do not force global cuts, so call with default strict parameters (no require_snapped enforcement here)
    new_segments, applied, info = _apply_split_core(self, seg_mask, line_mask, full_line_mask, seg_id, points, debug_img, line_thickness=line_thickness, endpoint_radius=endpoint_radius, seg_area=seg_area, edge_mask=edge_mask, require_snapped=False, snapped_endpoints=False)

    if not applied:
        return state, False, info

    # push history into both state and self
    _push_split_history(state, self)

    # Apply segments and update counts
    _apply_new_segments_counts(state, self, new_segments)

    # Update selection and inherit label if present
    new_id = _inherit_label_and_select(state, self, seg_id, info)

    return state, True, info

    # CRITICAL: Constrain line_mask to only pixels within the selected segment
    # This prevents the line from extending into neighboring segments
    line_mask = line_mask & seg_mask.astype(np.uint8) * 255
    
    print(f"Combined line mask pixels (within segment): {np.sum(line_mask > 0)}")
    
    # Add visualization to debug image
    # Show edge pixels in blue
    debug_img[edge_mask > 0] = [0, 0, 255]
    # Draw user's original points in yellow
    try:
        for poly in self.manual_polylines:
            for pt in poly:
                try:
                    cv2.circle(debug_img, tuple(pt), 5, (255, 255, 0), -1)
                except Exception:
                    pass
    except Exception:
        pass
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

    # CRITICAL: Constrain line_mask to only pixels within the selected segment
    # This prevents the line from extending into neighboring segments
    line_mask = line_mask & (seg_mask.astype(np.uint8) * 255)

    # Debug: report snapping and mask stats
    try:
        print(f"  DEBUG: all_open_polylines_snapped={all_open_polylines_snapped}, line_mask_pixels={int((line_mask>0).sum())}")
    except Exception:
        pass

    # Call the extracted split orchestration helper (strict snapping required)
    new_segments, applied, info = _apply_split_core(self, seg_mask, line_mask, full_line_mask, seg_id, points, debug_img, line_thickness=line_thickness, endpoint_radius=endpoint_radius, seg_area=seg_area, edge_mask=edge_mask, require_snapped=True, snapped_endpoints=all_open_polylines_snapped)

    if not applied:
        try:
            print(f"  DEBUG: split not applied, info={info}")
        except Exception:
            pass
        try:
            # Expose info to caller/UI for diagnostics and show concise message
            self._last_split_info = info
            reason = info.get('reason', 'no_split') if isinstance(info, dict) else str(info)
            try:
                self.set_manual_status(f"Split did not create any region (reason: {reason})")
            except Exception:
                pass
        except Exception:
            pass


    if applied:
        # apply results and handle UI state in caller
        self.segments = new_segments
        try:
            # infer new_seg_id from info
            new_seg_id = int(info.get('new_seg_id')) if info.get('new_seg_id') is not None else None
            if new_seg_id is not None:
                self.splitting_segment_id = new_seg_id
        except Exception:
            pass

        # Update counts and UI elements similarly to other success paths
        try:
            self.n_segments = int(self.segments.max())
        except Exception:
            self.n_segments = int(self.segments.max()) if self.segments is not None else 0
        try:
            self.set_manual_status("Split applied! Draw another or toggle off")
        except Exception:
            pass
        try:
            self._update_display_with_highlight(self.splitting_segment_id)
        except Exception:
            pass
        return

    # If combined cuts didn't yet produce multiple components, return and keep polylines
    if ready_to_split:
        # Delegate split apply logic to helper that updates a lightweight SplitState
        try:
            from labeling.core.split_core import apply_direct_split
        except Exception:
            apply_direct_split = None

        # Wrap current segments & labels in simple state for helper
        state = SplitState(segments=self.segments.copy(), manual_polylines=self.manual_polylines.copy(), segment_labels=self.segment_labels.copy(), split_history=getattr(self, 'split_history', []), n_segments=getattr(self, 'n_segments', 0), splitting_segment_id=getattr(self, 'splitting_segment_id', None))

        applied = False
        new_seg_id = None
        info = {}
        try:
            # Use the pure helper already implemented in split_core (if available)
            if apply_direct_split is not None:
                new_segments, new_seg_id, info = apply_direct_split(state.segments, seg_id, labeled_regions, region_sizes, min_side_px=min_side_size)
                if info.get('applied'):
                    # push undo history and apply update to 'self'
                    try:
                        self.split_history.append(self.segments.copy())
                        if len(self.split_history) > 10:
                            self.split_history.pop(0)
                        try:
                            self.set_undo_enabled(True)
                        except Exception:
                            pass
                    except Exception:
                        pass

                    # apply result
                    self.segments = new_segments
                    new_seg_id = int(new_seg_id)
                    # Inherit label if original had one
                    try:
                        orig_label = self.segment_labels.get(seg_id, None)
                        if orig_label is not None:
                            self.segment_labels[new_seg_id] = orig_label
                    except Exception:
                        pass
                    segments_added += 1
                    self.splitting_segment_id = new_seg_id
                    applied = True
        except Exception:
            applied = False

        if applied:
            try:
                print(f"  → Direct-split: moved pixels to seg {new_seg_id}")
            except Exception:
                pass

            # Finalize UI updates
            try:
                self.n_segments = int(self.segments.max())
            except Exception:
                self.n_segments = int(self.segments.max()) if self.segments is not None else 0
            try:
                self.set_manual_status("Split applied! Draw another or toggle off")
            except Exception:
                pass
            try:
                self.set_finalize_enabled(True)
            except Exception:
                pass
            try:
                self._clear_manual_line()
            except Exception:
                pass
            try:
                self._reset_submit_button()
            except Exception:
                pass
            try:
                self._update_display()
            except Exception:
                pass
            try:
                self._update_progress()
            except Exception:
                pass
            return
        else:
            # Not ready or apply failed - fall through to other fallbacks
            pass

    # If combined cuts didn't yet produce multiple components, return and keep polylines
    try:
        self.set_manual_status("No split yet - add more cuts or adjust lines")
        try:
            self.set_finalize_enabled(True)
        except Exception:
            pass
    except Exception:
        pass
# After fallbacks, if segments_added still 0, attempt direct split via pure helper only if strict snapping held
    if segments_added == 0:
        if not all_open_polylines_snapped:
            # do not attempt any fallback - inform user that endpoints were not snapped
            try:
                self.set_manual_status("Split failed: endpoints not snapped to edges/lines; adjust and try again")
            except Exception:
                pass
            segments_added = 0
            self.splitting_segment_id = None
        else:
            from labeling.core.split_core import apply_direct_split
            min_side_size = max(100, int(seg_area * 0.01))
            new_segments, new_seg_id, info = apply_direct_split(self.segments, seg_id, labeled_regions, region_sizes, min_side_px=min_side_size)
            if info.get('applied'):
                # replace segments and update metadata
                self.segments = new_segments
                segments_added = 1
                self.splitting_segment_id = new_seg_id
                try:
                    cv2.polylines(debug_img, [points], isClosed=False, color=(0, 255, 0), thickness=line_thickness)
                except Exception:
                    pass
                try:
                    print(f"  → Split successful: kept {info['side1_count']} as seg {seg_id}, moved {info['side2_count']} to seg {new_seg_id}")
                except Exception:
                    pass
                try:
                    self.set_manual_status(f"Segment {self.splitting_segment_id} selected - draw another line to split further")
                except Exception:
                    pass
                try:
                    self._update_display_with_highlight(self.splitting_segment_id)
                except Exception:
                    pass
            else:
                try:
                    cv2.drawMarker(debug_img, tuple(points[0]), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 40, 4)
                    cv2.drawMarker(debug_img, tuple(points[-1]), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 40, 4)
                except Exception:
                    pass
                try:
                    print(f"  → Split failed: {info.get('reason')} (min {min_side_size} pixels)")
                except Exception:
                    pass
                segments_added = 0
                self.splitting_segment_id = None
                try:
                    self.set_manual_status("Split failed: one side too small; redraw closer to midline")
                except Exception:
                    pass

            # Save debug image
            if self.images_folder:
                # Put debug folder at workspace level (parent of images folder)
                debug_folder = self.images_folder.parent / "debug"
                debug_folder.mkdir(exist_ok=True)
                debug_path = debug_folder / f"split_seg{seg_id}_{self.image_files[self.current_idx].stem}.png"
                cv2.imwrite(str(debug_path), cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
                try:
                    print(f"  Debug image saved: {debug_path}")
                except Exception:
                    pass
            try:
                old_n_segments = getattr(self, 'n_segments', 0)
            except Exception:
                old_n_segments = 0
            try:
                self.n_segments = int(self.segments.max())
            except Exception:
                self.n_segments = int(self.segments.max()) if self.segments is not None else 0
            try:
                print(f"\nSegment count: {old_n_segments} -> {self.n_segments} (added {segments_added})")
            except Exception:
                pass

            # Warn if no segments were actually split
            if segments_added == 0:
                try:
                    print(f"WARNING: No new segments created - line may be too thin or grazing edges")
                except Exception:
                    pass
                try:
                    print(f"TIP: Draw line through the middle/widest part of the segment")
                except Exception:
                    pass

            try:
                print(f"=== END DEBUG ===\n")
            except Exception:
                pass

            # Clear the drawn line(s). Preserve selection if the split succeeded so user can continue.
            keep_sel = None
            if segments_added == 1 and getattr(self, 'splitting_segment_id', None) is not None:
                keep_sel = int(self.splitting_segment_id)
            try:
                self._clear_manual_line()
            except Exception:
                pass
            # Restore selection if appropriate
            if keep_sel is not None:
                self.splitting_segment_id = keep_sel
                try:
                    self._update_display_with_highlight(self.splitting_segment_id)
                except Exception:
                    pass
            else:
                self.splitting_segment_id = None
            try:
                self._reset_submit_button()
            except Exception:
                pass
            try:
                self._update_display()
            except Exception:
                pass
            try:
                self._update_progress()
            except Exception:
                pass
            try:
                print(f"FINAL splitting_segment_id (end): {getattr(self, 'splitting_segment_id', None)}")
            except Exception:
                pass




# --- Endpoint snapping helpers (extracted from scripts/labeling_tool.py) ---

def snap_endpoints(p0: Tuple[int,int], p_last: Tuple[int,int], edge_coords: np.ndarray,
                   other_line_coords: np.ndarray, max_snap_dist: int) -> Tuple[Tuple[int,int], Tuple[int,int], str, str]:
    """Compute snapped endpoints for a drawn polyline.

    Returns: (edge_pt_0, edge_pt_last, chosen0_source, chosen_last_source)
    Sources are one of: 'edge' or 'line'. Each endpoint snaps to the nearest pixel
    from either the segment border (`edge_coords`) or the set of pixels belonging
    to other polylines (`other_line_coords`). If neither source is within `max_snap_dist`
    the original endpoint is returned (no snap).
    """
    # Convert inputs to numpy arrays for distance computations (use (y,x) ordering)
    p0_arr = np.array([p0[1], p0[0]])
    p1_arr = np.array([p_last[1], p_last[0]])

    # Distances to edge pixels
    if edge_coords.size == 0:
        dist_to_edges_0 = np.array([float('inf')])
        dist_to_edges_last = np.array([float('inf')])
    else:
        dist_to_edges_0 = np.linalg.norm(edge_coords - p0_arr, axis=1)
        dist_to_edges_last = np.linalg.norm(edge_coords - p1_arr, axis=1)

    nearest_edge_dist_0 = float(dist_to_edges_0.min()) if edge_coords.size > 0 else float('inf')
    nearest_edge_dist_last = float(dist_to_edges_last.min()) if edge_coords.size > 0 else float('inf')

    # Distances to other line pixels
    if other_line_coords is None or other_line_coords.size == 0:
        dist_to_line_0 = np.array([float('inf')])
        dist_to_line_last = np.array([float('inf')])
    else:
        dist_to_line_0 = np.linalg.norm(other_line_coords - p0_arr, axis=1)
        dist_to_line_last = np.linalg.norm(other_line_coords - p1_arr, axis=1)

    nearest_line_dist_0 = float(dist_to_line_0.min()) if (other_line_coords is not None and other_line_coords.size > 0) else float('inf')
    nearest_line_dist_last = float(dist_to_line_last.min()) if (other_line_coords is not None and other_line_coords.size > 0) else float('inf')

    # Choose for start: pick whichever source is closer (line vs edge). No max distance threshold.
    chosen0 = (int(p0[0]), int(p0[1]))
    chosen0_source = 'none'
    if nearest_line_dist_0 < nearest_edge_dist_0 and nearest_line_dist_0 < float('inf'):
        nearest_line_0 = other_line_coords[np.argmin(dist_to_line_0)]
        chosen0 = (int(nearest_line_0[1]), int(nearest_line_0[0]))
        chosen0_source = 'line'
    elif nearest_edge_dist_0 < float('inf'):
        nearest_edge_0 = edge_coords[np.argmin(dist_to_edges_0)]
        chosen0 = (int(nearest_edge_0[1]), int(nearest_edge_0[0]))
        chosen0_source = 'edge'
    else:
        chosen0_source = 'none'

    # Choose for last
    chosen_last = (int(p_last[0]), int(p_last[1]))
    chosen_last_source = 'none'
    if nearest_line_dist_last < nearest_edge_dist_last and nearest_line_dist_last < float('inf'):
        nearest_line_last = other_line_coords[np.argmin(dist_to_line_last)]
        chosen_last = (int(nearest_line_last[1]), int(nearest_line_last[0]))
        chosen_last_source = 'line'
    elif nearest_edge_dist_last < float('inf'):
        nearest_edge_last = edge_coords[np.argmin(dist_to_edges_last)]
        chosen_last = (int(nearest_edge_last[1]), int(nearest_edge_last[0]))
        chosen_last_source = 'edge'
    else:
        chosen_last_source = 'none'

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


def _snap_endpoints_for_polyline(p0: Tuple[int,int], p_last: Tuple[int,int], edge_coords: np.ndarray, other_polylines: list, seg_uint8_local: np.ndarray, max_snap_dist: int = 999999) -> Tuple[Tuple[int,int], Tuple[int,int], str, str]:
    """Helper to snap polyline endpoints to either nearest boundary edge or nearest other polyline pixel.

    - `other_polylines` is a list of polylines (each a list of (x,y) tuples). This function will rasterize
      them into a mask using `mask_from_polylines` when present.
    - Returns (snapped_start, snapped_end, source_start, source_end) where source is 'edge','line' or 'none'.
    """
    # Build other line coords if any
    try:
        if other_polylines:
            from labeling.core.split_core import mask_from_polylines
            other_mask = mask_from_polylines(other_polylines, seg_uint8_local.shape, thickness=1)
            other_coords = np.argwhere(other_mask > 0)
        else:
            other_coords = np.empty((0,2), dtype=int)
    except Exception:
        other_coords = np.empty((0,2), dtype=int)

    # Use the general snap_endpoints function which operates on coordinate arrays
    s0, s1, src0, src1 = snap_endpoints(p0, p_last, edge_coords, other_coords, max_snap_dist=max_snap_dist)

    # If endpoints collapsed, attempt to stretch via contour/project fallback
    try:
        s0, s1 = stretch_endpoints_if_collapsed(s0, s1, seg_uint8_local, p0, p_last)
    except Exception:
        pass

    return s0, s1, src0, src1
