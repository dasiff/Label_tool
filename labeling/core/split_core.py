"""Pure split helpers for manual split operations.

Functions here operate on numpy arrays and do not touch UI state or files.
They are unit-testable and deterministic.
"""
from typing import Tuple, List, Dict, Any
import numpy as np
from scipy import ndimage


def label_regions_after_cut(segment_mask: np.ndarray, line_mask: np.ndarray) -> Tuple[np.ndarray, int, List[Tuple[int, int]]]:
    """Given a binary segment mask and a line mask overlay, remove the line
    pixels from the segment and label the resulting connected components.

    Returns (labeled_regions, num_regions, region_sizes_sorted)
    where region_sizes_sorted is a list of (size, region_id) sorted descending.
    """
    # Ensure boolean masks
    seg_bool = (segment_mask != 0)
    cut_mask = (line_mask != 0)

    # Remove cut pixels from the segment
    seg_after = seg_bool & (~cut_mask)

    # Label connected components (4-connectivity)
    labeled, num = ndimage.label(seg_after)

    # Compute sizes
    region_sizes = []
    for rid in range(1, num + 1):
        size = int(np.sum(labeled == rid))
        region_sizes.append((size, rid))

    # Sort descending by size
    region_sizes.sort(reverse=True)

    # If segmentation didn't split and there are line pixels inside the segment, attempt a deterministic single-pixel connector
    if num < 2 and np.any(cut_mask):
        # Find connected components of the cut inside the segment
        cut_labeled, cut_num = ndimage.label(cut_mask)
        if cut_num >= 2:
            # Choose two largest cut components
            comp_sizes = []
            for cid in range(1, cut_num + 1):
                comp_sizes.append((int(np.sum(cut_labeled == cid)), cid))
            comp_sizes.sort(reverse=True)
            a_id = comp_sizes[0][1]
            b_id = comp_sizes[1][1]
            # BFS shortest path between any pixel in comp a and any in comp b constrained to seg_bool
            from collections import deque
            h, w = seg_bool.shape
            visited = -np.ones((h, w), dtype=np.int32)
            parents = -np.ones((h, w, 2), dtype=np.int32)
            dq = deque()
            ay, ax = np.where(cut_labeled == a_id)
            by, bx = np.where(cut_labeled == b_id)
            targets = set(zip(by.tolist(), bx.tolist()))
            for (y, x) in zip(ay.tolist(), ax.tolist()):
                visited[y, x] = 1
                parents[y, x] = (-1, -1)
                dq.append((y, x))
            found = False
            dest = None
            while dq and not found:
                y, x = dq.popleft()
                for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                    ny, nx = y+dy, x+dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if not seg_bool[ny, nx]:
                        continue
                    if visited[ny, nx] != -1:
                        continue
                    visited[ny, nx] = 1
                    parents[ny, nx] = (y, x)
                    if (ny, nx) in targets:
                        found = True
                        dest = (ny, nx)
                        break
                    dq.append((ny, nx))
            if found and dest is not None:
                # Reconstruct path from dest back to any a component pixel
                path = []
                y, x = dest
                while (y, x) != (-1, -1):
                    path.append((y, x))
                    py, px = parents[y, x]
                    y, x = int(py), int(px)
                # Add path pixels to cut_mask
                for (y, x) in path:
                    cut_mask[y, x] = True
                # Recompute labeling
                labeled, num = ndimage.label(seg_bool & (~cut_mask))
                region_sizes = []
                for rid in range(1, num + 1):
                    size = int(np.sum(labeled == rid))
                    region_sizes.append((size, rid))
                region_sizes.sort(reverse=True)
    return labeled, num, region_sizes


def apply_direct_split(segments: np.ndarray, seg_id: int, labeled_regions: np.ndarray, region_sizes: List[Tuple[int, int]], min_side_px: int) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Attempt to split `segments` by assigning the second-largest region
    (from `labeled_regions`) to a new segment id if both sides meet `min_side_px`.

    Returns (new_segments, new_seg_id_or_None, info dict)
    info contains keys: applied (bool), side1_count, side2_count, reason

    Note: `segments` is not modified in-place; a copy is returned.
    """
    res_info = {"applied": False, "reason": None, "side1_count": 0, "side2_count": 0}

    if len(region_sizes) < 2:
        res_info['reason'] = 'not_enough_regions'
        return segments.copy(), None, res_info

    side1_count, side1_id = region_sizes[0]
    side2_count, side2_id = region_sizes[1]

    res_info['side1_count'] = int(side1_count)
    res_info['side2_count'] = int(side2_count)

    if side1_count >= min_side_px and side2_count >= min_side_px:
        new_segments = segments.copy()
        new_seg_id = int(new_segments.max()) + 1
        new_segments[labeled_regions == side2_id] = new_seg_id
        res_info['applied'] = True
        res_info['new_seg_id'] = int(new_seg_id)
        res_info['reason'] = 'ok'
        return new_segments, new_seg_id, res_info
    else:
        res_info['reason'] = 'side_too_small'
        return segments.copy(), None, res_info


def mask_from_polylines(polylines: List[List[Tuple[int,int]]], shape: Tuple[int,int], thickness: int = 1) -> np.ndarray:
    """Rasterize polylines into a binary mask of shape (H,W).
    `polylines` is a list of polylines, each a list of (x,y) tuples.
    """
    import cv2
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polylines:
        if len(poly) >= 2:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=thickness)
            # Also draw endpoints to ensure connectivity
            cv2.circle(mask, tuple(pts[0]), radius=max(1, thickness), color=1, thickness=-1)
            cv2.circle(mask, tuple(pts[-1]), radius=max(1, thickness), color=1, thickness=-1)
    return mask


def build_combined_line_mask(polylines: List[List[Tuple[int,int]]], seg_mask: np.ndarray, shape: Tuple[int,int], thickness: int = 1, endpoint_radius: int = 2) -> np.ndarray:
    """Build a combined line mask from polylines, constrained to `seg_mask`.

    - Ensures endpoints are drawn as filled circles to maximize connectivity.
    - Returns binary mask (uint8) matching `shape` clipped to `seg_mask`.
    """
    import cv2
    h, w = shape
    full_mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polylines:
        if len(poly) >= 2:
            pts = np.array(poly, dtype=np.int32)
            cv2.polylines(full_mask, [pts], isClosed=False, color=1, thickness=thickness)
            cv2.circle(full_mask, tuple(pts[0]), radius=endpoint_radius, color=1, thickness=-1)
            cv2.circle(full_mask, tuple(pts[-1]), radius=endpoint_radius, color=1, thickness=-1)
    # Clip to inside the selected segment
    constrained = (full_mask > 0) & (seg_mask > 0)
    return constrained.astype(np.uint8)


def constrained_subtract_and_label(seg_mask: np.ndarray, line_mask: np.ndarray) -> Tuple[np.ndarray, int, List[Tuple[int,int]]]:
    """Subtract the `line_mask` from `seg_mask` and label the resulting components.
    Returns (labeled_regions, num_regions, region_sizes_sorted)
    """
    return label_regions_after_cut(seg_mask, line_mask)


def closed_polygon_handler(poly_pts: List[Tuple[int,int]], seg_mask: np.ndarray, min_side_px: int) -> Tuple[bool, np.ndarray, int]:
    """Handle a closed polygon drawn inside a segment.
    Returns (applied, new_seg_mask, area)
    - applied: whether a new segment should be created
    - new_seg_mask: boolean mask of the polygon interior constrained to seg_mask
    - area: pixel area of new_seg_mask
    """
    import cv2
    h, w = seg_mask.shape
    poly = np.array(poly_pts, dtype=np.int32)
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    try:
        cv2.fillPoly(poly_mask, [poly], 1)
    except Exception:
        return False, np.zeros_like(seg_mask, dtype=bool), 0
    constrained = (poly_mask > 0) & (seg_mask > 0)
    area = int(np.sum(constrained))
    if area >= min_side_px:
        return True, constrained, area
    else:
        return False, constrained, area


def line_disconnects(seg_mask: np.ndarray, line_mask: np.ndarray) -> bool:
    """Return True if removing `line_mask` from `seg_mask` disconnects the
    segment in either horizontal (left-right) or vertical (top-bottom) sense.
    Uses BFS limited to the bounding box of the segment for efficiency.
    """
    from collections import deque
    h, w = seg_mask.shape
    seg_without = (seg_mask > 0) & (~(line_mask > 0))
    if seg_without.sum() == 0:
        return True

    ys, xs = np.where(seg_mask > 0)
    minx, maxx = int(xs.min()), int(xs.max())
    miny, maxy = int(ys.min()), int(ys.max())

    # Helper to check connectivity between two opposite sides
    def connects_side(axis: str) -> bool:
        visited = np.zeros_like(seg_without, dtype=bool)
        q = deque()
        if axis == 'lr':
            # seed from leftmost column inside bbox
            for y in range(miny, maxy+1):
                if seg_without[y, minx]:
                    q.append((y, minx)); visited[y, minx] = True
            target_x = maxx
            while q:
                y, x = q.popleft()
                if x == target_x:
                    return True
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y+dy, x+dx
                    if ny < miny or ny > maxy or nx < minx or nx > maxx:
                        continue
                    if seg_without[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True; q.append((ny, nx))
            return False
        else:
            # top-bottom
            for x in range(minx, maxx+1):
                if seg_without[miny, x]:
                    q.append((miny, x)); visited[miny, x] = True
            target_y = maxy
            while q:
                y, x = q.popleft()
                if y == target_y:
                    return True
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = y+dy, x+dx
                    if ny < miny or ny > maxy or nx < minx or nx > maxx:
                        continue
                    if seg_without[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True; q.append((ny, nx))
            return False

    # If either left-right or top-bottom connectivity is broken, we declare disconnected
    lr = connects_side('lr')
    tb = connects_side('tb')
    return not (lr and tb)
