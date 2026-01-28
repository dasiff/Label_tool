"""Experimental rectangular-area detection prototype.

This is an experiment: coarse-scale contour+approximation detector that
searches for quadrilateral candidates on an image pyramid and returns
rectangle hypotheses (upscaled to the original image resolution).

The module is intentionally small and flagged experimental so it can be
replaced or removed later without affecting the main pipeline.
"""
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

EXPERIMENTAL = True


def detect_rectangles_pyramid(img_rgb: np.ndarray,
                               scales: Tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0),
                               min_area: int = 2000,
                               approx_eps_factor: float = 0.02,
                               use_hough: bool = True) -> List[Dict[str, Any]]:
    """Detect rectangular candidates using multi-scale contour approximation.

    Returns a list of dicts with at least the keys:
      - "quad": 4x2 list of points in original-image coordinates (if a quad was found)
      - "score": heuristic score (higher is better)
      - "area": approx area in px

    This is intentionally permissive and experimental.
    """
    h0, w0 = img_rgb.shape[:2]
    results: List[Dict[str, Any]] = []

    gray_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if img_rgb.ndim == 3 else img_rgb

    for scale in scales:
        if scale <= 0 or scale > 1.0:
            continue
        small = cv2.resize(gray_orig, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # Blur to reduce small occluders (trees, cars)
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Optionally run a coarse Hough-based detection at the coarsest scales to recover
        # long straight edges that survive occlusion (trees/cars). This is intentionally
        # conservative and only used on coarse scales to avoid noise.
        if use_hough and scale <= 0.125:
            hough_entries = _hough_rectangles_from_edges(edges, scale, h0, w0, min_area=min_area)
            results.extend(hough_entries)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < max(10, min_area * (scale * scale)):
                continue
            peri = cv2.arcLength(cnt, True)
            eps = max(1.0, approx_eps_factor * peri)
            approx = cv2.approxPolyDP(cnt, eps, True)

            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(float) / scale
                quad = _order_quad_clockwise(quad)
                score = 1.0
                # Compute rectangularity metric: area / bounding box area
                mask_contour = np.zeros((h0, w0), dtype=np.uint8)
                cv2.fillPoly(mask_contour, [np.round(quad).astype(int)], 255)
                area_full = mask_contour.sum() / 255
                box = cv2.minAreaRect(np.round(quad).astype(np.float32))
                box_pts = cv2.boxPoints(box)
                box_mask = np.zeros((h0, w0), dtype=np.uint8)
                cv2.fillPoly(box_mask, [np.round(box_pts).astype(int)], 255)
                box_area = box_mask.sum() / 255
                rect_score = float(area_full / max(1.0, box_area))
                # Favor larger-area detections: penalize tiny contours relative to min_area
                area_factor = min(1.0, float(area_full) / float(max(1, min_area)))
                # Combine confidence
                score *= rect_score * area_factor
                results.append({"quad": quad.tolist(), "score": score, "area": float(area_full)})
            else:
                # Consider min-area rectangle fallback for more robustness
                box = cv2.minAreaRect(cnt)
                box_pts = cv2.boxPoints(box)  # in small-scale coords
                box_pts_orig = (box_pts.astype(float) / scale)
                # Reject extremely small boxes
                box_w = box[1][0] / scale
                box_h = box[1][1] / scale
                if box_w * box_h < min_area:
                    continue
                # Heuristic score: aspect ratio and box area compared to contour area
                mask_cnt = np.zeros_like(small, dtype=np.uint8)
                cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
                area_cnt_full = cv2.contourArea(cnt) / (scale * scale)
                box_mask_full = np.zeros((h0, w0), dtype=np.uint8)
                cv2.fillPoly(box_mask_full, [np.round(box_pts_orig).astype(int)], 255)
                box_area_full = box_mask_full.sum() / 255
                rect_score = float(area_cnt_full / max(1.0, box_area_full))
                # Favor larger-area detections: penalize tiny boxes relative to min_area
                area_factor = min(1.0, float(box_area_full) / float(max(1, min_area)))
                # Upscale box points to full resolution
                results.append({"box": box_pts_orig.tolist(), "score": float(rect_score * 0.5 * area_factor), "area": float(box_area_full)})

    # Merge overlapping detections (simple NMS-ish by IoU on bounding boxes)
    merged = _nms_merge(results)
    # Sort by combined confidence and area to favor large, well-supported candidates
    merged.sort(key=lambda r: r.get("score", 0.0) * float(r.get("area", 1.0)), reverse=True)
    return merged


# --- Helpers ---

def _order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    """Order quad points in clockwise order starting with top-left."""
    # Compute centroid and angle
    cx, cy = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    pts_ordered = pts[order]
    # ensure starting point is the top-left
    min_idx = np.argmin(np.sum(pts_ordered, axis=1))
    pts_rot = np.roll(pts_ordered, -min_idx, axis=0)
    return pts_rot


# --- Hough / RANSAC helpers (coarse-scale) ---

def _line_length(l: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = l
    return float(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)


def _intersect_lines(l1, l2):
    """Intersect two infinite lines each defined by endpoints (x1,y1,x2,y2).
    Returns (x,y) float or None if parallel.
    """
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom
    return (px, py)


def _hough_rectangles_from_edges(edges_small: np.ndarray, scale: float, h0:int, w0:int, min_area:int=2000) -> List[Dict[str, Any]]:
    """Run Probabilistic Hough on a small-scale edge map and form rectangle hypotheses.

    Returns entries with 'box' points in original-image coords and heuristic 'score'.
    """
    h_s, w_s = edges_small.shape[:2]
    min_len = max(8, int(min(h_s, w_s) * 0.35))
    # Tune Hough parameters for coarse detection
    lines = cv2.HoughLinesP(edges_small, rho=1, theta=np.pi/180, threshold=40, minLineLength=min_len, maxLineGap=20)
    if lines is None:
        return []
    lines = [tuple(l[0]) for l in lines]
    # Filter and sort by length
    lines = sorted(lines, key=_line_length, reverse=True)

    # Classify lines into angle buckets (near-horizontal and near-vertical)
    horiz = []
    vert = []
    for l in lines:
        x1,y1,x2,y2 = l
        ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        if ang > 90:
            ang = 180 - ang
        if ang <= 20:
            horiz.append(l)
        elif ang >= 70:
            vert.append(l)
    # If insufficient lines, abort
    if len(horiz) < 2 or len(vert) < 2:
        return []

    candidates: List[Dict[str, Any]] = []
    # Limit to top few long lines to avoid combinatorial blowup
    top_h = horiz[:4]
    top_v = vert[:4]

    for i in range(len(top_h)):
        for j in range(i+1, len(top_h)):
            lh1, lh2 = top_h[i], top_h[j]
            for a in range(len(top_v)):
                for b in range(a+1, len(top_v)):
                    lv1, lv2 = top_v[a], top_v[b]
                    # Intersect to make 4 corners
                    p00 = _intersect_lines(lh1, lv1)
                    p10 = _intersect_lines(lh2, lv1)
                    p11 = _intersect_lines(lh2, lv2)
                    p01 = _intersect_lines(lh1, lv2)
                    if None in (p00, p10, p11, p01):
                        continue
                    pts = np.array([p00, p10, p11, p01], dtype=float)
                    # Scale points to original image coordinates
                    pts_orig = pts / scale
                    # Check bounding box validity
                    x1, y1 = pts_orig[:,0].min(), pts_orig[:,1].min()
                    x2, y2 = pts_orig[:,0].max(), pts_orig[:,1].max()
                    if x2 - x1 < 4 or y2 - y1 < 4:
                        continue
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < min_area:
                        continue
                    # Compute rectangularity: fill polygon area vs bbox area
                    mask = np.zeros((h0, w0), dtype=np.uint8)
                    cv2.fillPoly(mask, [np.round(pts_orig).astype(int)], 255)
                    area_full = mask.sum() / 255
                    rect_score = float(area_full / max(1.0, box_area))
                    # Line support: average fraction of line lengths relative to small dim
                    len_avg = ( _line_length(lh1) + _line_length(lh2) + _line_length(lv1) + _line_length(lv2) ) / 4.0
                    len_factor = min(1.0, len_avg / max(1, min(h_s, w_s)))
                    score = rect_score * (0.5 + 0.5 * len_factor)
                    candidates.append({"box": pts_orig.tolist(), "score": float(score), "area": float(box_area)})
    return candidates


def _rect_from_entry(entry: Dict[str, Any]) -> np.ndarray:
    """Return 4x2 rectangle points in float coords for an entry (box or quad)."""
    if "quad" in entry:
        return np.array(entry["quad"], dtype=float)
    if "box" in entry:
        return np.array(entry["box"], dtype=float)
    return np.zeros((0, 2), dtype=float)


def _bbox_from_pts(pts: np.ndarray) -> Tuple[int, int, int, int]:
    x_min = int(np.min(pts[:, 0]))
    y_min = int(np.min(pts[:, 1]))
    x_max = int(np.max(pts[:, 0]))
    y_max = int(np.max(pts[:, 1]))
    return x_min, y_min, x_max, y_max


def _iou_of_entries(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    a_pts = _rect_from_entry(a)
    b_pts = _rect_from_entry(b)
    if a_pts.size == 0 or b_pts.size == 0:
        return 0.0
    ax1, ay1, ax2, ay2 = _bbox_from_pts(a_pts)
    bx1, by1, bx2, by2 = _bbox_from_pts(b_pts)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1 + 1)
    ih = max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1 + 1) * (ay2 - ay1 + 1))
    area_b = max(1, (bx2 - bx1 + 1) * (by2 - by1 + 1))
    union = area_a + area_b - inter
    return float(inter) / float(union)


def _nms_merge(entries: List[Dict[str, Any]], iou_thresh: float = 0.5) -> List[Dict[str, Any]]:
    if not entries:
        return []
    keep: List[Dict[str, Any]] = []
    used = [False] * len(entries)
    for i, e in enumerate(entries):
        if used[i]:
            continue
        merged = e.copy()
        used[i] = True
        for j in range(i + 1, len(entries)):
            if used[j]:
                continue
            iou = _iou_of_entries(merged, entries[j])
            if iou > iou_thresh:
                # merge by picking the higher-score and keeping area/points from that one
                if entries[j].get("score", 0.0) > merged.get("score", 0.0):
                    merged = entries[j].copy()
                used[j] = True
        keep.append(merged)
    return keep


def draw_rectangles_debug(img_rgb: np.ndarray, entries: List[Dict[str, Any]]) -> np.ndarray:
    """Return an RGB image with rectangle overlays drawn for debugging."""
    out = img_rgb.copy()
    for idx, e in enumerate(entries):
        pts = _rect_from_entry(e)
        if pts.size == 0:
            continue
        pts_i = np.round(pts).astype(int)
        cv2.polylines(out, [pts_i], isClosed=True, color=(255, 0, 0), thickness=2)
        # draw center and score
        cx, cy = np.mean(pts, axis=0).astype(int)
        cv2.putText(out, f"{e.get('score', 0.0):.2f}", (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out
