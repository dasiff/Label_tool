import cv2
import numpy as np


def _compute_buffer_mask(self):
    """Compute buffer mask (bool) outside the property boundary within buffer distance."""
    if self.current_boundary is None:
        self.buffer_mask = None
        return
    h, w = self.clean_image.shape[:2]
    # Create ROI mask for interior (filled polygon)
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [self.current_boundary.astype(np.int32)], 255)
    # Create an edge mask of the boundary and compute distance transform outside
    edge_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(edge_mask, [self.current_boundary.astype(np.int32)], isClosed=True, color=255, thickness=1)
    invert = (edge_mask == 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(invert, cv2.DIST_L2, 5)
    if self.buffer_mode_var.get() == 'px':
        thresh = self.road_buffer_px
    else:
        val = self.road_buffer_pct
        thresh = min(h, w) * val
    buffer_mask = (dist <= thresh) & (roi_mask == 0)
    # Store as boolean mask
    self.buffer_mask = buffer_mask.astype(bool)


def _closest_point_on_boundary(self, x, y):
    """Return closest projected point on boundary and fraction along boundary length."""
    if self.current_boundary is None:
        return None, None
    pts = self.current_boundary
    # compute segment lengths and cumulative
    seg_starts = pts
    seg_ends = np.vstack([pts[1:], pts[0]])
    seg_vecs = seg_ends - seg_starts
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    best_dist = float('inf')
    best_pt = None
    best_frac = 0.0
    best_seg_idx = 0
    px = np.array([x, y])
    for i, (a, b, v, L) in enumerate(zip(seg_starts, seg_ends, seg_vecs, seg_lens)):
        if L == 0:
            continue
        t = np.dot(px - a, v) / (L * L)
        t_clamped = max(0.0, min(1.0, t))
        proj = a + t_clamped * v
        d = np.linalg.norm(proj - px)
        if d < best_dist:
            best_dist = d
            best_pt = proj
            frac = (cum[i] + L * t_clamped) / total if total > 0 else 0.0
            best_frac = frac % 1.0
            best_seg_idx = i
    return best_pt, best_frac
