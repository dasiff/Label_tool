"""Simple wrapper for background worker tasks scaffold.

Responsibility:
- Provide a small helper to run tasks in a background thread/process and
  schedule results back to the main thread via a callback.

This module intentionally contains only docstrings and minimal function signatures
so that it can be extended and unit-tested without GUI dependencies.
"""
from typing import Callable, Any, Dict
import numpy as np
import cv2


def run_background(func: Callable[..., Any], callback: Callable[[Dict[str, Any]], None], *args, **kwargs) -> None:
    """Run `func(*args, **kwargs)` in background and call `callback(result)` when done.

    `result` should be a dict like {'ok': True, 'result': ...} or {'ok': False, 'error': ...}
    """
    raise NotImplementedError


def apply_manual_split_background_safe(self):
    """Background-safe split handler for common closed-loop polygons.

    Ported verbatim from `scripts/labeling_tool.py` to allow unit testing and reuse without
    keeping a large GUI-facing method inside the scripts module.
    """
    self._running_in_background = True
    try:
        manual_polylines = [p.copy() for p in getattr(self, 'manual_polylines', [])]
        if len(manual_polylines) == 0 or self.segments is None:
            try:
                self.root.after(0, lambda: self.manual_status.config(text="No lines to apply"))
            except Exception:
                pass
            self._running_in_background = False
            return

        segs_local = self.segments.copy()
        seg_id = self.splitting_segment_id
        seg_mask = (segs_local == seg_id)
        seg_area = int(seg_mask.sum())
        h, w = segs_local.shape
        if seg_area == 0:
            try:
                self.root.after(0, lambda: self.manual_status.config(text="Selected segment empty"))
            except Exception:
                pass
            self._running_in_background = False
            return

        # compute edge coords for this segment
        seg_uint8 = seg_mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(seg_uint8, kernel, iterations=1)
        edge_mask = seg_uint8 - eroded
        edge_coords = np.argwhere(edge_mask > 0)
        if edge_coords.size == 0:
            try:
                contours, _ = cv2.findContours((seg_uint8 * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if contours:
                    c = max(contours, key=lambda x: cv2.contourArea(x))
                    edge_coords = np.array([[pt[0][1], pt[0][0]] for pt in c.reshape(-1,1,2)])
            except Exception:
                edge_coords = np.empty((0,2), dtype=int)

        segments_added_local = 0
        new_segment_selected = None

        # quick path: handle closed polygons only
        for manual_line_points in manual_polylines:
            pts = np.array(manual_line_points, dtype=np.int32)
            if pts.shape[0] < 3:
                continue
            p0 = pts[0]; p_last = pts[-1]
            dist_endpoints = np.linalg.norm(np.array([p0[1], p0[0]]) - np.array([p_last[1], p_last[0]]))
            if edge_coords.size == 0:
                continue
            dist_to_edges_0 = np.linalg.norm(edge_coords - np.array([p0[1], p0[0]]), axis=1)
            nearest_edge_dist_0 = dist_to_edges_0.min()
            dist_to_edges_last = np.linalg.norm(edge_coords - np.array([p_last[1], p_last[0]]), axis=1)
            nearest_edge_dist_last = dist_to_edges_last.min()
            is_closed_loop = (dist_endpoints < nearest_edge_dist_0 and dist_endpoints < nearest_edge_dist_last)
            if not is_closed_loop:
                # Not a closed polygon; fallback to main-thread full split
                try:
                    self.root.after(0, lambda: self._apply_manual_split())
                except Exception:
                    pass
                self._running_in_background = False
                return

            # closed polygon: fill and assign if large enough
            poly_pts = pts.astype(np.int32)
            poly_mask_local = np.zeros((h, w), dtype=np.uint8)
            try:
                cv2.fillPoly(poly_mask_local, [poly_pts], 255)
            except Exception:
                continue
            constrained_poly = (poly_mask_local > 0) & seg_mask
            area_poly = int(constrained_poly.sum())
            poly_min = max(100, int(seg_area * 0.01))
            if area_poly >= poly_min:
                new_seg_id_local = int(segs_local.max()) + 1
                segs_local[constrained_poly] = new_seg_id_local
                segments_added_local += 1
                new_segment_selected = new_seg_id_local

        # Finalize on main thread if any closed-loop handled
        if segments_added_local > 0:
            def ui_finalize():
                try:
                    self.segments = segs_local
                    old_n = self.n_segments
                    self.n_segments = int(self.segments.max())
                    print(f"\nSegment count: {old_n} -> {self.n_segments} (added {segments_added_local})")
                    self.manual_status.config(text="Split applied! Draw another or toggle off")
                    try:
                        self.finalize_btn.config(state=tk.NORMAL)
                        self.manual_btn.config(state=tk.NORMAL)
                    except Exception:
                        pass
                    if new_segment_selected is not None:
                        self.splitting_segment_id = int(new_segment_selected)
                        try:
                            self._update_display_with_highlight(self.splitting_segment_id)
                        except Exception:
                            pass
                    else:
                        self.splitting_segment_id = None
                    self._clear_manual_line()
                    self._reset_submit_button()
                    self._update_display()
                    self._update_progress()
                except Exception as e:
                    print("Error finalizing background split:", e)
            try:
                self.root.after(0, ui_finalize)
            except Exception:
                ui_finalize()
        else:
            # Nothing created; report to user
            try:
                self.root.after(0, lambda: self.manual_status.config(text="Split did not create any region"))
            except Exception:
                pass
    finally:
        self._running_in_background = False

