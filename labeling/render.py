"""Rendering helpers for the simplified labeling tool.

Responsibility:
- Produce annotated images and simple Matplotlib artists for UI display.
- Keep functions small and testable; do not perform I/O here (return arrays/artists).

Inputs: clean_image, segments, segment_labels, boundary polygon, options
Outputs: image array for display, list of artists (or rendering metadata)
"""
from typing import Any, Dict
import matplotlib.patches as mpatches
import numpy as np
import cv2
from labeling.config import CLASS_COLORS


def render_annotated_image(clean_image: Any, segments: Any, segment_labels: Dict[int, str], boundary: Any, **opts) -> Dict[str, Any]:
    """Return a dict with keys like 'image' (BGR ndarray), 'artists' (optional metadata)."""
    raise NotImplementedError


def _draw_editable_boundary(self):
    """Draw boundary polygon (vertex dragging disabled)."""
    if self.current_boundary is None:
        return

    # Don't try to remove artists - just clear the list
    # The ax.clear() in redraw will handle removing them
    self.boundary_vertices_artists.clear()

    # Draw only the polygon (vertex dragging disabled)
    poly = mpatches.Polygon(self.current_boundary, fill=False, edgecolor='yellow', linewidth=1, joinstyle='round')
    # Remove any previous polygon artist if present
    try:
        if getattr(self, '_boundary_poly_artist', None) is not None:
            try:
                self._boundary_poly_artist.remove()
            except Exception:
                pass
    except Exception:
        pass
    self._boundary_poly_artist = poly
    self.ax.add_patch(poly)
    # Avoid drawing access overlays while user is actively dragging the whole boundary
    if not getattr(self, 'dragging_boundary', False):
        self._draw_access_segments()


def _draw_access_segments(self):
    """Draw access segments on the current axis (used in both edit and display modes)."""
    if self.current_boundary is None:
        return
    if not hasattr(self, 'access_artists'):
        self.access_artists = []
    # Remove old artists
    for a in getattr(self, 'access_artists', []):
        try:
            a.remove()
        except:
            pass
    self.access_artists = []
    pts = self.current_boundary
    # Build cumulative lengths
    seg_starts = pts
    seg_ends = np.vstack([pts[1:], pts[0]])
    seg_vecs = seg_ends - seg_starts
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    def frac_to_point(frac):
        f = (frac % 1.0) * total
        # find segment containing f
        idx = np.searchsorted(cum, f, side='right') - 1
        idx = max(0, min(idx, len(seg_lens)-1))
        local_f = (f - cum[idx]) / (seg_lens[idx] if seg_lens[idx]>0 else 1e-6)
        pt = seg_starts[idx] + seg_vecs[idx] * local_f
        return pt
    for seg in self.boundary_access_segments:
        s, e = seg['start_frac'], seg['end_frac']
        # Skip full default 'no_access' to avoid a thick gold overlay for the entire boundary
        if seg.get('label') == 'no_access':
            # Compute segment length properly
            if e >= s:
                seg_len = e - s
            else:
                seg_len = (1.0 - s) + e
            if seg_len >= 0.99:
                continue
        # sample along shorter arc
        samples = np.linspace(s, e, num=50) if s <= e else np.linspace(s, e+1, num=50)
        pts_samples = np.array([frac_to_point(ss % 1.0) for ss in samples])
        if seg.get('label') == 'access_allowed':
            color = '#00FF00'
            lw = 3
            style = {'linewidth': lw, 'solid_capstyle': 'round'}
        else:
            # Subtle rendering for explicit 'no_access' segments (non-full)
            color = '#FFD700'
            lw = 1.0
            style = {'linewidth': lw, 'linestyle': '--'}
        artist, = self.ax.plot(pts_samples[:,0], pts_samples[:,1], color=color, **style)
        self.access_artists.append(artist)
    # Draw provisional first-click point
    if self.access_click_start is not None:
        pt = self.access_click_start['pt']
        artist = self.ax.scatter([pt[0]], [pt[1]], c='white', s=60, edgecolor='black', zorder=20)
        self.access_artists.append(artist)
    self.canvas.draw()


def _update_display(self):
    """Update the display with enhanced contrast for better visibility."""
    if self.segments is None:
        return

    # Create enhanced version if not cached
    if self.enhanced_image is None:
        lab = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        self.enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    rgb = self.enhanced_image

    h, w = rgb.shape[:2]

    # Start with original image (keep colors vivid!)
    display = rgb.copy().astype(float) / 255

    # Fill labeled segments with semi-transparent class color
    alpha = 0.4  # Blend factor
    print(f"DEBUG _update_display: segment_labels = {self.segment_labels}")
    for seg_id, class_name in self.segment_labels.items():
        print(f"DEBUG: Rendering segment {seg_id} as {class_name}")
        seg_mask = self.segments == seg_id
        pixels_in_seg = seg_mask.sum()
        print(f"DEBUG: Segment {seg_id} has {pixels_in_seg} pixels")
        color = CLASS_COLORS.get(class_name, [0.9, 0.9, 0.9])
        # Blend per-channel to avoid boolean-mask broadcasting issues
        for ch in range(3):
            chan = display[:, :, ch]
            chan[seg_mask] = chan[seg_mask] * (1 - alpha) + color[ch] * alpha
            display[:, :, ch] = chan

    # Dim area outside ROI+buffer to make ROI clearer (if boundary exists)
    try:
        if getattr(self, 'current_boundary', None) is not None:
            self._compute_buffer_mask()
            h, w = display.shape[:2]
            roi_mask_full = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(roi_mask_full, [self.current_boundary.astype(np.int32)], 255)
            roi_mask_bool = roi_mask_full > 0
            buffer_mask_full = self.buffer_mask.astype(bool) if getattr(self, 'buffer_mask', None) is not None else np.zeros((h, w), dtype=bool)
            roi_expanded = roi_mask_bool | buffer_mask_full
            # Dim outside the expanded ROI
            dim_mask = ~roi_expanded
            if dim_mask.any():
                mask3 = np.stack([dim_mask] * 3, axis=2)
                display[mask3] = display[mask3] * 0.45
    except Exception:
        pass

    # Draw manual split lines
    try:
        if hasattr(self, 'manual_polylines'):
            for line in self.manual_polylines:
                if len(line) >= 2:
                    pts = np.array(line, dtype=np.int32)
                    cv2.polylines(display, [pts], isClosed=False, color=(1.0, 0, 0), thickness=2)
    except Exception:
        pass

    # Draw segment boundaries (thin overlay) to make segmentation edges visible even when unlabeled
    try:
        if getattr(self, 'segments', None) is not None:
            seg_pad = np.pad(self.segments, 1, mode='constant', constant_values=0)
            edge_h = (seg_pad[:-2, 1:-1] != seg_pad[2:, 1:-1])
            edge_v = (seg_pad[1:-1, :-2] != seg_pad[1:-1, 2:])
            edges = edge_h | edge_v
            if edges.any():
                # Blend yellow onto edge pixels to make them visible
                edge_idx = np.where(edges)
                for ch in range(3):
                    c = display[:, :, ch]
                    c[edge_idx] = c[edge_idx] * 0.2 + (1.0 if ch in (0,1) else 0.0) * 0.8
                    display[:, :, ch] = c
    except Exception:
        pass

    # Draw highlighted segment overlay on top
    try:
        if getattr(self, 'last_clicked_segment', None) is not None:
            highlight_mask = (self.segments == self.last_clicked_segment)
            if highlight_mask.sum() > 0:
                color = [1.0, 1.0, 1.0]
                mask3 = np.stack([highlight_mask] * 3, axis=2)
                alpha_h = 0.35
                display[mask3] = display[mask3] * (1 - alpha_h) + np.array(color) * alpha_h
    except Exception:
        pass

    # Convert back to uint8 RGB for display with matplotlib
    display_img = (np.clip(display, 0, 1) * 255).astype(np.uint8)

    # Replace matplotlib image on the axis
    self.ax.clear()
    self.ax.imshow(display_img)
    self.ax.axis('off')
    self.canvas.draw()


def _update_display_with_highlight(self, highlight_seg_id):
    """Update display with a specific segment highlighted."""
    if self.segments is None:
        return
    
    # Create enhanced version if not cached
    if self.enhanced_image is None:
        lab = cv2.cvtColor(self.clean_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        self.enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    rgb = self.enhanced_image
    
    h, w = rgb.shape[:2]
    
    # Start with dimmed image
    display = rgb.copy().astype(float) / 255 * 0.5  # Dim everything
    
    # Highlight selected segment at full brightness
    highlight_mask = self.segments == highlight_seg_id
    display[highlight_mask] = (rgb.astype(float) / 255)[highlight_mask]  # Full brightness
    
    # Add bright border around highlighted segment
    seg_pad = np.pad(self.segments, 1, mode='constant', constant_values=0)
    edge_h = (seg_pad[:-2, 1:-1] == highlight_seg_id) != (seg_pad[2:, 1:-1] == highlight_seg_id)
    edge_v = (seg_pad[1:-1, :-2] == highlight_seg_id) != (seg_pad[1:-1, 2:] == highlight_seg_id)
    edges = edge_h | edge_v
    display[edges] = [1, 1, 0]  # Yellow border
    
    self.ax.clear()
    self.ax.imshow(display, extent=[0, w, h, 0], aspect='equal')
    
    # Add boundary
    if self.current_boundary is not None:
        poly = mpatches.Polygon(self.current_boundary, fill=False,
                               edgecolor='yellow', linewidth=1.5, joinstyle='round')
        self.ax.add_patch(poly)
    
    # Draw all manual polylines
    # Draw completed polylines in green
    for polyline in self.manual_polylines:
        points = np.array(polyline)
        artist, = self.ax.plot(points[:, 0], points[:, 1], 'g-', linewidth=2, marker='o', markersize=4)
        self.manual_line_artists.append(artist)
    
    # Draw current polyline being drawn in red
    if len(self.manual_line_points) > 0:
        points = np.array(self.manual_line_points)
        if len(points) == 1:
            artist, = self.ax.plot(points[:, 0], points[:, 1], 'ro', markersize=5)
        else:
            artist, = self.ax.plot(points[:, 0], points[:, 1], 'r-', linewidth=2, marker='o', markersize=4)
        self.manual_line_artists.append(artist)

    # Set title
    if self.boundary_access_mode or self.mode == 'access':
        self.ax.set_title("BOUNDARY ACCESS MODE: Click two points on boundary to add access segment")
    elif getattr(self, 'splitting_segment_id', None) is not None or len(self.manual_line_points) > 0:
        self.ax.set_title("SPLIT MODE: Draw lines (Esc=reselect, Space=new line, Enter=apply)")
    else:
        self.ax.set_title("LABEL MODE: Click segment to label (right-click to remove)")

    self.ax.axis('off')
    self.canvas.draw()
