"""Minimal UI adapter for the simplified labeling tool.

Responsibility:
- Create a small Tkinter + Matplotlib based UI (in later work) that calls the
  Controller methods and receives state updates for rendering.
- For now, this module contains only a stubbed `LabelingUI` class with docstrings
  describing the intended interactions.
"""

from typing import Any


class LabelingUI:
    """Thin UI wrapper that forwards events to a Controller instance.

    Intended responsibilities:
    - Build left-side control panel (load, approve, segment, save)
    - Build canvas area (matplotlib) to display images and overlays
    - Convert GUI events into controller calls and subscribe to controller updates
    """

    def __init__(self, controller: Any):
        self.controller = controller

    def build(self) -> None:
        """Build the widgets and bind callbacks."""
        raise NotImplementedError

    def run(self) -> None:
        """Start the UI mainloop (blocking)."""
        raise NotImplementedError


# Small UI helper function extracted for testability
def _update_title_for_mode(self):
    """Update title based on current mode."""
    # Mode selection has priority: ensure title reflects the actively selected mode
    try:
        if getattr(self, 'mode', None) == 'boundary':
            try:
                self.ax.set_title("Adjust boundary (drag corners to reshape, arrows=move, </>=rotate)")
            except Exception:
                pass
        elif getattr(self, 'mode', None) == 'segments':
            if getattr(self, 'splitting_segment_id', None) is not None or len(getattr(self, 'manual_line_points', [])) > 0:
                try:
                    self.ax.set_title("SPLIT MODE: Draw lines (Esc=reselect, Space=new line, Enter=apply)")
                except Exception:
                    pass
            else:
                try:
                    self.ax.set_title("SEGMENT MODE: Label, split, refine segments")
                except Exception:
                    pass
        elif getattr(self, 'mode', None) == 'label':
            try:
                self.ax.set_title("LABEL MODE: Click segment to label (right-click to remove)")
            except Exception:
                pass
        elif getattr(self, 'mode', None) == 'access':
            try:
                self.ax.set_title("ACCESS MODE: Boundary access and road painting (use buttons to enable)")
            except Exception:
                pass
        else:
            # Fallback to boundary-aware title
            if not getattr(self, 'boundary_approved', False):
                try:
                    self.ax.set_title("Adjust boundary (drag corners to reshape, arrows=move, </>=rotate)")
                except Exception:
                    pass
            elif getattr(self, 'splitting_segment_id', None) is not None or len(getattr(self, 'manual_line_points', [])) > 0:
                try:
                    self.ax.set_title("SPLIT MODE: Draw lines (Esc=reselect, Space=new line, Enter=apply)")
                except Exception:
                    pass
            else:
                try:
                    self.ax.set_title("LABEL MODE: Click segment to label (right-click to remove)")
                except Exception:
                    pass
    except Exception:
        pass
    try:
        self.canvas.draw()
    except Exception:
        pass


def _update_boundary_artists(self):
    """Fast update of existing boundary polygon only (vertex handles removed).
    If the polygon artist doesn't exist yet, create it via _draw_editable_boundary().
    """
    # If blit is available, try to blit the polygon update only
    if getattr(self, '_use_blit', False) and getattr(self, '_bg', None) is not None and getattr(self, '_renderer', None) is not None:
        try:
            self.canvas.restore_region(self._bg)
            try:
                self._boundary_poly_artist.set_xy(self.current_boundary)
            except Exception:
                try:
                    self._boundary_poly_artist.remove()
                except Exception:
                    pass
                self._boundary_poly_artist = mpatches.Polygon(self.current_boundary, fill=False, edgecolor='yellow', linewidth=1)
                self.ax.add_patch(self._boundary_poly_artist)
            try:
                self._boundary_poly_artist.draw(self._renderer)
            except Exception:
                pass
            try:
                self.canvas.blit(self.ax.bbox)
            except Exception:
                self.canvas.draw_idle()
            return
        except Exception:
            # Fall through to non-blit update
            pass

    # Non-blit update: update or recreate polygon artist and schedule a coalesced draw
    try:
        self._boundary_poly_artist.set_xy(self.current_boundary)
    except Exception:
        try:
            self._boundary_poly_artist.remove()
        except Exception:
            pass
        self._boundary_poly_artist = mpatches.Polygon(self.current_boundary, fill=False, edgecolor='yellow', linewidth=1)
        self.ax.add_patch(self._boundary_poly_artist)
    # Use coalesced draw to avoid frequent heavy GUI redraws
    try:
        if self._profile_drag:
            t_draw0 = time.time()
        self._schedule_coalesced_draw()
        if self._profile_drag:
            t_draw1 = time.time(); print(f"PROFILE: schedule_draw {t_draw1-t_draw0:.4f}s")
    except Exception:
        if self._profile_drag:
            t_draw0 = time.time()
        self.canvas.draw()
        if self._profile_drag:
            t_draw1 = time.time(); print(f"PROFILE: canvas_draw {t_draw1-t_draw0:.4f}s")
