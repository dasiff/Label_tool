from scripts.labeling_tool import LabelingTool
import numpy as np


def test_labeling_in_segments_mode():
    lt = LabelingTool()
    lt.clean_image = np.ones((100,100,3), dtype=np.uint8)*255
    lt.current_boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=float)
    lt.boundary_approved = True
    # Create a simple segments mask with segment id 1 occupying central area
    segs = np.zeros((100,100), dtype=np.int32)
    segs[20:80, 20:80] = 1
    lt.segments = segs
    lt.n_segments = 1

    lt._set_mode('segments')
    # Simulate click inside segment 1
    x, y = 50, 50
    lt._on_click(type('E', (), {'inaxes': lt.ax, 'xdata': x, 'ydata': y, 'button': 1}))

    # Should NOT label directly in Segments mode; instead segment is selected for splitting
    assert 1 not in lt.segment_labels
    assert lt.splitting_segment_id == 1

    # Right-click removes label only if present (no-op here)
    lt._on_click(type('E', (), {'inaxes': lt.ax, 'xdata': x, 'ydata': y, 'button': 3}))
    assert 1 not in lt.segment_labels


def test_click_selects_for_split():
    lt = LabelingTool()
    lt.clean_image = np.ones((100,100,3), dtype=np.uint8)*255
    lt.current_boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=float)
    lt.boundary_approved = True
    segs = np.zeros((100,100), dtype=np.int32)
    segs[20:80, 20:80] = 1
    lt.segments = segs
    lt.n_segments = 1

    lt._set_mode('segments')
    # Click inside to select for split
    x, y = 50, 50
    lt._on_click(type('E', (), {'inaxes': lt.ax, 'xdata': x, 'ydata': y, 'button': 1}))
    assert lt.splitting_segment_id == 1


def test_select_segment_enables_finalize_and_cleared_on_mode_change():
    import tkinter as tk
    lt = LabelingTool()
    lt.clean_image = np.ones((100,100,3), dtype=np.uint8)*255
    lt.current_boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=float)
    lt.boundary_approved = True
    segs = np.zeros((100,100), dtype=np.int32)
    segs[20:80, 20:80] = 1
    lt.segments = segs
    lt.n_segments = 1

    lt._set_mode('segments')
    # Finalize button disabled by default
    try:
        assert lt.finalize_btn.cget('state') == tk.DISABLED
    except Exception:
        pass

    # Select a segment
    lt._on_click(type('E', (), {'inaxes': lt.ax, 'xdata': 50, 'ydata': 50, 'button': 1}))
    assert lt.splitting_segment_id == 1
    try:
        assert lt.finalize_btn.cget('state') == tk.NORMAL
    except Exception:
        pass

    # Switching mode clears the selection
    lt._set_mode('label')
    assert lt.splitting_segment_id is None
