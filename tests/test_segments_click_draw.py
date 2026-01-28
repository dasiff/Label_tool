import numpy as np
from types import SimpleNamespace
from scripts.labeling_tool import LabelingTool


def test_select_segment_and_draw_points():
    t = LabelingTool()
    # Minimal image and approved boundary
    t.clean_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    t.current_boundary = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float)
    t.boundary_approved = True

    # Create a single segment region (id=1)
    segs = np.zeros((100, 100), dtype=np.int32)
    segs[15:60, 15:60] = 1
    t.segments = segs
    t.n_segments = 1

    # Ensure we're in segments mode
    t._set_mode('segments')

    # Click inside the segment to select it (and start the line)
    evt1 = SimpleNamespace(inaxes=t.ax, xdata=20.0, ydata=20.0, button=1)
    t._on_click(evt1)
    assert t.splitting_segment_id == 1, "Clicking a segment should select it for splitting"
    assert len(t.manual_line_points) == 1 and t.manual_line_points[0] == (20, 20), "Selecting a segment should start the manual line with the clicked point"

    # Click again (different point) to add a second point to the manual line
    evt2 = SimpleNamespace(inaxes=t.ax, xdata=30.0, ydata=30.0, button=1)
    t._on_click(evt2)
    assert len(t.manual_line_points) == 2 and t.manual_line_points[-1] == (30, 30), "Second left-click should append a second point to manual_line_points"


def test_click_far_outside_selects_other_segment_and_not_add_point():
    t = LabelingTool()
    t.clean_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    t.current_boundary = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float)
    t.boundary_approved = True

    # Two segments separated by gap
    segs = np.zeros((100, 100), dtype=np.int32)
    segs[15:50, 15:50] = 1
    segs[60:85, 60:85] = 2
    t.segments = segs
    t.n_segments = 2

    t._set_mode('segments')

    # Select segment 1
    t._on_click(SimpleNamespace(inaxes=t.ax, xdata=20.0, ydata=20.0, button=1))
    assert t.splitting_segment_id == 1

    # Click well inside segment 2 (far from 1) -> should switch selection to 2 and not append point
    t._on_click(SimpleNamespace(inaxes=t.ax, xdata=70.0, ydata=70.0, button=1))
    assert t.splitting_segment_id == 2
    assert len(t.manual_line_points) == 0


def test_click_near_selected_segment_appends_point_even_if_background():
    t = LabelingTool()
    t.clean_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    t.current_boundary = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float)
    t.boundary_approved = True

    segs = np.zeros((100, 100), dtype=np.int32)
    segs[20:60, 20:60] = 1
    t.segments = segs
    t.n_segments = 1

    t._set_mode('segments')
    t._on_click(SimpleNamespace(inaxes=t.ax, xdata=25.0, ydata=25.0, button=1))
    assert t.splitting_segment_id == 1

    # Click just outside the segment boundary but within proximity threshold (x=61 is one pixel outside if slice is exclusive)
    # Use a point just outside: (60, 40) - should be counted as 'near' and append
    t._on_click(SimpleNamespace(inaxes=t.ax, xdata=60.0, ydata=40.0, button=1))
    assert len(t.manual_line_points) == 1, "Click near selected segment should append a split point"


def test_press_enter_after_drawing_finishes_and_splits_segment():
    t = LabelingTool()
    t.clean_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    t.current_boundary = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float)
    t.boundary_approved = True

    # Create a single segment region (id=1)
    segs = np.zeros((100, 100), dtype=np.int32)
    segs[20:60, 20:60] = 1
    t.segments = segs
    t.n_segments = 1

    t._set_mode('segments')

    # Select segment and add two points to make a short line that bisects
    t._on_click(SimpleNamespace(inaxes=t.ax, xdata=30.0, ydata=30.0, button=1))
    t._on_click(SimpleNamespace(inaxes=t.ax, xdata=45.0, ydata=45.0, button=1))

    # Press Enter to finalize (run synchronously in tests)
    t._tk_available = False
    t._on_key_press(SimpleNamespace(key='enter'))

    # After finalize, either segments increased or a message indicates no split
    assert t.n_segments >= 1
    # If split applied, n_segments should be > 1
    assert t.n_segments > 1, "Pressing Enter after drawing two points should split the selected segment"


def test_segments_mode_does_not_show_class_section():
    from pathlib import Path
    src = Path(__file__).parent.parent.joinpath('scripts', 'labeling_tool.py').read_text(encoding='utf-8')
    # Find the segments branch
    start = src.find("elif mode == 'segments':")
    assert start != -1
    rest = src[start:]
    # Find end of this branch (next top-level elif/else/if at same indentation)
    end_rel = None
    for token in ['elif mode', 'elif ', '\n    else:', '\n        else:']:
        m = rest.find(token, 1)
        if m != -1:
            if end_rel is None or m < end_rel:
                end_rel = m
    end = start + (end_rel if end_rel is not None else len(rest))
    segment_block = src[start:end]
    assert "Class (click to label" not in segment_block, "Class selection should not be shown in 'segments' mode"
