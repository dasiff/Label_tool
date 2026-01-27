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

    assert 1 in lt.segment_labels
    assert lt.segment_labels[1] == lt.class_var.get()

    # Right-click should remove label
    lt._on_click(type('E', (), {'inaxes': lt.ax, 'xdata': x, 'ydata': y, 'button': 3}))
    assert 1 not in lt.segment_labels


def test_click_selects_for_split_when_manual_mode_on():
    lt = LabelingTool()
    lt.clean_image = np.ones((100,100,3), dtype=np.uint8)*255
    lt.current_boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=float)
    lt.boundary_approved = True
    segs = np.zeros((100,100), dtype=np.int32)
    segs[20:80, 20:80] = 1
    lt.segments = segs
    lt.n_segments = 1

    lt._set_mode('segments')
    lt.manual_mode = True
    # Click inside to select for split
    x, y = 50, 50
    lt._on_click(type('E', (), {'inaxes': lt.ax, 'xdata': x, 'ydata': y, 'button': 1}))
    assert lt.splitting_segment_id == 1
