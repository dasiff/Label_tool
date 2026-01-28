import tkinter as tk
from scripts.labeling_tool import LabelingTool
import numpy as np


def test_label_button_disabled_in_segments_mode():
    lt = LabelingTool()
    # Ensure label button enabled initially
    assert lt.mode_buttons['label'].cget('state') in ('normal', tk.NORMAL)
    # Switch to segments mode
    lt._set_mode('segments')
    # Label button should be disabled while in segments
    assert lt.mode_buttons['label'].cget('state') in ('disabled', tk.DISABLED)
    # Switch to label mode and ensure it's enabled again
    lt._set_mode('label')
    assert lt.mode_buttons['label'].cget('state') in ('normal', tk.NORMAL)


def test_switching_to_label_mode_shows_full_display():
    lt = LabelingTool()
    # Prepare a fake clean image and segments to simulate realistic state
    lt.clean_image = np.full((50, 50, 3), 120, dtype=np.uint8)  # mid-gray
    lt.enhanced_image = None  # force regeneration in update
    lt.segments = np.zeros((50, 50), dtype=np.int32)
    lt.segments[10:30, 10:30] = 1
    lt.segments[30:45, 30:45] = 2
    lt.boundary_approved = True

    # Simulate being in Segments mode with a highlighted segment
    lt._update_display_with_highlight(1)
    # Axis stub in headless tests exposes _last_img; when running with real Matplotlib
    # extract the image from the axis artists instead.
    if hasattr(lt.ax, '_last_img') and lt.ax._last_img is not None:
        img_before = lt.ax._last_img.copy()
    else:
        img_before = np.array(lt.ax.images[0].get_array())

    # Switch to Label mode — this should refresh to the full segments display
    lt._set_mode('label')
    if hasattr(lt.ax, '_last_img') and lt.ax._last_img is not None:
        img_after = lt.ax._last_img
    else:
        img_after = np.array(lt.ax.images[0].get_array())

    assert img_after is not None
    # Highlighted view is dimmed; full display should be brighter on average
    assert img_after.mean() > img_before.mean()
