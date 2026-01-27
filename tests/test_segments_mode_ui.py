import tkinter as tk
from scripts.labeling_tool import LabelingTool


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
