from scripts.labeling_tool import LabelingTool
import numpy as np
from tkinter import messagebox


def test_unapprove_boundary_allows_edit(monkeypatch):
    lt = LabelingTool()
    # Simulate a previously-saved image state
    lt.original_boundary = np.array([[10,10],[110,10],[110,110],[10,110]], dtype=float)
    lt.current_boundary = lt.original_boundary.copy()
    lt.boundary_approved = True
    lt.segment_labels = {1: 'parking_stalls'}
    # Mock messagebox to always confirm
    monkeypatch.setattr(messagebox, 'askyesno', lambda *a, **k: True)

    # Call unapprove which should set boundary_approved False and prepare adjustment view
    lt._unapprove_boundary()
    assert lt.boundary_approved is False
    # After unapproving, the manual button should say Split Mode (Off) and finalize disabled
    # Split toggle removed: ensure Apply button exists instead
    assert lt.finalize_btn is not None
    assert lt.finalize_btn.cget('state') in ('disabled', 'disabled') or lt.finalize_btn.cget('state') == 'disabled'
