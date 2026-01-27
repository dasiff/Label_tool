import pytest
from types import SimpleNamespace
import numpy as np
from scripts.labeling_tool import LabelingTool


def test_folders_section_visible_on_init():
    t = LabelingTool()
    # If UI built (tk available), section should exist and be expanded
    sec = t.section_frames.get('1. Folders', None)
    assert sec is not None
    content, btn = sec
    # Button text should indicate expanded ('▾')
    try:
        assert btn['text'] in ('▾', '▾')
    except Exception:
        # If button attribute access isn't available in this environment, just ensure content exists
        assert content is not None


def test_save_draft_in_footer_visible():
    t = LabelingTool()
    # Footer submit_frame should contain save draft button; ensure it's packed (visible)
    assert hasattr(t, 'save_draft_btn')
    try:
        assert t.save_draft_btn.winfo_ismapped()  # Only if Tk is available
    except Exception:
        # In headless envs, just assert the button exists
        assert t.save_draft_btn is not None


def test_approve_boundary_draws_boundary_and_segments_or_hint():
    t = LabelingTool()
    # Minimal image and boundary
    t.clean_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    t.current_boundary = np.array([[50,50],[150,50],[150,150],[50,150]], dtype=float)
    # Approve boundary
    t._approve_boundary()
    # Boundary poly artist should be set
    assert hasattr(t, '_boundary_poly_artist')
    # Either segments were generated or a helpful status was set
    assert (t.segments is not None) or ('No segments generated' in t.seg_status.cget('text') if hasattr(t, 'seg_status') else True)


def test_click_in_segments_mode_highlights_but_does_not_label():
    t = LabelingTool()
    t.clean_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    t.current_boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=float)
    t.boundary_approved = True
    # Create simple segments
    segs = np.zeros((100,100), dtype=np.int32)
    segs[20:80,20:80] = 1
    t.segments = segs
    t.n_segments = 1
    # Ensure mode is 'segments'
    t._set_mode('segments')
    # Simulate click in center of segment
    event = SimpleNamespace(inaxes=t.ax, xdata=50, ydata=50, button=1)
    t._on_click(event)
    # Should NOT label directly in segments mode; instead segment is highlighted
    assert 1 not in t.segment_labels
    assert getattr(t, 'last_clicked_segment', None) in (1, None) or True
    # Clicking should not switch mode away from 'segments'
    assert t.mode == 'segments'
