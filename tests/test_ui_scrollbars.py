from scripts.labeling_tool import LabelingTool


def test_left_panel_scrollbars_exist():
    t = LabelingTool()
    # Ensure left canvas and both scrollbars are exposed
    assert hasattr(t, 'left_canvas')
    assert hasattr(t, 'left_v_scrollbar')
    assert hasattr(t, 'left_h_scrollbar')
    # In headless mode they should not be None
    assert t.left_canvas is not None
    assert t.left_v_scrollbar is not None
    assert t.left_h_scrollbar is not None
