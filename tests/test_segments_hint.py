from scripts.labeling_tool import LabelingTool


def test_segments_hint_present():
    lt = LabelingTool()
    assert hasattr(lt, 'segments_hint_label')
    text = lt.segments_hint_label.cget('text')
    assert 'left-click' in text.lower() and 'split mode' in text.lower()
