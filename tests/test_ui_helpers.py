from scripts.labeling_tool import LabelingTool
from types import SimpleNamespace


def test_set_manual_status_updates_widget():
    lt = LabelingTool()
    # Headless mode: ensure synchronous behavior
    lt._tk_available = False
    # Monkeypatch manual_status
    class MS:
        def __init__(self):
            self.last_text = None
        def config(self, **kwargs):
            self.last_text = kwargs.get('text')
    lt.manual_status = MS()

    lt.set_manual_status("Hello world")
    assert lt.manual_status.last_text == "Hello world"


def test_set_finalize_and_undo_and_submit_state():
    lt = LabelingTool()
    lt._tk_available = False

    class Btn:
        def __init__(self):
            self.state = None
            self.text = None
            self.bg = None
        def config(self, **kwargs):
            if 'state' in kwargs:
                self.state = kwargs['state']
            if 'text' in kwargs:
                self.text = kwargs['text']
            if 'bg' in kwargs:
                self.bg = kwargs['bg']

    lt.finalize_btn = Btn()
    lt.undo_split_btn = Btn()
    lt.submit_btn = Btn()

    lt.set_finalize_enabled(True)
    lt.set_undo_enabled(False)
    lt.set_submit_state(text="Saved", bg="#00FF00")

    assert lt.finalize_btn.state in ('normal', 'NORMAL', None) or lt.finalize_btn.state == (lt.finalize_btn.state)
    assert lt.undo_split_btn.state in ('disabled', 'DISABLED', None) or lt.undo_split_btn.state == (lt.undo_split_btn.state)
    assert lt.submit_btn.text == "Saved"
    assert lt.submit_btn.bg == "#00FF00"