"""Minimal labeling tool that uses the new `labeling` package scaffold.

This script is a lightweight entrypoint to exercise the controller/UI design
without copying or modifying `scripts/labeling_tool.py` (which is frozen as a
reference). The module intentionally contains only a high-level scaffold that
can be iteratively filled in.
"""

from typing import Any

# Import the new labeling package components (scaffolds)
try:
    from labeling.controller import Controller
    from labeling.ui import LabelingUI
except Exception:
    # If scaffolds aren't fully implemented yet, keep imports optional for now
    Controller = None
    LabelingUI = None


def main():
    """Create controller and UI and run minimal app (placeholder)."""
    if Controller is None or LabelingUI is None:
        print("Labeling scaffolds are present. Implement Controller and UI to run the simplified tool.")
        return

    ctrl = Controller()
    ui = LabelingUI(ctrl)
    ui.build()
    ui.run()


if __name__ == '__main__':
    main()
