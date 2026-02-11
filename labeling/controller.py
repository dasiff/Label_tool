"""Controller (orchestrator) for the simplified labeling tool.

Responsibility:
- Maintain authoritative state: current image, boundary, segments, labels.
- Expose a small API for UI to call (load_image, approve_boundary, generate_segments,
  apply_split, label_segment, save_draft, submit_annotation).
- Coordinate background tasks via workers and call persistence when requested.

This scaffold intentionally contains no logic — only signatures and docstrings
so the module can be imported safely during incremental refactor.
"""

from typing import Any, Dict, Optional


class Controller:
    """High-level orchestrator for labeling workflows.

    Methods should be small, side-effecting and accept/return plain Python
    structures (numpy arrays are fine) so they are testable in headless mode.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # core state
        self.current_image = None
        self.current_boundary = None
        self.segments = None
        self.segment_labels = {}
        self.n_segments = 0

    # UI-facing API (minimal surface)
    def load_image(self, path: str) -> None:
        """Load image at path into controller state."""
        raise NotImplementedError

    def approve_boundary(self) -> None:
        """Mark boundary as approved and generate segments."""
        raise NotImplementedError

    def generate_segments(self, **kwargs) -> None:
        """Generate segments using core.segment module."""
        raise NotImplementedError

    def apply_manual_split(self, polylines) -> None:
        """Apply user-drawn split polylines to current segments."""
        raise NotImplementedError

    def label_segment(self, seg_id: int, class_name: str) -> None:
        """Assign a class to a segment id."""
        raise NotImplementedError

    def save_draft(self, folder: str) -> None:
        """Save a draft (non-destructive)."""
        raise NotImplementedError

    def submit_annotation(self, folder: str) -> None:
        """Finalize and write annotation outputs to disk."""
        raise NotImplementedError

    # Additional helpers for plugins and tests could be added here
