"""Atomic persistence helpers (drafts & submission) scaffold.

Responsibility:
- Provide functions like `save_draft(state, path)` and `submit_annotation(state, out_dir)`
  that perform atomic writes and return structured status objects.
Inputs: state dict (segments, labels, boundary, metadata), paths
Outputs: dict with status and any error messages
"""
from typing import Any, Dict


def save_draft(state: Dict[str, Any], draft_dir: str) -> Dict[str, Any]:
    """Save a draft atomically to draft_dir.

    Returns {'ok': True, 'files': [...]} or {'ok': False, 'error': '...'}.
    """
    raise NotImplementedError


def submit_annotation(state: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Write final annotation outputs (json, npy, png, masks) to out_dir."""
    raise NotImplementedError
