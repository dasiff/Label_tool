"""Labeling package - lightweight scaffold for a simplified labeling tool.

This package contains the minimal controller, UI, core, render, IO and worker
modules required to implement a second, simpler labeling tool while keeping
`scripts/labeling_tool.py` frozen as a reference.

No implementation here — only minimal classes and docstrings to guide the
refactor and to be expanded later.
"""

__all__ = [
    'controller',
    'ui',
    'core',
    'render',
    'io',
    'workers',
    'config'
]
