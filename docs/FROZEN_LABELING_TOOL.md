FROZEN LABELING TOOL

This repository snapshot freezes the file `scripts/labeling_tool.py` as the reference behavior for the labeling tool.

Details:
- Tag: `freeze/labeling_tool_v1`
- Branch: `frozen/labeling_tool_v1`
- Commit: `1a8c5c0` (see tag for full metadata)

Status:
- `scripts/labeling_tool.py` has been committed and the file attribute set to read-only on the filesystem.

Policy:
- Do not modify `scripts/labeling_tool.py` directly. Create a new simplified tool in a separate file (e.g., `scripts/labeling_tool_simple.py`).
- This frozen file serves as the authoritative reference for behavior and tests.
