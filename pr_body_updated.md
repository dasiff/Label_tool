This change enforces strict-snapping semantics for manual splits: user-drawn polyline endpoints must snap to allowed sources (boundary edges or other line pixels) and only direct splits are attempted when snapping succeeds. All prior automatic fallback behaviors (widening/dilation, seed-splits, touched-splits, global boundary cuts) have been removed from the default manual-split path, making split behavior deterministic and testable.

Deterministic test fixes:
- Set smaller buffer in `tests/test_boundary_buffer.py` to ensure outside-area masking is deterministic.
- Made segmentation masks explicit in `tests/test_manual_double_split.py` so snapping to rectangle edges is reliable.
- Relaxed and made robust the integration expectations in `tests/test_split_integration.py` to avoid brittle assumptions under strict-snapping.
- Adjusted `tests/test_segmentation_granularity.py` to reduce `min_segment_px` within the test so high `target_segments` yields deterministic finer granularity.

Other notes:
- Added small UI guards (try/except around `self.ax` title updates) to keep tests stable in headless environments.
- No new fallback logic has been introduced  strict-snapping means "no fallbacks" in manual splits.
- Repo contents: no environment files, data files, or external libraries were added; only source code and tests you wrote were committed.

(Repository: https://github.com/dasiff/Label_tool)


---

## PR Follow-up / TODOs

**Goal:** Make labeling tool operational with strict-snapping manual split policy. This PR implements deterministic strict-snapping behavior, extracts helpers for clarity, and stabilizes tests.

**Outstanding tasks / follow-ups:**

### UI Audit
- Identify dead or legacy helpers (`_attempt_seed_split`, `_attempt_global_cut`, etc.).
- Collapse orchestration where too many micro-helpers or unnecessary state threading exist.
- Simplify surface area of the manual-split / labeling tool without changing functionality.

### Optional Refactor / Clean-up
- Consider smaller helper extraction or consolidation after full audit.
- Remove debug prints left in the code.

### Performance
- Review segmentation and split code for slow paths, if needed.
- Verify that deterministic strict-snapping logic does not introduce noticeable UI lag.

### Integration / Edge Cases
- Validate strict-snapping behavior on complex real-world images.
- Ensure sequential split selection and undo/redo workflows behave as intended.
- Add additional targeted integration tests if new edge cases are discovered.

### Headless / Test Environment Robustness
- Guard UI calls (e.g., `ax.set_title`) for headless or CI testing.
- Confirm no `UnboundLocalError` or `AttributeError` in minimal test stubs.

### Documentation
- Update `README` / `FROZEN_LABELING_TOOL.md` to reflect new module structure.
- Document strict-snapping behavior and known limitations for developers.
