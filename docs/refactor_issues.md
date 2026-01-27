# Refactor Issues Log

- Date: 2026-01-26
- Context: During atomic extraction moves, some tests failed that should be logged for follow-up fixes.

Failures observed when moving `_simplify_component_mask` and `_compute_buffer_mask`:

1. tests/test_access_road_labeling.py::test_add_boundary_access_and_save
   - Error: UnboundLocalError in `_submit_annotation` (local variable `class_name` referenced when undefined).

2. tests/test_access_road_labeling.py::test_road_paint_and_constraints
   - Error: UnboundLocalError in `_submit_annotation` (same as above).

3. tests/test_access_road_labeling.py::test_shortcuts_toggle_and_escape
   - Error: AttributeError in `_toggle_manual_mode`: `manual_btn` missing in headless stub.

Notes:
- These are logged for later resolution. Proceeding with planned atomic moves per instruction.

**Additional notes (2026-01-26 2):**

- Validation: `_generate_segments` was exercised on the La Quinta image and produced expected output (Final segments: 6). ✅
- **Failing tests observed (recorded for follow-up):**
  - `tests/test_access_road_labeling.py::test_add_boundary_access_and_save` — UnboundLocalError in `_submit_annotation` (local `class_name` referenced before assignment).
  - `tests/test_access_road_labeling.py::test_road_paint_and_constraints` — UnboundLocalError in `_submit_annotation`.
  - `tests/test_access_road_labeling.py::test_shortcuts_toggle_and_escape` — AttributeError: `manual_btn` missing in headless stub (UI placeholders incomplete).

**Suggested TODOs:**
- Guard variable usage in `_submit_annotation` (ensure `class_name` is defined or provide early return). ⚠️
- Provide minimal UI stubs for tests (`manual_btn`, `finalize_btn`, etc.) or guard accesses in code paths used by headless tests. ⚠️
- Investigate layout/visibility for submit/save controls in the footer so Save Draft is not relied upon.


## UI observations (2026-01-26)

- User reported: *"Access & Roads: no tools appeared; Submit/Save not visible; Save Draft is a bandaid; Border Adjust & Access both highlighted.*"
  - Possible causes: footer/submit frame hidden by layout, floating save button off-screen, collapsed 'Access & Roads' section, or mode state mismatch (both 'boundary' and 'access' appearing highlighted).
  - Status: Needs manual UI investigation and possible fixes to ensure Save is always visible and Access panel elements are not hidden.

**Decision (2026-01-26):** UI refactors postponed during the mechanical extraction phase; a dedicated UI audit will be scheduled after core extractions complete. The failing access/road tests are logged above and will be addressed during that audit.

