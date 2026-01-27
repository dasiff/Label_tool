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

**Manual split failures (2026-01-26 3):**

- `tests/test_manual_double_split.py::test_sequential_splits_keep_selection`
  - Symptom: A manual split produced a combined line mask ("Combined line mask pixels (within segment): 212") but did **not** create additional segments — `n_segments` remained 1 (AssertionError).
  - Notes: Printed debug shows the line is created but removing the line did not yield >= 2 connected components. Possible causes: the constrained `line_mask` did not sufficiently clear pixels in `seg_mask`, widening attempts failed to produce multiple regions, or fallback seed-split logic was not triggered (background/main-thread path mismatch).
  - Suggested debugging actions:
    - Add debug prints for `num_regions`, sizes of labeled regions, `line_mask.sum()` at each stage, and each widen attempt result.
    - Add a focused unit test using a simple synthetic rectangle image that reproduces the sequence (draw vertical line, expect two regions) to isolate the geometry logic.
    - Verify the `ndimage.label` results and ensure the code path that schedules a main-thread fallback is reachable/testable.

- `tests/test_manual_double_split.py::test_closed_loop_creates_inner_segment`
  - Symptom: `AttributeError` in `labeling/render.py::_draw_editable_boundary`: `'LabelingTool' object has no attribute 'ax'` (triggered from `_update_display`).
  - Notes: Headless tests currently run without a minimal Matplotlib `ax`/`canvas` setup; rendering helpers assume these exist.
  - Suggested fixes:
    - Provide minimal headless stubs for `ax`, `canvas`, `_boundary_poly_artist`, and `access_artists` during test setup, or
    - Guard rendering helpers to short-circuit safely when `ax`/`canvas` are not present (and add tests to assert no exception).

**Next steps (short-term):**
- Add minimal headless rendering scaffolding so `render` helpers and `_draw_editable_boundary` do not raise in tests. ✅
- Add extra debug logging in `apply_manual_split` for region counts and widen attempts. ✅
- Add focused unit test reproducing splitting behavior and asserting `n_segments >= 2`. ✅

**Priority:** High for headless stubs and split debugging (blocking tests). UI polish deferred to the UI audit as previously decided.

---

**Record stored (2026-01-27 00:00 UTC):** Manual-split failures and UI AttributeError have been recorded here for follow-up during the refactor/debug phase. No further changes applied now; will address in the planned UI/split audit when you say go.



