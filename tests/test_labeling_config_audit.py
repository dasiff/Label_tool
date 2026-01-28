import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_no_direct_config_in_labeling_package():
    """Fail if any `.config(` calls are present inside the `labeling/` package.

    Core modules should not mutate widgets directly; they should use the UI helpers
    exposed by `LabelingTool` (e.g., `set_manual_status`, `set_seg_status`).
    """
    bad = []
    for p in (ROOT / 'labeling').rglob('*.py'):
        txt = p.read_text(encoding='utf-8')
        for m in re.finditer(r"\.config\s*\(", txt):
            # Allow a small whitelist: internal helper modules that are explicitly UI may be excluded
            # but by default flag any `.config(` in labeling/ for manual review.
            line_no = txt[:m.start()].count('\n') + 1
            bad.append((str(p.relative_to(ROOT)), line_no))
    assert not bad, f"Found .config(...) usages in labeling/ package: {bad}"