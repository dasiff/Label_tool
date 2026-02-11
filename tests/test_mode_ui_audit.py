import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_mode_button_configs_only_in_set_mode():
    """Fail if code directly mutates mode button appearance outside `_set_mode()`.

    This helps ensure `_set_mode()` remains the single place responsible for mode
    visual state and reduces the chance of inconsistent UI updates.
    """
    src = (ROOT / 'scripts' / 'labeling_tool.py').read_text(encoding='utf-8')

    # Find _set_mode block range
    m = re.search(r"def _set_mode\(self, mode: str\):", src)
    assert m, "_set_mode not found in labeling_tool.py"
    start = m.start()
    # Naive end: find next top-level 'def ' after start
    rest = src[start:]
    next_def = re.search(r"\ndef [^_].+\n", rest)
    end = start + next_def.start() if next_def else len(src)

    # All occurrences of mode_buttons.*.config(...)
    occurrences = []
    for mo in re.finditer(r"mode_buttons\[\s*['\"](?P<name>[^'\"]+)['\"]\s*\]\.config\(", src):
        idx = mo.start()
        occurrences.append((mo.group('name'), idx))

    # Any occurrence outside the _set_mode block is flagged
    out_of_place = [name for (name, idx) in occurrences if not (start <= idx <= end)]
    assert out_of_place == [], f"Found mode_buttons.config usages outside _set_mode(): {out_of_place}"
