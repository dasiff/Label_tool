import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = (ROOT / 'scripts' / 'labeling_tool.py').read_text()

# Find all top-level functions and their character ranges
func_pattern = re.compile(r"^def\s+(?P<name>\w+)\s*\(|^\s+def\s+(?P<iname>\w+)\s*\(", re.M)
matches = list(func_pattern.finditer(SRC))
func_ranges = []
for i, m in enumerate(matches):
    name = m.group('name') or m.group('iname')
    start = m.start()
    end = matches[i+1].start() if i+1 < len(matches) else len(SRC)
    func_ranges.append((name, start, end))

allowed_funcs = {'_build_ui', '_set_mode', '_toggle_section'}

# Find .pack( and .pack_forget( occurrences
pack_usages = [(m.start(), m.group(0)) for m in re.finditer(r"\.pack\s*\(|\.pack_forget\s*\(", SRC)]

out_of_place = []
for idx, snippet in pack_usages:
    # locate the function that contains this index
    containing = None
    for name, s, e in func_ranges:
        if s <= idx < e:
            containing = name
            break
    if containing is None:
        # top-level pack usage — flag as out-of-place
        out_of_place.append((idx, snippet, '<module>'))
    elif containing not in allowed_funcs:
        out_of_place.append((idx, snippet, containing))


def test_pack_usages_only_in_allowed_functions():
    assert not out_of_place, f"Found .pack/.pack_forget used outside {_build_ui, _set_mode, _toggle_section}: {out_of_place}"
