# Small runner to execute some test functions without pytest
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tests.test_split_core_helpers import (
    test_vertical_line_splits_rectangle,
    test_closed_polygon_inside_creates_mask,
    test_closed_polygon_too_small_is_rejected,
    test_line_disconnects_rectangle_internal_cut,
    test_line_disconnects_donut_single_cut_doesnt_disconnect,
)

functions = [
    test_vertical_line_splits_rectangle,
    test_closed_polygon_inside_creates_mask,
    test_closed_polygon_too_small_is_rejected,
    test_line_disconnects_rectangle_internal_cut,
    test_line_disconnects_donut_single_cut_doesnt_disconnect,
]

if __name__ == '__main__':
    for f in functions:
        try:
            f()
            print(f"{f.__name__}: PASS")
        except AssertionError as e:
            print(f"{f.__name__}: FAIL -> {e}")
        except Exception as e:
            print(f"{f.__name__}: ERROR -> {e}")
