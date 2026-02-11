import numpy as np
from scripts.labeling_tool import LabelingTool


def test_postprocess_merge_merges_small_sliver():
    lt = LabelingTool()
    # create fake clean_image for color means
    lt.clean_image = np.zeros((50, 80, 3), dtype=np.uint8) + 100
    # make region 1 (left large), region 2 (small sliver in middle), region 3 (right large)
    seg = np.zeros((50,80), dtype=np.int32)
    seg[:, :30] = 1
    seg[:, 30:31] = 2  # thin sliver column
    seg[:, 31:] = 3
    # paint colors: left=100, middle=102, right=100
    lt.clean_image[:, :30] = [100,100,100]
    lt.clean_image[:, 30:31] = [102,102,102]
    lt.clean_image[:, 31:] = [100,100,100]
    lt.segments = seg.copy()
    lt._grad_mag_full = np.zeros((50,80), dtype=float)  # no strong boundaries
    lt.target_segments = 3
    lt._postprocess_merge(target=3, min_size=200, boundary_grad_thresh=5.0)
    # After merge, there should be no tiny region remaining (< min_size)
    unique = np.unique(lt.segments)
    areas = [int((lt.segments == uid).sum()) for uid in unique if uid != 0]
    assert all(a >= 200 for a in areas)
    assert lt.n_segments <= 3


def test_postprocess_merge_respects_strong_boundary():
    lt = LabelingTool()
    lt.clean_image = np.zeros((40, 60, 3), dtype=np.uint8) + 100
    seg = np.zeros((40,60), dtype=np.int32)
    seg[:, :20] = 1
    seg[:, 20:22] = 2  # small area
    seg[:, 22:] = 3
    lt.segments = seg.copy()
    # set strong gradient across left boundary of sliver, so do not merge
    lt._grad_mag_full = np.zeros((40,60), dtype=float)
    lt._grad_mag_full[:, 19:21] = 100.0
    lt.target_segments = 3
    lt._postprocess_merge(target=3, min_size=200, boundary_grad_thresh=10.0)
    unique = np.unique(lt.segments)
    # sliver remains if boundary is strong
    assert 2 in unique
