import json
from pathlib import Path
import numpy as np

from scripts.labeling_tool import LabelingTool


def test_interior_mask_mapping(tmp_path):
    t = LabelingTool()
    try:
        t.root.withdraw()
    except Exception:
        pass
    t.clean_image = np.ones((100,100,3), dtype=np.uint8)*255
    t.current_boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=float)
    t.boundary_approved = True
    t.segments = np.ones((100,100), dtype=np.int32)
    t.n_segments = 1
    # Label segment 1 as building_roof
    t.segment_labels = {1: 'building_roof'}
    t.image_files = [Path('img.png')]
    t.current_idx = 0
    t.output_folder = tmp_path

    t._submit_annotation()
    data = json.load(open(tmp_path / 'img_labels.json'))
    assert 'interior_mask_file' in data
    import cv2
    interior_path = tmp_path / data['interior_mask_file']
    assert interior_path.exists()
    interior = cv2.imread(str(interior_path), cv2.IMREAD_UNCHANGED)
    # Should contain export id 2 for building_immobile inside
    assert (interior == 2).any()