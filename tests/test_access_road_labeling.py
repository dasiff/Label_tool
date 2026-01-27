import json
from pathlib import Path
import numpy as np
import pytest

from scripts.labeling_tool import LabelingTool


@pytest.fixture
def tool(tmp_path):
    # Instantiate and hide GUI
    t = LabelingTool()
    try:
        t.root.withdraw()
    except Exception:
        pass

    # Minimal state for tests
    t.clean_image = np.ones((200, 200, 3), dtype=np.uint8) * 255
    t.current_boundary = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=float)
    t.boundary_approved = True

    # Simple single-segment segmentation to allow saving
    t.segments = np.ones((200, 200), dtype=np.int32)
    t.n_segments = 1
    t.segment_labels = {}

    t.image_files = [Path('image1.png')]
    t.current_idx = 0
    t.output_folder = tmp_path
    return t


def test_add_boundary_access_and_save(tool, tmp_path):
    # Add an access segment by clicking two points on the boundary
    tool._add_boundary_access_click(100, 50, 1)  # top edge
    tool._add_boundary_access_click(150, 100, 1)  # right edge

    assert len(tool.boundary_access_segments) == 1
    seg = tool.boundary_access_segments[0]
    assert seg['label'] == 'access_allowed'

    # Submit and check JSON contains boundary_access.segments with pixel endpoints
    tool._submit_annotation()
    out = tmp_path / 'image1_labels.json'
    assert out.exists()

    data = json.load(open(out))
    assert 'boundary_access' in data
    assert data['boundary_access']['segments']
    s0 = data['boundary_access']['segments'][0]
    assert 'start_px' in s0 and 'end_px' in s0


def test_road_paint_and_constraints(tool, tmp_path):
    # Configure small buffer and compute mask
    tool.buffer_mode_var.set('px')
    tool.road_buffer_px = 20
    tool._compute_buffer_mask()

    # Paint just outside the top boundary within buffer (public road)
    tool.road_paint_var.set('public_road')
    tool._paint_road_at(100, 35, erase=False)
    assert tool.road_mask is not None
    assert tool.road_mask[35, 100] == 1

    # Painting inside the boundary should not set a pixel
    tool._paint_road_at(100, 100, erase=False)
    assert tool.road_mask[100, 100] == 0

    # Paint non_road negatives
    tool.road_paint_var.set('non_road')
    tool._paint_road_at(110, 35, erase=False)
    assert tool.road_mask[35, 110] == 2

    # Erase
    tool._paint_road_at(110, 35, erase=True)
    assert tool.road_mask[35, 110] == 0

    # Save and check that road_mask_file is present in JSON and has correct legend
    tool._submit_annotation()
    data = json.load(open(tmp_path / 'image1_labels.json'))
    assert 'road_mask_file' in data
    assert data['road_mask_info']['legend']['1'] == 'public_road'
    assert data['road_mask_info']['legend']['2'] == 'non_road'


def test_shortcuts_toggle_and_escape(tool):
    # 'm' toggle was removed; pressing 'm' should not change manual_mode or splitting selection
    from types import SimpleNamespace
    prev = tool.manual_mode
    tool._on_key_press(SimpleNamespace(key='m'))
    assert tool.manual_mode == prev
    assert getattr(tool, 'splitting_segment_id', None) is None

    # 'b' toggles boundary access mode
    assert not tool.boundary_access_mode
    tool._on_key_press(SimpleNamespace(key='b'))
    assert tool.boundary_access_mode
    tool._on_key_press(SimpleNamespace(key='b'))
    assert not tool.boundary_access_mode

    # 'r' toggles road mode
    assert not tool.road_mode_active
    tool._on_key_press(SimpleNamespace(key='r'))
    assert tool.road_mode_active
    tool._on_key_press(SimpleNamespace(key='r'))
    assert not tool.road_mode_active

    # Escape in manual mode clears selection and in-progress lines
    tool._toggle_manual_mode()
    tool.splitting_segment_id = 5
    tool.manual_polylines.append([(10,10),(20,20)])
    tool._on_key_press(SimpleNamespace(key='escape'))
    assert tool.splitting_segment_id is None
    assert tool.manual_polylines == []

