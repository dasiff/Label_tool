import numpy as np
from types import SimpleNamespace
from scripts.labeling_tool import LabelingTool
import labeling.workers as workers_core


def test_worker_schedules_main_thread_fallback():
    t = LabelingTool()
    t.clean_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    t.current_boundary = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=float)
    t.boundary_approved = True

    # Create a single segment
    segs = np.zeros((100, 100), dtype=np.int32)
    segs[20:60, 20:60] = 1
    t.segments = segs
    t.n_segments = 1

    # Select segment and add an open polyline (not closed)
    t.splitting_segment_id = 1
    t.manual_polylines = [[(30, 30), (45, 45)]]

    # Simulate running background worker
    workers_core.apply_manual_split_background_safe(t)

    # Worker should have scheduled a fallback to main thread when it couldn't handle open line
    assert getattr(t, '_background_scheduled_full_split', False) is True or getattr(t, '_last_split_info', None) is not None
