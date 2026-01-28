from scripts.labeling_tool import LabelingTool
import numpy as np
from tests.test_split_integration import make_rect_image

lt = LabelingTool()
img = make_rect_image()
lt.clean_image = img.copy()
seg = np.zeros_like(lt.clean_image[...,0], dtype=np.int32)
seg[40:160, 40:260] = 1
seg[80:120, 100:200] = 0
lt.segments = seg
lt.n_segments = 1
lt.splitting_segment_id = 1

lt.manual_polylines = [[(45,50),(155,50)]]
lt._apply_manual_split()
print('last_snap after first:', getattr(lt,'_last_snap_info', None))
lt.manual_polylines.append([(45,250),(155,250)])
lt._apply_manual_split()
print('last_snap after second:', getattr(lt,'_last_snap_info', None))
print('n_segments', lt.n_segments)
