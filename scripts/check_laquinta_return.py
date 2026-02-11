from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import cv2
from app.core.boundary import estimate_boundary_from_overlay

img_path = list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*La_Quinta*.png'))[0]
img = cv2.imread(str(img_path))
res = estimate_boundary_from_overlay(img)
print('RETURNED:', res)
print('polygon_px is None?', res.polygon_px is None)
print('warnings:', res.warnings)
if res.polygon_px is not None:
    print('len polygon:', len(res.polygon_px))
