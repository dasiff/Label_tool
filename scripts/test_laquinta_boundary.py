import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.boundary import estimate_boundary_from_overlay
import cv2
import numpy as np

img_path = list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*La_Quinta*.png'))[0]
img = cv2.imread(str(img_path))
result = estimate_boundary_from_overlay(img)
poly = np.array(result.polygon_px)

print(f'Total vertices: {len(poly)}')
print(f'First-last distance: {np.linalg.norm(poly[0] - poly[-1]):.1f}px\n')

print('Polygon edges:')
for i in range(len(poly)):
    next_i = (i+1) % len(poly)
    v = poly[next_i] - poly[i]
    edge_len = np.linalg.norm(v)
    angle = np.arctan2(v[1], v[0]) * 180 / np.pi
    print(f'  Edge {i:2d}->{next_i:2d}: {edge_len:6.1f}px at {angle:6.1f}°')
