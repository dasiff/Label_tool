import sys
sys.path.insert(0, 'g:/My Drive/parking_spaces')
import cv2
import numpy as np
from pathlib import Path
from app.core.boundary import _red_mask_hsv

img = cv2.imread(str(list(Path('g:/My Drive/parking_spaces/data/raw_images').glob('*La_Quinta*.png'))[0]))
mask = _red_mask_hsv(img)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
print(f'Total components: {num - 1}')
print('Component sizes:')
for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    print(f'  Component {i}: {area} pixels')

# Show which would be kept with 10% + 50px threshold
if num > 1:
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num)]
    largest = max(areas)
    print(f'\nLargest: {largest} pixels')
    print(f'10% threshold: {largest * 0.1:.0f} pixels')
    print('\nComponents kept:')
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area == largest:
            print(f'  Component {i}: {area} px (largest)')
        elif area >= largest * 0.1 and area >= 50:
            print(f'  Component {i}: {area} px (>10% and >50px)')
        else:
            print(f'  Component {i}: {area} px (FILTERED OUT)')
