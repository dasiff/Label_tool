import cv2
import numpy as np
from app.core.boundary import estimate_boundary_from_overlay

img = cv2.imread('data/raw_images/La_Quinta_Inn_by_Wyndham_Dallas_Uptown_4440_N_Central_Expy_Dallas_TX_75206_United_States.png')
result = estimate_boundary_from_overlay(img)
poly = np.array(result.polygon_px)

print('\nPolygon vertices:')
for i, v in enumerate(poly):
    print(f'{i}: ({v[0]:.0f}, {v[1]:.0f})')

# Check if it forms a proper outline
poly_closed = np.vstack([poly, poly[0:1]])
filled = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.fillPoly(filled, [poly_closed.astype(np.int32)], 255)
filled_area = np.sum(filled > 0)
print(f'\nFilled area: {filled_area} ({filled_area/(img.shape[0]*img.shape[1])*100:.1f}%)')

# Draw the polygon for visualization  
vis = img.copy()
cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 255, 255), 3)
for i, v in enumerate(poly):
    cv2.circle(vis, (int(v[0]), int(v[1])), 10, (0, 0, 255), -1)
    cv2.putText(vis, str(i), (int(v[0])+15, int(v[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

cv2.imwrite('data/debug/laquinta_vertices.png', vis)
print('Saved vertex visualization')
