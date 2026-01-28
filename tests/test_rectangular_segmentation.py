import numpy as np
import cv2
from labeling.core.rectangular import detect_rectangles_pyramid, draw_rectangles_debug


def _iou_box(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def _rect_to_bbox(pts):
    pts = np.array(pts)
    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
    return (int(x1), int(y1), int(x2), int(y2))


def test_detect_simple_rectangle():
    # Synthetic filled rectangle on white background
    h, w = 400, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    gt = (50, 80, 350, 300)
    cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0, 0, 0), -1)

    # Use default scales which now include coarser levels
    results = detect_rectangles_pyramid(img, min_area=1000)
    assert len(results) >= 1, "No rectangles detected on clean synthetic image"

    best = results[0]
    pts = best.get('quad') or best.get('box')
    assert pts is not None
    bbox = _rect_to_bbox(pts)
    iou = _iou_box(bbox, gt)
    assert iou > 0.7, f"IOU too low: {iou}"  # expect good overlap
    assert best.get('score', 0.0) > 0.3


def test_detect_occluded_rectangle():
    # Rectangle with partial occlusion to simulate trees
    h, w = 400, 400
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    gt = (60, 60, 340, 320)
    cv2.rectangle(img, (gt[0], gt[1]), (gt[2], gt[3]), (0, 0, 0), 6)
    # Add occluders (white circles over the edges)
    rng = np.random.RandomState(1)
    for cx in range(gt[0] + 20, gt[2], 40):
        cy = gt[1] + rng.randint(0, 40)
        cv2.circle(img, (cx, cy), 18, (255, 255, 255), -1)

    # Use default scales which include coarser levels to be robust to occlusion
    results = detect_rectangles_pyramid(img, min_area=800)
    assert len(results) >= 1, "No rectangles detected on occluded rectangle image"

    best = results[0]
    pts = best.get('quad') or best.get('box')
    bbox = _rect_to_bbox(pts)
    iou = _iou_box(bbox, gt)
    assert iou > 0.35, f"IOU too low under occlusion: {iou}"
    # Prototype is permissive and scores may be low under occlusion — accept small positive score
    assert best.get('score', 0.0) > 0.05

    # Optionally verify debug draw returns image-shaped array
    dbg = draw_rectangles_debug(img, results[:3])
    assert dbg.shape == img.shape

