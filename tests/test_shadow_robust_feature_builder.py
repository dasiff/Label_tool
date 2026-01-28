import numpy as np
from scripts.labeling_tool import LabelingTool


def make_shadowed_rect(w=200, h=100):
    # Create RGB image with a central rectangle (object) and a vertical shadow band
    img = np.ones((h, w, 3), dtype=np.uint8) * 200
    # rectangle: dark material (but similar hue)
    cv2 = __import__('cv2')
    cv2.rectangle(img, (40, 20), (160, 80), (80, 80, 140), -1)
    # shadow overlay as a left-to-right gradient reducing brightness
    for x in range(w):
        factor = 1.0 - 0.6 * (x / w)
        img[:, x, :] = np.clip(img[:, x, :].astype(float) * factor, 0, 255).astype(np.uint8)
    return img


def test_shadow_robust_features_emphasize_edges():
    lt = LabelingTool()
    img = make_shadowed_rect()
    # Resize down to small size used in segmentation
    small = __import__('cv2').resize(img, (100, 50), interpolation=__import__('cv2').INTER_AREA)
    # Default features
    lt.shadow_robust_var.set(False)
    feat_default = lt._build_segment_features(small)
    # Shadow-robust features
    lt.shadow_robust_var.set(True)
    feat_shadow = lt._build_segment_features(small)
    # Ensure shapes and ranges are correct
    assert feat_default.shape == feat_shadow.shape
    assert feat_shadow.dtype == np.uint8
    # Check that gradient channel has non-zero values when shadow_robust enabled
    # gradient channel is last channel per implementation
    grad = feat_shadow.astype(np.float32)[:, :, 2]
    assert grad.max() > 0
    # Further: gradient should highlight rectangle edges — sample along expected edge rows
    left_edge_x = 40 * (100 / 200)  # scaled location
    col = int(left_edge_x)
    # Some gradient presence at that column
    assert grad[:, max(0, col-1):col+2].mean() > 0
