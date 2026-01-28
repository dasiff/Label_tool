from types import SimpleNamespace
import os
import cv2
import numpy as np

# Ensure package import works when run from repo root
from pathlib import Path
repo_root = Path(__file__).resolve().parents[1]
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from scripts.labeling_tool import LabelingTool

# Image used by test
img_path = Path("data/raw_images/La_Quinta_Inn_by_Wyndham_Dallas_Uptown_4440_N_Central_Expy_Dallas_TX_75206_United_States.png")
img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(img_path)

h, w = img.shape[:2]
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
scale_factor = 0.5
small_h, small_w = int(h * scale_factor), int(w * scale_factor)
rgb_small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)

# Make a dummy self with shadow_robust_var on by default
dummy = SimpleNamespace()
dummy.shadow_robust_var = SimpleNamespace(get=lambda: True)

print("Calling wrapper _build_segment_features with shadow-robust = on")
# Call unbound method (wrapper) which prints timing
res = LabelingTool._build_segment_features(dummy, rgb_small)
print(f"Result shape: {res.shape}, dtype: {res.dtype}")
