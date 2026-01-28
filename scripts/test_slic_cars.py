import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from pathlib import Path

# Find all images and their sizes
raw_images_dir = Path("data/raw_images")
image_files = sorted(raw_images_dir.glob("*.png"))

# Look for the largest image (likely to be the complex parking lot with lots of cars)
largest_img = None
largest_size = 0

for f in image_files:
    bgr = cv2.imread(str(f))
    if bgr is not None:
        h, w = bgr.shape[:2]
        size = h * w
        if size > largest_size:
            largest_size = size
            largest_img = f

print(f"Using largest/most complex image: {largest_img.name}")
bgr = cv2.imread(str(largest_img))
print(f"Original size: {bgr.shape[1]}x{bgr.shape[0]} pixels")

# Scale down to manageable size (max 2000x2000)
scale = min(2000 / bgr.shape[1], 2000 / bgr.shape[0])
new_w = int(bgr.shape[1] * scale)
new_h = int(bgr.shape[0] * scale)
bgr_small = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
print(f"Scaled to: {rgb.shape[1]}x{rgb.shape[0]} pixels for testing")

# Test 4 segment counts
params_to_test = [
    {'n_segments': 150, 'compactness': 12, 'label': '150 (proposed)'},
    {'n_segments': 200, 'compactness': 12, 'label': '200'},
    {'n_segments': 250, 'compactness': 12, 'label': '250'},
    {'n_segments': 300, 'compactness': 12, 'label': '300'},
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

print("\nGenerating SLIC segmentations...")
for idx, params in enumerate(params_to_test):
    segments = slic(rgb, n_segments=params['n_segments'], compactness=12, sigma=1, start_label=1)
    marked = mark_boundaries(rgb, segments, color=(0.5, 0.5, 0), mode='thick')
    
    axes[idx].imshow(marked)
    axes[idx].set_title(
        f"{params['label']}\nGenerated: {segments.max()} segments",
        fontsize=11, fontweight='bold'
    )
    axes[idx].axis('off')
    
    print(f"  {params['label']}: {segments.max()} actual segments")

plt.tight_layout()
output_file = 'complex_parking_slic_test.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_file}")
print("\nObservations:")
print("- How are individual cars segmented?")
print("- Is parking surface cleanly separated from cars?")
print("- Are parking lines preserved?")
