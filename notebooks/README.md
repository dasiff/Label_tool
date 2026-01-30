Notebooks overview

This folder contains Jupyter notebooks and utilities for the Milestone-1 POC pipeline. The project is split into three notebooks with clear responsibilities:

- `train_boundary_model.ipynb`  – Train a boundary-transform regressor (dx, dy, theta) from aerial images using the red-overlay detection preprocessing pipeline. Outputs annotated images and transform predictions.

- `train_labeling_model.ipynb`  – Train a per-pixel labeling model (UNet-like) to classify each pixel. This notebook performs k‑fold training, inference UI, and saves *inferred* masks to `data/inferred_masks/` (see below).

- `mask_to_segments.ipynb`     – Convert inferred (or human) raster masks into final polygon segments (GeoJSON) using heuristics inspired by the labeling tool (morphology, Chaikin smoothing, Douglas–Peucker simplification, convex hull fallback, and gradient-based merging). Saves polygons & overlays to `data/segmented_images/`.

Key directories
- `data/raw_images/`         – Original images
- `data/segmented_images/`   – Final polygon outputs (GeoJSON + overlays) and human-labeled exports
- `data/inferred_masks/`     – Model-produced per-pixel masks and meta JSON (created by `train_labeling_model` inference)

Quick environment setup
1. Create a Python venv and install dependencies:

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   pip install -r requirements.txt

2. (Optional – Colab) Use the "Colab setup" cell at the top of notebooks to mount Drive and install missing packages.

Notebook: `train_labeling_model.ipynb` (pixel labeling)
- Configure the top cell (MODEL_NAME, MODEL_VERSION, RUN_MODE, TARGET_SIZE, MODE, NUM_FOLDS, MAX_EPOCHS, etc.).
- Run the Imports & env cell, then the K-fold training cell to evaluate model variance on the small dataset.
- Inference: use the inference UI cell (dropdown) to run a model on a raw image. The notebook saves:
  - `data/inferred_masks/<stem>_mask.png` (uint8 class ids)
  - `data/inferred_masks/<stem>_segmented.png` (overlay visualization)
  - `data/inferred_masks/<stem>_meta.json` (model version, checkpoint, timestamp, mean confidence)

Notebook: `mask_to_segments.ipynb` (mask → polygons)
- Use this notebook to preview and convert a mask from `data/inferred_masks/` into GeoJSON polygons saved to `data/segmented_images/`.
- Parameters you can tune: `pre_smooth` (bilateral), smoothing level (`off|low|med|high`), and `min_area` (default 1000 px to match the labeling tool).
- The script `scripts/masks_to_segments.py` performs the same algorithm (executable from the command line). Example usage:

   python scripts/masks_to_segments.py --mask data/inferred_masks/example_mask.png --out_dir data/segmented_images/

Notebook: `train_boundary_model.ipynb` (boundary transform)
- Configure and run the boundary training/inference cells (similar top-level config cell). Useful for Milestone 1 boundary-based workflows.

Testing & small-data advice
- This is a POC: with ~22 labeled images use heavy augmentations and k‑fold cross-validation to estimate variance. Avoid trusting single split results.
- Run `pytest tests/ -q` to execute repository tests (segmentation simplification tests included).

Developer notes
- Shared augmentation helper: `src/data/augmentations.py::get_common_augmentation()` is used by both notebooks to ensure consistent augmentations.
- Masks→segments heuristics are implemented in `scripts/masks_to_segments.py` following the labeling tool's approach (morphology, chaikin smoothing, DP simplification, convex hull fallback, and boundary-gradient based merging).
- Experiment logging: each training run records a metadata entry (run id, name, params, and results) under `experiments/` using `src/utils/experiments.record_experiment()`; the directory contains `experiments.jsonl`, `experiments.csv`, and per-run `<run_id>.json` files.

Next steps
- Want a convenience script to batch-infer all images and convert masks→polygons in one run? I can add a small CLI helper and a CI test that validates output sanity (area thresholds, number of vertices). Let me know and I will add it.
