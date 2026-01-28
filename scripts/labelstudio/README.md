# Label Studio Integration

This directory contains tools for preparing training data using Label Studio (offline from the main API).

## Overview

Label Studio is used for annotating aerial parking lot images to create ground truth for ML models (Milestone 2+). The workflow is:

1. **Set up Label Studio** locally or on shared storage
2. **Upload images** to a Label Studio project
3. **Annotate parking boundaries** (draw polygons)
4. **Export annotations** as JSON
5. **Convert to masks** using `export_to_masks.py` for model training

## Running Label Studio

### Option 1: Local Installation

```bash
pip install label-studio
label-studio
```

Then open http://localhost:8080 in your browser.

### Option 2: Docker

```bash
docker run -it -p 8080:8080 \
  -v $(pwd)/label_studio_data:/label-studio/data \
  heartexlabs/label-studio:latest
```

## Connecting to Shared Storage

For team collaboration, store images and annotations on shared storage (Google Drive, S3, etc.):

1. **Upload images** to shared location
2. **In Label Studio**, configure cloud import:
   - Project Settings → Cloud Storage
   - Choose provider (S3, Azure, etc.) and credentials
   - Select folder containing images
3. **Export annotations** back to cloud after labeling

## Exporting Annotations

1. In Label Studio, go to Project → Export
2. Select **JSON** format
3. Download `project.json`

## Converting to Masks

Once you have annotations, convert them to class masks:

```bash
python scripts/labelstudio/export_to_masks.py \
  --input project.json \
  --output masks/ \
  --image_dir path/to/images
```

This generates:
- `masks/<image_id>_mask.png` – Binary mask (parking boundary = 255, background = 0)
- `masks/<image_id>_meta.json` – Metadata (confidence, original polygon)

## Annotation Format

Label Studio exports polygons in JSON:

```json
{
  "id": 1,
  "image": "image1.png",
  "annotations": [
    {
      "result": [
        {
          "value": {
            "points": [[10, 20], [50, 20], [50, 60], [10, 60]],
            "polygonlabels": ["parking_boundary"]
          },
          "type": "polygonlabels"
        }
      ]
    }
  ]
}
```

## Next Steps (Milestone 2)

- Train boundary detection model on exported masks
- Fine-tune with active learning (hardest examples)
- Deploy trained model in `app/models/`
