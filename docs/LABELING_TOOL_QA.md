# Parking Spaces Labeling Tool - Q&A Session Progress

**Date:** January 13, 2026  
**Status:** In Progress - Need to resume segment splitting questions

---

## ✅ DECISIONS FINALIZED

### SLIC Parameters
- **n_segments:** 50 (initial default, within ROI only)
- **compactness:** 12
- **sigma:** 1
- **start_label:** 1
- **convert2lab:** True (LAB color space for perceptual uniformity)
- **mask:** ROI mask (parcel interior only - buffer zone NOT segmented)
- **Max resolution:** 1000px longest side (scaled down from 2000px)
- **Refinement:** "Request Finer Segments" button → increases to ~100-200 segments

**Rationale:**
- LAB color space better separates vegetation from pavement, roofs from ground
- 1000px resolution is 4x faster than 2000px, still sufficient detail
- Starting with 50 segments avoids overwhelming labelers for simple properties
- ROI-only segmentation avoids wasting segments outside parcel
- User can request more segments if property is complex/detailed

---

### ROI Strategy
- **Segmented area:** Inside parcel polygon ONLY
- **Buffer zone:** Visible for context but NOT segmented
- **Buffer width:** 50 pixels (from original spec)
- **Buffer purpose:** Road access marking only

**Why:** Most image area is irrelevant; segmenting only parcel interior reduces labeler workload significantly

---

### Road Access Marking
- **Interaction:** Click-and-drag line tool (Option B)
- **Line width:** 24 pixels (typical driveway at zoom 21: ~0.1 m/px × 24 ≈ 2.4m)
- **Multiple access points:** Yes, allowed
- **No overlapping:** New line replaces old if overlapping
- **Undo:** Step backwards through actions
- **Eraser tool:** Click line to delete it
- **Storage:** Rasterized in `label_mask.png` as class `road_access`
- **Class ID:** To be determined (0 or 7?)

**Workflow:**
1. Select "Road Access" tool
2. Click starting point on parcel edge
3. Drag to ending point on road (in buffer zone)
4. Line drawn with 24px width

---

### Semantic Classes (Updated)
- road / public ROW
- sidewalk / curb
- parking_stalls (painted/marked spaces)
- parking_access (drive aisles / access lanes)
- building / roof
- vegetation
- pool_or_equipment

**Total:** 7 classes (note: road access **segments** are stored separately via the Access tool)

**Class Definitions:**

**parking_stalls** - Individual marked parking stalls or areas clearly intended for parking
- **Includes:** Painted spaces, striping, and clearly delineated stall areas
- **Excludes:** Drive aisles (use `parking_access`)
- **Rationale:** Useful for counting individual parking spots

**parking_access** - Drive aisles and maneuvering areas that provide access to stalls
- **Includes:** Drive lanes, turning aisles, access ramps inside parking area
- **Excludes:** Public roads outside the parcel (use `road`)
- **Rationale:** Indicates drivable surface that contributes to layout but not individual stalls

**vegetation** - Any living plant matter visible on the ground
- **Includes:** Grass lawns, bushes, hedges, flower beds, mulch beds with plants, trees in landscaped areas
- **Excludes:** Trees in parking lots (cars can park under them - label ground as `parking_stalls` or `parking_access` instead)
- **Shadow handling:** Label the ground surface, not the canopy shadow (pavement under tree shadow = `parking_stalls` or `parking_access`)

**vegetation** - Any living plant matter visible on the ground
- **Includes:** Grass lawns, bushes, hedges, flower beds, mulch beds with plants, trees in landscaped areas
- **Excludes:** Trees in parking lots (cars can park under them - label ground as `parking_surface` instead)
- **Shadow handling:** Label the ground surface, not the canopy shadow (pavement under tree shadow = `parking_surface`)
- **Rationale:** Identifies areas that cannot be used for parking due to ground cover

**pool_or_equipment** - Ground-level obstacles that occupy space but could theoretically be removed
- **Includes:** Swimming pools, hot tubs, HVAC units on ground, generators, transformers, dumpster enclosures, trash pads, electrical boxes, storage containers
- **Excludes:** Rooftop equipment (label as `building` instead)
- **Rationale:** Identifies removable obstacles for parking expansion feasibility

**Common Feature Guidance:**
- **Fences/walls:** Too narrow to label - ignore
- **Sheds/storage buildings:** Label as `building`
- **Carports:** Label ground as `parking_surface` if obviously part of parking area
- **Awnings/covered walkways:** Don't affect labeling - label ground surface beneath
- **Gravel/permeable driveways:** Label as `parking_surface` (function matters, not material)
- **Dumpsters (movable):** Label what's underneath (typically `parking_surface`)
- **Trees in parking lots:** Label ground as `parking_surface` (cars park under canopy)

---

### Annotation Workflow & Finalization

**Phase 1: Boundary Review** (happens FIRST, before segmentation)
1. Image loads with detected boundary polygon overlay (yellow)
2. User reviews boundary quality
3. Decision:
   - ✅ "Boundary Looks Good" → Proceeds to Phase 2 (SLIC segmentation)
   - ❌ "Bad Boundary" → Saves metadata, skips to next image (no segmentation/labeling)

**Phase 2: Segment Labeling** (only if boundary approved in Phase 1)
1. SLIC segmentation generated within approved ROI
2. User labels segments + marks road access
3. Submit when validation passes

**Completion Requirements:**
- All segments ≥100 pixels must have class assigned
- At least one road access line (or "No Road Access" checked)
- Small segments (<100px) auto-ignored in validation

**Validation Rules:**
- **Block submission if:** Unlabeled segments ≥100px exist
- **Warning (can override) if:** No road access marked and "No Road Access" unchecked
- **Progress indicator:** "87/95 large segments labeled (8 tiny segments auto-ignored)"

**Annotation States:**
1. **Boundary Review** - Awaiting user approval/rejection
2. **In Progress** - Boundary approved, labeling incomplete
3. **Complete** - All validation passed, ready for training
4. **Bad Boundary** - Marked unusable, excluded from training

---

### Output File Format: `segment_labels.json`

**Structure:**
```json
{
  "image_id": "75206_Greenville_Ave",
  "timestamp": "2026-01-14T15:32:10Z",
  "boundary_status": "approved",
  "slic_params": {
    "n_segments": 100,
    "compactness": 12,
    "sigma": 1,
    "resolution": [1000, 1000]
  },
  "segments": {
    "1": {"class": "parking_stalls", "area_px": 12500},
    "2": {"class": "building", "area_px": 8200},
    "3": {"class": "vegetation", "area_px": 450},
    "42": {"class": "road", "area_px": 5100},
    "101": {"class": "parking_stalls", "area_px": 1200},
    "102": {"class": "sidewalk", "area_px": 800}
  },
  "road_access": {
    "lines": [
      {"start": [120, 450], "end": [180, 520], "width_px": 24}
    ],
    "none": false
  },
  "metadata": {
    "labeler": "user_id_or_name",
    "duration_seconds": 180,
    "splits_performed": 2,
    "original_segment_count": 98,
    "final_segment_count": 100
  }
}
```

**For "Bad Boundary" images:**
```json
{
  "image_id": "75225_Knox_St",
  "timestamp": "2026-01-14T15:35:22Z",
  "boundary_status": "rejected",
  "reason": "Boundary incomplete - missing east edge",
  "metadata": {
    "labeler": "user_id_or_name"
  }
}
```

**Key Design Decisions:**
- Segment IDs are strings (handles new IDs from splits: "101", "102")
- Area stored per segment (helps training pipeline filter tiny segments)
- Road access lines store pixel coordinates (converted to lat/lon in post-processing)
- Splits tracked in metadata (original vs final segment count)
- Bad boundary images get minimal JSON (no segments/labels)

---

### Edge Case Handling

**SLIC produces fewer segments than requested:**
- Proceed with labeling using actual segment count
- User can request "Finer Segments" button if more detail needed
- No warning necessary - segment count varies naturally by parcel complexity

**Minimum parcel size / Too small for parking:**
- No hard minimum - if parcel is too small to fit any parking, still label what's visible
- Label small parcels as surrounding area type (e.g., all `vegetation` or `building`)
- Useful for training model to recognize non-parking parcels

**Polygon partially/entirely outside image bounds:**
- Automatically crop boundary polygon to image bounds
- Proceed with labeling cropped region
- No error - common with edge parcels

**Multiple disconnected boundary polygons:**
- Should not occur (convex hull produces single polygon)
- If detected: Mark as "Bad Boundary" error
- Don't attempt to label - flag for boundary detection improvement

---

## ✅ DECISIONS FINALIZED (CONTINUED)

### Segment Splitting Tool

**Decided:**
- **Activation:** Click "Split Tool" button (enters split mode)
- **Definition:** Click multiple vertices to define polyline cut
- **Boundary constraint:** Split line must start/end on segment boundary (auto-snaps if close)
- **Finalize split:** Double-click on last vertex OR press Enter key
- **Cancel split:** Press Escape OR right-click
- **Auto-snap distance:** 10 pixels from segment boundary
- **Visual feedback:** Yellow dashed line preview + green circle indicator when snap activates
- **Multiple pieces:** If split creates 3+ pieces, largest keeps original ID, all others get new sequential IDs (sorted by area descending)
- **ID assignment example:** Segment #42 splits into 3 pieces (500px, 200px, 150px) → 500px stays #42, others become next available IDs

**Rationale:**
- Double-click is intuitive "done" gesture (CAD tools standard); Enter provides keyboard workflow
- 10px snap tolerance is forgiving for mouse precision at 1000px resolution
- Visual preview essential for usability - users must see what they're drawing
- Deterministic ID assignment (by area) ensures consistency

---

### Multi-Image Workflow

**Image Loading:**
- User selects input folder containing images (file dialog)
- User selects output directory for saving results (file dialog)
- Tool loads all images from selected folder

**Navigation:**
- Next/Previous buttons to move sequentially
- Jump to specific image: Text input field "Go to image #: [___]" + Go button
- Display current position: "Image 5 of 50"

**Progress Tracking:**
- Show labeled count: "15/50 images labeled (30%)"
- Visual indicator for each image state:
  - ⚪ Not started
  - 🟡 In progress (boundary approved, labeling incomplete)
  - ✅ Complete (submitted)
  - ❌ Bad boundary (rejected)

**Re-editing:**
- User can revisit any previously labeled image
- Load existing `segment_labels.json` and restore state
- Can modify labels and re-submit (overwrites previous)
- Warning prompt: "This image was already labeled. Re-open for editing?"

**File Management:**
- Input folder: Contains source images (PNG/JPEG)
- Output folder: Saves `segment_labels.json` + `label_mask.png` per image
- Output filename format: `{image_id}_labels.json`, `{image_id}_mask.png`

---

### Tool Architecture

**Framework:** Napari (Python desktop app for scientific image annotation)
- Layer-based UI with custom widgets for labeling controls
- Built-in polygon/point/line tools
- Easy to add custom interactions (segment splitting, road access drawing)

---

## ❓ QUESTIONS NOT YET ADDRESSED

None - all specification questions resolved!

---

## 📋 SPECIFICATION COMPLETE

**ALL QUESTIONS RESOLVED** ✅

1. ✅ SLIC parameters (50 initial segments, refinement available)
2. ✅ ROI strategy (parcel interior only)
3. ✅ Road access marking (click-drag tool, 24px width)
4. ✅ Semantic class definitions (7 classes with clear guidelines)
5. ✅ Segment splitting tool (double-click, 10px snap, visual preview)
6. ✅ Boundary review workflow (approve/reject before segmentation)
7. ✅ Annotation finalization and validation rules
8. ✅ Output file format (`segment_labels.json`)
9. ✅ Edge case handling
10. ✅ Multi-image workflow (navigation, progress, re-editing)
11. ✅ Tool architecture (Napari framework)

**Next step:** Begin Napari labeling tool implementation

---

## 🎯 IMPLEMENTATION READINESS

**Complete specification ready for development:**
- SLIC generation with exact parameters
- ROI mask creation (parcel interior only)
- Road access click-drag tool
- Segment labeling with 7 classes
- Segment splitting tool (polyline cut with auto-snap)
- Boundary review workflow (approve/reject before segmentation)
- Validation rules (100px minimum segment size, road access required)
- JSON export format with metadata
- Multi-image navigation and progress tracking
- Napari-based desktop application

**No blockers remaining - ready to implement**
