# Labeling Tool Updates - January 19, 2026

## ✅ Completed Changes

1. **Removed "small segments ignored" text** - Simplified progress display to just show "Labeled: X% of pixels"

2. **Updated building labels for parallax** - Split "building" into:
   - `building_side` (walls visible from angle)
   - `building_roof` (overhead view of roofs)

3. **Changed segment boundaries to yellow** - Changed from pink to yellow for better visibility

4. **Added property name display** - Shows property address under progress (extracted from filename with underscores replaced by spaces)

5. **Fixed progress label not resetting** - Progress now correctly resets to "Labeled: 0% of pixels" when loading a new image

6. **Added "Edit Boundary" button** - After approving boundary, you can click "✏️ Edit Boundary" to return to boundary adjustment mode

7. **Fixed histogram equalization timing** - Enhanced image is now pre-generated when image loads, so there's no sudden visual change when you start adjusting the boundary

8. **Fixed instruction visibility** - Title "Adjust boundary (arrows=move, </>=rotate) and press Enter to proceed" now stays visible during all boundary adjustments

## 🚧 Still To Implement

### High Priority

**Item 7: Auto-deselect segment when clicking far outside**
- When in labeling mode, if you click >30px outside the current segment, it should deselect and select the clicked segment
- Currently segments stay selected until you click a new one

### Medium Priority - Requires New Features

**Item 2: Road access as brush tool**
- Convert road_access from a class label to a brush-based drawing tool
- Would allow freeform marking of driveway/access areas
- Needs new UI controls and drawing mode

**Item 3: On-street parking brush**
- Add brush tool for marking "on-street parking" pixels OUTSIDE the boundary
- Needs separate layer/mask for exterior annotations
- Would require updates to save format to include exterior parking regions

## Implementation Notes

### For Item 7 (Auto-deselect)
This would be implemented in the `_on_click` method in the labeling section. Need to:
1. Track which segment is currently "selected" (if any)
2. When clicking a new segment, check if it's the same as current
3. If different and click is >30px from segment boundary, switch selection
4. Add visual feedback for "selected" segment (highlight or outline)

### For Items 2 & 3 (Brush Tools)
These require:
1. New brush mode toggle button
2. Brush size selector
3. Mouse painting logic (track drag events, paint to mask)
4. Separate mask layers for:
   - Road access (within boundary)
   - On-street parking (outside boundary)
5. Updated save/load to handle brush masks
6. Visual overlay showing brushed regions

Estimated effort: ~4-6 hours for brush implementation

## Breaking Changes

- **Class names changed**: Any existing labels with "building" will need migration
- **Save format**: May need to update for Items 2&3 if implemented

## Testing Recommendations

1. Test boundary editing after approval - verify segments are cleared properly
2. Verify progress resets correctly when navigating images
3. Check that histogram equalization doesn't cause visual jumps
4. Verify property names display correctly for various filename formats
