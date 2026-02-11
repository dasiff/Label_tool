import os
import json
import shutil
from pathlib import Path
import numpy as np
from scripts.labeling_tool import LabelingTool

def test_save_draft_creates_files(tmp_path):
    # Setup tool in headless mode
    lt = LabelingTool()
    # minimal state
    lt.clean_image = np.zeros((64,64,3), dtype=np.uint8)
    lt.segments = np.zeros((64,64), dtype=np.int32)
    lt.segments[10:40,10:40] = 1
    lt.segment_labels = {1: 'parking_stalls'}
    lt.current_boundary = np.array([[0,0],[63,0],[63,63],[0,63]])

    # Ensure drafts dir is under tmp_path
    drafts_dir = tmp_path / 'data' / 'drafts'
    drafts_dir.mkdir(parents=True, exist_ok=True)

    # Temporarily change cwd so saved drafts go into tmp_path
    old_cwd = Path.cwd()
    try:
        os.chdir(str(tmp_path))
        lt._save_draft()
        # Wait (with retries) for files to appear in data/drafts to avoid intermittent Windows timing/race issues
        import time
        found = False
        jfile = None
        for _ in range(20):  # up to 2 seconds
            files = list((tmp_path / 'data' / 'drafts').glob('*_draft.json'))
            if files:
                jfile = files[0]
                found = True
                break
            time.sleep(0.1)
        assert found, f"No draft json file found in {(tmp_path / 'data' / 'drafts')}"
        with open(jfile, 'r') as f:
            data = json.load(f)
        assert data.get('note') == 'draft'
        assert data.get('segments_file') is not None
        segfile = (tmp_path / 'data' / 'drafts') / data['segments_file']
        # Wait for segments file to appear as well
        found_seg = False
        for _ in range(20):
            if segfile.exists():
                found_seg = True
                break
            time.sleep(0.1)
        assert found_seg, f"Segments file not found: {segfile}"
        arr = np.load(str(segfile))
        assert arr.shape == (64,64)
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(str(tmp_path))
