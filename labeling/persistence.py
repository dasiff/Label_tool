"""Persistence helpers for saving drafts and annotations.

This module is intended to contain the file I/O and atomic write logic that was
previously embedded in the GUI's `_save_draft` method. The implementation is
copied verbatim and remains synchronous; the GUI will run it in a background
thread when appropriate.
"""
from pathlib import Path
import json
import os
import shutil
from datetime import datetime
import numpy as np


def save_draft(self):
    """Save a draft of the current work (non-destructive). Runs in a separate thread to avoid blocking the UI."""
    try:
        try:
            print("DEBUG _save_draft worker start")
        except Exception:
            pass
        drafts_dir = (Path.cwd() / 'data' / 'drafts').resolve()
        drafts_dir.mkdir(parents=True, exist_ok=True)
        # Choose image name if available
        if getattr(self, 'image_files', None) and len(self.image_files) > 0:
            image_name = self.image_files[self.current_idx].stem
        else:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_name = f'draft_{ts}'
        base = drafts_dir / image_name
        # Ensure parent exists
        base.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Diagnostics for unexpected write failures
            print(f"DEBUG draft dir: {base.parent.resolve()}, exists={base.parent.exists()}")
        except Exception:
            pass
        # Save segments
        if self.segments is not None:
            seg_path = base.with_name(base.name + '_segments.npy')
            # Ensure parent exists (defensive)
            try:
                seg_path.parent.mkdir(parents=True, exist_ok=True)
                with open(seg_path, 'wb') as _f:
                    pass
            except Exception:
                pass
            try:
                # Diagnostics
                try:
                    print(f"DEBUG seg_path: {seg_path}, parent_exists={seg_path.parent.exists()}")
                except Exception:
                    pass
                # Write to a temp file and atomically replace to avoid file locking races on Windows
                # Ensure parent directory exists (retry a few times to mitigate transient OS races on Windows)
                for _i in range(3):
                    try:
                        seg_path.parent.mkdir(parents=True, exist_ok=True)
                        break
                    except Exception:
                        import time
                        time.sleep(0.01)
                if not seg_path.parent.exists():
                    # If, for some reason, the direct parent doesn't exist (transient), fallback to drafts_dir
                    try:
                        print(f"WARNING: seg_path.parent missing ({seg_path.parent}) - falling back to drafts_dir {drafts_dir}")
                    except Exception:
                        pass
                    tmp_dir = drafts_dir
                else:
                    tmp_dir = seg_path.parent
                # Create an atomic temporary filename under the determined tmp_dir
                import uuid
                tmp_name = tmp_dir / (seg_path.name + f'.tmp.{uuid.uuid4().hex}.npy')
                np.save(str(tmp_name), self.segments)
                try:
                    os.replace(str(tmp_name), str(seg_path))
                    try:
                        print(f"DEBUG saved segments -> {seg_path}")
                    except Exception:
                        pass
                except Exception:
                    # fallback to copy
                    try:
                        shutil.copy2(str(tmp_name), str(seg_path))
                        try:
                            print(f"DEBUG copied segments -> {seg_path}")
                        except Exception:
                            pass
                    finally:
                        try:
                            os.unlink(str(tmp_name))
                        except Exception:
                            pass
            except Exception as _e:
                import traceback as _tb
                print('Failed saving segments to:', repr(str(seg_path)))
                print(_tb.format_exc())
                raise
            seg_file = seg_path
        else:
            seg_file = None
        # Build metadata
        data = {
            'image_name': image_name,
            'timestamp': datetime.now().isoformat(),
            'boundary_polygon_px': [[float(x), float(y)] for x, y in self.current_boundary] if getattr(self, 'current_boundary', None) is not None else None,
            'boundary_transform': {'dx': float(getattr(self, 'boundary_dx', 0.0)), 'dy': float(getattr(self, 'boundary_dy', 0.0)), 'theta_deg': float(getattr(self, 'boundary_theta', 0.0))},
            'segment_labels': {str(k): v for k, v in getattr(self, 'segment_labels', {}).items()},
            'manual_polylines': getattr(self, 'manual_polylines', []),
            'splitting_segment_id': int(getattr(self, 'splitting_segment_id', -1)) if getattr(self, 'splitting_segment_id', None) is not None else None,
            'min_segment_px': int(getattr(self, 'min_segment_px', 1000)),
            'target_segments': int(getattr(self, 'target_segments', 50)),
            'segments_file': seg_file.name if seg_file is not None else None,
            'note': 'draft',
        }
        json_path = base.with_name(base.name + '_draft.json')
        try:
            # Write JSON atomically avoiding NamedTemporaryFile to reduce Windows race issues
            import uuid
            tmp_json = json_path.parent / (json_path.name + f'.tmp.{uuid.uuid4().hex}.json')
            with open(tmp_json, 'w', encoding='utf-8') as _tf:
                _tf.write(json.dumps(data, indent=2))
                try:
                    _tf.flush()
                    os.fsync(_tf.fileno())
                except Exception:
                    pass
            try:
                os.replace(str(tmp_json), str(json_path))
                try:
                    print(f"DEBUG saved json -> {json_path}")
                except Exception:
                    pass
            except Exception:
                try:
                    shutil.copy2(str(tmp_json), str(json_path))
                    try:
                        print(f"DEBUG copied json -> {json_path}")
                    except Exception:
                        pass
                finally:
                    try:
                        os.unlink(str(tmp_json))
                    except Exception:
                        pass
        except Exception as e:
            import traceback as _tb
            print('Failed writing json_path:', repr(str(json_path)))
            print(_tb.format_exc())
            raise
        # Optionally copy to output folder as well
        if getattr(self, 'output_folder', None) is not None:
            try:
                out_dir = Path(self.output_folder)
                out_dir.mkdir(parents=True, exist_ok=True)
                if seg_file is not None:
                    shutil.copy2(str(seg_file), str(out_dir / seg_file.name))
                shutil.copy2(str(json_path), str(out_dir / json_path.name))
            except Exception:
                pass

        # Notify user on main thread
        try:
            self.set_manual_status(f"Draft saved: {json_path.name}")
            try:
                self.root.after(0, lambda: __import__('tkinter').messagebox.showinfo('Draft saved', f"Draft saved: {json_path}"))
            except Exception:
                pass
        except Exception:
            pass
        try:
            print("DEBUG drafts dir listing:", [p.name for p in drafts_dir.iterdir()])
        except Exception:
            pass
        try:
            print("DEBUG _save_draft worker done")
        except Exception:
            pass
    except Exception as e:
        print('Error saving draft:', e)
        try:
            self.root.after(0, lambda: __import__('tkinter').messagebox.showerror('Draft save failed', str(e)))
        except Exception:
            pass
