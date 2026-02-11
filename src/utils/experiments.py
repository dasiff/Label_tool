import os
import json
import csv
from datetime import datetime
from pathlib import Path
import hashlib
import subprocess


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def git_commit_hash():
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return out
    except Exception:
        return None


def _run_id(name: str):
    now = datetime.utcnow().isoformat(timespec="seconds")
    raw = f"{name}-{now}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def record_experiment(
    name: str,
    params: dict,
    results: dict,
    out_dir: str = "experiments",
    model_path: str = None,
    extra: dict = None,
):
    """Record an experiment run.

    - Appends a JSON line to `experiments/experiments.jsonl` for easy programmatic consumption.
    - Appends a row to `experiments/experiments.csv` for quick viewing in Excel.
    - Writes a per-run JSON file `experiments/<run_id>.json` with full metadata.

    Returns the run_id and JSON path.
    """
    _ensure_dir(out_dir)
    run_id = _run_id(name)
    timestamp = datetime.utcnow().isoformat() + "Z"
    commit = git_commit_hash()

    payload = {
        "run_id": run_id,
        "name": name,
        "timestamp": timestamp,
        "git_commit": commit,
        "params": params,
        "results": results,
        "model_path": model_path,
    }
    if extra:
        payload["extra"] = extra

    # Append JSONL
    jsonl_path = os.path.join(out_dir, "experiments.jsonl")
    with open(jsonl_path, "a", encoding="utf8") as f:
        f.write(json.dumps(payload) + "\n")

    # Append CSV (flatten results & params)
    csv_path = os.path.join(out_dir, "experiments.csv")
    # columns: run_id, name, timestamp, git_commit, model_path, <params..>, <results..>
    flat = {
        "run_id": run_id,
        "name": name,
        "timestamp": timestamp,
        "git_commit": commit,
        "model_path": model_path,
    }
    # merge params/results with some ordering
    for k, v in (params or {}).items():
        flat[f"param_{k}"] = v
    for k, v in (results or {}).items():
        flat[f"result_{k}"] = v

    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)

    # Write per-run JSON
    run_json_path = os.path.join(out_dir, f"{run_id}.json")
    with open(run_json_path, "w", encoding="utf8") as f:
        json.dump(payload, f, indent=2)

    return run_id, run_json_path
