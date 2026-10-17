"""Data lineage + experiment tracking without a server.

Every dataset directory has manifest.json (generator config, seed, git sha, sha256 of each file, row counts).
Every training run has runs/<run_id>/run.json (config, dataset version + manifest hash, git sha, metrics, artifact
hashes) and is appended to runs/registry.jsonl, so any model traces back to the exact data and code that made it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def obj_hash(obj, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def git_sha() -> str | None:
    try:
        sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.call(["git", "-C", str(ROOT), "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def write_manifest(data_dir: Path, config: dict, extra: dict | None = None) -> dict:
    files = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(data_dir.glob("*.parquet"))}
    m = {"created": now(), "git_sha": git_sha(), "config": config, "config_hash": obj_hash(config), "files": files,
         **(extra or {})}
    m["dataset_hash"] = obj_hash({k: v["sha256"] for k, v in files.items()})
    (data_dir / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


def verify_manifest(data_dir: Path) -> dict:
    m = json.loads((data_dir / "manifest.json").read_text())
    for name, info in m["files"].items():
        if sha256(data_dir / name) != info["sha256"]:
            raise RuntimeError(f"{data_dir / name} does not match its manifest: data changed after it was recorded")
    return m


def record_run(run_id: str, config: dict, data_dir: Path, metrics: dict, artifacts: dict[str, Path]) -> dict:
    m = verify_manifest(data_dir)
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rec = {"run_id": run_id, "created": now(), "git_sha": git_sha(), "config": config, "config_hash": obj_hash(config),
           "dataset": {"dir": data_dir.name, "dataset_hash": m["dataset_hash"], "config_hash": m["config_hash"]},
           "metrics": metrics,
           "artifacts": {k: {"path": str(v.relative_to(ROOT)), "sha256": sha256(v)} for k, v in artifacts.items()}}
    (run_dir / "run.json").write_text(json.dumps(rec, indent=2))
    with open(ROOT / "runs" / "registry.jsonl", "a") as f:
        f.write(json.dumps({k: rec[k] for k in ("run_id", "created", "git_sha", "config_hash", "dataset", "metrics")}) + "\n")
    return rec
