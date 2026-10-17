"""Export the site-attention model to ONNX, write its manifest, validate, and record the run (lineage)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emtrack.data import DATA, load  # noqa: E402
from emtrack.lineage import record_run, verify_manifest  # noqa: E402
from emtrack.package import export, validate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ck = ROOT / "runs" / "site_attention" / "model.pt"
    m = verify_manifest(DATA)
    metrics = json.loads((ROOT / "reports" / "headline.json").read_text()) if (ROOT / "reports" / "headline.json").exists() else {}
    pkg = export(ck, ROOT / "package", load("train"), m["dataset_hash"], metrics)
    env = load("test_clinic").env_id.iloc[0]
    site, pool = load("test_clinic"), load("test_clinic", pool=True)
    clean, cpool = load("clean"), load("clean", pool=True)
    ce = clean.env_id.iloc[0]
    res = validate(pkg, ck, site[site.env_id == env], pool[pool.env_id == env], clean[clean.env_id == ce], cpool[cpool.env_id == ce])
    print(json.dumps({k: v for k, v in res.items() if k != "units_guard_detail"}, indent=1))
    record_run("site_attention", {"model": "site_attention", "train_script": "scripts/train_attention.py"}, DATA,
               {"package_validation_passed": res["passed"], **metrics}, {"model_pt": ck, "onnx": pkg / "model.onnx"})
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
