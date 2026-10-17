"""Dependency-light runtime for a packaged model: load, guard inputs against the manifest, run.

    python -m emtrack.package_runtime package      # self-check: loads the package and runs a smoke inference
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


class CorrectionModel:
    def __init__(self, pkg: str | Path):
        pkg = Path(pkg)
        self.manifest = json.loads((pkg / "manifest.json").read_text())
        self.sess = ort.InferenceSession(str(pkg / "model.onnx"), providers=["CPUExecutionProvider"])
        f = self.manifest["features"]
        self.lo = np.array([x["accept_lo"] for x in f])
        self.hi = np.array([x["accept_hi"] for x in f])
        self.max_bad = self.manifest["limits"]["max_out_of_range_frac"]

    def guard(self, X: np.ndarray) -> float:
        return float(((X < self.lo) | (X > self.hi)).mean())

    def __call__(self, q_x: np.ndarray, ctx_x: np.ndarray, ctx_y: np.ndarray) -> np.ndarray:
        """q_x (Q,F) live frames; ctx_x (K,F), ctx_y (K,6) the site calibration set. Returns (Q,6) [mm, deg]."""
        bad = self.guard(q_x)
        if bad > self.max_bad:
            raise ValueError(f"{bad:.1%} of input values outside the trained range: check units (mm, deg) and column order")
        K = len(ctx_x)
        if K == 0:
            ctx_x, ctx_y = np.zeros((1, q_x.shape[1])), np.zeros((1, 6))
        mask = np.zeros((1, max(K, 1)), bool) if K else np.ones((1, 1), bool)
        return self.sess.run(None, {"ctx_x": ctx_x[None].astype(np.float32), "ctx_y": ctx_y[None].astype(np.float32),
                                    "ctx_mask": mask, "q_x": q_x[None].astype(np.float32)})[0][0]


if __name__ == "__main__":
    m = CorrectionModel(sys.argv[1] if len(sys.argv) > 1 else "package")
    mid = np.array([[(f["train_p0.1"] + f["train_p99.9"]) / 2 for f in m.manifest["features"]]])
    out = m(mid, np.empty((0, mid.shape[1])), np.empty((0, 6)))
    # synthetic mid-range input: only checks the graph loads and runs; the numbers themselves mean nothing
    assert out.shape == (1, 6) and np.isfinite(out).all()
    print("package OK:", m.manifest["model"], "| onnx", m.manifest["onnx_sha256"], "| dataset", m.manifest["dataset_hash"])
