"""ONNX packaging with an explicit contract, and a validator that refuses packages that break it.

A package directory holds:
  model.onnx      raw features in, correction out (normalisation is inside the graph)
  manifest.json   input/output schema with units, per-feature training ranges, dataset hash, git sha, metrics
The validator checks: schema, ONNX/PyTorch parity, calibration-set permutation invariance, a units guard (a
metres-vs-millimetres mistake must be caught), clean-room behaviour (no metal -> near-zero correction) and output
bounds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from .lineage import git_sha, sha256
from .simulate import TARGET_COLS


def unit_of(col: str) -> str:
    base = col.rsplit("_l", 1)[0]
    if base in ("rx1_px", "rx1_py", "rx1_pz", "rx2_px", "rx2_py", "rx2_pz", "nat_px", "nat_py", "nat_pz", "rigid_sep_err"):
        return "mm"
    if base in ("rigid_rel_angle", "imu_tilt_disc_deg"):
        return "deg"
    if base.startswith("imu_"):
        return "g"
    if base.endswith(("logres", "logB")):
        return "log(normalised field)"
    return "dimensionless"


OUTPUT_UNITS = {"dp_x": "mm", "dp_y": "mm", "dp_z": "mm", "dth_x": "deg", "dth_y": "deg", "dth_z": "deg"}


class _Graph(torch.nn.Module):
    def __init__(self, net, mu, sd, ysd):
        super().__init__()
        self.net = net
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.register_buffer("ysd", torch.tensor(ysd, dtype=torch.float32))

    def forward(self, ctx_x, ctx_y, ctx_mask, q_x):
        out = self.net((ctx_x - self.mu) / self.sd, ctx_y / self.ysd, ctx_mask, (q_x - self.mu) / self.sd)
        return out * self.ysd


def export(ckpt_path: Path, out_dir: Path, train_df: pd.DataFrame, dataset_hash: str, metrics: dict) -> Path:
    from .attention import SiteAttention

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = SiteAttention(ck["arch"]["d_feat"])
    net.load_state_dict(ck["state"])
    g = _Graph(net.eval(), ck["mu"], ck["sd"], ck["ysd"]).eval()
    F = len(ck["cols"])
    ex = (torch.zeros(1, 8, F), torch.zeros(1, 8, 6), torch.zeros(1, 8, dtype=torch.bool), torch.zeros(1, 16, F))
    out_dir.mkdir(parents=True, exist_ok=True)
    # torch.export-based exporter: the legacy tracer baked the example K into the attention reshapes
    K, Q = torch.export.Dim("K", min=1, max=4096), torch.export.Dim("Q", min=1, max=65536)
    onnx_prog = torch.onnx.export(g, ex, input_names=["ctx_x", "ctx_y", "ctx_mask", "q_x"], output_names=["correction"],
                                  dynamo=True, dynamic_shapes={"ctx_x": {1: K}, "ctx_y": {1: K}, "ctx_mask": {1: K}, "q_x": {1: Q}})
    onnx_prog.save(str(out_dir / "model.onnx"), external_data=False)
    X = train_df[ck["cols"]].to_numpy(np.float64)
    lo, hi = np.quantile(X, 0.001, 0), np.quantile(X, 0.999, 0)
    span = hi - lo
    manifest = {
        "model": "site_attention", "onnx_sha256": sha256(out_dir / "model.onnx"), "git_sha": git_sha(),
        "dataset_hash": dataset_hash, "metrics": metrics,
        "inputs": {
            "q_x": {"shape": ["batch", "Q", F], "dtype": "float32", "columns": ck["cols"]},
            "ctx_x": {"shape": ["batch", "K", F], "dtype": "float32", "columns": ck["cols"]},
            "ctx_y": {"shape": ["batch", "K", 6], "dtype": "float32", "columns": TARGET_COLS, "units": OUTPUT_UNITS},
            "ctx_mask": {"shape": ["batch", "K"], "dtype": "bool", "meaning": "True = padding"}},
        "features": [{"name": c, "unit": unit_of(c), "train_p0.1": float(a), "train_p99.9": float(b),
                      "accept_lo": float(a - 0.5 * s), "accept_hi": float(b + 0.5 * s)} for c, a, b, s in zip(ck["cols"], lo, hi, span, strict=True)],
        "outputs": {"correction": {"columns": TARGET_COLS, "units": OUTPUT_UNITS,
                                   "apply": "p_true = p_native + dp (cassette frame, mm); R_true = exp(dth in deg -> rad) @ R_native"}},
        "limits": {"max_abs_dp_mm": 300.0, "max_abs_dth_deg": 30.0, "max_out_of_range_frac": 0.02},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out_dir


def check_inputs(manifest: dict, X: np.ndarray) -> dict:
    """Runtime guard: fraction of values outside the accepted range per feature. Catches unit mistakes."""
    lo = np.array([f["accept_lo"] for f in manifest["features"]])
    hi = np.array([f["accept_hi"] for f in manifest["features"]])
    bad = (X < lo) | (X > hi)
    frac = float(bad.mean())
    worst = [manifest["features"][i]["name"] for i in np.argsort(-bad.mean(0))[:5] if bad[:, i].any()]
    return {"out_of_range_frac": frac, "ok": frac <= manifest["limits"]["max_out_of_range_frac"], "worst": worst}


def run_onnx(sess, cx, cy, cm, qx):
    return sess.run(None, {"ctx_x": cx.astype(np.float32), "ctx_y": cy.astype(np.float32), "ctx_mask": cm, "q_x": qx.astype(np.float32)})[0]


def validate(pkg: Path, ckpt_path: Path, site: pd.DataFrame, site_pool: pd.DataFrame, clean: pd.DataFrame,
             clean_pool: pd.DataFrame, seed: int = 0) -> dict:
    from .predictors import Attention

    m = json.loads((pkg / "manifest.json").read_text())
    sess = ort.InferenceSession(str(pkg / "model.onnx"), providers=["CPUExecutionProvider"])
    cols = m["inputs"]["q_x"]["columns"]
    rng = np.random.default_rng(seed)
    checks = {}
    checks["onnx_hash_matches_manifest"] = sha256(pkg / "model.onnx") == m["onnx_sha256"]
    ins = {i.name: i for i in sess.get_inputs()}
    checks["schema"] = set(ins) == set(m["inputs"]) and ins["q_x"].shape[-1] == len(cols)
    calib = site_pool.iloc[rng.choice(len(site_pool), 60, replace=False)]
    q = site.iloc[:200]
    cx, cy = calib[cols].to_numpy()[None], calib[TARGET_COLS].to_numpy()[None]
    cm, qx = np.zeros((1, len(calib)), bool), q[cols].to_numpy()[None]
    y_onnx = run_onnx(sess, cx, cy, cm, qx)[0]
    y_torch = Attention(ckpt_path, device="cpu")(q, calib)
    checks["parity_max_abs_diff"] = float(np.abs(y_onnx - y_torch).max())
    checks["parity_ok"] = checks["parity_max_abs_diff"] < 1e-2
    perm = rng.permutation(len(calib))
    y_perm = run_onnx(sess, cx[:, perm], cy[:, perm], cm, qx)[0]
    checks["permutation_invariance_max_diff"] = float(np.abs(y_perm - y_onnx).max())
    checks["permutation_invariant"] = checks["permutation_invariance_max_diff"] < 1e-3
    g_ok = check_inputs(m, q[cols].to_numpy())
    mm_cols = [i for i, f in enumerate(m["features"]) if f["unit"] == "mm"]
    Xm = q[cols].to_numpy().copy()
    Xm[:, mm_cols] *= 1e-3  # a client sending metres
    g_bad = check_inputs(m, Xm)
    checks["units_guard_accepts_real_input"] = g_ok["ok"]
    checks["units_guard_rejects_metres"] = not g_bad["ok"]
    checks["units_guard_detail"] = {"real": g_ok, "metres": g_bad}
    cal_c = clean_pool.iloc[rng.choice(len(clean_pool), 60, replace=False)]
    yc = run_onnx(sess, cal_c[cols].to_numpy()[None], cal_c[TARGET_COLS].to_numpy()[None], np.zeros((1, 60), bool),
                  clean[cols].to_numpy()[None])[0]
    checks["clean_room_median_dp_mm"] = float(np.median(np.linalg.norm(yc[:, :3], axis=1)))
    checks["clean_room_median_dth_deg"] = float(np.median(np.linalg.norm(yc[:, 3:], axis=1)))
    checks["clean_room_invariant"] = checks["clean_room_median_dp_mm"] < 2.0 and checks["clean_room_median_dth_deg"] < 0.3
    checks["output_bounds"] = bool((np.abs(y_onnx[:, :3]) < m["limits"]["max_abs_dp_mm"]).all()
                                   and (np.abs(y_onnx[:, 3:]) < m["limits"]["max_abs_dth_deg"]).all())
    keys = ["onnx_hash_matches_manifest", "schema", "parity_ok", "permutation_invariant", "units_guard_accepts_real_input",
            "units_guard_rejects_metres", "clean_room_invariant", "output_bounds"]
    checks["passed"] = all(bool(checks[k]) for k in keys)
    (pkg / "validation.json").write_text(json.dumps(checks, indent=1, default=float))
    return checks
