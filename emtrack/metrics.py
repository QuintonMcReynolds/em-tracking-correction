"""Pose-error metrics after applying a predicted correction, aggregated per environment (the unit that matters)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rotations import angle_between, rotvec_to_matrix
from .simulate import TARGET_COLS

TARGET_MM, TARGET_DEG = 10.0, 1.0


def corrected_errors(df: pd.DataFrame, pred: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """pred (N,6) = predicted [dp mm, dth deg]; None = no correction (native tracker)."""
    y = df[TARGET_COLS].to_numpy(np.float64)
    p = np.zeros_like(y) if pred is None else np.asarray(pred, np.float64)
    pos = np.linalg.norm(y[:, :3] - p[:, :3], axis=1)
    # true R = exp(dth_true) R_nat ; corrected R = exp(dth_pred) R_nat -> error = angle(exp(dth_true), exp(dth_pred))
    ang = angle_between(rotvec_to_matrix(np.radians(y[:, 3:])), rotvec_to_matrix(np.radians(p[:, 3:])))
    return pos, ang


def per_env(df: pd.DataFrame, pred: np.ndarray | None) -> pd.DataFrame:
    pos, ang = corrected_errors(df, pred)
    t = pd.DataFrame({"env_id": df["env_id"].to_numpy(), "family": df["family"].to_numpy(), "pos": pos, "ang": ang})
    g = t.groupby(["env_id", "family"])
    out = g.agg(pos_med=("pos", "median"), pos_p95=("pos", lambda x: np.percentile(x, 95)),
                ang_med=("ang", "median"), ang_p95=("ang", lambda x: np.percentile(x, 95))).reset_index()
    out["meets_target"] = (out.pos_med < TARGET_MM) & (out.ang_med < TARGET_DEG)
    return out


def summarize(env_table: pd.DataFrame) -> dict:
    return {"n_envs": int(len(env_table)), "pos_med_mm": float(env_table.pos_med.median()),
            "pos_p95_mm": float(env_table.pos_p95.median()), "ang_med_deg": float(env_table.ang_med.median()),
            "ang_p95_deg": float(env_table.ang_p95.median()), "envs_meeting_target": float(env_table.meets_target.mean()),
            "worst_env_pos_med_mm": float(env_table.pos_med.max())}
