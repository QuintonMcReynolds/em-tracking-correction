from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "v1"
EVAL_SPLITS = ["test_clinic", "near_emitter", "under_table", "clutter", "strong", "clean"]


LOGRES_MAX = -3.0       # receiver fit residual above this -> the tracker reports the frame invalid
POS_BOUNDS = (450.0, 100.0, 1000.0)


def valid_mask(df: pd.DataFrame) -> np.ndarray:
    """Tracker-side validity check from observable signals only (never the true error): native position inside a
    plausible volume, both receivers fit the dipole model, and the two receivers agree on their rigid separation."""
    px, py, pz = (df[f"nat_p{c}_l0"].to_numpy() for c in "xyz")
    inside = (np.abs(px) < POS_BOUNDS[0]) & (np.abs(py) < POS_BOUNDS[0]) & (pz > POS_BOUNDS[1]) & (pz < POS_BOUNDS[2])
    fit = np.maximum(df["rx1_logres_l0"].to_numpy(), df["rx2_logres_l0"].to_numpy()) <= LOGRES_MAX
    rigid = np.abs(df["rigid_sep_err_l0"].to_numpy()) <= 50
    return inside & fit & rigid


@lru_cache(maxsize=32)
def load_all(split: str, pool: bool = False) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{split}{'_pool' if pool else ''}.parquet")


@lru_cache(maxsize=32)
def load(split: str, pool: bool = False) -> pd.DataFrame:
    """Valid frames only: what a correction model ever sees in deployment."""
    d = load_all(split, pool)
    return d[valid_mask(d)].reset_index(drop=True)


def valid_rate(split: str) -> float:
    return float(valid_mask(load_all(split)).mean())
