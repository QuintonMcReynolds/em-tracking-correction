"""Per-site correction from K calibration poses.

fieldmap   : the classic approach. Ridge-regularised polynomial (degree 2) map from the native pose (position + 6-D
             rotation) to the correction, fit on the site's calibration poses only.
hybrid     : a global model's prediction plus a field map fit to the global model's residuals at the site.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from .simulate import TARGET_COLS

POSE_COLS = ["nat_px_x", "nat_px_y", "nat_px_z"] + [f"nat_r6x_{k}" for k in range(6)]


def fit_fieldmap(calib: pd.DataFrame, target: np.ndarray | None = None, degree: int = 2, alpha: float | None = None):
    y = calib[TARGET_COLS].to_numpy() if target is None else target
    k = len(calib)
    if k < 10:
        return None
    alpha = alpha if alpha is not None else 30.0 / np.sqrt(k)
    deg = 1 if k < 60 else degree       # too few points for a quadratic in 9 variables (55 terms)
    m = make_pipeline(StandardScaler(), PolynomialFeatures(deg), Ridge(alpha=alpha))
    m.fit(calib[POSE_COLS].to_numpy(), y)
    return m


def predict_fieldmap(m, df: pd.DataFrame) -> np.ndarray:
    if m is None:
        return np.zeros((len(df), 6))
    return m.predict(df[POSE_COLS].to_numpy())
