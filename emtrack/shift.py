"""Live distribution-shift detection: is this frame like the data the correction model was trained on?

Mahalanobis distance (Ledoit-Wolf covariance) on the signals that describe *the environment and the fit*, not the
pose itself: reference-receiver deviation, receiver fit residuals, rigid-body consistency, IMU tilt disagreement, and
the native position (so frames outside the trained volume are flagged too). The alarm threshold is a quantile of
in-distribution validation scores, chosen up front, not tuned on the shifted data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

SHIFT_COLS = ([f"ref_dS{a}{b}_l0" for a in range(3) for b in range(3)] + ["ref_dS_norm_l0", "rx1_logres_l0", "rx2_logres_l0",
              "rigid_sep_err_l0", "rigid_rel_angle_l0", "imu_tilt_disc_deg_l0", "rx1_logB_l0", "rx2_logB_l0",
              "nat_px_l0", "nat_py_l0", "nat_pz_l0"])


class ShiftDetector:
    def __init__(self, cols: list[str] = SHIFT_COLS, quantile: float = 0.99):
        self.cols, self.quantile = cols, quantile

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame | None = None):
        X = train[self.cols].to_numpy(np.float64)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.cov = LedoitWolf().fit((X - self.mu) / self.sd)
        ref = train if calib is None else calib
        self.threshold = float(np.quantile(self.score(ref), self.quantile))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        Z = (df[self.cols].to_numpy(np.float64) - self.mu) / self.sd
        return np.sqrt(np.maximum(self.cov.mahalanobis(Z), 0))

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        return self.score(df) > self.threshold

    def to_dict(self) -> dict:
        return {"cols": self.cols, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "precision": self.cov.precision_.tolist(), "threshold": self.threshold, "quantile": self.quantile}


POSE_COLS_L0 = ["nat_px_l0", "nat_py_l0", "nat_pz_l0"] + [f"nat_r6_{k}_l0" for k in range(6)]


class SupportGate:
    """Is this pose inside the region the model was trained on? Mean distance to the k nearest training poses in
    (position / 50 mm, 6-D rotation) space, alarm above the in-distribution quantile. Mahalanobis alone misses this:
    a frame can be typical in every signal and still sit where the model never saw data."""

    def __init__(self, k: int = 10, quantile: float = 0.99, max_bank: int = 30000, seed: int = 0):
        self.k, self.quantile, self.max_bank, self.seed = k, quantile, max_bank, seed

    @staticmethod
    def _coords(df: pd.DataFrame) -> np.ndarray:
        X = df[POSE_COLS_L0].to_numpy(np.float64)
        return np.c_[X[:, :3] / 50.0, X[:, 3:]]

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame | None = None):
        from sklearn.neighbors import NearestNeighbors

        X = self._coords(train)
        if len(X) > self.max_bank:
            X = X[np.random.default_rng(self.seed).choice(len(X), self.max_bank, replace=False)]
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(X)
        ref = calib if calib is not None else train.sample(min(len(train), 5000), random_state=self.seed)
        self.threshold = float(np.quantile(self.score(ref), self.quantile))
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        d, _ = self.nn.kneighbors(self._coords(df))
        return d.mean(1)

    def flag(self, df: pd.DataFrame) -> np.ndarray:
        return self.score(df) > self.threshold
