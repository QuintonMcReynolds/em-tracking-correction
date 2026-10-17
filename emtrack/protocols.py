"""Site calibration capture protocols: which K static poses to record at a new site.

Time model (stated assumption): handheld capture with a tracked reference, ~4 s per pose (position, hold 4 frames,
move on), so 900 poses ~ 60 minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_POSE = 4.0


def minutes(k: int) -> float:
    return k * SECONDS_PER_POSE / 60


def _pose_coords(pool: pd.DataFrame) -> np.ndarray:
    p = pool[["true_px", "true_py", "true_pz"]].to_numpy() / 200.0           # ~unit scale
    r = pool[[f"true_r6_{k}" for k in range(6)]].to_numpy()                    # rotation, 6-D
    return np.c_[p, r]


def select(pool: pd.DataFrame, k: int, protocol: str, rng: np.random.Generator) -> pd.DataFrame:
    if k <= 0:
        return pool.iloc[:0]
    if protocol == "random":
        return pool.iloc[rng.choice(len(pool), min(k, len(pool)), replace=False)]
    if protocol == "convenient":
        # what an unguided operator tends to do: mostly near-upright beam, mostly central positions.
        # Weighted sampling (not a hard filter) so the bias holds at every budget K.
        tilt = np.degrees(np.arccos(np.clip(_beam_cos(pool), -1, 1)))
        rxy = np.hypot(pool.true_px.to_numpy(), pool.true_py.to_numpy())
        w = np.exp(-tilt / 5.0) * np.exp(-rxy / 70.0)
        return pool.iloc[rng.choice(len(pool), min(k, len(pool)), replace=False, p=w / w.sum())]
    if protocol == "coverage":
        # farthest-point sampling in (position, orientation) space: spreads poses over the whole problem space
        X = _pose_coords(pool)
        idx = [int(rng.integers(len(X)))]
        d = np.linalg.norm(X - X[idx[0]], axis=1)
        for _ in range(min(k, len(X)) - 1):
            j = int(np.argmax(d))
            idx.append(j)
            d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
        return pool.iloc[idx]
    if protocol == "stratified":
        # representative coverage: k-means in (position, orientation) space, take the pool pose nearest each centre.
        # Unlike farthest-point sampling it doesn't spend the first poses on the extreme corners.
        from sklearn.cluster import KMeans

        X = _pose_coords(pool)
        k = min(k, len(X))
        km = KMeans(n_clusters=k, n_init=1, random_state=int(rng.integers(1 << 31)), max_iter=50).fit(X)
        d = ((X[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1) if len(X) * k < 4e6 else None
        if d is None:
            from sklearn.metrics import pairwise_distances_argmin
            idx = pairwise_distances_argmin(km.cluster_centers_, X)
        else:
            idx = d.argmin(0)
        idx = list(dict.fromkeys(int(i) for i in idx))      # de-duplicate, keep order
        rest = [i for i in rng.permutation(len(X)) if i not in set(idx)][: k - len(idx)]
        return pool.iloc[idx + rest]
    raise ValueError(protocol)


def _beam_cos(pool: pd.DataFrame) -> np.ndarray:
    """cos of the angle between the beam (-z_body) and straight down (-z_world) = R[2,2] (third row, third column)."""
    # 6-D rep = [R[:,0], R[:,1]]; R[:,2] = R[:,0] x R[:,1]
    c0 = pool[[f"true_r6_{k}" for k in range(3)]].to_numpy()
    c1 = pool[[f"true_r6_{k}" for k in range(3, 6)]].to_numpy()
    return np.cross(c0, c1)[:, 2]
