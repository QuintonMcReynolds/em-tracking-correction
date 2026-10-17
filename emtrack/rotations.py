"""Rotation helpers (numpy). Convention: R maps body-frame vectors into the world (cassette) frame."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(rv).as_matrix()


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def angle_between(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between batches of rotation matrices."""
    rel = np.einsum("...ji,...jk->...ik", Ra, Rb)
    c = np.clip((np.trace(rel, axis1=-2, axis2=-1) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(c))


def to_6d(R: np.ndarray) -> np.ndarray:
    """Continuous 6-D rotation representation (first two columns, Zhou et al. 2019)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def super_fibonacci(n: int, seed: float = 0.5) -> np.ndarray:
    """n near-uniform unit quaternions (Alexa 2022, 'Super-Fibonacci spirals'), returned as rotation matrices."""
    phi, psi = np.sqrt(2.0), 1.533751168755204288118041
    i = np.arange(n) + seed
    s = i / n
    r, R_ = np.sqrt(s), np.sqrt(1.0 - s)
    alpha, beta = 2 * np.pi * i / phi, 2 * np.pi * i / psi
    q = np.stack([r * np.sin(alpha), r * np.cos(alpha), R_ * np.sin(beta), R_ * np.cos(beta)], -1)  # x y z w
    return Rotation.from_quat(q).as_matrix()
