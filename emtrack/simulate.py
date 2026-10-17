"""Environment sampling and dataset generation.

Sensors (5): two 3-axis EM receivers on the handheld emitter, a fixed EM reference receiver on the cassette, and
accelerometers on both the emitter and the cassette. Each row is one frame of a capture with the current frame plus
3 lag frames: 4 x 68 = 272 feature columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .physics import Environment, Metal, measure, primary
from .rotations import angle_between, matrix_to_rotvec, rotvec_to_matrix, to_6d
from .solver import solve

RX_OFFSETS = np.array([[0.0, 60.0, 40.0], [0.0, -60.0, 40.0]])   # receivers in the emitter body frame (mm)
REF_POS = np.array([260.0, 160.0, 25.0])                          # reference receiver on the cassette (mm)
REF_R = np.eye(3)
WORK_LO, WORK_HI = np.array([-200.0, -200.0, 350.0]), np.array([200.0, 200.0, 750.0])
WORK_C = (WORK_LO + WORK_HI) / 2
MAX_TILT_DEG = 40.0
LAGS = 4

TRAIN_FAMILY = "clinic"
STRESS_FAMILIES = ["near_emitter", "under_table", "clutter", "strong", "clean"]


# ---------------------------------------------------------------- environments
def _rand_unit(rng, n=None):
    v = rng.normal(size=(3,) if n is None else (n, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _metal(rng, kind, pos: np.ndarray, scale: float = 1.0) -> Metal:
    kind = str(kind)
    if kind == "sphere":
        a = rng.uniform(60, 180) * scale
        A = -((a * 1e-3) ** 3 / 2) * np.eye(3)
        return Metal(kind, pos, A, a)
    if kind == "plate":
        b = rng.uniform(150, 380) * scale
        n = np.array([0, 0, 1.0]) if rng.random() < 0.6 else _rand_unit(rng)
        A = -0.42 * (b * 1e-3) ** 3 * np.outer(n, n)
        return Metal(kind, pos, A, b)
    L = rng.uniform(200, 600) * scale
    u = _rand_unit(rng)
    A = 0.04 * (L * 1e-3) ** 3 * np.outer(u, u)
    return Metal("rod", pos, A, L)


MIN_TX_DIST = 380.0  # metal is never right on top of the field generator (that saturates the tracker)


def _shell_pos(rng, rmin: float, rmax: float) -> np.ndarray:
    while True:
        p = WORK_C + _rand_unit(rng) * rng.uniform(rmin, rmax)
        if p[2] > -350 and np.linalg.norm(p) > MIN_TX_DIST:
            return p


def sample_environment(env_id: str, family: str, rng: np.random.Generator) -> Environment:
    kinds = ["sphere", "plate", "rod"]
    w = [0.4, 0.3, 0.3]
    metals: list[Metal] = []
    if family == "clinic":
        for _ in range(rng.integers(1, 5)):
            metals.append(_metal(rng, rng.choice(kinds, p=w), _shell_pos(rng, 420, 900), 1.25))
    elif family == "near_emitter":       # an instrument tray / hand-held tool right next to the working volume
        metals.append(_metal(rng, rng.choice(["sphere", "rod"]), _shell_pos(rng, 240, 330), 0.85))
        metals.append(_metal(rng, rng.choice(kinds, p=w), _shell_pos(rng, 420, 900)))
    elif family == "under_table":        # large horizontal plate under the cassette (procedure table)
        b = rng.uniform(220, 320)
        pos = np.array([rng.uniform(-80, 80), rng.uniform(-80, 80), rng.uniform(-420, -330)])
        metals.append(Metal("plate", pos, -0.42 * (b * 1e-3) ** 3 * np.outer([0, 0, 1.0], [0, 0, 1.0]), b))
        metals.append(_metal(rng, rng.choice(kinds, p=w), _shell_pos(rng, 420, 900)))
    elif family == "clutter":
        for _ in range(rng.integers(6, 10)):
            metals.append(_metal(rng, rng.choice(kinds, p=w), _shell_pos(rng, 420, 900), 0.8))
    elif family == "strong":
        for _ in range(rng.integers(1, 5)):
            metals.append(_metal(rng, rng.choice(kinds, p=w), _shell_pos(rng, 480, 900), 1.6))
    elif family == "clean":
        pass
    else:
        raise ValueError(family)
    tilt = rotvec_to_matrix(_rand_unit(rng) * np.radians(rng.uniform(0, 12)) * np.array([1, 1, 0]))
    return Environment(env_id, family, metals, tilt @ np.array([0.0, 0.0, -1.0]))


# ---------------------------------------------------------------- poses
def sample_orientations(rng, n: int, max_tilt: float = MAX_TILT_DEG) -> np.ndarray:
    """Beam pointing down (body -z ~ world -z) tilted up to max_tilt, any roll about the beam."""
    roll = rotvec_to_matrix(np.c_[np.zeros(n), np.zeros(n), rng.uniform(0, 2 * np.pi, n)])
    ax = rng.normal(size=(n, 2))
    ax = np.c_[ax / np.linalg.norm(ax, axis=1, keepdims=True), np.zeros(n)]
    tilt = rotvec_to_matrix(ax * np.radians(max_tilt) * np.sqrt(rng.uniform(0, 1, n))[:, None])
    return tilt @ roll


def trajectories(rng, n_traj: int, T: int):
    """Slow hand motion: smooth random walks in position (~3 mm/frame) and orientation (~0.6 deg/frame)."""
    p0 = rng.uniform(WORK_LO, WORK_HI, (n_traj, 3))
    R0 = sample_orientations(rng, n_traj)
    v = rng.normal(0, 3.0, (n_traj, 3))
    w = rng.normal(0, np.radians(0.6), (n_traj, 3))
    P, R = np.zeros((n_traj, T, 3)), np.zeros((n_traj, T, 3, 3))
    P[:, 0], R[:, 0] = p0, R0
    for t in range(1, T):
        v = 0.8 * v + rng.normal(0, 1.5, v.shape)
        w = 0.8 * w + rng.normal(0, np.radians(0.3), w.shape)
        P[:, t] = np.clip(P[:, t - 1] + v, WORK_LO, WORK_HI)
        R[:, t] = rotvec_to_matrix(w) @ R[:, t - 1]
    return P, R


# ---------------------------------------------------------------- simulation of one capture batch
def _frame_features(P, R, env: Environment, rng) -> tuple[np.ndarray, dict]:
    """P (N,3), R (N,3,3) true emitter poses -> per-frame features (N,68) plus native estimates."""
    N = len(P)
    feats, nat_p, nat_R = [], [], []
    for off in RX_OFFSETS:
        pr = P + np.einsum("nij,j->ni", R, off)
        S = measure(pr, R, env, rng)
        # tracking-style initialisation near the truth: the solver converges to the biased optimum next to it
        pe, Re, res = solve(S, pr + rng.normal(0, 5, pr.shape), R @ rotvec_to_matrix(rng.normal(0, 0.02, (N, 3))))
        nrm = np.linalg.norm(S, axis=(1, 2))
        feats += [(S / nrm[:, None, None]).reshape(N, 9), np.log(nrm)[:, None], pe, to_6d(Re), np.log(res + 1e-9)[:, None]]
        nat_p.append(pe - np.einsum("nij,j->ni", Re, off))
        nat_R.append(Re)
    # emitter pose from the two receivers: mean position, chordal-mean rotation
    pe = (nat_p[0] + nat_p[1]) / 2
    U, _, Vt = np.linalg.svd(nat_R[0] + nat_R[1])
    D = np.ones((N, 3))
    D[:, 2] = np.sign(np.linalg.det(U @ Vt))
    Re = U @ (D[:, :, None] * Vt)
    feats += [pe, to_6d(Re)]
    sep_true = np.linalg.norm(RX_OFFSETS[0] - RX_OFFSETS[1])
    sep = np.linalg.norm((nat_p[0] + np.einsum("nij,j->ni", nat_R[0], RX_OFFSETS[0]))
                         - (nat_p[1] + np.einsum("nij,j->ni", nat_R[1], RX_OFFSETS[1])), axis=1)
    feats += [(sep - sep_true)[:, None], angle_between(nat_R[0], nat_R[1])[:, None]]
    # reference receiver: true pose known, so its deviation from the model is a direct environment fingerprint
    Sref = measure(np.repeat(REF_POS[None], N, 0), np.repeat(REF_R[None], N, 0), env, rng)
    Sref0 = np.einsum("ji,njk->nik", REF_R, primary(np.repeat(REF_POS[None], N, 0)))
    dref = (Sref - Sref0) / np.linalg.norm(Sref0, axis=(1, 2), keepdims=True)
    feats += [dref.reshape(N, 9), np.linalg.norm(dref.reshape(N, 9), axis=1)[:, None]]
    # accelerometers (static assumption): specific force = -g, gravity is undistorted by metal
    g = env.gravity
    acc_c = np.repeat(-g[None], N, 0) + rng.normal(0, 0.01, (N, 3))
    acc_e = np.einsum("nji,j->ni", R, -g) + rng.normal(0, 0.01, (N, 3))
    pred_e = np.einsum("nji,nj->ni", Re, acc_c)
    cosang = np.sum(pred_e * acc_e, 1) / (np.linalg.norm(pred_e, axis=1) * np.linalg.norm(acc_e, axis=1))
    feats += [acc_e, acc_c, np.degrees(np.arccos(np.clip(cosang, -1, 1)))[:, None]]
    return np.concatenate(feats, 1).astype(np.float32), {"p": pe, "R": Re}


def frame_feature_names() -> list[str]:
    n = []
    for i in (1, 2):
        n += [f"rx{i}_S{a}{b}" for a in range(3) for b in range(3)] + [f"rx{i}_logB"]
        n += [f"rx{i}_p{c}" for c in "xyz"] + [f"rx{i}_r6_{k}" for k in range(6)] + [f"rx{i}_logres"]
    n += [f"nat_p{c}" for c in "xyz"] + [f"nat_r6_{k}" for k in range(6)]
    n += ["rigid_sep_err", "rigid_rel_angle"]
    n += [f"ref_dS{a}{b}" for a in range(3) for b in range(3)] + ["ref_dS_norm"]
    n += [f"imu_e_a{c}" for c in "xyz"] + [f"imu_c_a{c}" for c in "xyz"] + ["imu_tilt_disc_deg"]
    return n


FRAME_COLS = frame_feature_names()
FEATURE_COLS = [f"{c}_l{lag}" for lag in range(LAGS) for c in FRAME_COLS]
TARGET_COLS = ["dp_x", "dp_y", "dp_z", "dth_x", "dth_y", "dth_z"]


def simulate_sequences(P: np.ndarray, R: np.ndarray, env: Environment, rng, keep_from: int = LAGS - 1) -> pd.DataFrame:
    """P (n,T,3), R (n,T,3,3) -> rows for frames t >= keep_from, with lag features 0..LAGS-1."""
    n, T = P.shape[:2]
    F, nat = _frame_features(P.reshape(-1, 3), R.reshape(-1, 3, 3), env, rng)
    F = F.reshape(n, T, -1)
    natp, natR = nat["p"].reshape(n, T, 3), nat["R"].reshape(n, T, 3, 3)
    rows = []
    for t in range(keep_from, T):
        lagged = np.concatenate([F[:, t - lag] for lag in range(LAGS)], 1)
        dp = P[:, t] - natp[:, t]
        dth = np.degrees(matrix_to_rotvec(R[:, t] @ np.transpose(natR[:, t], (0, 2, 1))))
        meta = np.c_[np.arange(n), np.full(n, t), P[:, t], to_6d(R[:, t]), natp[:, t], to_6d(natR[:, t])]
        rows.append(np.c_[lagged, dp, dth, meta])
    cols = FEATURE_COLS + TARGET_COLS + ["seq", "frame"] + [f"true_p{c}" for c in "xyz"] + [f"true_r6_{k}" for k in range(6)] \
        + [f"nat_px_{c}" for c in "xyz"] + [f"nat_r6x_{k}" for k in range(6)]
    arr = np.concatenate(rows)
    df = pd.DataFrame(arr, columns=cols)
    df[FEATURE_COLS + TARGET_COLS] = df[FEATURE_COLS + TARGET_COLS].astype(np.float32)
    extra = pd.DataFrame({"env_id": env.env_id, "family": env.family,
                          "pos_err_mm": np.linalg.norm(arr[:, len(FEATURE_COLS):len(FEATURE_COLS) + 3], axis=1),
                          "ang_err_deg": np.linalg.norm(arr[:, len(FEATURE_COLS) + 3:len(FEATURE_COLS) + 6], axis=1)})
    return pd.concat([df, extra], axis=1)


def static_poses_df(P: np.ndarray, R: np.ndarray, env: Environment, rng) -> pd.DataFrame:
    """Calibration captures: the operator holds each pose for LAGS frames (lag features are independent re-reads)."""
    Pt = np.repeat(P[:, None], LAGS, 1)
    Rt = np.repeat(R[:, None], LAGS, 1)
    return simulate_sequences(Pt, Rt, env, rng, keep_from=LAGS - 1)
