"""The tracker's native pose solver: batched Levenberg-Marquardt fitting the *undistorted* dipole model to readings.

Near metal the readings no longer match the model, so the least-squares optimum is biased: that bias is the
distortion the learned models correct. Runs in float64 on CPU (MPS has no float64).
"""
from __future__ import annotations

import numpy as np
import torch

MM = 1e-3


def _rotvec_to_matrix(rv: torch.Tensor) -> torch.Tensor:
    """Rodrigues on the unnormalised skew matrix: smooth (and differentiable) through zero rotation."""
    x, y, z = rv[..., 0], rv[..., 1], rv[..., 2]
    o = torch.zeros_like(x)
    K = torch.stack([torch.stack([o, -z, y], -1), torch.stack([z, o, -x], -1), torch.stack([-y, x, o], -1)], -2)
    th2 = (rv * rv).sum(-1)[..., None, None] + 1e-24
    th = torch.sqrt(th2)
    a = torch.sin(th) / th
    b = (1 - torch.cos(th)) / th2
    eye = torch.eye(3, dtype=rv.dtype).expand(K.shape)
    return eye + a * K + b * (K @ K)


def _model(x: torch.Tensor, R0: torch.Tensor) -> torch.Tensor:
    """x = (p_mm(3), delta_rotvec(3)) relative to a reference orientation R0 -> predicted S flattened (9)."""
    p, dr = x[..., :3], x[..., 3:]
    R = _rotvec_to_matrix(dr) @ R0
    r = p * MM
    d = torch.linalg.norm(r, dim=-1, keepdim=True)
    rh = r / d
    B = (3 * rh[..., :, None] * rh[..., None, :] - torch.eye(3, dtype=x.dtype)) / d[..., None] ** 3  # (...,3,3) col j = coil j
    S = R.transpose(-1, -2) @ B
    return S.reshape(S.shape[:-2] + (9,))


_H = torch.tensor([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 1e-6], dtype=torch.float64)  # mm, rad


def _jacobian(f, x: torch.Tensor) -> torch.Tensor:
    """Central finite differences, all 6 directions evaluated in one batched call (much faster than vmap(jacfwd))."""
    E = torch.diag(_H)                                           # (6,6)
    xs = torch.cat([x[:, None, :] + E, x[:, None, :] - E], 1)   # (n,12,6)
    y = f_batched(f, xs)                                         # (n,12,9)
    return ((y[:, :6] - y[:, 6:]) / (2 * _H)[None, :, None]).transpose(1, 2)


def f_batched(f, xs: torch.Tensor) -> torch.Tensor:
    n, k, d = xs.shape
    return f(xs.reshape(n * k, d), repeat=k).reshape(n, k, 9)


def solve(S: np.ndarray, p0: np.ndarray, R0: np.ndarray, iters: int = 25, chunk: int = 50000):
    """Fit pose to readings. S (N,3,3), initial p0 (N,3) mm, R0 (N,3,3). Returns p (N,3), R (N,3,3), rms residual (N,)."""
    out_p, out_R, out_res = [], [], []
    for i in range(0, len(S), chunk):
        s = torch.tensor(S[i:i + chunk].reshape(-1, 9), dtype=torch.float64)
        scale = s.norm(dim=-1, keepdim=True)
        R0t = torch.tensor(R0[i:i + chunk], dtype=torch.float64)
        x = torch.cat([torch.tensor(p0[i:i + chunk], dtype=torch.float64), torch.zeros(len(s), 3, dtype=torch.float64)], -1)
        lam = torch.full((len(s), 1, 1), 1e-3, dtype=torch.float64)
        def f(xx, repeat: int = 1):
            if repeat == 1:
                return _model(xx, R0t) / scale
            return _model(xx, R0t.repeat_interleave(repeat, 0)) / scale.repeat_interleave(repeat, 0)

        def resid(xx):
            return f(xx) - s / scale

        r = resid(x)
        cost = (r ** 2).sum(-1)
        for _ in range(iters):
            J = _jacobian(f, x)  # (n,9,6)
            JtJ = J.transpose(1, 2) @ J
            g = J.transpose(1, 2) @ r[..., None]
            diag = torch.diag_embed(torch.diagonal(JtJ, dim1=1, dim2=2))
            step = -torch.linalg.solve(JtJ + lam * (diag + 1e-12 * torch.eye(6, dtype=x.dtype)), g)[..., 0]
            xn = x + step
            rn = resid(xn)
            cn = (rn ** 2).sum(-1)
            ok = cn < cost
            x = torch.where(ok[:, None], xn, x)
            r = torch.where(ok[:, None], rn, r)
            cost = torch.where(ok, cn, cost)
            lam = torch.where(ok[:, None, None], lam * 0.3, lam * 10).clamp(1e-9, 1e6)
        R = _rotvec_to_matrix(x[:, 3:]) @ R0t
        out_p.append(x[:, :3].numpy())
        out_R.append(R.numpy())
        out_res.append(torch.sqrt(cost / 9).numpy())
    return np.concatenate(out_p), np.concatenate(out_R), np.concatenate(out_res)
