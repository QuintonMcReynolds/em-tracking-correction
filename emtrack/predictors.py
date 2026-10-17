"""Load trained models behind one interface: predict(site_rows, calib_rows) -> (N,6) corrections."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from .attention import SiteAttention
from .simulate import TARGET_COLS
from .sitefit import fit_fieldmap, predict_fieldmap

ROOT = Path(__file__).resolve().parents[1]


class GlobalLGB:
    def __init__(self, run: str = "global_lgb_all"):
        d = ROOT / "runs" / run
        self.models = [lgb.Booster(model_file=str(d / f"lgb_{t}.txt")) for t in TARGET_COLS]
        self.cols = self.models[0].feature_name()

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.cols].to_numpy(np.float32)
        return np.stack([m.predict(X) for m in self.models], 1)


class Attention:
    def __init__(self, path: Path = ROOT / "runs" / "site_attention" / "model.pt", device: str | None = None):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.cols = ck["cols"]
        self.net = SiteAttention(ck["arch"]["d_feat"])
        self.net.load_state_dict(ck["state"])
        self.dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.net = self.net.to(self.dev).eval()
        self.mu, self.sd, self.ysd = (torch.tensor(ck[k], device=self.dev) for k in ("mu", "sd", "ysd"))

    @torch.no_grad()
    def __call__(self, df: pd.DataFrame, calib: pd.DataFrame | None, chunk: int = 4096) -> np.ndarray:
        K = 0 if calib is None else len(calib)
        if calib is not None and K:
            cx = (torch.tensor(calib[self.cols].to_numpy(np.float32), device=self.dev) - self.mu) / self.sd
            cy = torch.tensor(calib[TARGET_COLS].to_numpy(np.float32), device=self.dev) / self.ysd
            cm = torch.zeros(1, K, dtype=torch.bool, device=self.dev)
        else:
            cx = torch.zeros(1, 1, len(self.cols), device=self.dev)
            cy = torch.zeros(1, 1, 6, device=self.dev)
            cm = torch.ones(1, 1, dtype=torch.bool, device=self.dev)
        out = []
        X = df[self.cols].to_numpy(np.float32)
        for i in range(0, len(X), chunk):
            q = (torch.tensor(X[i:i + chunk], device=self.dev) - self.mu) / self.sd
            out.append((self.net(cx[None] if cx.dim() == 2 else cx.reshape(1, -1, len(self.cols)),
                                 cy.reshape(1, -1, 6), cm.reshape(1, -1), q[None])[0] * self.ysd).cpu().numpy())
        return np.concatenate(out)


def fieldmap(df: pd.DataFrame, calib: pd.DataFrame) -> np.ndarray:
    return predict_fieldmap(fit_fieldmap(calib), df)


def hybrid(glob: GlobalLGB, df: pd.DataFrame, calib: pd.DataFrame) -> np.ndarray:
    base = glob(df)
    if len(calib) < 10:
        return base
    resid = calib[TARGET_COLS].to_numpy() - glob(calib)
    return base + predict_fieldmap(fit_fieldmap(calib, target=resid), df)


def attention_fieldmap(attn: Attention, df: pd.DataFrame, calib: pd.DataFrame) -> np.ndarray:
    """Attention prediction, then a field map fit to the attention model's residuals on the calibration poses.
    The attention model generalises from little data; the field map keeps improving as the budget grows."""
    base = attn(df, calib)
    if len(calib) < 60:
        return base
    resid = calib[TARGET_COLS].to_numpy() - attn(calib, calib)
    return base + predict_fieldmap(fit_fieldmap(calib, target=resid), df)
