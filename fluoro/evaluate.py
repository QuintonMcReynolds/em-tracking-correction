"""Compare denoisers against the current baseline on held-out radiographs, and screen for hallucinated structure.

Methods (display domain, mu / MU_MAX):
  single           one frame, no processing
  gauss3           3x3 Gaussian on one frame
  avg4             mean of 4 frames                         (4x the dose of one frame)
  gauss3+avg4      the current device baseline
  n2n_single       Noise2Noise U-Net on one frame
  n2n_avg4         Noise2Noise U-Net on the 4-frame mean
  sup_single       supervised U-Net on one frame (upper bound: needs clean targets)
Screens:
  fracture SSIM    structural similarity inside the annotated fracture boxes (is diagnostic detail kept?)
  false edges      edges in the output > 2 px from any edge in the clean image, per 1000 clean-edge pixels
  wire test        a 1-px wire of low contrast added to the scene: recovered contrast / true contrast
  flat field       uniform phantom: residual structure (high-pass std) and mean bias; ideal output is flat
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.feature import canny
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluoro.model import UNet, make_input  # noqa: E402
from fluoro.sim import DOSES, frames, load_split, to_display  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["single", "gauss3", "avg4", "gauss3+avg4", "n2n_single", "n2n_avg4", "sup_single"]


def gauss3(y):
    return gaussian_filter(y, 0.85, truncate=1.2)        # 3x3 support


class Nets:
    def __init__(self):
        self.m = {}
        for k in ("n2n_single", "n2n_avg4", "sup_single"):
            p = ROOT / "runs" / f"denoise_{k}" / "model.pt"
            if p.exists():
                ck = torch.load(p, map_location="cpu", weights_only=False)
                net = UNet()
                net.load_state_dict(ck["state"])
                self.m[k] = (net.eval(), ck["n_frames"])

    @torch.no_grad()
    def __call__(self, k, y, n0):
        net, nf = self.m[k]
        H, W = y.shape
        ph, pw = (-H) % 4, (-W) % 4
        yp = np.pad(y, ((0, ph), (0, pw)), mode="reflect")
        x = torch.tensor(yp[None, None])
        out = net(make_input(x, torch.tensor([float(n0)]), nf))[0, 0].numpy()
        return out[:H, :W]


def apply_all(nets, F, n0):
    """F (8,H,W) frames -> dict of display-domain outputs."""
    y1, y4 = F[0], F[:4].mean(0)
    out = {"single": y1, "gauss3": gauss3(y1), "avg4": y4, "gauss3+avg4": gauss3(y4)}
    for k in nets.m:
        out[k] = nets(k, y4 if k == "n2n_avg4" else y1, n0)
    return {k: to_display(v) for k, v in out.items()}


def false_edges(out, clean, sigma=2.0):
    ec, eo = canny(clean, sigma), canny(out, sigma)
    if ec.sum() == 0:
        return 0.0
    far = distance_transform_edt(~ec) > 2
    return 1000.0 * float((eo & far).sum()) / float(ec.sum())


def main(max_images: int = 61, seed: int = 0):
    nets = Nets()
    imgs, boxes = load_split("test")
    rng = np.random.default_rng(seed)
    rows = []
    for i, (I, bb) in enumerate(zip(imgs[:max_images], boxes[:max_images], strict=True)):
        for n0 in DOSES:
            F = frames(I, n0, 8, rng)
            outs = apply_all(nets, F, n0)
            for k, o in outs.items():
                r = {"img": i, "dose": n0, "method": k, "psnr": peak_signal_noise_ratio(I, o, data_range=1.0),
                     "ssim": structural_similarity(I, o, data_range=1.0), "false_edges": false_edges(o, I)}
                fs = [structural_similarity(I[y0:y1, x0:x1], o[y0:y1, x0:x1], data_range=1.0)
                      for x0, y0, x1, y1 in bb if (y1 - y0) >= 7 and (x1 - x0) >= 7]
                r["fracture_ssim"] = float(np.mean(fs)) if fs else np.nan
                rows.append(r)
            # wire test: a faint 1-px wire across the image
            for c in (0.04, 0.08):
                Iw = I.copy()
                row = I.shape[0] // 2 + 7
                Iw[row, 10:-10] = np.clip(Iw[row, 10:-10] + c, 0, 1)
                ow = apply_all(nets, frames(Iw, n0, 8, rng), n0)
                for k, o in ow.items():
                    rec = o[row, 10:-10].mean() - 0.5 * (o[row - 3, 10:-10].mean() + o[row + 3, 10:-10].mean())
                    tru = Iw[row, 10:-10].mean() - 0.5 * (Iw[row - 3, 10:-10].mean() + Iw[row + 3, 10:-10].mean())
                    rows.append({"img": i, "dose": n0, "method": k, f"wire_transfer_{c}": float(rec / tru)})
        print(f"image {i + 1}/{min(max_images, len(imgs))}", flush=True)
    # flat-field phantoms
    flat = []
    for level in (0.3, 0.6):
        I = np.full((192, 192), level, np.float32)
        for n0 in DOSES:
            outs = apply_all(nets, frames(I, n0, 8, rng), n0)
            for k, o in outs.items():
                hp = o - gaussian_filter(o, 6)
                flat.append({"level": level, "dose": n0, "method": k, "hp_std": float(hp[16:-16, 16:-16].std()),
                             "bias": float(o[16:-16, 16:-16].mean() - level),
                             "false_edges_px": int(canny(o, 2.0)[16:-16, 16:-16].sum())})
    df, fl = pd.DataFrame(rows), pd.DataFrame(flat)
    out = ROOT / "reports"
    df.to_parquet(out / "denoise_eval.parquet", index=False)
    fl.to_parquet(out / "denoise_flat.parquet", index=False)
    agg = df.groupby(["method", "dose"]).mean(numeric_only=True).drop(columns=["img"]).round(4)
    flat_agg = fl.groupby(["method", "dose"]).mean(numeric_only=True).round(5).reset_index()
    summary = {"by_method_dose": agg.reset_index().to_dict(orient="records"), "flat": flat_agg.to_dict(orient="records")}
    (out / "denoise_summary.json").write_text(json.dumps(summary, indent=1))
    print(agg.to_string())


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
