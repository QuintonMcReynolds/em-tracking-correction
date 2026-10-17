"""Static-scene fluoroscopy simulation from radiographs.

A radiograph (bright = attenuating) is treated as the line-integral map: mu = MU_MAX * I. Each fluoroscopy frame
is an independent photon-count image  N ~ Poisson(N0 * exp(-mu)) + Normal(0, sigma_e), at N0 photons per pixel
in air. Models work on normalised counts y = N / N0 (unbiased for the transmission t = exp(-mu), which is what makes
Noise2Noise valid); images are compared in the display domain I_hat = -log(t_hat) / MU_MAX.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

MU_MAX = 3.0
SIGMA_E = 2.0
DOSES = [100, 200, 400, 800]          # photons / pixel in air: low-dose fluoroscopy regime
ROOT = Path(__file__).resolve().parents[1] / "data" / "fracatlas" / "raw" / "Fracatlas"


def load_split(split: str, max_side: int = 384) -> tuple[list[np.ndarray], list[list]]:
    """Images in [0,1] (downscaled), and YOLO fracture boxes scaled to pixel coords."""
    idir = ROOT / split / "images"
    ldir = ROOT / split / ("lables" if (ROOT / split / "lables").exists() else "labels")
    imgs, boxes = [], []
    for f in sorted(idir.glob("*.jpg")):
        im = Image.open(f).convert("L")
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BILINEAR)
        a = np.asarray(im, np.float32) / 255.0
        h, w = a.shape
        bb = []
        lf = ldir / (f.stem + ".txt")
        if lf.exists():
            for line in lf.read_text().split("\n"):
                p = line.split()
                if len(p) == 5:
                    _, cx, cy, bw, bh = map(float, p)
                    bb.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h), int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
        imgs.append(a)
        boxes.append(bb)
    return imgs, boxes


def transmission(I: np.ndarray) -> np.ndarray:
    return np.exp(-MU_MAX * I)


def frames(I: np.ndarray, n0: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """n independent normalised-count frames (n, H, W) of the static scene I."""
    lam = n0 * transmission(I)
    N = rng.poisson(np.broadcast_to(lam, (n,) + I.shape)).astype(np.float32) + rng.normal(0, SIGMA_E, (n,) + I.shape)
    return (N / n0).astype(np.float32)


def to_display(y: np.ndarray) -> np.ndarray:
    return np.clip(-np.log(np.clip(y, 1e-3, None)) / MU_MAX, 0, 1)
