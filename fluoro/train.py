"""Train denoisers on random 128 px crops of FracAtlas training radiographs, dose drawn per sample.

n2n_single : input one frame, target an independent frame of the same static scene (no clean data needed)
n2n_avg4   : input mean of 4 frames, target mean of 4 other frames
sup_single : input one frame, target the clean transmission (upper bound; needs clean data, which fluoroscopy lacks)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluoro.model import UNet, make_input  # noqa: E402
from fluoro.sim import DOSES, frames, load_split, transmission  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def crops(imgs, rng, n, size=128):
    out = np.zeros((n, size, size), np.float32)
    for i in range(n):
        im = imgs[rng.integers(len(imgs))]
        h, w = im.shape
        y, x = rng.integers(0, h - size + 1), rng.integers(0, w - size + 1)
        c = im[y:y + size, x:x + size]
        if rng.random() < 0.5:
            c = c[:, ::-1]
        out[i] = c
    return out


def main(kind: str = "n2n_single", steps: int = 6000, seed: int = 0, lr_milli: float = 2.0):
    lr = lr_milli / 1000.0
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    imgs, _ = load_split("train")
    imgs = [im for im in imgs if min(im.shape) >= 128]
    net = UNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    nf = 4 if kind == "n2n_avg4" else 1
    t0, log = time.time(), []
    for step in range(1, steps + 1):
        C = crops(imgs, rng, 24)
        n0 = rng.choice(DOSES, len(C)).astype(np.float32)
        xin, tgt = [], []
        for c, d in zip(C, n0, strict=True):
            f = frames(c, d, 2 * nf, rng)
            xin.append(f[:nf].mean(0))
            tgt.append(transmission(c) if kind == "sup_single" else f[nf:].mean(0))
        x = torch.tensor(np.stack(xin)[:, None], device=dev)
        y = torch.tensor(np.stack(tgt)[:, None], device=dev)
        pred = net(make_input(x, torch.tensor(n0, device=dev), nf))
        loss = torch.mean((pred - y) ** 2)        # L2: Noise2Noise needs the mean-seeking loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            log.append({"step": step, "loss": float(loss), "sec": round(time.time() - t0)})
            print(kind, json.dumps(log[-1]), flush=True)
    out = ROOT / "runs" / f"denoise_{kind}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.cpu().state_dict(), "kind": kind, "n_frames": nf, "steps": steps, "lr": lr, "log": log}, out / "model.pt")


if __name__ == "__main__":
    # usage: train.py KIND [steps] [seed] [lr in units of 1e-4 * 10, i.e. 2 -> 2e-3]
    a = sys.argv[1:]
    main(a[0], *(int(x) for x in a[1:3]), *(float(x) for x in a[3:4]))
