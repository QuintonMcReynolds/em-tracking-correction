"""Meta-train the encoder-decoder attention model: each episode = one training site, K calibration poses from its
pool as context, 64 trajectory frames from the same site as queries. K varies per episode (0..320), so one model
serves every collection budget, including K = 0."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emtrack.attention import SiteAttention  # noqa: E402
from emtrack.data import load  # noqa: E402
from emtrack.simulate import FEATURE_COLS, TARGET_COLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K_CHOICES = [0, 10, 20, 40, 80, 160, 320]


def site_tensors(df, pool, cols):
    out = {}
    for env, g in df.groupby("env_id"):
        p = pool[pool.env_id == env]
        if len(g) < 16:      # site where the tracker is almost never valid: nothing to learn from
            continue
        out[env] = (g[cols].to_numpy(np.float32), g[TARGET_COLS].to_numpy(np.float32),
                    p[cols].to_numpy(np.float32), p[TARGET_COLS].to_numpy(np.float32))
    return out


def main(steps: int = 12000, seed: int = 0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cols = FEATURE_COLS
    tr, trp = load("train"), load("train", pool=True)
    va, vap = load("val"), load("val", pool=True)
    X = tr[cols].to_numpy(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ysd = tr[TARGET_COLS].to_numpy(np.float32).std(0) + 1e-6
    norm = {"mu": torch.tensor(mu), "sd": torch.tensor(sd), "ysd": torch.tensor(ysd)}
    S_tr, S_va = site_tensors(tr, trp, cols), site_tensors(va, vap, cols)
    envs = list(S_tr)
    net = SiteAttention(len(cols)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    Bsz, Q, _Kmax = 16, 64, max(K_CHOICES)

    def batch(S, env_list, K):
        cx = np.zeros((Bsz, max(K, 1), len(cols)), np.float32)
        cy = np.zeros((Bsz, max(K, 1), 6), np.float32)
        cm = np.ones((Bsz, max(K, 1)), bool)
        qx = np.zeros((Bsz, Q, len(cols)), np.float32)
        qy = np.zeros((Bsz, Q, 6), np.float32)
        for b in range(Bsz):
            fx, fy, px, py = S[env_list[rng.integers(len(env_list))]]
            qi = rng.choice(len(fx), Q, replace=len(fx) < Q)
            qx[b], qy[b] = fx[qi], fy[qi]
            k = min(K, len(px))  # some sites have fewer valid calibration poses: pad + mask the rest
            if k:
                ci = rng.choice(len(px), k, replace=False)
                cx[b, :k], cy[b, :k], cm[b, :k] = px[ci], py[ci], False
        t = lambda a: torch.tensor(a, device=dev)  # noqa: E731
        return ((t(cx) - norm["mu"].to(dev)) / norm["sd"].to(dev), t(cy) / norm["ysd"].to(dev), t(cm),
                (t(qx) - norm["mu"].to(dev)) / norm["sd"].to(dev), t(qy) / norm["ysd"].to(dev))

    t0, log = time.time(), []
    for step in range(1, steps + 1):
        net.train()
        K = int(rng.choice(K_CHOICES))
        cx, cy, cm, qx, qy = batch(S_tr, envs, K)
        loss = torch.nn.functional.huber_loss(net(cx, cy, cm, qx), qy, delta=1.0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == steps:
            net.eval()
            val = {}
            with torch.no_grad():
                for K in (0, 40, 320):
                    errs = []
                    for _ in range(8):
                        cx, cy, cm, qx, qy = batch(S_va, list(S_va), K)
                        pr = net(cx, cy, cm, qx) * norm["ysd"].to(dev)
                        errs.append(torch.linalg.norm(pr[..., :3] - qy[..., :3] * norm["ysd"].to(dev)[:3], dim=-1).median().item())
                    val[K] = round(float(np.mean(errs)), 2)
            log.append({"step": step, "loss": round(loss.item(), 4), "val_median_pos_mm": val, "sec": round(time.time() - t0)})
            print(json.dumps(log[-1]), flush=True)
    run_dir = ROOT / "runs" / "site_attention"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.cpu().state_dict(), "mu": mu, "sd": sd, "ysd": ysd, "cols": cols,
                "arch": {"d_feat": len(cols)}, "log": log, "steps": steps, "K_choices": K_CHOICES}, run_dir / "model.pt")
    (run_dir / "train_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
