"""Global correction models (no data from the target site): LightGBM, XGBoost, MLP.

Trained on 240 clinic environments, early-stopped on 40 *other* environments (grouped by environment, never by row).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emtrack.data import DATA, EVAL_SPLITS, load  # noqa: E402
from emtrack.lineage import record_run  # noqa: E402
from emtrack.metrics import per_env, summarize  # noqa: E402
from emtrack.simulate import FEATURE_COLS, TARGET_COLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def fit_lgb(Xtr, Ytr, Xva, Yva, cols, params):
    models = []
    for j in range(Ytr.shape[1]):
        dtr = lgb.Dataset(Xtr, Ytr[:, j], feature_name=cols, free_raw_data=False)
        dva = lgb.Dataset(Xva, Yva[:, j], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append(m)
    return models


def pred_lgb(models, X):
    return np.stack([m.predict(X, num_iteration=m.best_iteration) for m in models], 1)


def fit_xgb(Xtr, Ytr, Xva, Yva):
    m = xgb.XGBRegressor(n_estimators=3000, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.5,
                         min_child_weight=20, tree_method="hist", early_stopping_rounds=100, multi_strategy="one_output_per_tree")
    m.fit(Xtr, Ytr, eval_set=[(Xva, Yva)], verbose=False)
    return m


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_out=6, h=512):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, h), torch.nn.GELU(), torch.nn.Linear(h, h), torch.nn.GELU(),
                                       torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, Ytr, Xva, Yva, epochs=40, seed=0):
    torch.manual_seed(seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    ysd = Ytr.std(0) + 1e-6
    net = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    Xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor(Ytr / ysd, dtype=torch.float32)
    Xv = torch.tensor((Xva - mu) / sd, dtype=torch.float32, device=dev)
    best, best_state = 1e9, None
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 1024):
            b = perm[i:i + 1024]
            loss = torch.nn.functional.huber_loss(net(Xt[b].to(dev)), Yt[b].to(dev), delta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = net(Xv).cpu().numpy() * ysd
        v = float(np.median(np.linalg.norm(pv[:, :3] - Yva[:, :3], axis=1)))
        if v < best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    net = net.cpu().eval()
    return {"net": net, "mu": mu, "sd": sd, "ysd": ysd}


def pred_mlp(m, X):
    with torch.no_grad():
        return m["net"](torch.tensor((X - m["mu"]) / m["sd"], dtype=torch.float32)).numpy() * m["ysd"]


def evaluate(predict, cols) -> dict:
    res = {}
    for s in EVAL_SPLITS:
        df = load(s)
        res[s] = summarize(per_env(df, predict(df[cols].to_numpy(np.float32))))
    return res


def main(which: str = "lgb,xgb,mlp", feature_set: str = "all"):
    tr, va = load("train"), load("val")
    cols = FEATURE_COLS
    if feature_set == "no_imu":
        cols = [c for c in FEATURE_COLS if not c.startswith("imu_")]
    elif feature_set == "no_ref":
        cols = [c for c in FEATURE_COLS if not c.startswith("ref_")]
    elif feature_set == "lag0":
        cols = [c for c in FEATURE_COLS if c.endswith("_l0")]
    Xtr, Ytr = tr[cols].to_numpy(np.float32), tr[TARGET_COLS].to_numpy(np.float32)
    Xva, Yva = va[cols].to_numpy(np.float32), va[TARGET_COLS].to_numpy(np.float32)
    out = {}
    for w in which.split(","):
        t0 = time.time()
        run_id = f"global_{w}_{feature_set}"
        run_dir = ROOT / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        if w == "lgb":
            params = {"objective": "huber", "alpha": 3.0, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
                      "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "verbose": -1,
                      "num_threads": 8}
            models = fit_lgb(Xtr, Ytr, Xva, Yva, cols, params)
            for j, m in enumerate(models):
                m.save_model(str(run_dir / f"lgb_{TARGET_COLS[j]}.txt"))
            pred = lambda X: pred_lgb(models, X)  # noqa: E731
            cfg = {"model": "lightgbm", "params": params, "features": feature_set, "n_features": len(cols)}
            art = {f"lgb_{t}": run_dir / f"lgb_{t}.txt" for t in TARGET_COLS}
        elif w == "xgb":
            m = fit_xgb(Xtr, Ytr, Xva, Yva)
            m.save_model(str(run_dir / "xgb.json"))
            pred = m.predict
            cfg = {"model": "xgboost", "params": m.get_params(), "features": feature_set, "n_features": len(cols)}
            art = {"xgb": run_dir / "xgb.json"}
        else:
            m = fit_mlp(Xtr, Ytr, Xva, Yva)
            torch.save({"state": m["net"].state_dict(), "mu": m["mu"], "sd": m["sd"], "ysd": m["ysd"], "cols": cols},
                       run_dir / "model.pt")
            pred = lambda X, m=m: pred_mlp(m, X)  # noqa: E731
            cfg = {"model": "mlp", "hidden": 512, "layers": 3, "loss": "huber", "features": feature_set, "n_features": len(cols)}
            art = {"mlp": run_dir / "model.pt"}
        metrics = evaluate(pred, cols)
        metrics["train_seconds"] = round(time.time() - t0, 1)
        record_run(run_id, cfg, DATA, metrics, art)
        out[run_id] = metrics
        print(run_id, f"{metrics['train_seconds']}s", json.dumps({s: {k: round(v, 2) for k, v in metrics[s].items() if k in
              ('pos_med_mm', 'ang_med_deg', 'envs_meeting_target')} for s in EVAL_SPLITS}), flush=True)
    return out


if __name__ == "__main__":
    main(*sys.argv[1:])
