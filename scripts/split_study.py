"""How much does the split strategy flatter the model? Same LightGBM, three ways of holding out data.

random   : rows shuffled; every test environment also appears in training (leaks the metal configuration)
grouped  : whole environments held out (GroupKFold-style): the deployment question 'new site, no calibration'
spatial  : held-out environments AND held-out region of space: train on the central volume, test on the outer
           shell (extrapolation in position, which is where the field changes fastest)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emtrack.data import load  # noqa: E402
from emtrack.metrics import per_env, summarize  # noqa: E402
from emtrack.simulate import FEATURE_COLS, TARGET_COLS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"objective": "huber", "alpha": 3.0, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "num_threads": 8}


def fit_predict(tr, te, rounds=600):
    X, Xt = tr[FEATURE_COLS].to_numpy(np.float32), te[FEATURE_COLS].to_numpy(np.float32)
    return np.stack([lgb.train(PARAMS, lgb.Dataset(X, tr[t].to_numpy()), rounds).predict(Xt) for t in TARGET_COLS], 1)


def inner(df):
    return (df.true_px.abs() < 120) & (df.true_py.abs() < 120) & (df.true_pz.between(450, 650))


def main(seed: int = 0):
    rng = np.random.default_rng(seed)
    d = load("train")
    envs = d.env_id.unique()
    rng.shuffle(envs)
    test_envs = set(envs[:48])
    res = {}
    # random rows (leaky): 20% of rows, from all environments
    m = rng.random(len(d)) < 0.2
    res["random_rows"] = summarize(per_env(d[m], fit_predict(d[~m], d[m])))
    # grouped by environment
    te = d.env_id.isin(test_envs)
    res["grouped_envs"] = summarize(per_env(d[te], fit_predict(d[~te], d[te])))
    # spatial + grouped: central volume of training envs -> outer shell of held-out envs
    tr_s, te_s = d[~te & inner(d)], d[te & ~inner(d)]
    res["grouped_spatial"] = summarize(per_env(te_s, fit_predict(tr_s, te_s)))
    res["native_on_grouped_test"] = summarize(per_env(d[te], None))
    res["native_on_spatial_test"] = summarize(per_env(te_s, None))
    out = ROOT / "reports" / "split_study.json"
    out.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(f"{k:24s} pos_med {v['pos_med_mm']:.2f} mm  ang_med {v['ang_med_deg']:.2f} deg  meets {v['envs_meeting_target']:.0%}")


if __name__ == "__main__":
    main()
