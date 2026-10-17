"""Two uses of the shift detector.

A. Spatial gate. Correction model trained on the central volume only (as in split_study.py); detector fit on the same
   training rows. In the outer shell, frames the detector flags fall back to the native pose.
B. Environment shift. Detector fit on all training environments, threshold = 99th percentile on validation
   environments. Alarm rates per stress family, and how well a site's alarm rate predicts that the global
   correction will miss the +-10 mm / 1 deg target there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from split_study import fit_predict, inner  # noqa: E402

from emtrack.data import EVAL_SPLITS, load  # noqa: E402
from emtrack.metrics import per_env, summarize  # noqa: E402
from emtrack.predictors import GlobalLGB  # noqa: E402
from emtrack.shift import ShiftDetector, SupportGate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    res = {}
    # ---- A: spatial gate
    d = load("train")
    envs = d.env_id.unique()
    rng = np.random.default_rng(0)
    rng.shuffle(envs)
    te = d.env_id.isin(set(envs[:48]))
    tr_s, te_s = d[~te & inner(d)], d[te & ~inner(d)]
    pred = fit_predict(tr_s, te_s)
    # the gate's threshold comes from held-out rows *inside* the trained volume (in-distribution), never the shell
    calib_in = d[te & inner(d)]
    maha = ShiftDetector().fit(tr_s, calib=calib_in)
    gate = SupportGate().fit(tr_s, calib=calib_in.sample(min(len(calib_in), 5000), random_state=0))
    flag = gate.flag(te_s) | maha.flag(te_s)
    gated = np.where(flag[:, None], 0.0, pred)
    res["spatial_gate"] = {
        "flag_rate_outer_shell": float(flag.mean()),
        "flag_rate_outer_shell_mahalanobis_only": float(maha.flag(te_s).mean()),
        "flag_rate_inner_holdout": float((gate.flag(calib_in) | maha.flag(calib_in)).mean()),
        "native": summarize(per_env(te_s, None)), "ungated": summarize(per_env(te_s, pred)),
        "gated": summarize(per_env(te_s, gated))}
    # ---- B: environment shift
    det = ShiftDetector().fit(load("train"), calib=load("val"))
    glob = GlobalLGB()
    rates, y_fail, s_env = {}, [], []
    for s in EVAL_SPLITS:
        df = load(s)
        f = det.flag(df)
        rates[s] = float(f.mean())
        pe = per_env(df, glob(df))
        fr = df.assign(f=f).groupby("env_id").f.mean()
        y_fail += list(~pe.set_index("env_id").loc[fr.index, "meets_target"].to_numpy())
        s_env += list(fr.to_numpy())
    res["env_shift"] = {"threshold": det.threshold, "alarm_rate": rates,
                        "auroc_alarm_rate_predicts_global_miss": float(roc_auc_score(y_fail, s_env)),
                        "n_envs": len(y_fail), "n_miss": int(sum(y_fail))}
    (ROOT / "reports" / "shift_study.json").write_text(json.dumps(res, indent=2))
    a = res["spatial_gate"]
    print(f"spatial: flag outer {a['flag_rate_outer_shell']:.0%} / inner {a['flag_rate_inner_holdout']:.1%}; "
          f"pos_med native {a['native']['pos_med_mm']:.1f}, ungated {a['ungated']['pos_med_mm']:.1f}, gated {a['gated']['pos_med_mm']:.1f} mm")
    print("alarm rates", {k: round(v, 3) for k, v in rates.items()}, "AUROC", round(res["env_shift"]["auroc_alarm_rate_predicts_global_miss"], 3))


if __name__ == "__main__":
    main()
