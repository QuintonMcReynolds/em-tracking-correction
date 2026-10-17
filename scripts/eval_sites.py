"""Site-adaptation evaluation: methods x calibration budget K x capture protocol, per unseen environment.

Writes reports/site_eval.parquet (one row per env x method x K x protocol x repeat) and reports/site_eval_summary.json.
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emtrack.data import EVAL_SPLITS, load  # noqa: E402
from emtrack.metrics import per_env  # noqa: E402
from emtrack.predictors import Attention, GlobalLGB, attention_fieldmap, fieldmap, hybrid  # noqa: E402
from emtrack.protocols import minutes, select  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
K_GRID = [0, 15, 30, 60, 120, 240, 450, 900]
STATS = ["pos_med", "pos_p95", "ang_med", "ang_p95", "meets_target"]


def main(splits: str = ",".join(EVAL_SPLITS), protocols: str = "random,coverage,convenient", repeats: int = 2,
         out_name: str = "site_eval"):
    glob, attn = GlobalLGB(), Attention()
    rows = []
    t0 = time.time()
    for split in splits.split(","):
        df_all, pool_all = load(split), load(split, pool=True)
        for env, df in df_all.groupby("env_id"):
            pool = pool_all[pool_all.env_id == env]
            base = {"split": split, "env_id": env}
            for name, pred in (("native", None), ("global_lgb", glob(df))):
                e = per_env(df, pred).iloc[0][STATS].to_dict()
                rows.append({**base, "method": name, "K": 0, "minutes": 0.0, "protocol": "-", "rep": 0, **e})
            for proto in protocols.split(","):
                for rep in range(int(repeats) if proto != "coverage" else 1):
                    rng = np.random.default_rng(1000 * rep + zlib.crc32(env.encode()))
                    for K in K_GRID:
                        calib = select(pool, K, proto, rng)
                        preds = {"site_attention": attn(df, calib if K else None)}
                        if K >= 15:
                            preds["fieldmap"] = fieldmap(df, calib)
                            preds["global+fieldmap"] = hybrid(glob, df, calib)
                        if K >= 60:
                            preds["attention+fieldmap"] = attention_fieldmap(attn, df, calib)
                        for name, pred in preds.items():
                            e = per_env(df, pred).iloc[0][STATS].to_dict()
                            rows.append({**base, "method": name, "K": K, "minutes": minutes(K), "protocol": proto, "rep": rep, **e})
        print(split, f"{time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    res.to_parquet(out / f"{out_name}.parquet", index=False)
    if out_name != "site_eval":
        return res
    summ = (res.groupby(["split", "method", "protocol", "K"])
            .agg(pos_med=("pos_med", "median"), ang_med=("ang_med", "median"), meets=("meets_target", "mean")).reset_index())
    (out / "site_eval_summary.json").write_text(summ.to_json(orient="records", indent=1))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:])
