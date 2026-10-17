"""Generate every split in configs/data.yaml -> data/<version>/{split}.parquet, {split}_pool.parquet, envs.json, manifest.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from emtrack.lineage import write_manifest  # noqa: E402
from emtrack.simulate import (  # noqa: E402
    WORK_HI,
    WORK_LO,
    sample_environment,
    sample_orientations,
    simulate_sequences,
    static_poses_df,
    trajectories,
)

ROOT = Path(__file__).resolve().parents[1]


def main(cfg_path: str = "configs/data.yaml"):
    cfg = yaml.safe_load((ROOT / cfg_path).read_text())
    out = ROOT / "data" / cfg["version"]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    envs_meta = {}
    for split, sc in cfg["splits"].items():
        t0 = time.time()
        rows, pools = [], []
        for k in range(sc["n_envs"]):
            env = sample_environment(f"{split}-{k:03d}", sc["family"], rng)
            envs_meta[env.env_id] = env.to_dict()
            P, R = trajectories(rng, sc["n_traj"], sc["T"])
            rows.append(simulate_sequences(P, R, env, rng))
            n = sc["pool"]  # static calibration poses available at this site (protocols choose from these)
            pools.append(static_poses_df(rng.uniform(WORK_LO, WORK_HI, (n, 3)), sample_orientations(rng, n), env, rng))
        pd.concat(rows, ignore_index=True).to_parquet(out / f"{split}.parquet", index=False)
        pd.concat(pools, ignore_index=True).to_parquet(out / f"{split}_pool.parquet", index=False)
        print(f"{split}: {sum(len(r) for r in rows)} rows, {sum(len(p) for p in pools)} pool rows ({time.time() - t0:.0f}s)", flush=True)
    (out / "envs.json").write_text(json.dumps(envs_meta, indent=1))
    m = write_manifest(out, cfg)
    print("dataset_hash", m["dataset_hash"])


if __name__ == "__main__":
    main(*sys.argv[1:])
