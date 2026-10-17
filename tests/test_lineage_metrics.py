import json

import numpy as np
import pandas as pd
import pytest

from emtrack import lineage
from emtrack.metrics import corrected_errors, per_env
from emtrack.simulate import TARGET_COLS


def test_manifest_detects_tampering(tmp_path):
    pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / "x.parquet")
    m = lineage.write_manifest(tmp_path, {"seed": 1})
    assert lineage.verify_manifest(tmp_path)["dataset_hash"] == m["dataset_hash"]
    pd.DataFrame({"a": [1, 3]}).to_parquet(tmp_path / "x.parquet")
    with pytest.raises(RuntimeError):
        lineage.verify_manifest(tmp_path)
    assert json.loads((tmp_path / "manifest.json").read_text())["config"] == {"seed": 1}


def test_perfect_correction_gives_zero_error():
    rng = np.random.default_rng(0)
    y = rng.normal(0, [5, 5, 5, 1, 1, 1], (50, 6))
    df = pd.DataFrame(y, columns=TARGET_COLS).assign(env_id="e", family="f")
    pos, ang = corrected_errors(df, y)
    assert pos.max() < 1e-9 and ang.max() < 1e-4
    pos0, _ = corrected_errors(df, None)
    assert np.allclose(pos0, np.linalg.norm(y[:, :3], axis=1))
    t = per_env(df, y)
    assert bool(t.meets_target.iloc[0])
