import numpy as np
import pandas as pd

from emtrack.protocols import _beam_cos, minutes, select
from emtrack.simulate import (
    FEATURE_COLS,
    STRESS_FAMILIES,
    TARGET_COLS,
    sample_environment,
    simulate_sequences,
    static_poses_df,
    trajectories,
)


def test_feature_layout():
    assert len(FEATURE_COLS) == 272 and len(set(FEATURE_COLS)) == 272


def test_clean_environment_has_tiny_targets_and_full_layout():
    rng = np.random.default_rng(0)
    env = sample_environment("c", "clean", rng)
    P, R = trajectories(rng, 6, 8)
    df = simulate_sequences(P, R, env, rng)
    assert set(FEATURE_COLS + TARGET_COLS) <= set(df.columns)
    assert df.pos_err_mm.median() < 2.0 and df.ang_err_deg.median() < 0.5
    assert len(df) == 6 * (8 - 3)


def test_all_families_sample():
    rng = np.random.default_rng(0)
    for fam in ["clinic"] + STRESS_FAMILIES:
        env = sample_environment("e", fam, rng)
        assert (len(env.metals) == 0) == (fam == "clean")
        assert np.isclose(np.linalg.norm(env.gravity), 1.0)


def _pool(n=400):
    rng = np.random.default_rng(3)
    env = sample_environment("p", "clinic", rng)
    from emtrack.simulate import WORK_HI, WORK_LO, sample_orientations
    return static_poses_df(rng.uniform(WORK_LO, WORK_HI, (n, 3)), sample_orientations(rng, n), env, rng)


def test_protocols_sizes_and_bias():
    pool = _pool()
    rng = np.random.default_rng(0)
    for proto in ("random", "coverage", "convenient", "stratified"):
        assert len(select(pool, 40, proto, rng)) == 40
    tilt = lambda d: np.degrees(np.arccos(np.clip(_beam_cos(d), -1, 1))).mean()  # noqa: E731
    assert tilt(select(pool, 40, "convenient", rng)) < 0.6 * tilt(pool)    # biased toward upright poses
    cov = select(pool, 40, "coverage", rng)
    spread = lambda d: d[["true_px", "true_py", "true_pz"]].std().mean()  # noqa: E731
    assert spread(cov) > spread(select(pool, 40, "random", np.random.default_rng(1)))
    assert minutes(900) == 60.0
    assert isinstance(select(pool, 0, "random", rng), pd.DataFrame)
