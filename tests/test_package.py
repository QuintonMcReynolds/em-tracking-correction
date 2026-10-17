"""Runs only when a trained package exists (after `make package`)."""
import json
from pathlib import Path

import numpy as np
import pytest

from emtrack.package import check_inputs, unit_of

PKG = Path(__file__).resolve().parents[1] / "package"


def test_units_mapping():
    assert unit_of("nat_px_l0") == "mm" and unit_of("rigid_rel_angle_l2") == "deg" and unit_of("imu_e_ax_l1") == "g"
    assert unit_of("rx1_S01_l0") == "dimensionless"


@pytest.mark.skipif(not (PKG / "manifest.json").exists(), reason="no package built")
def test_recorded_validation_passed_and_guard_works():
    v = json.loads((PKG / "validation.json").read_text())
    assert v["passed"], v
    m = json.loads((PKG / "manifest.json").read_text())
    mid = np.array([[(f["train_p0.1"] + f["train_p99.9"]) / 2 for f in m["features"]]])
    assert check_inputs(m, mid)["ok"]
    far = mid.copy()
    far[0, :] = 1e6
    assert not check_inputs(m, far)["ok"]
