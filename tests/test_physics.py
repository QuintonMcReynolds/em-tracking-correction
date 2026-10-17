import numpy as np

from emtrack.physics import Environment, Metal, dipole_field, measure, primary
from emtrack.rotations import angle_between, rotvec_to_matrix
from emtrack.solver import solve


def test_dipole_on_axis_and_equator():
    m = np.array([0.0, 0.0, 1.0])
    on_axis = dipole_field(np.array([[0.0, 0.0, 1000.0]]), m)[0]     # 1 m on axis: 2m/r^3
    equator = dipole_field(np.array([[1000.0, 0.0, 0.0]]), m)[0]     # equator: -m/r^3
    assert np.allclose(on_axis, [0, 0, 2.0]) and np.allclose(equator, [0, 0, -1.0])


def test_field_falls_off_as_inverse_cube():
    p = np.array([[0.0, 300.0, 400.0]])
    assert np.isclose(np.linalg.norm(primary(p)) / np.linalg.norm(primary(2 * p)), 8.0)


def test_measurement_rotates_with_receiver():
    rng = np.random.default_rng(0)
    p = np.array([[50.0, -80.0, 500.0]])
    R = rotvec_to_matrix(rng.normal(0, 0.5, (1, 3)))
    S = measure(p, R, None)
    assert np.allclose(R[0] @ S[0], primary(p)[0])


def test_solver_exact_without_metal_and_biased_with_it():
    rng = np.random.default_rng(1)
    N = 300
    p = np.c_[rng.uniform(-150, 150, (N, 2)), rng.uniform(400, 700, N)]
    R = rotvec_to_matrix(rng.normal(0, 0.3, (N, 3)))
    pe, Re, res = solve(measure(p, R, None), p + rng.normal(0, 10, p.shape), R)
    assert np.abs(pe - p).max() < 1e-3 and angle_between(Re, R).max() < 1e-3 and res.max() < 1e-8
    env = Environment("t", "t", [Metal("sphere", np.array([500.0, 0, 400]), -(0.2 ** 3 / 2) * np.eye(3), 200)])
    pe, Re, res = solve(measure(p, R, env), p, R)
    err = np.linalg.norm(pe - p, axis=1)
    assert np.median(err) > 2.0 and np.median(res) > 1e-4   # metal biases the pose and leaves a fit residual
