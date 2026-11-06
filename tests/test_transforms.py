import numpy as np
import pytest

from mmm.transforms import adstock_half_life, geometric_adstock, hill, hill_derivative


def reference_adstock(x, theta):
    out, a = np.empty_like(x, dtype=float), x[0]
    for t, v in enumerate(x):
        a = theta * a + (1 - theta) * v
        out[t] = a
    return out


@pytest.mark.parametrize("theta", [0.0, 0.1, 0.5, 0.85])
def test_adstock_matches_recursive_definition(theta):
    x = np.random.default_rng(0).lognormal(size=60)
    np.testing.assert_allclose(geometric_adstock(x, theta), reference_adstock(x, theta), rtol=1e-12)


def test_adstock_2d_is_columnwise():
    x = np.random.default_rng(1).lognormal(size=(40, 3))
    out = geometric_adstock(x, 0.6)
    for j in range(3):
        np.testing.assert_allclose(out[:, j], reference_adstock(x[:, j], 0.6), rtol=1e-12)


def test_adstock_is_normalized():
    """Constant spend stays at steady state, and total effect is conserved."""
    np.testing.assert_allclose(geometric_adstock(np.full(30, 7.0), 0.7), 7.0)
    impulse = np.r_[0.0, 100.0, np.zeros(400)]
    assert geometric_adstock(impulse, 0.7).sum() == pytest.approx(100.0, rel=1e-9)


def test_half_life():
    theta = 0.6
    hl = adstock_half_life(theta)
    assert theta ** hl == pytest.approx(0.5)
    assert adstock_half_life(0.0) == 0.0


def test_adstock_rejects_bad_theta():
    with pytest.raises(ValueError):
        geometric_adstock(np.ones(5), 1.0)


def test_hill_properties():
    assert hill(2.0, 2.0, 1.7) == pytest.approx(0.5)                  # K is the half-saturation point
    assert hill(0.0, 1.0, 2.0) == 0.0
    u = np.linspace(0, 20, 200)
    h = hill(u, 1.2, 1.5)
    assert np.all(np.diff(h) > 0) and h.max() < 1


@pytest.mark.parametrize("s", [0.7, 1.0, 2.5])
def test_hill_derivative_matches_finite_difference(s):
    u, k, eps = np.linspace(0.05, 5, 50), 1.3, 1e-6
    fd = (hill(u + eps, k, s) - hill(u - eps, k, s)) / (2 * eps)
    np.testing.assert_allclose(hill_derivative(u, k, s), fd, rtol=1e-5)


def test_hill_derivative_limits_at_zero():
    assert hill_derivative(0.0, 2.0, 1.0) == pytest.approx(0.5)
    assert hill_derivative(0.0, 2.0, 2.0) == 0.0
    assert np.isinf(hill_derivative(0.0, 2.0, 0.6))
    # vectorized over per-item k and s
    out = hill_derivative(np.zeros(3), np.array([2.0, 2.0, 2.0]), np.array([0.6, 1.0, 2.0]))
    assert np.isinf(out[0]) and out[1] == pytest.approx(0.5) and out[2] == 0.0
