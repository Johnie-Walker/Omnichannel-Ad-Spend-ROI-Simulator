import numpy as np
import pytest

from mmm.optimizer import BudgetOptimizer, ResponseSurface


def surface(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return ResponseSurface(
        channel=np.array([f"C{i}" for i in range(n)]), region=np.array(["R"] * n),
        beta=rng.uniform(2e4, 9e4, n), norm=rng.uniform(5e3, 2e4, n),
        k=rng.uniform(0.5, 2.0, n), s=rng.uniform(0.8, 2.5, n))


def current(surf):
    return surf.norm * 52            # spend at the historical mean


def test_marginal_roi_is_gradient_of_response():
    s = surface()
    x, eps = current(s), 1.0
    fd = (s.response(x + eps) - s.response(x - eps)) / (2 * eps)
    np.testing.assert_allclose(s.marginal_roi(x), fd, rtol=1e-5)


def test_budget_bounds_and_kkt():
    s = surface()
    x0 = current(s)
    opt = BudgetOptimizer(s, x0, 0.3, 2.5)
    r = opt.optimize(budget=x0.sum(), n_starts=12)
    assert r.optimal.sum() == pytest.approx(x0.sum(), rel=1e-9)
    assert np.all(r.optimal >= r.lower - 1e-6) and np.all(r.optimal <= r.upper + 1e-6)
    assert r.kkt_ok, r.kkt_max_gap
    assert r.optimal_response >= r.current_response


def test_beats_random_feasible_allocations():
    s = surface(seed=3)
    x0 = current(s)
    opt = BudgetOptimizer(s, x0, 0.3, 2.5)
    r = opt.optimize(budget=x0.sum(), n_starts=12)
    rng = np.random.default_rng(9)
    lb, ub = x0 * 0.3, x0 * 2.5
    for _ in range(2000):
        w = rng.dirichlet(np.ones(len(x0))) * (ub - lb)
        x = lb + w * (x0.sum() - lb.sum()) / w.sum()
        if np.all(x <= ub):
            assert s.response(x).sum() <= r.optimal_response * (1 + 1e-9)


def test_symmetric_concave_items_split_equally():
    s = ResponseSurface(np.array(["A", "B"]), np.array(["R", "R"]), np.array([1e5, 1e5]),
                        np.array([1e4, 1e4]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    x0 = np.array([8e5, 2e5])
    r = BudgetOptimizer(s, x0, 0.01, 100).optimize(budget=1e6, n_starts=4)
    np.testing.assert_allclose(r.optimal, [5e5, 5e5], rtol=1e-5)


def test_infeasible_budget_raises():
    s = surface()
    with pytest.raises(ValueError):
        BudgetOptimizer(s, current(s), 0.5, 1.5).optimize(budget=current(s).sum() * 2)
