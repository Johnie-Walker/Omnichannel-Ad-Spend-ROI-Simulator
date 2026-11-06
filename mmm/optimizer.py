"""Non-linear budget allocation across channel x region line items.

Planning response of a line item with annual budget X (flat weekly flighting,
so normalized adstock is in steady state at X/52):

    R(X)  = 52 * beta * Hill( X / (52 * n); K, S )
    R'(X) = beta * Hill'( X / (52 * n) ) / n          (marginal ROI)

Problem:   max_X  sum_i R_i(X_i)
           s.t.   sum_i X_i = B,   lb_i <= X_i <= ub_i

Solved with SciPy SLSQP using analytic gradients. Hill curves with S > 1 are
S-shaped (non-concave), so the solver is multi-started and the best solution
is kept. Optimality is verified with the KKT conditions: every interior line
item ends at the same marginal ROI (the budget's shadow price, lambda).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config as cfg
from .transforms import hill, hill_derivative

_SCALE = 1e6        # optimize in $M for good conditioning


@dataclass
class ResponseSurface:
    """Steady-state annual response for a flat list of line items."""
    channel: np.ndarray
    region: np.ndarray
    beta: np.ndarray
    norm: np.ndarray
    k: np.ndarray
    s: np.ndarray

    @classmethod
    def from_params(cls, params: pd.DataFrame) -> "ResponseSurface":
        return cls(params["Channel"].to_numpy(), params["Region"].to_numpy(),
                   params["Beta"].to_numpy(float), params["Norm"].to_numpy(float),
                   params["K"].to_numpy(float), params["S"].to_numpy(float))

    def _u(self, annual: np.ndarray) -> np.ndarray:
        return np.asarray(annual, float) / (cfg.WEEKS_PER_YEAR * self.norm)

    def response(self, annual: np.ndarray) -> np.ndarray:
        """Annual incremental sales per line item."""
        return cfg.WEEKS_PER_YEAR * self.beta * hill(self._u(annual), self.k, self.s)

    def marginal_roi(self, annual: np.ndarray) -> np.ndarray:
        """d(incremental sales) / d(spend) per line item."""
        return self.beta * hill_derivative(self._u(annual), self.k, self.s) / self.norm


@dataclass
class OptimizationResult:
    budget: float
    current: np.ndarray
    optimal: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    current_response: float
    optimal_response: float
    shadow_price: float          # lambda: marginal ROI of the last dollar
    kkt_max_gap: float           # max relative mROI gap among interior items
    kkt_ok: bool
    n_starts: int
    n_converged: int

    @property
    def incremental_lift(self) -> float:
        return self.optimal_response / self.current_response - 1.0


class BudgetOptimizer:
    def __init__(self, surface: ResponseSurface, current: np.ndarray,
                 min_mult: float = cfg.MIN_SPEND_MULT, max_mult: float = cfg.MAX_SPEND_MULT):
        self.surface = surface
        self.current = np.asarray(current, float)
        self.min_mult, self.max_mult = min_mult, max_mult

    def _starts(self, lb, ub, budget, n, rng):
        """Feasible starting points: current mix, uniform split of headroom, random."""
        starts = [self.current * budget / self.current.sum()]
        head = ub - lb
        starts.append(lb + head * (budget - lb.sum()) / head.sum())
        for _ in range(max(n - 2, 0)):
            w = rng.dirichlet(np.ones(len(lb))) * head
            x = lb + w * (budget - lb.sum()) / w.sum()
            starts.append(x)
        return [np.clip(x, lb, ub) for x in starts]

    def optimize(self, budget: float = cfg.ANNUAL_BUDGET, n_starts: int = cfg.OPTIMIZER_STARTS,
                 seed: int = cfg.SEED) -> OptimizationResult:
        lb, ub = self.current * self.min_mult, self.current * self.max_mult
        if not lb.sum() <= budget <= ub.sum():
            raise ValueError(f"budget {budget:,.0f} infeasible for guardrails "
                             f"[{lb.sum():,.0f}, {ub.sum():,.0f}]")
        surf = self.surface

        def f(xm):
            return -surf.response(xm * _SCALE).sum() / _SCALE

        def grad(xm):
            return -surf.marginal_roi(xm * _SCALE)

        cons = [{"type": "eq", "fun": lambda xm: xm.sum() - budget / _SCALE,
                 "jac": lambda xm: np.ones_like(xm)}]
        bounds = list(zip(lb / _SCALE, ub / _SCALE))
        rng = np.random.default_rng(seed)
        best, converged = None, 0
        for x0 in self._starts(lb, ub, budget, n_starts, rng):
            res = minimize(f, x0 / _SCALE, jac=grad, bounds=bounds, constraints=cons,
                           method="SLSQP", options={"ftol": 1e-12, "maxiter": 500})
            feasible = abs(res.x.sum() - budget / _SCALE) < 1e-6
            if res.success and feasible:
                converged += 1
                if best is None or res.fun < best.fun:
                    best = res
        if best is None:
            raise RuntimeError("SLSQP failed from every start")

        x = np.clip(best.x * _SCALE, lb, ub)
        x *= budget / x.sum()                       # remove float residue on the constraint
        lam, gap, ok = self.kkt(x, lb, ub)
        return OptimizationResult(
            budget=budget, current=self.current * budget / self.current.sum(), optimal=x,
            lower=lb, upper=ub,
            current_response=float(surf.response(self.current * budget / self.current.sum()).sum()),
            optimal_response=float(surf.response(x).sum()),
            shadow_price=lam, kkt_max_gap=gap, kkt_ok=ok,
            n_starts=n_starts, n_converged=converged)

    def kkt(self, x, lb, ub, rtol: float = 1e-3, tol: float = 1e-3):
        """Check first-order optimality; returns (lambda, max interior gap, ok)."""
        m = self.surface.marginal_roi(x)
        at_lb = x <= lb * (1 + rtol)
        at_ub = x >= ub * (1 - rtol)
        interior = ~(at_lb | at_ub)
        if interior.any():
            lam = float(np.median(m[interior]))
            gap = float(np.max(np.abs(m[interior] / lam - 1.0)))
        else:                                       # every item on a guardrail
            lam = float((m[at_lb].max() + m[at_ub].min()) / 2)
            gap = 0.0
        ok = (gap <= tol
              and bool(np.all(m[at_lb] <= lam * (1 + tol)))
              and bool(np.all(m[at_ub] >= lam * (1 - tol))))
        return lam, gap, ok

    def result_table(self, res: OptimizationResult) -> pd.DataFrame:
        s = self.surface
        cur, opt = res.current, res.optimal
        status = np.where(opt <= res.lower * (1 + 1e-3), "At floor",
                          np.where(opt >= res.upper * (1 - 1e-3), "At ceiling", "Interior"))
        out = pd.DataFrame({
            "Channel": s.channel, "Region": s.region,
            "CurrentAnnualSpend": cur, "OptimalAnnualSpend": opt,
            "SpendChangePct": opt / cur - 1.0,
            "CurrentIncrementalSales": s.response(cur), "OptimalIncrementalSales": s.response(opt),
            "CurrentMarginalROI": s.marginal_roi(cur), "OptimalMarginalROI": s.marginal_roi(opt),
            "MinAnnualSpend": res.lower, "MaxAnnualSpend": res.upper, "Constraint": status})
        out["CurrentROI"] = out.CurrentIncrementalSales / out.CurrentAnnualSpend
        out["OptimalROI"] = out.OptimalIncrementalSales / out.OptimalAnnualSpend
        return out

    def frontier(self, budgets=cfg.FRONTIER_BUDGETS, n_starts: int = 12) -> pd.DataFrame:
        rows = []
        for b in budgets:
            r = self.optimize(b, n_starts=n_starts)
            rows.append(dict(Budget=b, OptimalIncrementalSales=r.optimal_response,
                             CurrentMixIncrementalSales=r.current_response,
                             OptimalROI=r.optimal_response / b,
                             CurrentMixROI=r.current_response / b,
                             MarginalROI=r.shadow_price))
        return pd.DataFrame(rows)
