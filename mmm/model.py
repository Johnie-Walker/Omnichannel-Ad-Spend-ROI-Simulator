"""Marketing Mix Model with Adstock carry-over and Hill saturation.

Model, for week t and region r (sales normalized by the regional mean):

    y[t,r] / ybar[r] = alpha[r] + gamma . controls[t,r]
                       + sum_c beta[c,r] * Hill( Adstock(x[c,:,r]; theta_c)[t] / n[c,r]; K_c, S_c )
                       + eps[t,r]

* Adstock decay (theta), half-saturation (K) and shape (S) are shared by all
  regions of a channel (pooled curve shape - far more data per parameter).
* Effect sizes beta[c,r] are region specific and constrained >= 0.
* n[c,r] is the mean weekly spend of the line item, so K is "multiples of
  typical spend" and comparable across channels and regions.

Estimation uses *variable projection*: for any candidate (theta, K, S) the
model is linear in (alpha, gamma, beta), which is solved exactly by bounded
least squares. SciPy's differential evolution searches the 15 non-linear
parameters globally on the resulting profile SSE, then polishes with L-BFGS-B.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.linalg
from scipy.optimize import differential_evolution, lsq_linear, minimize

from . import config as cfg
from .transforms import adstock_half_life, geometric_adstock, hill


@dataclass
class FitDiagnostics:
    r2: float
    mape: float
    nrmse: float
    durbin_watson: float


@dataclass
class MarketingMixModel:
    channels: list[str] = field(default_factory=lambda: list(cfg.CHANNELS))
    regions: list[str] = field(default_factory=lambda: list(cfg.REGIONS))
    fourier_order: int = cfg.FOURIER_ORDER
    pooling_lambda: float = cfg.POOLING_LAMBDA
    seed: int = cfg.SEED

    # fitted state
    theta: np.ndarray | None = None          # (C,)
    k: np.ndarray | None = None              # (C,)
    s: np.ndarray | None = None              # (C,)
    beta: np.ndarray | None = None           # (C, R) weekly $ at full saturation
    norm: np.ndarray | None = None           # (C, R) mean weekly spend
    ybar: np.ndarray | None = None           # (R,)
    control_coef: np.ndarray | None = None   # (p,)
    control_names: list[str] | None = None
    n_fit_weeks: int | None = None
    optimizer_message: str = ""

    # ------------------------------------------------------------------ data prep
    def _arrays(self, panel: pd.DataFrame):
        p = panel.sort_values(["Region", "Date"])
        dates = np.sort(p["Date"].unique())
        n_t, n_r = len(dates), len(self.regions)

        def grid(col: str) -> np.ndarray:
            return (p.pivot(index="Date", columns="Region", values=col)
                    .reindex(index=dates, columns=self.regions).to_numpy(float))

        spend = np.stack([grid(ch) for ch in self.channels])     # (C, T, R)
        y = grid("Sales")                                        # (T, R)
        price = grid("PriceIndex")
        holiday = grid("Holiday")
        if np.isnan(spend).any() or np.isnan(y).any():
            raise ValueError("panel must be a complete week x region grid")
        return pd.DatetimeIndex(dates), spend, y, price, holiday, n_t, n_r

    def _controls(self, n_t: int, price: np.ndarray, holiday: np.ndarray):
        """Control design, region-major rows (r*T + t)."""
        n_r = len(self.regions)
        t = np.arange(n_t)
        cols, names = [], []
        for ri, reg in enumerate(self.regions):                 # regional intercepts
            d = np.zeros((n_t, n_r)); d[:, ri] = 1.0
            cols.append(d); names.append(f"Intercept[{reg}]")
        cols.append(np.repeat(((t - t.mean()) / cfg.WEEKS_PER_YEAR)[:, None], n_r, 1))
        names.append("Trend")
        for k in range(1, self.fourier_order + 1):
            w = 2 * np.pi * k * t / cfg.WEEKS_PER_YEAR
            cols += [np.repeat(np.sin(w)[:, None], n_r, 1), np.repeat(np.cos(w)[:, None], n_r, 1)]
            names += [f"Sin{k}", f"Cos{k}"]
        cols.append(holiday); names.append("Holiday")
        cols.append(price - 1.0); names.append("PriceIndex")
        z = np.stack([c.T.reshape(-1) for c in cols], axis=1)  # (R*T, p)
        return z, names

    def _media_design(self, spend, theta, k, s, norm):
        """Hill(Adstock) design matrix (R*T, C*R) with one column per line item."""
        n_c, n_t, n_r = spend.shape
        m = np.zeros((n_r * n_t, n_c * n_r))
        for ci in range(n_c):
            h = hill(geometric_adstock(spend[ci], theta[ci]) / norm[ci], k[ci], s[ci])
            for ri in range(n_r):
                m[ri * n_t:(ri + 1) * n_t, ci * n_r + ri] = h[:, ri]
        return m

    def _pooling_rows(self, n_controls: int) -> np.ndarray:
        """Partial-pooling penalty rows on the ROI scale.

        For fixed curve shapes, a line item's ROI is proportional to
        b[c,r] * ybar[r] / n[c,r]. Penalizing each region's deviation from the
        channel's mean of that quantity shrinks regional ROIs toward the
        national ROI - a ridge form of a hierarchical prior. (Pooling the raw
        effects instead would bias totals whenever spend is skewed by region.)
        """
        n_c, n_r = len(self.channels), len(self.regions)
        w = self.ybar[None, :] / self.norm                       # (C, R) b -> ROI scale
        w = w / w.mean(axis=1, keepdims=True)                    # unit-free, mean 1
        center = np.eye(n_r) - np.full((n_r, n_r), 1.0 / n_r)
        block = np.zeros((n_c * n_r, n_c * n_r))
        for ci in range(n_c):
            sl = slice(ci * n_r, (ci + 1) * n_r)
            block[sl, sl] = center * w[ci][None, :]
        rows = np.hstack([np.zeros((n_c * n_r, n_controls)), block])
        return np.sqrt(self.pooling_lambda) * rows

    def _solve_linear(self, x: np.ndarray, y: np.ndarray, n_controls: int):
        """Bounded (media >= 0) penalized least squares; returns (coef, penalized SSE)."""
        if self.pooling_lambda > 0:
            x = np.vstack([x, self._pooling_rows(n_controls)])
            y = np.r_[y, np.zeros(x.shape[0] - len(y))]
        # QR with column pivoting: robust where SVD-based gelsd can fail to
        # converge on the ill-conditioned candidates the global search visits.
        coef = scipy.linalg.lstsq(x, y, lapack_driver="gelsy", check_finite=False)[0]
        if (coef[n_controls:] < 0).any():
            lb = np.r_[np.full(n_controls, -np.inf), np.zeros(x.shape[1] - n_controls)]
            coef = lsq_linear(x, y, bounds=(lb, np.inf), method="bvls").x
        r = y - x @ coef
        return coef, float(r @ r)

    # ----------------------------------------------------------------------- fit
    def _setup(self, panel: pd.DataFrame, n_fit_weeks: int | None):
        """Fix the normalizers and return what the objective needs."""
        _, spend, y, price, holiday, n_t, n_r = self._arrays(panel)
        n_fit = n_t if n_fit_weeks is None else n_fit_weeks
        self.norm = spend[:, :n_fit].mean(axis=1)                # (C, R)
        if (self.norm <= 0).any():
            raise ValueError("every channel x region needs positive spend in the fit window")
        self.ybar = y[:n_fit].mean(axis=0)
        yn = (y / self.ybar).T.reshape(-1)                       # region-major
        z, self.control_names = self._controls(n_t, price, holiday)
        rows = (np.arange(n_r * n_t) % n_t) < n_fit
        self.n_fit_weeks = n_fit
        return spend, yn, z, rows

    def _unpack(self, v: np.ndarray):
        n_c = len(self.channels)
        return np.asarray(v[:n_c]), np.asarray(v[n_c:2 * n_c]), np.asarray(v[2 * n_c:])

    def objective(self, v: np.ndarray, spend, yn, z, rows) -> float:
        """Negative log posterior (up to a constant), sigma profiled out.

        N/2 * log(SSE) is the Gaussian likelihood with the noise variance
        concentrated out; the log-normal priors on K and S regularize curve
        shapes that the data alone cannot pin down.
        """
        th, kk, ss = self._unpack(v)
        x = np.hstack([z, self._media_design(spend, th, kk, ss, self.norm)])[rows]
        _, sse = self._solve_linear(x, yn[rows], z.shape[1])
        nll = 0.5 * rows.sum() * np.log(sse)
        prior = (((np.log(kk) - cfg.K_PRIOR[0]) / cfg.K_PRIOR[1]) ** 2).sum() / 2 \
              + (((np.log(ss) - cfg.S_PRIOR[0]) / cfg.S_PRIOR[1]) ** 2).sum() / 2
        return float(nll + prior)

    def fit(self, panel: pd.DataFrame, n_fit_weeks: int | None = None,
            maxiter: int = cfg.DE_MAXITER, popsize: int = cfg.DE_POPSIZE) -> "MarketingMixModel":
        spend, yn, z, rows = self._setup(panel, n_fit_weeks)
        n_c = len(self.channels)
        bounds = [cfg.THETA_BOUNDS] * n_c + [cfg.K_BOUNDS] * n_c + [cfg.S_BOUNDS] * n_c
        res = differential_evolution(self.objective, bounds, args=(spend, yn, z, rows),
                                     seed=self.seed, popsize=popsize, maxiter=maxiter,
                                     tol=1e-8, init="sobol", mutation=(0.5, 1.0),
                                     recombination=0.7, polish=True)
        self.optimizer_message = f"{res.message} (nfev={res.nfev}, objective={res.fun:.4f})"
        self.theta, self.k, self.s = self._unpack(res.x)
        self._solve_coefficients(spend, yn, z, rows)
        return self

    def fit_local(self, panel: pd.DataFrame, start: np.ndarray) -> "MarketingMixModel":
        """Local L-BFGS-B refit from a known good point (used by the bootstrap)."""
        spend, yn, z, rows = self._setup(panel, None)
        n_c = len(self.channels)
        bounds = [cfg.THETA_BOUNDS] * n_c + [cfg.K_BOUNDS] * n_c + [cfg.S_BOUNDS] * n_c
        res = minimize(self.objective, start, args=(spend, yn, z, rows),
                       method="L-BFGS-B", bounds=bounds)
        self.optimizer_message = f"{res.message} (nfev={res.nfev})"
        self.theta, self.k, self.s = self._unpack(res.x)
        self._solve_coefficients(spend, yn, z, rows)
        return self

    def curve_vector(self) -> np.ndarray:
        return np.r_[self.theta, self.k, self.s]

    def set_curve_params(self, panel: pd.DataFrame, theta, k, s,
                         n_fit_weeks: int | None = None) -> "MarketingMixModel":
        """Fix (theta, K, S) and solve only the linear part (used for audits)."""
        spend, yn, z, rows = self._setup(panel, n_fit_weeks)
        self.theta, self.k, self.s = (np.asarray(a, float) for a in (theta, k, s))
        self._solve_coefficients(spend, yn, z, rows)
        return self

    def _solve_coefficients(self, spend, yn, z, rows) -> None:
        x = np.hstack([z, self._media_design(spend, self.theta, self.k, self.s, self.norm)])
        p = z.shape[1]
        coef, _ = self._solve_linear(x[rows], yn[rows], p)
        self.control_coef = coef[:p]
        self.beta = coef[p:].reshape(len(self.channels), -1) * self.ybar[None, :]  # -> weekly $

    # ------------------------------------------------------------------- predict
    def decompose(self, panel: pd.DataFrame) -> dict[str, np.ndarray | pd.DatetimeIndex]:
        """Actual, predicted, base and per-channel contributions, each (T, R) in $."""
        dates, spend, y, price, holiday, n_t, n_r = self._arrays(panel)
        z, _ = self._controls(n_t, price, holiday)
        base = (z @ self.control_coef).reshape(n_r, n_t).T * self.ybar[None, :]
        contrib = {}
        for ci, ch in enumerate(self.channels):
            u = geometric_adstock(spend[ci], self.theta[ci]) / self.norm[ci]
            contrib[ch] = self.beta[ci][None, :] * hill(u, self.k[ci], self.s[ci])
        pred = base + sum(contrib.values())
        return dict(dates=dates, actual=y, predicted=pred, base=base,
                    contributions=contrib, spend=spend)

    @staticmethod
    def diagnostics(actual: np.ndarray, predicted: np.ndarray) -> FitDiagnostics:
        """MAPE on region-weeks; R2, NRMSE and Durbin-Watson on the national
        weekly series (pooling regions would inflate R2 with between-region
        variance)."""
        mape = float(np.mean(np.abs((actual - predicted) / actual)))
        a, p = actual.sum(axis=1), predicted.sum(axis=1)
        resid = a - p
        r2 = 1.0 - resid @ resid / ((a - a.mean()) @ (a - a.mean()))
        nrmse = float(np.sqrt(np.mean(resid ** 2)) / (a.max() - a.min()))
        dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2))
        return FitDiagnostics(float(r2), mape, nrmse, dw)

    # -------------------------------------------------------------------- export
    def params_table(self) -> pd.DataFrame:
        rows = []
        for ci, ch in enumerate(self.channels):
            for ri, reg in enumerate(self.regions):
                rows.append(dict(Channel=ch, Region=reg, Theta=self.theta[ci],
                                 HalfLifeWeeks=adstock_half_life(self.theta[ci]),
                                 K=self.k[ci], S=self.s[ci], Beta=self.beta[ci, ri],
                                 Norm=self.norm[ci, ri]))
        return pd.DataFrame(rows)

    def controls_table(self) -> pd.DataFrame:
        return pd.DataFrame({"Term": self.control_names, "Coefficient": self.control_coef})
