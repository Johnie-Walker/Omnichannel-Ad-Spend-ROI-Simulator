"""Moving-block residual bootstrap for parameter and ROI uncertainty.

Each replicate rebuilds sales as fitted + resampled residuals (blocks of
consecutive weeks, all regions together, so autocorrelation and cross-region
correlation survive), then refits the full non-linear model locally from the
point estimate. The replicate response surfaces give confidence intervals for
channel ROI and for the projected lift of the recommended plan.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import config as cfg
from .model import MarketingMixModel


def _resample_blocks(n_t: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n_t - block + 1, size=int(np.ceil(n_t / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n_t]


def _replicate(args) -> pd.DataFrame:
    panel, fitted, resid, start, regions, block, seed, pooling = args
    rng = np.random.default_rng(seed)
    idx = _resample_blocks(resid.shape[0], block, rng)
    y_star = fitted + resid[idx]
    dates = np.sort(panel["Date"].unique())
    grid = pd.DataFrame(y_star, index=pd.DatetimeIndex(dates), columns=regions)
    star = panel.copy()
    star["Sales"] = grid.stack().reindex(
        pd.MultiIndex.from_frame(star[["Date", "Region"]])).to_numpy()
    m = MarketingMixModel(pooling_lambda=pooling).fit_local(star, start)
    out = m.params_table()
    out["Replicate"] = seed
    return out


def block_bootstrap(model: MarketingMixModel, panel: pd.DataFrame,
                    n_boot: int = cfg.BOOTSTRAP_REPLICATES, block: int = cfg.BOOTSTRAP_BLOCK,
                    seed: int = cfg.SEED, workers: int | None = None) -> pd.DataFrame:
    """Return the stacked params tables of all replicates."""
    dec = model.decompose(panel)
    fitted, resid = dec["predicted"], dec["actual"] - dec["predicted"]
    start = model.curve_vector()
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_boot)
    jobs = [(panel, fitted, resid, start, model.regions, block, int(s), model.pooling_lambda)
            for s in seeds]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        reps = list(pool.map(_replicate, jobs))
    return pd.concat(reps, ignore_index=True)
