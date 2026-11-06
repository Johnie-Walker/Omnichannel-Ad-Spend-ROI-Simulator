"""Simulate weekly regional sales and media spend with a known ground truth.

Because the true Adstock / Hill / ROI parameters are known, the fitted model
can be audited against them (parameter recovery, ROI recovery, and the
"true" lift of the optimized plan) - something real data never allows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .transforms import geometric_adstock, hill


@dataclass
class SimulatedData:
    panel: pd.DataFrame            # one row per week x region
    truth_params: pd.DataFrame     # one row per channel x region
    truth_contrib: pd.DataFrame    # true weekly media contribution, long format


def _spend_pattern(kind: str, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Relative weekly spend shape (T x R) for one channel."""
    n_t, n_r = len(t), len(cfg.REGIONS)
    if kind == "burst":                                   # TV flights: 4 weeks on / 2 off
        on = (t % 6) < 4
        noise = rng.lognormal(0.0, 0.30, size=(n_t, n_r))
        return on[:, None] * noise
    if kind == "pulse":                                   # Radio: random 2-4 week flights
        on = np.zeros(n_t, dtype=bool)                    # (independent of the TV calendar)
        i, state = 0, True
        while i < n_t:
            length = int(rng.integers(2, 5)) if state else int(rng.integers(1, 4))
            on[i:i + length] = state
            i, state = i + length, not state
        noise = rng.lognormal(0.0, 0.20, size=(n_t, n_r))
        return on[:, None] * noise
    if kind == "always_on_seasonal":                      # Search follows demand
        season = 1.0 + 0.30 * np.sin(2 * np.pi * (t - 35) / cfg.WEEKS_PER_YEAR)
        pattern = season[:, None] * rng.lognormal(0.0, 0.25, size=(n_t, n_r))
    elif kind == "always_on":
        pattern = rng.lognormal(0.0, 0.40, size=(n_t, n_r))
    else:
        raise ValueError(f"unknown flighting pattern {kind!r}")
    return _geo_go_dark(pattern, t, rng)


def _geo_go_dark(pattern: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Geo holdout tests: twice a year a region pauses the channel for 4 weeks.

    Always-on channels otherwise never visit low spend, which leaves the level
    of their contribution unidentified (collinear with the intercept).
    """
    out = pattern.copy()
    n_r = pattern.shape[1]
    order = rng.permutation(n_r)
    test = 0
    for y in range(len(t) // cfg.WEEKS_PER_YEAR):
        for half in range(2):                               # one test per half-year
            lo = y * cfg.WEEKS_PER_YEAR + half * 26
            start = lo + int(rng.integers(2, 26 - 6))
            out[start:start + 4, order[test % n_r]] = 0.0
            test += 1
    return out


def generate(seed: int = cfg.SEED) -> SimulatedData:
    rng = np.random.default_rng(seed)
    t = np.arange(cfg.N_WEEKS)
    dates = pd.date_range(cfg.START_DATE, periods=cfg.N_WEEKS, freq="W-MON")
    year_idx = t // cfg.WEEKS_PER_YEAR
    n_r = len(cfg.REGIONS)

    # ---- media spend: (T x R) per channel, scaled to the yearly budgets -------
    spend = {}
    for ci, ch in enumerate(cfg.CHANNELS):
        truth = cfg.CHANNEL_TRUTH[ch]
        pattern = _spend_pattern(truth.flighting, t, rng)
        split = np.array([cfg.REGION_SPEND_SPLIT[r][ci] for r in cfg.REGIONS])
        x = np.zeros_like(pattern)
        for y, budget in enumerate(cfg.YEARLY_SPEND):
            rows = year_idx == y
            block = pattern[rows]
            target = budget * truth.spend_share * split          # per region
            x[rows] = block / block.sum(axis=0) * target
        spend[ch] = x

    # ---- non-media base ----------------------------------------------------------
    week_of_year = dates.isocalendar().week.to_numpy().astype(int)
    holiday = np.isin(week_of_year, [47, 50, 51]).astype(float)
    price = np.empty((cfg.N_WEEKS, n_r))
    price[0] = 1.0
    shocks = rng.normal(0.0, 0.012, size=(cfg.N_WEEKS, n_r))
    for i in range(1, cfg.N_WEEKS):
        price[i] = 1.0 + 0.7 * (price[i - 1] - 1.0) + shocks[i]
    price -= 0.05 * holiday[:, None]                            # promo price cuts

    weekly_base = cfg.ANNUAL_BASE_SALES / cfg.WEEKS_PER_YEAR
    shape = (1.0
             + 0.05 * (t - t.mean()) / cfg.WEEKS_PER_YEAR
             + 0.10 * np.sin(2 * np.pi * (t - 35) / cfg.WEEKS_PER_YEAR)
             + 0.04 * np.cos(4 * np.pi * t / cfg.WEEKS_PER_YEAR)
             + 0.18 * holiday)
    size = np.array([cfg.REGION_SIZE[r] for r in cfg.REGIONS])
    base = size[None, :] * weekly_base * (shape[:, None] - 1.2 * (price - 1.0))

    # ---- true media response ----------------------------------------------------------
    contrib = {}
    truth_rows = []
    for ci, ch in enumerate(cfg.CHANNELS):
        tr = cfg.CHANNEL_TRUTH[ch]
        x = spend[ch]
        norm = x.mean(axis=0)                                   # mean weekly spend per region
        eff = np.array([cfg.REGION_EFFICIENCY[r][ci] for r in cfg.REGIONS])
        h1 = hill(1.0, tr.k, tr.s)
        beta = tr.roi * eff * norm / h1                         # weekly $ at full saturation
        u = geometric_adstock(x, tr.theta) / norm
        contrib[ch] = beta * hill(u, tr.k, tr.s)
        for ri, reg in enumerate(cfg.REGIONS):
            truth_rows.append(dict(Channel=ch, Region=reg, Theta=tr.theta, K=tr.k, S=tr.s,
                                   Beta=beta[ri], Norm=norm[ri]))

    noise = rng.normal(0.0, cfg.NOISE_SD, size=(cfg.N_WEEKS, n_r)) * (size * weekly_base)
    sales = base + sum(contrib.values()) + noise

    # ---- long panel -------------------------------------------------------------------
    frames = []
    for ri, reg in enumerate(cfg.REGIONS):
        f = pd.DataFrame({"Date": dates, "Region": reg, "Sales": sales[:, ri],
                          "PriceIndex": price[:, ri], "Holiday": holiday})
        for ch in cfg.CHANNELS:
            f[ch] = spend[ch][:, ri]
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)

    contrib_long = pd.concat(
        [pd.DataFrame({"Date": dates, "Region": reg, "Channel": ch,
                       "TrueContribution": contrib[ch][:, ri]})
         for ch in cfg.CHANNELS for ri, reg in enumerate(cfg.REGIONS)],
        ignore_index=True)

    return SimulatedData(panel=panel, truth_params=pd.DataFrame(truth_rows),
                         truth_contrib=contrib_long)
