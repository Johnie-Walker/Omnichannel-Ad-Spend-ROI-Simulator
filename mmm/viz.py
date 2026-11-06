"""Static report figures (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import config as cfg  # noqa: E402
from .optimizer import ResponseSurface  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8985"            # neutral for "current / actual" reference series
ACCENT = cfg.CHANNEL_COLORS["TV"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK_2,
    "ytick.color": INK_2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5, "font.family": "DejaVu Sans",
    "lines.linewidth": 2.0, "figure.dpi": 110,
})

def _fmt_m(v, _):
    s = f"{v / 1e6:,.2f}".rstrip("0").rstrip(".")   # $0.25M, $1.5M, $2M - never rounds ticks
    return f"${s}M"


money_m = FuncFormatter(_fmt_m)
money_k = FuncFormatter(lambda v, _: f"${v / 1e3:,.0f}K")


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def actual_vs_predicted(dates, actual, predicted, holdout_pred, n_holdout, path: Path):
    a, p, h = actual.sum(axis=1), predicted.sum(axis=1), holdout_pred.sum(axis=1)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axvspan(dates[-n_holdout], dates[-1], color=GRID, alpha=0.6, lw=0)
    ax.text(dates[-n_holdout], 0.02, "  holdout (26 wk)", transform=ax.get_xaxis_transform(),
            va="bottom", color=INK_2, fontsize=8.5)
    ax.plot(dates, a, color=REFERENCE, lw=1.4, label="Actual sales")
    ax.plot(dates, p, color=ACCENT, label="MMM fit (all data)")
    ax.plot(dates[-n_holdout:], h[-n_holdout:], color=cfg.CHANNEL_COLORS["Paid Search"], ls="--",
            label="Out-of-sample forecast (fit on first 130 wk)")
    ax.yaxis.set_major_formatter(money_k)
    ax.set_ylabel("Weekly sales, all regions")
    ax.set_title("Model fit: actual vs predicted weekly sales")
    ax.legend(loc="upper left", ncol=3)
    _save(fig, path)


def roi_by_channel(summary: pd.DataFrame, path: Path):
    s = summary.set_index("Channel").loc[cfg.CHANNELS[::-1]]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8.5, 4))
    err = np.vstack([s.HistoricalROI - s.HistoricalROI_P05, s.HistoricalROI_P95 - s.HistoricalROI])
    ax.barh(y, s.HistoricalROI, height=0.55, color=ACCENT, label="Estimated ROI (90% bootstrap CI)")
    ax.errorbar(s.HistoricalROI, y, xerr=err, fmt="none", ecolor=INK, elinewidth=1.2, capsize=3)
    ax.scatter(s.TrueHistoricalROI, y, marker="D", s=40, color=INK, zorder=5,
               edgecolor=SURFACE, linewidth=1.5, label="Ground truth (simulation)")
    ax.axvline(1.0, color=INK_2, lw=1, ls=":")
    ax.text(1.0, 0.01, " break-even", transform=ax.get_xaxis_transform(), color=INK_2,
            fontsize=8, va="bottom")
    for yi, v in zip(y, s.HistoricalROI):
        ax.text(0.05, yi, f"{v:.2f}", va="center", color="white", fontsize=8.5, fontweight="bold")
    ax.set_yticks(y, s.index)
    ax.set_xlabel("Incremental sales per $1 of spend (3-year history)")
    ax.set_title("Promotional ROI by channel")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _save(fig, path)


def _national_curve(surface: ResponseSurface, current: np.ndarray, ch: str, idx: np.ndarray):
    mask = surface.channel == ch
    sub = ResponseSurface(*(getattr(surface, f)[mask] for f in
                            ("channel", "region", "beta", "norm", "k", "s")))
    cur = current[mask]
    spend = np.array([(cur * i).sum() for i in idx])
    sales = np.array([sub.response(cur * i).sum() for i in idx])
    mroi = np.array([(cur * i * sub.marginal_roi(cur * i)).sum() / (cur * i).sum()
                     if i > 0 else np.nan for i in idx])
    return spend, sales, mroi


def response_curves(surface: ResponseSurface, table: pd.DataFrame, path: Path):
    idx = np.linspace(0, cfg.MAX_SPEND_MULT * 1.2, 121)
    cur = table.CurrentAnnualSpend.to_numpy()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), sharey=False)
    for ax, ch in zip(axes, cfg.CHANNELS):
        spend, sales, _ = _national_curve(surface, cur, ch, idx)
        ax.plot(spend, sales, color=cfg.CHANNEL_COLORS[ch])
        t = table[table.Channel == ch]
        for col_s, col_r, color, lab in [("CurrentAnnualSpend", "CurrentIncrementalSales", REFERENCE, "Current"),
                                         ("OptimalAnnualSpend", "OptimalIncrementalSales", INK, "Optimized")]:
            ax.scatter(t[col_s].sum(), t[col_r].sum(), s=64, color=color, zorder=5,
                       edgecolor=SURFACE, linewidth=2, label=lab)
        ax.set_title(ch, fontsize=10.5)
        ax.xaxis.set_major_formatter(money_m)
        ax.yaxis.set_major_formatter(money_m)
        ax.set_xlabel("Annual spend")
    axes[0].set_ylabel("Annual incremental sales")
    axes[0].legend(loc="lower right")
    fig.suptitle("Hill response curves (steady state, all regions)", x=0.01, ha="left",
                 fontweight="bold", color=INK)
    _save(fig, path)


def marginal_roi_curves(surface: ResponseSurface, table: pd.DataFrame, shadow_price: float, path: Path):
    idx = np.linspace(0.05, cfg.MAX_SPEND_MULT, 100)
    cur = table.CurrentAnnualSpend.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for ch in cfg.CHANNELS:
        _, _, mroi = _national_curve(surface, cur, ch, idx)
        ax.plot(idx * 100, mroi, color=cfg.CHANNEL_COLORS[ch], label=ch)
    ax.axhline(1.0, color=INK_2, lw=1, ls=":")
    ax.text(102, 1.0, "break-even (mROI = 1.00)", color=INK_2, fontsize=8, va="bottom")
    ax.axhline(shadow_price, color=INK, lw=1, ls="--")
    ax.text(102, shadow_price, f"optimal plan equalizes mROI at {shadow_price:.2f}",
            color=INK, fontsize=8, va="bottom")
    ax.axvline(100, color=GRID, lw=1.2)
    ax.set_xlim(0, cfg.MAX_SPEND_MULT * 100 + 5)
    ax.set_ylim(0, min(np.nanmax(ax.get_ylim()), 8))
    ax.set_xlabel("Spend as % of current plan")
    ax.set_ylabel("Marginal ROI (sales from next $1)")
    ax.set_title("Marginal ROI curves: where the next dollar works hardest")
    ax.legend(loc="upper right", ncol=5)
    _save(fig, path)


def budget_reallocation(table: pd.DataFrame, path: Path):
    g = table.groupby("Channel")[["CurrentAnnualSpend", "OptimalAnnualSpend"]].sum().loc[cfg.CHANNELS[::-1]]
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh(y + 0.19, g.CurrentAnnualSpend, height=0.36, color=REFERENCE, label="Current plan")
    ax.barh(y - 0.19, g.OptimalAnnualSpend, height=0.36, color=ACCENT, label="Optimized plan")
    for yi, (c, o) in zip(y, g.to_numpy()):
        ax.text(max(c, o) * 1.02, yi, f"{o / c - 1:+.0%}", va="center", color=INK, fontsize=9)
    ax.set_yticks(y, g.index)
    ax.xaxis.set_major_formatter(money_m)
    ax.set_xlim(0, g.to_numpy().max() * 1.18)
    ax.set_xlabel("Annual spend ($5M total in both plans)")
    ax.set_title("Budget reallocation by channel")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    _save(fig, path)


def budget_frontier(frontier: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.plot(frontier.Budget, frontier.CurrentMixIncrementalSales, color=REFERENCE, marker="o",
            ms=5, label="Current mix, scaled")
    ax.plot(frontier.Budget, frontier.OptimalIncrementalSales, color=ACCENT, marker="o", ms=5,
            label="Optimized mix")
    ax.axvline(cfg.ANNUAL_BUDGET, color=INK_2, lw=1, ls=":")
    ax.text(cfg.ANNUAL_BUDGET, ax.get_ylim()[0], " $5M plan", color=INK_2, fontsize=8, va="bottom")
    ax.xaxis.set_major_formatter(money_m)
    ax.yaxis.set_major_formatter(money_m)
    ax.set_xlabel("Annual media budget")
    ax.set_ylabel("Annual incremental sales")
    ax.set_title("Budget efficiency frontier")
    ax.legend(loc="upper left")
    _save(fig, path)
