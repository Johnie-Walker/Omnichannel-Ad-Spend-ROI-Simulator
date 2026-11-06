"""Central configuration for the Omnichannel Ad Spend & ROI Simulator.

Everything that defines the scenario lives here: the channel/region universe,
the planning budget, optimizer guardrails, model search bounds and the
ground-truth parameters used to simulate the historical data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
POWERBI_DIR = PROJECT_ROOT / "powerbi"
POWERBI_DATA_DIR = POWERBI_DIR / "data"

SEED = 42

# --- Universe -----------------------------------------------------------------
CHANNELS = ["TV", "Paid Search", "Paid Social", "Display", "Radio"]
REGIONS = ["North", "South", "East", "West"]
START_DATE = "2023-01-02"          # a Monday; weekly grain (W-MON)
N_WEEKS = 156                      # 3 years of history
WEEKS_PER_YEAR = 52

# --- Planning -----------------------------------------------------------------
ANNUAL_BUDGET = 5_000_000.0        # budget to reallocate (USD / year)
# Guardrails per channel x region line item, as multiples of current spend.
# They keep the plan executable (no line item cut >70% or scaled past 2.5x).
MIN_SPEND_MULT = 0.30
MAX_SPEND_MULT = 2.50
OPTIMIZER_STARTS = 24              # multi-start SLSQP (Hill curves can be S-shaped)
FRONTIER_BUDGETS = [2.5e6, 3.0e6, 3.5e6, 4.0e6, 4.5e6, 5.0e6, 5.5e6,
                    6.0e6, 6.5e6, 7.0e6, 7.5e6, 8.0e6]

# --- Model estimation ---------------------------------------------------------
HOLDOUT_WEEKS = 26                 # last 26 weeks held out for validation
THETA_BOUNDS = (0.0, 0.85)         # adstock retention per week
K_BOUNDS = (0.20, 4.00)            # Hill half-saturation (x mean weekly spend)
S_BOUNDS = (0.50, 3.50)            # Hill shape
# Weakly-informative log-normal priors (MAP estimation). Medians: half-saturation
# at typical spend, mildly S-shaped curve. Wide enough for the data to dominate.
K_PRIOR = (0.0, 0.60)              # log K ~ N(mu, sd)
S_PRIOR = (0.405, 0.50)            # log S ~ N(log 1.5, sd)
# Partial pooling of regional media ROIs toward the channel's national ROI
# (ridge penalty on within-channel deviations; see model._pooling_rows).
POOLING_LAMBDA = 20.0
DE_POPSIZE = 10
DE_MAXITER = 200
FOURIER_ORDER = 2                  # yearly seasonality harmonics
BOOTSTRAP_REPLICATES = 200         # moving-block residual bootstrap
BOOTSTRAP_BLOCK = 4                # weeks per resampled block


@dataclass(frozen=True)
class ChannelTruth:
    """Ground-truth response parameters used only by the data simulator."""
    spend_share: float   # share of the historical budget
    theta: float         # adstock retention
    k: float             # Hill half-saturation, in units of mean weekly spend
    s: float             # Hill shape (>1 = S-curve)
    roi: float           # average ROI at mean historical spend (revenue per $)
    flighting: str       # weekly spend pattern


# Historical mix is deliberately TV/Display-heavy - the classic "legacy mix"
# that an MMM exposes: TV is past its efficient point and Display under-performs,
# while Paid Search / Paid Social still sit on the steep part of their curves.
CHANNEL_TRUTH: dict[str, ChannelTruth] = {
    #                       share  theta  K     S     ROI
    "TV":          ChannelTruth(0.38, 0.60, 0.45, 2.00, 2.60, "burst"),
    "Paid Search": ChannelTruth(0.14, 0.10, 2.20, 1.00, 4.60, "always_on_seasonal"),
    "Paid Social": ChannelTruth(0.12, 0.30, 1.60, 1.30, 3.40, "always_on"),
    "Display":     ChannelTruth(0.22, 0.25, 1.00, 1.00, 1.08, "always_on"),
    "Radio":       ChannelTruth(0.14, 0.45, 1.00, 1.60, 1.90, "pulse"),
}

# Region market size (share of national base sales) and the share of each
# channel's historical budget that region received.
REGION_SIZE = {"North": 0.30, "South": 0.25, "East": 0.25, "West": 0.20}

# Regional efficiency of each channel (multiplies the channel ROI).
REGION_EFFICIENCY = {
    #               TV    Search Social Display Radio
    "North":       (1.10, 0.95, 0.85, 1.00, 1.15),
    "South":       (1.05, 1.00, 0.95, 0.90, 1.10),
    "East":        (0.90, 1.05, 1.10, 1.05, 0.90),
    "West":        (0.85, 1.05, 1.25, 1.05, 0.80),
}

# Historical regional split of each channel's budget (legacy, size-driven,
# not efficiency-driven).
REGION_SPEND_SPLIT = {
    #               TV    Search Social Display Radio
    "North":       (0.34, 0.28, 0.26, 0.30, 0.34),
    "South":       (0.27, 0.25, 0.24, 0.25, 0.28),
    "East":        (0.22, 0.25, 0.26, 0.25, 0.22),
    "West":        (0.17, 0.22, 0.24, 0.20, 0.16),
}

ANNUAL_BASE_SALES = 35_000_000.0   # national non-media revenue per year
YEARLY_SPEND = [4.60e6, 4.80e6, 5.00e6]
NOISE_SD = 0.025                   # weekly noise, share of regional mean sales

CHANNEL_COLORS = {                 # validated categorical slots 1-5 (light)
    "TV": "#2a78d6",
    "Paid Search": "#eb6834",
    "Paid Social": "#1baf7a",
    "Display": "#eda100",
    "Radio": "#e87ba4",
}
