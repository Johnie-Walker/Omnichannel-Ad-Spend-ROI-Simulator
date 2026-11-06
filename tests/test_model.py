"""Data simulator checks and end-to-end ground-truth recovery (slow: one full fit)."""
import numpy as np
import pytest

from mmm import config as cfg
from mmm.data_generator import generate
from mmm.model import MarketingMixModel


@pytest.fixture(scope="module")
def sim():
    return generate(cfg.SEED)


@pytest.fixture(scope="module")
def fitted(sim):
    return MarketingMixModel().fit(sim.panel)


def test_simulated_budgets(sim):
    p = sim.panel
    assert len(p) == cfg.N_WEEKS * len(cfg.REGIONS)
    year = (p.Date - p.Date.min()).dt.days // 7 // cfg.WEEKS_PER_YEAR
    totals = p.groupby(year)[cfg.CHANNELS].sum().sum(axis=1).to_numpy()
    np.testing.assert_allclose(totals, cfg.YEARLY_SPEND, rtol=1e-9)
    assert (p[cfg.CHANNELS] >= 0).all().all()


def test_geo_go_dark_tests_present(sim):
    for ch in ["Paid Search", "Paid Social", "Display"]:
        assert (sim.panel[ch] == 0).sum() == 4 * 2 * (cfg.N_WEEKS // cfg.WEEKS_PER_YEAR)


def test_fit_quality(sim, fitted):
    dec = fitted.decompose(sim.panel)
    d = MarketingMixModel.diagnostics(dec["actual"], dec["predicted"])
    assert d.r2 > 0.9 and d.mape < 0.03 and 1.5 < d.durbin_watson < 2.5


def test_global_search_beats_truth(sim, fitted):
    """The estimator finds a posterior mode at least as good as the true curve
    parameters - remaining error is statistical, not an optimizer failure."""
    spend, yn, z, rows = fitted._setup(sim.panel, None)
    truth = [cfg.CHANNEL_TRUTH[c] for c in cfg.CHANNELS]
    v_true = np.r_[[t.theta for t in truth], [t.k for t in truth], [t.s for t in truth]]
    assert fitted.objective(fitted.curve_vector(), spend, yn, z, rows) <= \
        fitted.objective(v_true, spend, yn, z, rows) + 1e-9


def test_roi_recovery(sim, fitted):
    dec = fitted.decompose(sim.panel)
    true = sim.truth_contrib.groupby("Channel").TrueContribution.sum()
    for ci, ch in enumerate(cfg.CHANNELS):
        err = dec["contributions"][ch].sum() / true[ch] - 1
        assert abs(err) < 0.20, f"{ch}: ROI error {err:+.1%}"
    total_err = sum(dec["contributions"][c].sum() for c in cfg.CHANNELS) / true.sum() - 1
    assert abs(total_err) < 0.05


def test_adstock_recovery(sim, fitted):
    # Display (ROI ~1, smallest signal) has weakly identified carry-over timing:
    # its decay is allowed a wider band even though its ROI is recovered tightly.
    tol = {"Display": 0.20}
    for ci, ch in enumerate(cfg.CHANNELS):
        assert abs(fitted.theta[ci] - cfg.CHANNEL_TRUTH[ch].theta) < tol.get(ch, 0.10), ch
