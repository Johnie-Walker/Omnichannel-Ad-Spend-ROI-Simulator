"""End-to-end pipeline: simulate -> validate -> fit -> bootstrap -> optimize -> export.

    .\\.venv\\Scripts\\python.exe run_pipeline.py              # full run (~3-4 min)
    .\\.venv\\Scripts\\python.exe run_pipeline.py --bootstrap 40
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from mmm import config as cfg
from mmm import viz
from mmm.data_generator import generate
from mmm.model import MarketingMixModel
from mmm.optimizer import BudgetOptimizer, ResponseSurface
from mmm.powerbi_export import simulate_like_dax, write_powerbi
from mmm.transforms import adstock_half_life, geometric_adstock, hill


def log(msg: str, t0: float):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def current_plan(panel: pd.DataFrame, params: pd.DataFrame) -> np.ndarray:
    """Last 52 weeks of spend per line item, scaled to the planning budget."""
    last = panel[panel.Date > panel.Date.max() - pd.Timedelta(weeks=cfg.WEEKS_PER_YEAR)]
    spend = last.groupby("Region")[cfg.CHANNELS].sum()
    x = np.array([spend.loc[r, c] for c, r in zip(params.Channel, params.Region)])
    return x * cfg.ANNUAL_BUDGET / x.sum()


def historical_contribution(panel: pd.DataFrame, params: pd.DataFrame) -> pd.Series:
    """Total historical media contribution per channel for a params table."""
    out = {}
    for ch in cfg.CHANNELS:
        total = 0.0
        for _, p in params[params.Channel == ch].iterrows():
            x = panel.loc[panel.Region == p.Region].sort_values("Date")[ch].to_numpy()
            total += (p.Beta * hill(geometric_adstock(x, p.Theta) / p.Norm, p.K, p.S)).sum()
        out[ch] = total
    return pd.Series(out)


def channel_mroi(surface: ResponseSurface, x: np.ndarray) -> pd.Series:
    m = surface.marginal_roi(x)
    df = pd.DataFrame({"Channel": surface.channel, "w": x, "wm": x * m})
    g = df.groupby("Channel").sum()
    return g.wm / g.w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=cfg.BOOTSTRAP_REPLICATES)
    args = ap.parse_args()
    t0 = time.time()
    data_dir = cfg.OUTPUT_DIR / "data"
    for d in (data_dir, cfg.FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. data -------------------------------------------------------------------------
    sim = generate(cfg.SEED)
    panel = sim.panel
    panel.to_csv(data_dir / "simulated_panel.csv", index=False)
    sim.truth_params.to_csv(data_dir / "ground_truth_params.csv", index=False)
    log(f"simulated {len(panel)} region-weeks x {len(cfg.CHANNELS)} channels", t0)

    # 2. out-of-sample validation -------------------------------------------------------
    n_train = cfg.N_WEEKS - cfg.HOLDOUT_WEEKS
    val = MarketingMixModel().fit(panel, n_fit_weeks=n_train)
    dv = val.decompose(panel)
    d_train = MarketingMixModel.diagnostics(dv["actual"][:n_train], dv["predicted"][:n_train])
    d_test = MarketingMixModel.diagnostics(dv["actual"][n_train:], dv["predicted"][n_train:])
    log(f"validation fit: train MAPE {d_train.mape:.2%} | holdout MAPE {d_test.mape:.2%}, "
        f"R2 {d_test.r2:.3f}", t0)

    # 3. final model on all data -----------------------------------------------------------
    model = MarketingMixModel().fit(panel)
    dec = model.decompose(panel)
    d_full = MarketingMixModel.diagnostics(dec["actual"], dec["predicted"])
    params = model.params_table()
    log(f"final fit: R2 {d_full.r2:.4f}, MAPE {d_full.mape:.2%}, DW {d_full.durbin_watson:.2f} "
        f"| {model.optimizer_message}", t0)

    # 4. bootstrap -----------------------------------------------------------------------------
    from mmm.uncertainty import block_bootstrap
    boot = block_bootstrap(model, panel, n_boot=args.bootstrap)
    n_rep = boot.Replicate.nunique()
    log(f"bootstrap: {n_rep} block-bootstrap refits", t0)

    # 5. optimization ------------------------------------------------------------------------------
    x_cur = current_plan(panel, params)
    surface = ResponseSurface.from_params(params)
    opt = BudgetOptimizer(surface, x_cur)
    res = opt.optimize()
    table = opt.result_table(res)
    frontier = opt.frontier()
    base_next = dec["base"][-cfg.WEEKS_PER_YEAR:].sum(axis=0)              # per region
    total_cur = base_next.sum() + res.current_response
    total_lift = (res.optimal_response - res.current_response) / total_cur
    log(f"optimizer: {res.n_converged}/{res.n_starts} starts converged, KKT ok={res.kkt_ok} "
        f"(gap {res.kkt_max_gap:.1e}), lambda={res.shadow_price:.3f}", t0)
    log(f"PROJECTED INCREMENTAL SALES LIFT {res.incremental_lift:.2%} "
        f"(${res.current_response / 1e6:.2f}M -> ${res.optimal_response / 1e6:.2f}M); "
        f"total sales lift {total_lift:.2%}", t0)

    # bootstrap distribution of the recommended plan's lift + ROI CIs
    lifts, hist_roi, plan_roi = [], [], []
    hist_spend = panel[cfg.CHANNELS].sum()
    for _, rep in boot.groupby("Replicate"):
        rep = rep.set_index(["Channel", "Region"]).loc[list(zip(params.Channel, params.Region))].reset_index()
        sb = ResponseSurface.from_params(rep)
        lifts.append(sb.response(res.optimal).sum() / sb.response(res.current).sum() - 1)
        hist_roi.append(historical_contribution(panel, rep) / hist_spend)
    lifts = np.array(lifts)
    hist_roi = pd.DataFrame(hist_roi)
    lift_ci = np.percentile(lifts, [5, 95])
    log(f"lift 90% CI [{lift_ci[0]:.2%}, {lift_ci[1]:.2%}]; P(lift > 0) = {(lifts > 0).mean():.1%}", t0)

    # ground-truth audit (possible only because the data are simulated)
    tp = sim.truth_params.set_index(["Channel", "Region"]).loc[
        list(zip(params.Channel, params.Region))].reset_index()
    ts = ResponseSurface.from_params(tp)
    true_cur, true_opt = ts.response(res.current).sum(), ts.response(res.optimal).sum()
    true_best = BudgetOptimizer(ts, x_cur).optimize()
    log(f"ground truth: recommended plan delivers {true_opt / true_cur - 1:.2%} "
        f"(true optimum {true_best.incremental_lift:.2%})", t0)

    # 6. tables ------------------------------------------------------------------------------------
    hist_contrib = pd.Series({ch: dec["contributions"][ch].sum() for ch in cfg.CHANNELS})
    true_contrib = sim.truth_contrib.groupby("Channel").TrueContribution.sum()
    g = table.groupby("Channel")[["CurrentAnnualSpend", "OptimalAnnualSpend",
                                  "CurrentIncrementalSales", "OptimalIncrementalSales"]].sum()
    mroi_cur, mroi_opt = channel_mroi(surface, res.current), channel_mroi(surface, res.optimal)
    summary = pd.DataFrame([{
        "Channel": ch, "ChannelOrder": i + 1,
        "HistoricalSpend": hist_spend[ch], "HistoricalIncrementalSales": hist_contrib[ch],
        "HistoricalROI": hist_contrib[ch] / hist_spend[ch],
        "HistoricalROI_P05": hist_roi[ch].quantile(0.05), "HistoricalROI_P95": hist_roi[ch].quantile(0.95),
        "TrueHistoricalROI": true_contrib[ch] / hist_spend[ch],
        "CurrentPlanSpend": g.loc[ch, "CurrentAnnualSpend"], "OptimalPlanSpend": g.loc[ch, "OptimalAnnualSpend"],
        "SpendChangePct": g.loc[ch, "OptimalAnnualSpend"] / g.loc[ch, "CurrentAnnualSpend"] - 1,
        "CurrentPlanROI": g.loc[ch, "CurrentIncrementalSales"] / g.loc[ch, "CurrentAnnualSpend"],
        "OptimalPlanROI": g.loc[ch, "OptimalIncrementalSales"] / g.loc[ch, "OptimalAnnualSpend"],
        "CurrentMarginalROI": mroi_cur[ch], "OptimalMarginalROI": mroi_opt[ch],
        "Theta": model.theta[i], "HalfLifeWeeks": adstock_half_life(model.theta[i]),
        "K": model.k[i], "S": model.s[i],
        "TrueTheta": cfg.CHANNEL_TRUTH[ch].theta, "TrueK": cfg.CHANNEL_TRUTH[ch].k,
        "TrueS": cfg.CHANNEL_TRUTH[ch].s,
    } for i, ch in enumerate(cfg.CHANNELS)])
    summary.to_csv(cfg.OUTPUT_DIR / "channel_summary.csv", index=False)
    table.to_csv(cfg.OUTPUT_DIR / "optimal_allocation.csv", index=False)
    frontier.to_csv(cfg.OUTPUT_DIR / "budget_frontier.csv", index=False)
    model.controls_table().to_csv(cfg.OUTPUT_DIR / "control_coefficients.csv", index=False)
    boot.to_csv(cfg.OUTPUT_DIR / "bootstrap_params.csv", index=False)

    metrics = {
        "annual_budget": cfg.ANNUAL_BUDGET,
        "current_incremental_sales": res.current_response,
        "optimized_incremental_sales": res.optimal_response,
        "incremental_sales_lift": res.incremental_lift,
        "incremental_sales_lift_ci90": lift_ci.tolist(),
        "prob_lift_positive": float((lifts > 0).mean()),
        "projected_base_sales": float(base_next.sum()),
        "total_sales_lift": total_lift,
        "current_roi": res.current_response / cfg.ANNUAL_BUDGET,
        "optimized_roi": res.optimal_response / cfg.ANNUAL_BUDGET,
        "shadow_price_marginal_roi": res.shadow_price,
        "kkt_ok": res.kkt_ok, "kkt_max_gap": res.kkt_max_gap,
        "optimizer_starts_converged": f"{res.n_converged}/{res.n_starts}",
        "fit_r2": d_full.r2, "fit_mape": d_full.mape, "fit_nrmse": d_full.nrmse,
        "durbin_watson": d_full.durbin_watson,
        "holdout_r2": d_test.r2, "holdout_mape": d_test.mape, "train_mape": d_train.mape,
        "truth_lift_of_recommended_plan": true_opt / true_cur - 1,
        "truth_optimal_lift": true_best.incremental_lift,
        "truth_current_incremental_sales": true_cur,
        "bootstrap_replicates": int(n_rep),
    }
    (cfg.OUTPUT_DIR / "summary.json").write_text(json.dumps(metrics, indent=2))

    # 7. figures -------------------------------------------------------------------------------
    holdout_pred = dv["predicted"]
    viz.actual_vs_predicted(dec["dates"], dec["actual"], dec["predicted"], holdout_pred,
                            cfg.HOLDOUT_WEEKS, cfg.FIGURE_DIR / "01_actual_vs_predicted.png")
    viz.roi_by_channel(summary, cfg.FIGURE_DIR / "02_roi_by_channel.png")
    viz.response_curves(surface, table, cfg.FIGURE_DIR / "03_response_curves.png")
    viz.marginal_roi_curves(surface, table, res.shadow_price, cfg.FIGURE_DIR / "04_marginal_roi_curves.png")
    viz.budget_reallocation(table, cfg.FIGURE_DIR / "05_budget_reallocation.png")
    viz.budget_frontier(frontier, cfg.FIGURE_DIR / "06_budget_frontier.png")
    log("figures written", t0)

    # 8. Power BI ------------------------------------------------------------------------------
    dates = dec["dates"]
    regions = cfg.REGIONS
    fact_media = pd.concat([pd.DataFrame({
        "Date": dates, "Region": r, "Channel": ch,
        "Spend": dec["spend"][ci][:, ri], "Contribution": dec["contributions"][ch][:, ri]})
        for ci, ch in enumerate(cfg.CHANNELS) for ri, r in enumerate(regions)], ignore_index=True)
    media_total = sum(dec["contributions"].values())
    fact_sales = pd.concat([pd.DataFrame({
        "Date": dates, "Region": r, "ActualSales": dec["actual"][:, ri],
        "PredictedSales": dec["predicted"][:, ri], "BaseSales": dec["base"][:, ri],
        "MediaSales": media_total[:, ri],
        "HoldoutPrediction": np.where(np.arange(len(dates)) >= n_train, holdout_pred[:, ri], np.nan),
        "Split": np.where(np.arange(len(dates)) >= n_train, "Holdout", "Train")})
        for ri, r in enumerate(regions)], ignore_index=True)
    model_params = params.merge(table, on=["Channel", "Region"])
    idx = np.round(np.arange(0, cfg.MAX_SPEND_MULT + 1e-9, 0.05), 2)
    curves = []
    for i, p in model_params.iterrows():
        x = p.CurrentAnnualSpend * idx
        one = ResponseSurface.from_params(model_params.loc[[i]])
        resp = np.array([one.response(np.array([v]))[0] for v in x])
        mr = np.array([one.marginal_roi(np.array([v]))[0] for v in x])
        curves.append(pd.DataFrame({"Channel": p.Channel, "Region": p.Region, "SpendIndex": idx,
                                    "AnnualSpend": x, "IncrementalSales": resp,
                                    "MarginalROI": np.where(np.isfinite(mr), mr, np.nan)}))
    curves = pd.concat(curves, ignore_index=True)
    dim_date = pd.DataFrame({"Date": dates})
    dim_date["Year"] = dim_date.Date.dt.year
    dim_date["Quarter"] = "Q" + dim_date.Date.dt.quarter.astype(str)
    dim_date["YearQuarter"] = dim_date.Year.astype(str) + " " + dim_date.Quarter
    dim_date["Month"] = dim_date.Date.dt.month
    dim_date["MonthName"] = dim_date.Date.dt.strftime("%b")
    dim_date["WeekOfYear"] = dim_date.Date.dt.isocalendar().week.astype("int64")
    dim_channel = pd.DataFrame({"Channel": cfg.CHANNELS, "ChannelOrder": range(1, 6),
                                "ChannelGroup": ["Offline", "Online", "Online", "Online", "Offline"],
                                "ChannelColor": [cfg.CHANNEL_COLORS[c] for c in cfg.CHANNELS]})
    dim_region = pd.DataFrame({"Region": regions, "RegionOrder": range(1, 5),
                               "MarketSizeShare": [cfg.REGION_SIZE[r] for r in regions],
                               "ProjectedBaseSales": base_next})
    diagnostics = pd.DataFrame([
        ("In-sample R2", d_full.r2, "All data"), ("In-sample MAPE", d_full.mape, "All data"),
        ("NRMSE", d_full.nrmse, "All data"), ("Durbin-Watson", d_full.durbin_watson, "All data"),
        ("Holdout R2", d_test.r2, "Last 26 weeks"), ("Holdout MAPE", d_test.mape, "Last 26 weeks"),
        ("Incremental lift (optimized)", res.incremental_lift, "Plan"),
        ("Incremental lift CI90 low", lift_ci[0], "Plan"), ("Incremental lift CI90 high", lift_ci[1], "Plan"),
        ("Total sales lift (optimized)", total_lift, "Plan"),
        ("Shadow price (marginal ROI)", res.shadow_price, "Plan"),
        ("Ground-truth lift of plan", true_opt / true_cur - 1, "Simulation audit"),
    ], columns=["Metric", "Value", "Scope"])

    # DAX parity cases: set the slicers as listed and compare the cards
    scenarios = [("Current plan, all 100%", "Current Plan", {}),
                 ("Optimized plan, all 100%", "Optimized Plan", {}),
                 ("Current: TV 70%, Search 150%, Social 150%, Display 50%", "Current Plan",
                  {"TV": 0.7, "Paid Search": 1.5, "Paid Social": 1.5, "Display": 0.5}),
                 ("Optimized: TV 120%", "Optimized Plan", {"TV": 1.2}),
                 ("Current: Display 0%", "Current Plan", {"Display": 0.0})]
    tests = []
    for name, plan, lv in scenarios:
        r = simulate_like_dax(model_params, plan, lv)
        row = {"Scenario": name, "PlanBaseline": plan}
        row.update({f"{ch} Spend Level": lv.get(ch, 1.0) for ch in cfg.CHANNELS})
        row.update({f"Expected {k}": v for k, v in r.items()})
        tests.append(row)
    scenario_tests = pd.DataFrame(tests)
    assert abs(scenario_tests.loc[1, "Expected IncrementalSales"] - res.optimal_response) < 1e-6 * res.optimal_response

    pbip = write_powerbi({
        "DimChannel": dim_channel, "DimRegion": dim_region, "DimDate": dim_date,
        "FactMediaWeekly": fact_media, "FactSalesWeekly": fact_sales,
        "ModelParameters": model_params, "ResponseCurves": curves,
        "ChannelSummary": summary, "BudgetFrontier": frontier,
        "ModelDiagnostics": diagnostics, "ScenarioTests": scenario_tests})
    log(f"Power BI project written: {pbip}", t0)


if __name__ == "__main__":
    main()
