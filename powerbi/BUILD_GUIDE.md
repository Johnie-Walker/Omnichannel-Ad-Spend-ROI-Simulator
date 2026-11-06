# Power BI Simulator: Build Guide

The pipeline generates a **Power BI Project (PBIP)** whose semantic model is
complete: 11 CSV-backed tables, 10 relationships, 5 DAX what-if spend
parameters, a plan selector, a spend-curve axis and 36 measures. The report
pages are created empty. Follow the layout below to place the visuals (about 20
minutes).

## 1. Open the project

1. Run `run_pipeline.py` first. It writes `powerbi/data/*.csv` and the PBIP.
2. Open `powerbi/OmnichannelROISimulator.pbip` in Power BI Desktop (2024 or
   later). On older builds, enable *File > Options > Preview features >
   Power BI Project (.pbip) save option* and *Store semantic model using TMDL format*.
3. Click **Refresh**. The CSVs load from the `DataFolder` parameter, which is set
   to this project's absolute path. If you move the folder, update it under
   *Transform data > Edit parameters > DataFolder* (keep the trailing `\`).
4. *View > Themes > Browse for themes* and pick `powerbi/theme.json`. Its channel
   colors match the Python figures.

**Fallback (no PBIP support):** *Get data > Text/CSV* for each file in
`powerbi/data` except `ScenarioTests.csv`. Create the relationships listed in
section 5, create the calculated tables from the header of `DAX_Measures.dax`,
then paste each measure from that file into a `_Measures` table.

## 2. How the simulator works

`ModelParameters` holds one row per channel x region line item with the fitted
curve (`Beta`, `Norm`, `K`, `S`) and both plans' spend. The measures re-evaluate
the Hill curve live:

```
annual spend X   = baseline plan spend x channel spend level
incremental      = 52 x Beta x u^S / (u^S + K^S),     u = X / (52 x Norm)
marginal ROI     = Beta x S x K^S x u^(S-1) / (u^S + K^S)^2 / Norm
```

Every slider move gives the exact model response; no lookup grid is involved.
Marginal ROI at channel or total level is spend-weighted. It equals the sales
from the next $1 when that $1 is spread pro rata over the line items in
context. Below 1.00 the next dollar does not pay for itself.

## 3. Report layout

### Page 1: Executive Summary
| Visual | Fields |
|---|---|
| 4 cards | `Optimized Incremental Lift %`, `Current Plan Incremental Sales`, `Optimized Plan Incremental Sales`, `Budget Target` |
| Clustered bar | Y: `DimChannel[Channel]`; X: `Current Plan Spend`, `Optimized Plan Spend` |
| Table | `ChannelSummary`: Channel, HistoricalROI, CurrentMarginalROI, OptimalMarginalROI, SpendChangePct |
| Slicer | `DimRegion[Region]` |

### Page 2: Budget Simulator
| Visual | Fields |
|---|---|
| Slicer (tile) | `'Plan Baseline'[Plan]`, single select |
| 5 slicers (single-value slider) | `'TV Spend Level'[TV Spend Level]`, and the same for Paid Search, Paid Social, Display, Radio |
| Slicer | `DimRegion[Region]` |
| Cards | `Sim Spend`, `Sim Budget Status`, `Sim Incremental Sales`, `Sim Incremental Lift %`, `Sim ROI`, `Sim Marginal ROI`, `Sim Total Sales Lift %` |
| Clustered bar | Y: `DimChannel[Channel]`; X: `Current Plan Spend`, `Sim Spend` |
| Table | `DimChannel[Channel]`, `Sim Spend`, `Sim Spend Change %`, `Sim Incremental Sales`, `Sim ROI`, `Sim Marginal ROI` |

Story for stakeholders: start from *Current Plan* and move dollars out of the
channels with the lowest `Sim Marginal ROI` into the highest until the marginal
ROIs converge, keeping `Sim Budget Status` at "On budget". Switch to *Optimized
Plan* to see the SciPy solution, where the marginal ROIs are equal.

### Page 3: Response Curves & Marginal ROI
| Visual | Fields |
|---|---|
| Line chart | X: `'Spend Curve'[Spend Index]` (axis type *Continuous*); Y: `Curve Incremental Sales`; Legend: `DimChannel[Channel]` |
| Line chart | X: `'Spend Curve'[Spend Index]`; Y: `Curve Marginal ROI`, `Break-even Line`; Legend: `DimChannel[Channel]` |
| Slicer | `DimRegion[Region]` |

Spend Index 100% marks today's spend. The second chart shows where each channel
crosses break-even.

### Page 4: Model Validation
| Visual | Fields |
|---|---|
| Line chart | X: `DimDate[Date]`; Y: `Actual Sales`, `Predicted Sales`, `Holdout Prediction` |
| Table | `ModelDiagnostics`: Metric, Value, Scope |
| Bar chart | `ChannelSummary`: Channel x HistoricalROI (add TrueHistoricalROI to show the simulation audit) |
| Line chart | `BudgetFrontier`: Budget x OptimalIncrementalSales, CurrentMixIncrementalSales |

## 4. Verify the DAX against Python

`powerbi/data/ScenarioTests.csv` lists five slicer settings with the values the
Python mirror of the DAX produces (`mmm/powerbi_export.simulate_like_dax`). Set
the slicers as listed, with Region cleared, and the cards `Sim Spend`, `Sim
Incremental Sales`, `Sim ROI` and `Sim Marginal ROI` should match to the dollar.

## 5. Relationships (all many-to-one, single direction)

| From (many) | To (one) |
|---|---|
| FactMediaWeekly[Channel] / [Region] / [Date] | DimChannel / DimRegion / DimDate |
| FactSalesWeekly[Region] / [Date] | DimRegion / DimDate |
| ModelParameters[Channel] / [Region] | DimChannel / DimRegion |
| ResponseCurves[Channel] / [Region] | DimChannel / DimRegion |
| ChannelSummary[Channel] | DimChannel |

The what-if tables, `Plan Baseline` and `Spend Curve` are intentionally
disconnected; the measures read them with `SELECTEDVALUE`.
