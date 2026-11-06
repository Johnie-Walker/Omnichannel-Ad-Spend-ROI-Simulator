"""Power BI deliverables: CSV star schema, DAX measures and a PBIP project.

The semantic model (tables, relationships, what-if parameters, measures) is
written as a Power BI Project (PBIP) in TMDL, so it opens directly in Power BI
Desktop. Every DAX measure is defined once below and emitted both into the
TMDL and into a standalone ``.dax`` file for copy-paste builds.

The simulator measures re-evaluate the fitted Hill curves live in DAX from
the ``ModelParameters`` table, so slicer moves give exact model responses -
not lookups into a pre-computed grid.
"""
from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

PROJECT = "OmnichannelROISimulator"
_NS = uuid.UUID("6f1c2a52-8f0e-4c55-9d1e-0a9b7c3e5d21")

CUR = r"\$#,0"
ROI = "0.00"
PCT = "0.0%"


def _tag(*parts: str) -> str:
    """Deterministic lineage tag so regenerating the model is diff-stable."""
    return str(uuid.uuid5(_NS, "|".join(parts)))


def _q(name: str) -> str:
    """Quote a TMDL object name unless it is a plain identifier."""
    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", name) else f"'{name}'"


# ----------------------------------------------------------------------------- DAX
def _var(ch: str) -> str:
    return "_m" + ch.replace(" ", "")


def _param_table(ch: str) -> str:
    return f"{ch} Spend Level"


def _param_measure(ch: str) -> str:
    return f"{ch} Spend Level Value"


def _header() -> str:
    lines = ['VAR _plan = [Plan Baseline Selected]']
    lines += [f"VAR {_var(ch)} = [{_param_measure(ch)}]" for ch in cfg.CHANNELS]
    return "\n".join(lines)


def _row_spend(indent: str = "    ") -> str:
    switch = ", ".join(f'"{ch}", {_var(ch)}' for ch in cfg.CHANNELS)
    return (f'{indent}VAR _base = IF ( _plan = "Optimized Plan", ModelParameters[OptimalAnnualSpend], '
            f'ModelParameters[CurrentAnnualSpend] )\n'
            f'{indent}VAR _mult = SWITCH ( ModelParameters[Channel], {switch}, 1 )\n'
            f'{indent}VAR _x = _base * _mult')


def _row_hill(indent: str = "    ") -> str:
    return (f"{indent}VAR _u = DIVIDE ( _x, 52 * ModelParameters[Norm] )\n"
            f"{indent}VAR _us = POWER ( _u, ModelParameters[S] )\n"
            f"{indent}VAR _ks = POWER ( ModelParameters[K], ModelParameters[S] )")


def _row_mroi(indent: str = "    ") -> str:
    # dR/dX = Beta * S * K^S * u^(S-1) / (u^S + K^S)^2 / Norm ; limits at u = 0
    return (f"{indent}VAR _mroi =\n"
            f"{indent}    IF ( _u > 0,\n"
            f"{indent}        ModelParameters[Beta] * ModelParameters[S] * _ks * POWER ( _u, ModelParameters[S] - 1 )\n"
            f"{indent}            / POWER ( _us + _ks, 2 ) / ModelParameters[Norm],\n"
            f"{indent}        IF ( ModelParameters[S] > 1, 0,\n"
            f"{indent}            IF ( ModelParameters[S] = 1, DIVIDE ( ModelParameters[Beta], ModelParameters[K] * ModelParameters[Norm] ) ) ) )")


@dataclass
class Measure:
    name: str
    expression: str
    fmt: str
    folder: str
    description: str


def build_measures() -> list[Measure]:
    m: list[Measure] = []
    add = lambda *a: m.append(Measure(*a))  # noqa: E731

    # ---- 1. historical (model decomposition) -----------------------------------
    f = "1. Historical"
    add("Historical Spend", "SUM ( FactMediaWeekly[Spend] )", CUR, f,
        "Actual media spend over the modeled history.")
    add("Historical Incremental Sales", "SUM ( FactMediaWeekly[Contribution] )", CUR, f,
        "Sales attributed to media by the MMM (Adstock + Hill decomposition).")
    add("Historical ROI", "DIVIDE ( [Historical Incremental Sales], [Historical Spend] )", ROI, f,
        "Incremental sales per $1 of historical spend.")
    add("Actual Sales", "SUM ( FactSalesWeekly[ActualSales] )", CUR, f, "Observed sales.")
    add("Predicted Sales", "SUM ( FactSalesWeekly[PredictedSales] )", CUR, f, "MMM in-sample fit.")
    add("Holdout Prediction", "SUM ( FactSalesWeekly[HoldoutPrediction] )", CUR, f,
        "Out-of-sample forecast from the validation model (last 26 weeks only).")
    add("Base Sales", "SUM ( FactSalesWeekly[BaseSales] )", CUR, f,
        "Sales the model attributes to non-media drivers (trend, seasonality, price, holidays).")
    add("Media Share of Sales",
        "DIVIDE ( CALCULATE ( [Historical Incremental Sales], REMOVEFILTERS ( DimChannel ) ), [Actual Sales] )",
        PCT, f, "Share of actual sales driven by paid media.")
    add("Fit MAPE",
        "AVERAGEX ( FactSalesWeekly, DIVIDE ( ABS ( FactSalesWeekly[ActualSales] - FactSalesWeekly[PredictedSales] ), FactSalesWeekly[ActualSales] ) )",
        PCT, f, "Mean absolute percentage error of the in-sample fit.")

    # ---- 2. plans (static, from the Python optimizer) ------------------------------
    f = "2. Plans"
    add("Current Plan Spend", "SUM ( ModelParameters[CurrentAnnualSpend] )", CUR, f,
        "Last 52 weeks of spend, scaled to the $5M annual budget.")
    add("Current Plan Incremental Sales", "SUM ( ModelParameters[CurrentIncrementalSales] )", CUR, f,
        "Projected annual incremental sales of the current plan.")
    add("Optimized Plan Spend", "SUM ( ModelParameters[OptimalAnnualSpend] )", CUR, f,
        "SciPy SLSQP allocation of the same budget.")
    add("Optimized Plan Incremental Sales", "SUM ( ModelParameters[OptimalIncrementalSales] )", CUR, f,
        "Projected annual incremental sales of the optimized plan.")
    add("Optimized Incremental Lift %",
        "DIVIDE ( [Optimized Plan Incremental Sales] - [Current Plan Incremental Sales], [Current Plan Incremental Sales] )",
        PCT, f, "Lift in media-driven sales from reallocating the same budget.")
    add("Projected Base Sales", "SUM ( DimRegion[ProjectedBaseSales] )", CUR, f,
        "Next-year non-media sales (last 52 weeks of modeled base).")
    add("Budget Target",
        "CALCULATE ( [Current Plan Spend], REMOVEFILTERS ( DimChannel ) )", CUR, f,
        "Annual budget in the current filter context ($5M nationally).")

    # ---- 3. simulator (live DAX on the fitted curves) ---------------------------
    f = "3. Simulator"
    add("Plan Baseline Selected", "SELECTEDVALUE ( 'Plan Baseline'[Plan], \"Current Plan\" )", "", f,
        "Starting plan the spend-level sliders scale.")
    add("Sim Spend",
        f"{_header()}\nRETURN\n    SUMX ( ModelParameters,\n{_row_spend('        ')}\n        RETURN _x )",
        CUR, f, "Annual spend of the simulated plan (baseline plan x channel spend levels).")
    add("Sim Incremental Sales",
        f"{_header()}\nRETURN\n    SUMX ( ModelParameters,\n{_row_spend('        ')}\n{_row_hill('        ')}\n"
        "        RETURN 52 * ModelParameters[Beta] * DIVIDE ( _us, _us + _ks ) )",
        CUR, f, "Steady-state annual incremental sales: 52 x Beta x Hill(weekly spend / Norm; K, S).")
    add("Sim ROI", "DIVIDE ( [Sim Incremental Sales], [Sim Spend] )", ROI, f,
        "Average ROI of the simulated plan.")
    body_mroi = (f"{_row_spend('        ')}\n{_row_hill('        ')}\n{_row_mroi('        ')}")
    add("Sim Marginal ROI",
        f"{_header()}\nVAR _spend = [Sim Spend]\n"
        f"VAR _weighted =\n    SUMX ( ModelParameters,\n{body_mroi}\n        RETURN _x * _mroi )\n"
        f"VAR _avg =\n    AVERAGEX ( ModelParameters,\n{body_mroi}\n        RETURN _mroi )\n"
        "RETURN IF ( _spend > 0, DIVIDE ( _weighted, _spend ), _avg )",
        ROI, f, "Incremental sales from the next $1, spread pro rata over the line items in context. "
                "Below 1.00 the next dollar loses money.")
    add("Sim Incremental Lift %",
        "DIVIDE ( [Sim Incremental Sales] - [Current Plan Incremental Sales], [Current Plan Incremental Sales] )",
        PCT, f, "Simulated vs current-plan incremental sales.")
    add("Sim Total Sales",
        "[Projected Base Sales] + CALCULATE ( [Sim Incremental Sales], REMOVEFILTERS ( DimChannel ) )",
        CUR, f, "Projected base + simulated media sales (all channels).")
    add("Sim Total Sales Lift %",
        "VAR _cur = [Projected Base Sales] + CALCULATE ( [Current Plan Incremental Sales], REMOVEFILTERS ( DimChannel ) )\n"
        "RETURN DIVIDE ( [Sim Total Sales] - _cur, _cur )",
        PCT, f, "Simulated vs current-plan total sales.")
    add("Sim Budget Variance",
        "CALCULATE ( [Sim Spend], REMOVEFILTERS ( DimChannel ) ) - [Budget Target]", CUR, f,
        "Simulated spend minus the budget (positive = over budget).")
    add("Sim Budget Status",
        "VAR _v = [Sim Budget Variance]\n"
        "RETURN\n    SWITCH ( TRUE (),\n        ABS ( _v ) < 1, \"On budget\",\n"
        "        _v > 0, \"Over budget by \" & FORMAT ( _v, \"$#,0\" ),\n"
        "        \"Under budget by \" & FORMAT ( -_v, \"$#,0\" ) )",
        "", f, "Text status for a card.")
    add("Sim Spend Change %", "DIVIDE ( [Sim Spend] - [Current Plan Spend], [Current Plan Spend] )", PCT, f,
        "Simulated vs current spend.")

    # ---- 4. response curves ----------------------------------------------------------
    f = "4. Response Curves"
    curve_row = ("        VAR _x = ModelParameters[CurrentAnnualSpend] * _idx\n" + _row_hill("        "))
    add("Curve Incremental Sales",
        "VAR _idx = SELECTEDVALUE ( 'Spend Curve'[Spend Index] )\nRETURN\n    IF ( NOT ISBLANK ( _idx ),\n"
        f"        SUMX ( ModelParameters,\n{curve_row}\n"
        "            RETURN 52 * ModelParameters[Beta] * DIVIDE ( _us, _us + _ks ) ) )",
        CUR, f, "Annual incremental sales when every line item in context runs at Spend Index x current spend.")
    add("Curve Spend",
        "VAR _idx = SELECTEDVALUE ( 'Spend Curve'[Spend Index] )\n"
        "RETURN IF ( NOT ISBLANK ( _idx ), [Current Plan Spend] * _idx )", CUR, f,
        "Annual spend at the curve point.")
    add("Curve Marginal ROI",
        "VAR _idx = SELECTEDVALUE ( 'Spend Curve'[Spend Index] )\nVAR _spend = [Current Plan Spend] * _idx\n"
        "VAR _weighted =\n    SUMX ( ModelParameters,\n"
        f"{curve_row}\n{_row_mroi('        ')}\n        RETURN _x * _mroi )\n"
        "RETURN IF ( NOT ISBLANK ( _idx ) && _spend > 0, DIVIDE ( _weighted, _spend ) )",
        ROI, f, "Marginal ROI along the response curve; where it crosses 1.00 is break-even.")
    add("Break-even Line", "1", ROI, f, "Reference line: marginal ROI = 1.00.")
    return m


# --------------------------------------------------------------------- python mirror
def simulate_like_dax(params: pd.DataFrame, plan: str, levels: dict[str, float]) -> dict[str, float]:
    """Exact Python mirror of the simulator measures (used for parity tests)."""
    base = params["OptimalAnnualSpend"] if plan == "Optimized Plan" else params["CurrentAnnualSpend"]
    x = base.to_numpy(float) * params["Channel"].map(levels).fillna(1.0).to_numpy(float)
    beta, norm = params["Beta"].to_numpy(float), params["Norm"].to_numpy(float)
    k, s = params["K"].to_numpy(float), params["S"].to_numpy(float)
    u = x / (52 * norm)
    us, ks = u ** s, k ** s
    sales = 52 * beta * us / (us + ks)
    with np.errstate(divide="ignore", invalid="ignore"):
        mroi = np.where(u > 0, beta * s * ks * u ** (s - 1) / (us + ks) ** 2 / norm,
                        np.where(s > 1, 0.0, np.where(s == 1, beta / (k * norm), np.nan)))
    spend = x.sum()
    weighted = np.where(x > 0, x * mroi, 0.0).sum()        # DAX: BLANK rows drop out of SUMX
    return dict(Spend=spend, IncrementalSales=sales.sum(), ROI=sales.sum() / spend,
                MarginalROI=float(weighted / spend) if spend > 0 else float(np.nanmean(mroi)))


# --------------------------------------------------------------------------- TMDL
_DTYPE = {"string": ("string", "type text"), "int64": ("int64", "Int64.Type"),
          "double": ("double", "type number"), "dateTime": ("dateTime", "type date")}


def _col_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "dateTime"
    if pd.api.types.is_integer_dtype(series):
        return "int64"
    if pd.api.types.is_float_dtype(series):
        return "double"
    return "string"


def _indent(text: str, tabs: int) -> str:
    pad = "\t" * tabs
    return "\n".join(pad + line for line in text.splitlines())


def _m_table_tmdl(name: str, df: pd.DataFrame, sort_by: dict[str, str] | None = None,
                  fmt: dict[str, str] | None = None) -> str:
    sort_by, fmt = sort_by or {}, fmt or {}
    out = [f"table {_q(name)}", f"\tlineageTag: {_tag(name)}", ""]
    types = []
    for col in df.columns:
        t = _col_type(df[col])
        types.append((col, t))
        out.append(f"\tcolumn {_q(col)}")
        out.append(f"\t\tdataType: {_DTYPE[t][0]}")
        if col in fmt:
            out.append(f"\t\tformatString: {fmt[col]}")
        elif t == "dateTime":
            out.append("\t\tformatString: yyyy-mm-dd")
        out.append(f"\t\tlineageTag: {_tag(name, col)}")
        out.append(f"\t\tsummarizeBy: {'sum' if t == 'double' and col in fmt and fmt[col] == CUR else 'none'}")
        out.append(f"\t\tsourceColumn: {col}")
        if col in sort_by:
            out.append(f"\t\tsortByColumn: {_q(sort_by[col])}")
        out += ["", "\t\tannotation SummarizationSetBy = Automatic", ""]
    type_list = ", ".join(f'{{"{c}", {_DTYPE[t][1]}}}' for c, t in types)
    m = (f'let\n'
         f'    Source = Csv.Document(File.Contents(DataFolder & "{name}.csv"), '
         f'[Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]),\n'
         f'    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),\n'
         f'    Typed = Table.TransformColumnTypes(Promoted, {{{type_list}}}, "en-US")\n'
         f'in\n'
         f'    Typed')
    out += [f"\tpartition {_q(name)} = m", "\t\tmode: import", "\t\tsource =", _indent(m, 4), "",
            "\tannotation PBI_ResultType = Table", ""]
    return "\n".join(out)


def _measure_tmdl(ms: Measure, owner: str) -> str:
    lines = [f"\t/// {ms.description}"]
    expr = ms.expression
    if "\n" in expr:
        lines.append(f"\tmeasure {_q(ms.name)} =")
        lines.append(_indent(expr, 3))
    else:
        lines.append(f"\tmeasure {_q(ms.name)} = {expr}")
    if ms.fmt:
        lines.append(f"\t\tformatString: {ms.fmt}")
    lines.append(f"\t\tdisplayFolder: {ms.folder}")
    lines.append(f"\t\tlineageTag: {_tag(owner, ms.name)}")
    return "\n".join(lines) + "\n"


def _what_if_tmdl(ch: str) -> str:
    t = _param_table(ch)
    hi = cfg.MAX_SPEND_MULT
    return "\n".join([
        f"table '{t}'",
        f"\tlineageTag: {_tag(t)}",
        "",
        f"\t/// Spend level for {ch} as a share of the baseline plan (100% = unchanged).",
        f"\tmeasure '{_param_measure(ch)}' = SELECTEDVALUE ( '{t}'[{t}], 1 )",
        "\t\tformatString: 0%",
        f"\t\tlineageTag: {_tag(t, 'value')}",
        "",
        f"\tcolumn '{t}'",
        "\t\tdataType: double",
        "\t\tformatString: 0%",
        f"\t\tlineageTag: {_tag(t, 'col')}",
        "\t\tsummarizeBy: none",
        "\t\tsourceColumn: [Value]",
        "",
        "\t\textendedProperty ParameterMetadata =",
        "\t\t\t\t{",
        '\t\t\t\t  "version": 0',
        "\t\t\t\t}",
        "",
        "\t\tannotation SummarizationSetBy = User",
        "",
        f"\tpartition '{t}' = calculated",
        "\t\tmode: import",
        f"\t\tsource = GENERATESERIES ( 0, {hi}, 0.05 )",
        "",
    ])


def _calc_tables_tmdl() -> dict[str, str]:
    plan = "\n".join([
        "table 'Plan Baseline'",
        f"\tlineageTag: {_tag('Plan Baseline')}",
        "",
        "\tcolumn Plan",
        "\t\tdataType: string",
        f"\t\tlineageTag: {_tag('Plan Baseline', 'Plan')}",
        "\t\tsummarizeBy: none",
        "\t\tsourceColumn: [Plan]",
        "\t\tsortByColumn: 'Plan Order'",
        "",
        "\t\tannotation SummarizationSetBy = Automatic",
        "",
        "\tcolumn 'Plan Order'",
        "\t\tdataType: int64",
        "\t\tisHidden",
        f"\t\tlineageTag: {_tag('Plan Baseline', 'Plan Order')}",
        "\t\tsummarizeBy: none",
        "\t\tsourceColumn: [Plan Order]",
        "",
        "\t\tannotation SummarizationSetBy = Automatic",
        "",
        "\tpartition 'Plan Baseline' = calculated",
        "\t\tmode: import",
        '\t\tsource = DATATABLE ( "Plan", STRING, "Plan Order", INTEGER, { { "Current Plan", 1 }, { "Optimized Plan", 2 } } )',
        "",
    ])
    curve = "\n".join([
        "table 'Spend Curve'",
        f"\tlineageTag: {_tag('Spend Curve')}",
        "",
        "\tcolumn 'Spend Index'",
        "\t\tdataType: double",
        "\t\tformatString: 0%",
        f"\t\tlineageTag: {_tag('Spend Curve', 'idx')}",
        "\t\tsummarizeBy: none",
        "\t\tsourceColumn: [Value]",
        "",
        "\t\tannotation SummarizationSetBy = User",
        "",
        "\tpartition 'Spend Curve' = calculated",
        "\t\tmode: import",
        "\t\tsource = GENERATESERIES ( 0, 3, 0.05 )",
        "",
    ])
    measures = [_measure_tmdl(ms, "_Measures") for ms in build_measures()]
    mtab = "\n".join([
        "table '_Measures'",
        f"\tlineageTag: {_tag('_Measures')}",
        "",
        *measures,
        "\tcolumn Value",
        "\t\tdataType: int64",
        "\t\tisHidden",
        f"\t\tlineageTag: {_tag('_Measures', 'Value')}",
        "\t\tsummarizeBy: none",
        "\t\tsourceColumn: [Value]",
        "",
        "\t\tannotation SummarizationSetBy = Automatic",
        "",
        "\tpartition '_Measures' = calculated",
        "\t\tmode: import",
        "\t\tsource = { 0 }",
        "",
    ])
    out = {"Plan Baseline": plan, "Spend Curve": curve, "_Measures": mtab}
    for ch in cfg.CHANNELS:
        out[_param_table(ch)] = _what_if_tmdl(ch)
    return out


RELATIONSHIPS = [
    ("FactMediaWeekly", "Channel", "DimChannel", "Channel"),
    ("FactMediaWeekly", "Region", "DimRegion", "Region"),
    ("FactMediaWeekly", "Date", "DimDate", "Date"),
    ("FactSalesWeekly", "Region", "DimRegion", "Region"),
    ("FactSalesWeekly", "Date", "DimDate", "Date"),
    ("ModelParameters", "Channel", "DimChannel", "Channel"),
    ("ModelParameters", "Region", "DimRegion", "Region"),
    ("ResponseCurves", "Channel", "DimChannel", "Channel"),
    ("ResponseCurves", "Region", "DimRegion", "Region"),
    ("ChannelSummary", "Channel", "DimChannel", "Channel"),
]

COLUMN_FORMATS = {
    "FactMediaWeekly": {"Spend": CUR, "Contribution": CUR},
    "FactSalesWeekly": {"ActualSales": CUR, "PredictedSales": CUR, "BaseSales": CUR,
                        "MediaSales": CUR, "HoldoutPrediction": CUR},
    "DimRegion": {"ProjectedBaseSales": CUR},
}
SORT_BY = {"DimChannel": {"Channel": "ChannelOrder"}, "DimRegion": {"Region": "RegionOrder"},
           "DimDate": {"MonthName": "Month"}}


def write_powerbi(tables: dict[str, pd.DataFrame], out_dir: Path = cfg.POWERBI_DIR) -> Path:
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(data_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")

    sm = out_dir / f"{PROJECT}.SemanticModel"
    rp = out_dir / f"{PROJECT}.Report"
    for d in (sm, rp):
        if d.exists():
            shutil.rmtree(d)
    (sm / "definition" / "tables").mkdir(parents=True)
    rp.mkdir(parents=True)

    model_tables = [n for n in tables if n != "ScenarioTests"]
    calc = _calc_tables_tmdl()
    all_tables = model_tables + list(calc)

    def w(path: Path, text: str):
        path.write_text(text, encoding="utf-8")

    w(sm / "definition.pbism", json.dumps({"version": "4.0", "settings": {}}, indent=2))
    w(sm / "definition" / "database.tmdl", "database\n\tcompatibilityLevel: 1567\n")
    query_order = json.dumps(["DataFolder"] + model_tables)
    w(sm / "definition" / "model.tmdl", "\n".join([
        "model Model",
        "\tculture: en-US",
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
        "\tdiscourageImplicitMeasures",
        "\tsourceQueryCulture: en-US",
        "\tdataAccessOptions",
        "\t\tlegacyRedirects",
        "\t\treturnErrorValuesAsNull",
        "",
        f"annotation PBI_QueryOrder = {query_order}",
        "",
        "annotation __PBI_TimeIntelligenceEnabled = 0",
        "",
        *[f"ref table {_q(t)}" for t in all_tables],
        "",
    ]))
    folder = str(data_dir.resolve()) + "\\"
    w(sm / "definition" / "expressions.tmdl", "\n".join([
        f'expression DataFolder = "{folder}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]',
        f"\tlineageTag: {_tag('DataFolder')}",
        "",
        "\tannotation PBI_ResultType = Text",
        "",
    ]))
    for name in model_tables:
        w(sm / "definition" / "tables" / f"{name}.tmdl",
          _m_table_tmdl(name, tables[name], SORT_BY.get(name), COLUMN_FORMATS.get(name)))
    for name, text in calc.items():
        w(sm / "definition" / "tables" / f"{name}.tmdl", text)
    w(sm / "definition" / "relationships.tmdl", "\n".join(
        f"relationship {_tag('rel', a, b, c, d)}\n\tfromColumn: {_q(a)}.{_q(b)}\n\ttoColumn: {_q(c)}.{_q(d)}\n"
        for a, b, c, d in RELATIONSHIPS))

    w(rp / "definition.pbir", json.dumps(
        {"version": "4.0", "datasetReference": {"byPath": {"path": f"../{PROJECT}.SemanticModel"}}}, indent=2))
    pages = ["Executive Summary", "Budget Simulator", "Response Curves & Marginal ROI", "Model Validation"]
    w(rp / "report.json", json.dumps({
        "config": json.dumps({"version": "5.43", "themeCollection": {}, "activeSectionIndex": 0}),
        "layoutOptimization": 0,
        "sections": [{"config": "{}", "displayName": p, "displayOption": 1, "filters": "[]",
                      "height": 720.0, "name": f"ReportSection{i}", "ordinal": i,
                      "visualContainers": [], "width": 1280.0} for i, p in enumerate(pages)],
    }, indent=2))
    w(out_dir / f"{PROJECT}.pbip", json.dumps({
        "version": "1.0", "artifacts": [{"report": {"path": f"{PROJECT}.Report"}}],
        "settings": {"enableAutoRecovery": True}}, indent=2))
    w(out_dir / ".gitignore", "**/.pbi/localSettings.json\n**/.pbi/cache.abf\n")

    # standalone DAX + theme
    dax = ["// Omnichannel Ad Spend & ROI Simulator - DAX measures (generated; do not edit by hand)",
           "// Home table: _Measures.  What-if parameter tables:"]
    for ch in cfg.CHANNELS:
        dax.append(f"//   '{_param_table(ch)}' = GENERATESERIES ( 0, {cfg.MAX_SPEND_MULT}, 0.05 )"
                   f"   ->  [{_param_measure(ch)}] = SELECTEDVALUE ( '{_param_table(ch)}'[{_param_table(ch)}], 1 )")
    dax += ["//   'Plan Baseline' = DATATABLE ( \"Plan\", STRING, \"Plan Order\", INTEGER, "
            "{ { \"Current Plan\", 1 }, { \"Optimized Plan\", 2 } } )",
            "//   'Spend Curve'   = GENERATESERIES ( 0, 3, 0.05 )   (rename column to [Spend Index])", ""]
    current = None
    for ms in build_measures():
        if ms.folder != current:
            current = ms.folder
            dax += ["", f"// ===== {current} " + "=" * 50]
        dax += [f"// {ms.description}", f"[{ms.name}] =", ms.expression, f"// format: {ms.fmt or 'text'}", ""]
    w(out_dir / "DAX_Measures.dax", "\n".join(dax))
    w(out_dir / "theme.json", json.dumps({
        "name": "Omnichannel ROI",
        "dataColors": list(cfg.CHANNEL_COLORS.values()) + ["#008300", "#6250d6", "#e34948"],
        "background": "#fcfcfb", "foreground": "#0b0b0b", "tableAccent": "#2a78d6",
        "good": "#008300", "bad": "#e34948", "neutral": "#8a8985",
    }, indent=2))
    return out_dir / f"{PROJECT}.pbip"
