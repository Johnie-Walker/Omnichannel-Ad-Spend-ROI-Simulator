"""Static checks on the generated DAX/TMDL and parity of the DAX math with Python."""
import re

import numpy as np
import pandas as pd
import pytest

from mmm import config as cfg
from mmm.optimizer import ResponseSurface
from mmm.powerbi_export import PROJECT, build_measures, simulate_like_dax

TABLES_DIR = cfg.POWERBI_DIR / f"{PROJECT}.SemanticModel" / "definition" / "tables"
COL_REF = re.compile(r"('[^']+'|[A-Za-z_][A-Za-z0-9_]*)\[([^\]]+)\]")
MEASURE_REF = re.compile(r"(?<![\w'\]])\[([^\]]+)\]")


def _strip_strings(expr: str) -> str:
    return re.sub(r'"[^"]*"', '""', expr)


def test_parentheses_balanced():
    for m in build_measures():
        e = _strip_strings(m.expression)
        depth = 0
        for ch in e:
            depth += {"(": 1, ")": -1}.get(ch, 0)
            assert depth >= 0, m.name
        assert depth == 0, m.name


def test_measure_references_resolve():
    measures = {m.name for m in build_measures()}
    measures |= {f"{ch} Spend Level Value" for ch in cfg.CHANNELS} | {"Plan Baseline Selected"}
    for m in build_measures():
        e = COL_REF.sub("", _strip_strings(m.expression))
        for ref in MEASURE_REF.findall(e):
            assert ref in measures, f"{m.name} -> [{ref}]"


@pytest.mark.skipif(not TABLES_DIR.exists(), reason="run run_pipeline.py first")
def test_column_references_resolve():
    cols = {}
    for f in TABLES_DIR.glob("*.tmdl"):
        text = f.read_text(encoding="utf-8")
        name = re.match(r"table ('?)(.+?)\1\n", text).group(2)
        cols[name] = set(re.findall(r"\n\tcolumn '?(.+?)'?\n", text))
    for m in build_measures():
        for table, col in COL_REF.findall(_strip_strings(m.expression)):
            table = table.strip("'")
            assert table in cols, f"{m.name}: unknown table {table}"
            assert col in cols[table], f"{m.name}: unknown column {table}[{col}]"


def _params():
    rng = np.random.default_rng(4)
    rows = []
    for ch in cfg.CHANNELS:
        for r in cfg.REGIONS:
            cur = rng.uniform(1e5, 5e5)
            rows.append(dict(Channel=ch, Region=r, Beta=rng.uniform(1e4, 6e4), Norm=cur / 52 * rng.uniform(0.8, 1.2),
                             K=rng.uniform(0.4, 2), S=rng.choice([0.8, 1.0, 1.6]),
                             CurrentAnnualSpend=cur, OptimalAnnualSpend=cur * rng.uniform(0.3, 2.5)))
    return pd.DataFrame(rows)


def test_dax_mirror_matches_response_surface():
    p = _params()
    surf = ResponseSurface.from_params(p)
    levels = {"TV": 0.7, "Paid Search": 1.5, "Display": 0.0}
    x = p.CurrentAnnualSpend.to_numpy() * p.Channel.map(levels).fillna(1.0).to_numpy()
    out = simulate_like_dax(p, "Current Plan", levels)
    assert out["Spend"] == pytest.approx(x.sum())
    assert out["IncrementalSales"] == pytest.approx(surf.response(x).sum(), rel=1e-12)
    pos = x > 0
    expected_m = (x[pos] * surf.marginal_roi(x)[pos]).sum() / x.sum()
    assert out["MarginalROI"] == pytest.approx(expected_m, rel=1e-12)
    opt = simulate_like_dax(p, "Optimized Plan", {})
    assert opt["IncrementalSales"] == pytest.approx(surf.response(p.OptimalAnnualSpend.to_numpy()).sum())
