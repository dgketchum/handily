"""Synthetic tests for the WP2 two-surface identifiability feasibility harness.

No disk dependencies: build a synthetic collocated-well population in memory and
assert that the three analysis stages (per-cell Spearman ordering, 2-component
BIC preference, metadata->component AUC) separate an ORDERED nest population
(deep completions sample systematically deeper water) from unstructured noise.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ts = _load("v02_two_surface_feasibility")


def _finish_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the grid/metadata columns the analyses expect (cells pre-assigned)."""
    df = df.copy()
    df["ix"] = df["cell"].astype(np.int64)
    df["iy"] = 0
    df["state"] = "XX"
    # construction metadata: a screen_bottom just under the completion, a casing
    # in the upper third, a head above the screen, and a modest record length.
    df["screen_bottom"] = df["comp_m"]
    df["screen_bottom_present"] = 1.0
    df["casing_depth"] = df["comp_m"] / 3.0
    df["casing_depth_present"] = 1.0
    # benign construction geometry (function of completion depth only, NOT dtw)
    # so it cannot circularly predict a dtw-derived component label
    df["head_above_screen"] = df["comp_m"] * 0.25
    df["head_above_screen_present"] = 1.0
    df["obs_count"] = 12.0
    df["confinement_confidence"] = 0.3
    return df


def _ordered_population(seed=0, n_cells=200, per_cell=8):
    """Nested wells: two completion clusters (~20 m shallow, ~60 m deep); deep
    completions carry +20 m DTW (a systematic downward vertical gradient)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        base_dtw = rng.uniform(5.0, 40.0)
        for j in range(per_cell):
            deep = j % 2 == 0
            comp = (60.0 if deep else 20.0) + rng.normal(0, 2.0)
            dtw = base_dtw + (20.0 if deep else 0.0) + rng.normal(0, 1.0)
            rows.append({"cell": c, "comp_m": comp, "mean_dtw": dtw})
    return _finish_columns(pd.DataFrame(rows))


def _noise_population(seed=1, n_cells=400, per_cell=8):
    """Completion depth and DTW independent within each cell (no ordering)."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        base_dtw = rng.uniform(5.0, 40.0)
        for _ in range(per_cell):
            comp = rng.uniform(15.0, 65.0)
            # single surface: modest within-cell dispersion, no vertical order
            dtw = base_dtw + rng.normal(0, 1.5)
            rows.append({"cell": c, "comp_m": comp, "mean_dtw": dtw})
    return _finish_columns(pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# ordered nest population -> structure recovered
# --------------------------------------------------------------------------- #
def test_ordered_population_positive_spearman():
    df = _ordered_population()
    base = ts.cell_base_table(df)
    _, rho = ts.vertical_ordering(df, base, seed=0)
    assert float(np.median(rho.dropna())) > 0.5


def test_ordered_population_prefers_two_components():
    df = _ordered_population()
    base = ts.cell_base_table(df)
    panel, _, labelled = ts.two_component_analysis(df, base, seed=0, cap=10_000)
    assert panel["overall_frac_prefers_2"] > 0.5
    assert len(labelled) > 0


def test_ordered_population_metadata_auc_high():
    df = _ordered_population()
    base = ts.cell_base_table(df)
    _, _, labelled = ts.two_component_analysis(df, base, seed=0, cap=10_000)
    auc = ts.metadata_component_auc(labelled, base, ts.META_FEATURES, seed=0)
    assert auc["oof_auc"] > 0.8


# --------------------------------------------------------------------------- #
# noise population -> no recoverable structure
# --------------------------------------------------------------------------- #
def test_noise_population_no_two_component_and_low_auc():
    df = _noise_population()
    base = ts.cell_base_table(df)
    panel, _, labelled = ts.two_component_analysis(df, base, seed=0, cap=10_000)
    assert panel["overall_frac_prefers_2"] < 0.2
    auc = ts.metadata_component_auc(labelled, base, ts.META_FEATURES, seed=0)
    # AUC may be degenerate if too few spurious splits; otherwise must be ~chance
    assert auc.get("oof_auc", 0.5) < 0.6


def test_noise_population_spearman_near_zero():
    df = _noise_population()
    base = ts.cell_base_table(df)
    _, rho = ts.vertical_ordering(df, base, seed=0)
    assert abs(float(np.median(rho.dropna()))) < 0.15
