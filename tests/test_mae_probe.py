"""Unit tests for the MAE probe reporting logic: metric-panel structure and the
summary writer (regression for the wide-basin section, which must index the nested
panel's "overall" cell, not treat the panel as a flat cell).

Loads utils/probe_mae_embeddings.py by path (repo test convention).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pmp = _load("probe_mae_embeddings")


def test_panel_structure_and_values():
    obs = np.array([1.0, 3.0, 7.0, 20.0, 40.0])
    pred = obs + np.array([0.5, -0.5, 1.0, -2.0, 4.0])  # residuals
    p = pmp.panel(obs, pred)
    # nested: overall + one cell per depth band
    assert "overall" in p
    for lo, hi in pmp.DEPTH_BANDS:
        assert f"{lo}-{hi}m" in p
    o = p["overall"]
    assert o["n"] == 5
    assert o["MAD_m"] == float(np.median(np.abs(pred - obs)))
    assert abs(o["bias_m"] - float(np.mean(pred - obs))) < 1e-9
    # depth banding is [lo, hi): obs=1 in 0-2, obs=3 in 2-5, obs=7 in 5-10, etc.
    assert p["0-2m"]["n"] == 1
    assert p["30-infm"]["n"] == 1


def test_panel_empty_band_returns_n_zero():
    obs = np.array([1.0, 1.5])  # all in 0-2 band
    pred = obs + 0.1
    p = pmp.panel(obs, pred)
    assert p["30-infm"] == {"n": 0}
    assert p["0-2m"]["n"] == 2


def test_write_summary_renders_wide_basin_section(tmp_path):
    # synthetic report mirroring main(): wide-slice values are FULL panels (nested),
    # so the summary must reach into ["overall"] -- the bug this guards against.
    obs = np.array([1.0, 4.0, 8.0, 25.0])
    predb = obs + 2.0
    predc = obs + 1.0
    rep = {
        "population_n": 4,
        "n_splits": 8,
        "locked_huc4_excluded": ["0707", "1019", "1605"],
        "point_covariates": ["pt_slope_deg"],
        "baseline_b_point_only": pmp.panel(obs, predb),
        "wide_basin_slice": {
            "huc2": ["13", "15", "16"],
            "n": 2,
            "baseline_b_point_only": pmp.panel(obs[:2], predb[:2]),
        },
        "arms": {
            "pyr_wide": {
                "emb_dim": 128,
                "a_embedding_only": pmp.panel(obs, predc),
                "c_embedding_plus_point": pmp.panel(obs, predc),
                "decision_c_vs_b": {
                    "delta_mad_c_minus_b_m": -1.0,
                    "ci95_low_m": -2.0,
                    "ci95_high_m": -0.5,
                    "frac_c_better": 1.0,
                },
                "wide_basin": {
                    "c_embedding_plus_point": pmp.panel(obs[:2], predc[:2]),
                    "delta_mad_c_minus_b_m": -0.5,
                },
            }
        },
    }
    out = tmp_path / "probe_summary.md"
    pmp._write_summary(out, rep)  # must not raise KeyError
    text = out.read_text()
    assert "Wide-basin slice" in text
    assert "pyr_wide" in text
    # baseline + one arm row rendered in the wide-basin table
    assert text.count("c: emb+point") >= 1


def test_build_contrasts_sign_and_presence():
    # constant-magnitude residuals so median-based MAD is exact and deterministic.
    n = 40
    obs = np.zeros(n)
    groups = np.array(["01", "02"] * (n // 2))
    preds_a = {
        "mae": np.full(n, 2.0),  # MAD 2
        "aef": np.full(n, 1.0),  # MAD 1
        "aefmae": np.full(n, 0.5),  # MAD 0.5
    }
    preds_c = {k: v for k, v in preds_a.items()}
    con = pmp.build_contrasts(obs, preds_a, preds_c, groups, seed=0)
    # all six specs present when all three arms exist
    assert set(con) == {
        "aef_vs_mae_embonly",
        "aef_vs_mae_withpoint",
        "aefmae_vs_aef_embonly",
        "aefmae_vs_mae_embonly",
        "aefmae_vs_aef_withpoint",
        "aefmae_vs_mae_withpoint",
    }
    # AEF vs MAE: MAD(aef) - MAD(mae) = 1 - 2 = -1 (AEF better), frac_y_better == 1
    d = con["aef_vs_mae_embonly"]
    assert abs(d["delta_mad_y_minus_x_m"] - (-1.0)) < 1e-9
    assert d["frac_y_better"] == 1.0
    # combo vs AEF: 0.5 - 1 = -0.5 (combo better)
    assert abs(con["aefmae_vs_aef_embonly"]["delta_mad_y_minus_x_m"] - (-0.5)) < 1e-9


def test_build_contrasts_new_arm_judged_against_incumbents():
    # a v2 candidate arm (not in CONTRAST_ORDER) must appear as y against EVERY
    # incumbent, with the historical trio labels unchanged.
    n = 40
    obs = np.zeros(n)
    groups = np.array(["01", "02"] * (n // 2))
    preds = {
        "aef": np.full(n, 1.0),
        "mae": np.full(n, 2.0),
        "aefmae": np.full(n, 0.5),
        "v2hard": np.full(n, 0.25),
    }
    con = pmp.build_contrasts(obs, preds, preds, groups, seed=0)
    expected_pairs = {
        "aef_vs_mae",
        "aefmae_vs_aef",
        "aefmae_vs_mae",
        "v2hard_vs_mae",
        "v2hard_vs_aef",
        "v2hard_vs_aefmae",
    }
    assert set(con) == {
        f"{p}_{t}" for p in expected_pairs for t in ("embonly", "withpoint")
    }
    # v2hard MAD 0.25 vs arm-of-record aefmae MAD 0.5 -> delta -0.25, v2hard better
    d = con["v2hard_vs_aefmae_embonly"]
    assert abs(d["delta_mad_y_minus_x_m"] - (-0.25)) < 1e-9
    assert d["frac_y_better"] == 1.0


def test_build_contrasts_skips_absent_arms():
    n = 20
    obs = np.zeros(n)
    groups = np.array(["01", "02"] * (n // 2))
    preds = {"aef": np.full(n, 1.0), "mae": np.full(n, 2.0)}  # no aefmae
    con = pmp.build_contrasts(obs, preds, preds, groups, seed=0)
    assert set(con) == {"aef_vs_mae_embonly", "aef_vs_mae_withpoint"}


def test_write_summary_renders_nv_and_contrasts(tmp_path):
    obs = np.array([1.0, 4.0, 8.0, 25.0])
    predb = obs + 2.0
    predc = obs + 1.0
    rep = {
        "population_n": 4,
        "n_splits": 8,
        "locked_huc4_excluded": ["0707", "1019", "1605"],
        "point_covariates": ["pt_slope_deg"],
        "baseline_b_point_only": pmp.panel(obs, predb),
        "wide_basin_slice": {
            "huc2": ["13", "15", "16"],
            "n": 2,
            "baseline_b_point_only": pmp.panel(obs[:2], predb[:2]),
        },
        "nv_slice": {
            "huc2": ["16"],
            "n": 2,
            "baseline_b_point_only": pmp.panel(obs[:2], predb[:2]),
        },
        "arms": {
            "aef": {
                "emb_dim": 64,
                "a_embedding_only": pmp.panel(obs, predc),
                "c_embedding_plus_point": pmp.panel(obs, predc),
                "decision_c_vs_b": {
                    "delta_mad_c_minus_b_m": -1.0,
                    "ci95_low_m": -2.0,
                    "ci95_high_m": -0.5,
                    "frac_c_better": 1.0,
                },
                "wide_basin": {
                    "c_embedding_plus_point": pmp.panel(obs[:2], predc[:2]),
                    "delta_mad_c_minus_b_m": -0.5,
                },
                "nv": {
                    "c_embedding_plus_point": pmp.panel(obs[:2], predc[:2]),
                    "delta_mad_c_minus_b_m": -0.4,
                },
            }
        },
        "contrasts": {
            "aef_vs_mae_embonly": {
                "delta_mad_y_minus_x_m": -0.7,
                "ci95_low_m": -1.2,
                "ci95_high_m": -0.1,
                "frac_y_better": 0.98,
            }
        },
    }
    out = tmp_path / "probe_summary.md"
    pmp._write_summary(out, rep)
    text = out.read_text()
    assert "NV closed-basin slice" in text
    assert "Cross-arm contrasts" in text
    assert "aef_vs_mae_embonly" in text
