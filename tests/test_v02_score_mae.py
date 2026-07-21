"""Unit tests for the MAE-arm scorer: population filters (unconfined-only,
sacrificial-HUC4 lockout, water-row exclusion, common-footprint join) and the
slice-panel structure. Loads utils/v02_score_mae.py by path (repo test convention).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load("v02_score_mae")


def test_slice_panel_structure_and_paired():
    df = pd.DataFrame(
        {
            "obs_dtw_m": [1.0, 3.0, 7.0, 20.0, 40.0, 1.5, 8.0, 25.0],
            "mae_dtw_m": [1.2, 3.1, 6.5, 18.0, 44.0, 1.4, 8.5, 22.0],
            "base_dtw_m": [1.8, 4.0, 8.0, 15.0, 50.0, 2.5, 9.0, 30.0],
        }
    )
    p = sm.slice_panel(df, np.ones(len(df), bool), "all")
    assert p["n"] == 8
    for arm in ("mae_arm", "baseline"):
        assert set(p[arm]) == {"overall", "by_depth_band", "shallow_skill"}
        assert {"n", "mad_m", "bias_mean_m", "median_resid_m", "rmse_m"} <= set(
            p[arm]["overall"]
        )
    # paired improvement carries a bootstrap CI on the MAD skill
    assert "mad_skill" in p["paired_overall"]
    assert len(p["paired_overall"]["mad_skill_ci95"]) == 2
    # per-band paired reductions exist for every contract band
    for lo, hi in sm.DEPTH_BANDS:
        assert sm._band_label(lo, hi) in p["paired_by_depth_band"]


def _write_bundle(tmp: Path):
    """Minimal arm-OOF + baseline + panels parquets covering the filter cases:
    one confined well, one sacrificial-HUC4 well, one water pseudo-row, three clean
    unconfined wells (one in HUC2=16 NV, one in HUC2=13 wide-basin)."""
    ids = ["clean_nv", "clean_wide", "clean_other", "confined", "sac", "water"]
    arm = pd.DataFrame(
        {
            "canonical_id": ids,
            "is_water_pseudo": [False, False, False, False, False, True],
            "obs_dtw_m": [2.0, 5.0, 9.0, 3.0, 4.0, np.nan],
            "gnn_dtw_m": [2.3, 5.5, 8.0, 3.1, 4.2, 0.0],
            "huc2": ["16", "13", "18", "18", "07", "16"],
            "cv_fold": [0, 1, 2, 3, 4, 0],
        }
    )
    base = pd.DataFrame(
        {
            "canonical_id": ids,
            "is_water_pseudo": arm["is_water_pseudo"],
            "gnn_dtw_m": [2.8, 6.0, 9.5, 3.6, 4.9, 0.0],
        }
    )
    panels = pd.DataFrame(
        {
            "canonical_id": ids,
            "is_water_pseudo": arm["is_water_pseudo"],
            "huc4": ["1601", "1301", "1801", "1801", "0707", "1601"],
            "confinement_class": [
                "unconfined",
                "unconfined_marginal",
                "unconfined",
                "confined",
                "unconfined",
                None,
            ],
        }
    )
    (tmp / "arm").mkdir()
    arm.to_parquet(tmp / "arm" / "gnn_oof_predictions.parquet")
    base.to_parquet(tmp / "baseline.parquet")
    panels.to_parquet(tmp / "panels.parquet")


def test_load_and_join_filters(tmp_path):
    _write_bundle(tmp_path)

    class Args:
        arm_oof = str(tmp_path / "arm" / "gnn_oof_predictions.parquet")
        baseline = str(tmp_path / "baseline.parquet")
        panels = str(tmp_path / "panels.parquet")

    df, diag = sm.load_and_join(Args)
    # only the 3 clean unconfined wells survive (confined + sacrificial + water dropped)
    assert diag["n_scored_unconfined"] == 3
    assert diag["n_confined_dropped"] == 1
    assert diag["n_sacrificial_dropped"] == 1
    assert set(df["canonical_id"]) == {"clean_nv", "clean_wide", "clean_other"}
    assert set(df["confinement_class"]) <= set(sm.UNCONFINED)
    # NV slice = HUC2 16 -> exactly clean_nv; wide = 13/15/16 -> clean_nv + clean_wide
    assert (df["huc2"] == "16").sum() == 1
    assert df["huc2"].isin(sm.WIDE_BASIN_HUC2).sum() == 2
