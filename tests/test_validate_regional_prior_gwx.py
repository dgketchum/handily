"""Unit tests for utils/validate_regional_prior_gwx_wells.py (pure functions).

Covers the blocked-CV scoring contract: block assignment matches the covariate
builder, OOF predictions never use a well's own spatial block (leak-free), DTW is
clipped at zero, and RBF-TPS recovers a planar WTE field out-of-fold.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_MOD = (
    Path(__file__).resolve().parents[1]
    / "utils"
    / "validate_regional_prior_gwx_wells.py"
)
_spec = importlib.util.spec_from_file_location(
    "validate_regional_prior_gwx_wells", _MOD
)
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)


def test_assign_blocks_matches_covariate_builder():
    """assign_blocks reproduces build_wte_covariate_table's integer-divide label."""
    x = np.array([0.0, 19999.0, 20001.0, -1.0, 45000.0])
    y = np.array([0.0, 0.0, 40000.0, 0.0, 12000.0])
    got = vr.assign_blocks(x, y, 20000)
    want = (
        (pd.Series(x) // 20000).astype("int64").astype(str)
        + "_"
        + (pd.Series(y) // 20000).astype("int64").astype(str)
    ).to_numpy()
    assert list(got) == list(want)


def _two_block_wells():
    # Block A near (0,0) -> "0_0"; block B near (100km,100km) -> "5_5".
    return pd.DataFrame(
        {
            "x5070": [0.0, 100.0, 200.0, 100_000.0, 100_100.0, 100_200.0],
            "y5070": [0.0, 100.0, 0.0, 100_000.0, 100_000.0, 100_100.0],
            "obs_wte_m": [10.0, 11.0, 12.0, 50.0, 51.0, 52.0],
            "dem_m": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )


def test_crossfit_oof_leak_free():
    """A block's OOF prediction must equal a fit on the complement only."""
    wells = _two_block_wells()
    blocks = vr.assign_blocks(
        wells["x5070"].to_numpy(), wells["y5070"].to_numpy(), 20000
    )
    assert set(blocks) == {"0_0", "5_5"}
    spec = vr.parse_method_name("idw_k3_p2")
    oof = vr.crossfit_oof_dtw(wells, blocks, spec, n_splits=2)

    a = blocks == "0_0"
    bsel = blocks == "5_5"
    xy = wells[["x5070", "y5070"]].to_numpy("float64")
    ywte = wells["obs_wte_m"].to_numpy("float64")
    dem = wells["dem_m"].to_numpy("float64")
    w = np.ones(len(wells))
    # B predicted from A only; A predicted from B only.
    predB = np.clip(
        dem[bsel] - vr.regional_fit_predict(xy[a], ywte[a], w[a], xy[bsel], spec),
        0,
        None,
    )
    predA = np.clip(
        dem[a] - vr.regional_fit_predict(xy[bsel], ywte[bsel], w[bsel], xy[a], spec),
        0,
        None,
    )
    assert np.allclose(oof[bsel], predB)
    assert np.allclose(oof[a], predA)


def test_crossfit_oof_dtw_clips_negative():
    """dem - pred_wte < 0 (water above ground) is clipped to 0, never negative."""
    wells = _two_block_wells()
    wells["obs_wte_m"] = 100.0  # both blocks high WTE
    wells.loc[3:, "dem_m"] = 0.0  # block B ground below the interpolated table
    blocks = vr.assign_blocks(
        wells["x5070"].to_numpy(), wells["y5070"].to_numpy(), 20000
    )
    spec = vr.parse_method_name("idw_k3_p2")
    oof = vr.crossfit_oof_dtw(wells, blocks, spec, n_splits=2)
    assert np.all(oof >= 0)
    assert oof[blocks == "5_5"].min() == pytest.approx(0.0)


def test_rbf_tps_oof_recovers_planar_wte():
    """Out-of-fold RBF-TPS recovers a planar WTE field across spatial blocks."""
    gx, gy = np.meshgrid(np.linspace(0, 120_000, 8), np.linspace(0, 120_000, 8))
    x, y = gx.ravel(), gy.ravel()
    wte = 0.0008 * x + 0.0005 * y + 20.0  # gentle plane, ~96 m + 60 m range
    dem = np.full_like(x, 1000.0)  # high enough that dem - wte stays positive
    wells = pd.DataFrame({"x5070": x, "y5070": y, "obs_wte_m": wte, "dem_m": dem})
    blocks = vr.assign_blocks(x, y, 20000)
    assert np.unique(blocks).size >= 5
    spec = vr.parse_method_name("rbf_tps_s25")
    oof_dtw = vr.crossfit_oof_dtw(wells, blocks, spec, n_splits=5)
    recovered_wte = dem - oof_dtw  # no clip triggered (dem high)
    mad = np.median(np.abs(recovered_wte - wte))
    assert mad < 2.0
