"""Unit tests for the deep-aquifer-datum helpers in build_conus_graph_inputs.py.

Covers the two properties the deep regional datum depends on:
  * deep_well_mask is LOCAL (per-HUC6 quantile), so a locally-deep well in a
    shallow regime is selected while a globally-deep-but-locally-shallow well is
    not -- the whole point of not using a single global cut. Sparse HUC6s fall
    back to the HUC4 threshold.
  * crossfit_deep_idw is leak-free (a held-out fold is predicted only from deep
    wells in OTHER folds), predicts for ALL wells, and guards thin deep folds
    (kk = min(k, n_deep_train)).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_conus_graph_inputs.py"
_spec = importlib.util.spec_from_file_location("build_conus_graph_inputs", _MOD)
bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bc)


def test_deep_well_mask_is_local_not_global():
    # HUC6 100100 (HUC4 1001): shallow regime, dtw 1..8 -> q0.75=6.25 -> {7,8}.
    # HUC6 100200 (HUC4 1002): deep regime, dtw 100..130 -> q0.75=122.5 -> {130}.
    wells = pd.DataFrame(
        {
            "huc8": ["10010000"] * 8 + ["10020000"] * 4,
            "mean_dtw": [1, 2, 3, 4, 5, 6, 7, 8, 100, 110, 120, 130],
        }
    )
    mask = bc.deep_well_mask(wells, quantile=0.75, unit="huc6", min_per_unit=2)
    # locally-deep shallow-regime wells ARE selected (dtw 7,8)
    assert mask[6] and mask[7]
    # a globally-deep but locally-shallow well is NOT selected (dtw 100 < 122.5)
    assert not mask[8]
    # only the locally-deepest of the deep regime is selected (dtw 130)
    assert mask[11] and not mask[9] and not mask[10]


def test_deep_well_mask_sparse_huc6_falls_back_to_huc4():
    # HUC4 1001: dense HUC6 100101 (40 wells) + sparse HUC6 100100 (3 wells).
    # min_per_unit=30 -> sparse 100100 uses the HUC4 quartile, not its own.
    dense_dtw = list(range(1, 41))  # HUC6 100101
    sparse_dtw = [100, 100, 100]  # HUC6 100100 (above the HUC4 q0.75)
    wells = pd.DataFrame(
        {
            "huc8": ["10010100"] * 40 + ["10010000"] * 3,
            "mean_dtw": dense_dtw + sparse_dtw,
        }
    )
    mask = bc.deep_well_mask(wells, quantile=0.75, unit="huc6", min_per_unit=30)
    # HUC4 1001 q0.75 over all 43 wells is well below 100, so the sparse trio
    # (fallback to HUC4 threshold) are all deep.
    assert mask[40] and mask[41] and mask[42]


def test_crossfit_deep_idw_leak_free_and_predicts_all():
    # Deep wells on a line, one per fold; a query sits on top of the same-fold
    # deep well. Leak-free => its prediction comes from the NEXT-nearest deep
    # well in another fold, not its own.
    q = [0.1, 0.0]
    d = [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]
    xy_all = np.array([q] + d, dtype="float64")
    fold_all = np.array([0, 0, 1, 2, 3])
    xy_deep = np.array(d, dtype="float64")
    dtw_deep = np.array([100.0, 200.0, 300.0, 400.0])
    fold_deep = np.array([0, 1, 2, 3])

    pred = bc.crossfit_deep_idw(
        xy_all, xy_deep, dtw_deep, fold_all, fold_deep, k=2, power=6.0
    )
    assert np.isfinite(pred).all()  # predicts for ALL wells
    # query in fold 0: excludes the same-fold deep well (dtw 100), so the
    # prediction is pulled toward the next-nearest other-fold well (dtw 200).
    assert abs(pred[0] - 200.0) < abs(pred[0] - 100.0)
    assert 200.0 <= pred[0] < 300.0


def test_crossfit_deep_idw_thin_fold_guard():
    # k larger than the deep training pool per held-out fold -> kk clamps, no crash.
    xy_all = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype="float64")
    fold_all = np.array([0, 1, 2, 3])
    xy_deep = xy_all.copy()
    dtw_deep = np.array([10.0, 20.0, 30.0, 40.0])
    fold_deep = fold_all.copy()
    pred = bc.crossfit_deep_idw(
        xy_all, xy_deep, dtw_deep, fold_all, fold_deep, k=10, power=2.0
    )
    assert np.isfinite(pred).all()
