"""Unit tests for utils/build_wte_regional_prior.py (pure functions only)."""

import importlib.util
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point

# utils/ is a script dir, not a package -- load the module by path.
_MOD_PATH = (
    Path(__file__).resolve().parents[1] / "utils" / "build_wte_regional_prior.py"
)
_spec = importlib.util.spec_from_file_location("build_wte_regional_prior", _MOD_PATH)
brp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brp)


# --- 17.1 IDW exact point --------------------------------------------------- #
def test_idw_exact_point():
    train_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    train_y = np.array([100.0, 200.0])
    query_xy = np.array([[0.0, 0.0]])
    pred = brp.idw_predict(train_xy, train_y, query_xy, k=2, power=2.0)
    assert pred[0] == pytest.approx(100.0)


# --- 17.2 IDW weighted midpoint --------------------------------------------- #
def test_idw_midpoint_equal_weights():
    train_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    train_y = np.array([100.0, 200.0])
    query_xy = np.array([[5.0, 0.0]])
    pred = brp.idw_predict(train_xy, train_y, query_xy, k=2, power=2.0)
    assert pred[0] == pytest.approx(150.0)


def test_idw_weight_skews_prediction():
    train_xy = np.array([[0.0, 0.0], [10.0, 0.0]])
    train_y = np.array([100.0, 200.0])
    query_xy = np.array([[5.0, 0.0]])
    # heavier weight on the second point pulls the equidistant query toward 200
    pred = brp.idw_predict(
        train_xy, train_y, query_xy, train_weight=np.array([1.0, 3.0]), k=2, power=2.0
    )
    assert pred[0] == pytest.approx(175.0)


# --- 17.3 group-fold leakage ------------------------------------------------ #
def test_group_fold_no_leakage():
    # two well-separated blocks, each a distinct constant WTE; held-out preds
    # must come from the *other* block and so cannot equal the held-out label.
    xs = [0, 100, 200] + [100_000, 100_100, 100_200]
    block = ["A"] * 3 + ["B"] * 3
    label = [10.0] * 3 + [90.0] * 3
    df = gpd.GeoDataFrame(
        {
            "x_5070": xs,
            "y_5070": [0] * 6,
            "obs_wte_m": label,
            "weight_model": [1.0] * 6,
            "block_20km": block,
        },
        geometry=[Point(x, 0) for x in xs],
        crs=5070,
    )
    folds = brp.make_group_folds(df, "block_20km")
    spec = brp.MethodSpec("idw_k2_p2", "idw", {"k": 2, "power": 2.0})
    pred = brp.crossfit_regional_method(df, folds, spec)
    assert np.isfinite(pred).all()
    # block A rows predicted from block B's constant (90) and vice versa
    assert np.allclose(pred[:3], 90.0)
    assert np.allclose(pred[3:], 10.0)
    assert not np.allclose(pred, df["obs_wte_m"].to_numpy())


# --- 17.4 metric test ------------------------------------------------------- #
def test_score_dtw_product_metrics():
    pred_dtw = np.array([1.0, 4.0, 3.0, 50.0])
    obs_dtw = np.array([2.0, 3.0, 8.0, 60.0])
    dem = 100.0
    pred_wte = dem - pred_dtw
    obs_wte = dem - obs_dtw
    w = np.ones(4)
    m = brp.score_dtw_product(pred_dtw, obs_dtw, pred_wte, obs_wte, w)

    assert m["n"] == 4
    assert m["mad"] == pytest.approx(3.0)  # median(|[-1,1,-5,-10]|)
    assert m["bias"] == pytest.approx(-3.0)  # median([-1,1,-5,-10])
    assert m["wmad"] == pytest.approx(4.25)
    assert m["mean_bias"] == pytest.approx(-3.75)
    assert m["rmse"] == pytest.approx(np.sqrt(127.0 / 4.0))
    # obs shallow {0,1}=2, pred shallow {0,1,2}=3, TP {0,1}=2
    assert m["n_obs_shallow"] == 2
    assert m["n_pred_shallow"] == 3
    assert m["shallow_recall"] == pytest.approx(1.0)
    assert m["shallow_precision"] == pytest.approx(2.0 / 3.0)
    assert m["shallow_f1"] == pytest.approx(0.8)
    # WTE residual is the exact sign flip of the DTW error
    assert m["wte_bias"] == pytest.approx(-m["bias"])
    assert m["wte_mean_bias"] == pytest.approx(-m["mean_bias"])


def test_score_dtw_product_drops_nonfinite():
    pred_dtw = np.array([1.0, np.nan, 3.0])
    obs_dtw = np.array([2.0, 5.0, 8.0])
    pred_wte = np.array([99.0, 95.0, 97.0])
    obs_wte = np.array([98.0, 95.0, 92.0])
    w = np.ones(3)
    m = brp.score_dtw_product(pred_dtw, obs_dtw, pred_wte, obs_wte, w)
    assert m["n"] == 2


def test_wavg_zero_weight_is_nan():
    assert np.isnan(brp._wavg(np.array([1.0, 2.0]), np.array([0.0, 0.0])))


# --- 17.5 column derivation sign conventions -------------------------------- #
def test_regional_method_column_signs():
    wells = gpd.GeoDataFrame(
        {
            "dem_m": [1000.0],
            "obs_dtw_m": [20.0],
            "obs_wte_m": [980.0],
            "fac_wte_m": [985.0],
            "fac_dtw_m": [15.0],
        },
        geometry=[Point(0, 0)],
        crs=5070,
    )
    brp.add_regional_method_columns(wells, "m", np.array([990.0]))
    r = wells.iloc[0]
    assert r["regional_wte_oof__m"] == pytest.approx(990.0)
    assert r["regional_dtw_raw_oof__m"] == pytest.approx(10.0)
    assert r["regional_dtw_oof__m"] == pytest.approx(10.0)
    assert r["regional_residual_wte__m"] == pytest.approx(-10.0)
    assert r["regional_residual_dtw__m"] == pytest.approx(-10.0)
    assert r["fac_minus_regional_wte__m"] == pytest.approx(-5.0)  # 985 - 990
    assert r["fac_minus_regional_dtw__m"] == pytest.approx(5.0)  # 15 - 10


def test_clip_dtw_above_ground():
    # predicted WTE above the DEM -> raw DTW negative, clipped to 0
    dtw_raw, dtw_clip, wte_clip = brp.clip_dtw(np.array([1005.0]), np.array([1000.0]))
    assert dtw_raw[0] == pytest.approx(-5.0)
    assert dtw_clip[0] == pytest.approx(0.0)
    assert wte_clip[0] == pytest.approx(1000.0)


# --- method-name parsing + leakage guard ------------------------------------ #
def test_method_name_roundtrip():
    for k, p in [(16, 2.0), (32, 1.5), (64, 3.0)]:
        name = brp._idw_name(k, p)
        spec = brp.parse_method_name(name)
        assert spec.family == "idw"
        assert spec.params["k"] == k
        assert spec.params["power"] == pytest.approx(p)
    spec = brp.parse_method_name("rbf_tps_smooth100")
    assert spec.family == "rbf"
    assert spec.params["kernel"] == "thin_plate_spline"
    assert spec.params["smoothing"] == pytest.approx(100.0)


def test_residual_features_are_not_leaky():
    feats = brp._residual_features("idw_k32_p2")
    assert not (set(feats) & brp.DISALLOWED_RESIDUAL_FEATURES)
    assert "ma_dtw_m" not in feats
    assert "dem_m" not in feats
    assert "x_5070" not in feats


def test_build_method_specs_defaults():
    import argparse

    # a namespace mimicking unset CLI args -> the default method ladder
    ns = argparse.Namespace(
        methods=None, idw_k=None, idw_power=None, rbf_kernel=None, rbf_smoothing=None
    )
    specs = brp.build_method_specs(ns, {})
    names = {s.name for s in specs}
    assert {"idw_k16_p2", "idw_k32_p2", "idw_k64_p2"} <= names
    assert {"rbf_tps_s25", "rbf_tps_s100", "rbf_tps_s400"} <= names


# --- nested-CV leak-free invariants ----------------------------------------- #
def _multiblock_df(n_blocks=5, per_block=3):
    rows = []
    for b in range(n_blocks):
        for j in range(per_block):
            rows.append(
                {
                    "x_5070": b * 50_000.0 + j * 100.0,
                    "y_5070": j * 100.0,
                    "obs_wte_m": 1000.0 + 40.0 * b + j,
                    "weight_model": 1.0,
                    "block_20km": f"b{b}",
                }
            )
    return gpd.GeoDataFrame(
        rows, geometry=[Point(r["x_5070"], r["y_5070"]) for r in rows], crs=5070
    )


def test_global_oof_equals_fit_on_outer_train():
    # the residual model reuses the global OOF regional prior as its leak-free
    # held-out prediction; this only holds if the global OOF for a test fold
    # equals a direct fit on that fold's outer-training rows.
    df = _multiblock_df()
    folds = brp.make_group_folds(df, "block_20km")
    spec = brp.MethodSpec("idw_k4_p2", "idw", {"k": 4, "power": 2.0})
    oof = brp.crossfit_regional_method(df, folds, spec)
    xy = df[["x_5070", "y_5070"]].to_numpy("float64")
    y = df["obs_wte_m"].to_numpy("float64")
    w = df["weight_model"].to_numpy("float64")
    for tr, te in folds:
        direct = brp.regional_fit_predict(xy[tr], y[tr], w[tr], xy[te], spec)
        assert np.allclose(oof[te], direct)


def test_grouped_folds_requires_two_blocks():
    with pytest.raises(ValueError):
        brp._grouped_folds(np.array(["only_one"] * 5))
    folds = brp._grouped_folds(np.array(["a", "a", "b", "b", "c", "c"]))
    assert len(folds) >= 2


def test_inner_train_folds_exclude_outer_test_block():
    # the inner CV used to build residual training targets is restricted to the
    # outer-train rows, so the held-out outer block can never enter inner training.
    df = _multiblock_df()
    folds = brp.make_group_folds(df, "block_20km")
    for tr, _ in folds:
        train_df = df.iloc[tr]
        inner = brp._grouped_folds(train_df["block_20km"].to_numpy())
        train_blocks = set(train_df["block_20km"])
        for itr, ite in inner:
            seen = set(train_df["block_20km"].to_numpy()[itr]) | set(
                train_df["block_20km"].to_numpy()[ite]
            )
            assert seen <= train_blocks
