"""Leak-free regional WTE prior + residual diagnostics from the point covariates.

First practical step of the hybrid water-table plan
(``notes/plans/hybrid_wte.md``):

    WTE_hat  = WTE_regional_prior + local_expression_model
    DTW_hat  = DEM - WTE_hat

This script does NOT rasterize. It builds a cross-fitted, point-level regional
WTE prior (simple spatial interpolation of water-table well elevations) and a
residual diagnostic package that answers:

  1. How strong is a well-derived regional aquifer surface by itself?
  2. Where does FAC10 agree or disagree with that regional surface?
  3. Do existing FAC10 / topo / climate / recharge / ET / BFI covariates explain
     the residual WTE structure?

Discipline (see CLAUDE.md and the plan):
  * Train + score water-table labels only (``is_water_table_label``, i.e.
    confinement_class in {unconfined, unconfined_marginal}). Confined /
    likely-confined / artesian / unknown never enter loss or headline metrics.
  * Springs are shallow-discharge constraints, never regional-aquifer labels --
    they are excluded from interpolation here.
  * Ma is a benchmark only. It is scored for comparison and flagged
    ``benchmark_only``; it never enters interpolation, feature selection, the
    residual model, or any tuning.
  * Every label-derived prediction is out-of-fold under spatial blocked CV
    (``block_20km`` by default), so numbers reflect transfer, not memorization.
  * The regional prior may use all finite water-table WTE labels; FAC comparisons
    are reported on the common FAC footprint group.

Run::

    uv run python utils/build_wte_regional_prior.py \\
        --config configs/wte/nm_rga_hybrid_v0.toml --overwrite

Outputs land under ``<out_dir>.parent/regional_prior`` (regional_prior_*.{csv,
parquet,json}); the optional residual model adds residual_model_*. See
``notes/plans/hybrid_wte.md`` for the broader strategy.
"""

import argparse
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

SHALLOW_M = 5.0
DEEP_M = 30.0
DEPTH_BINS = [-np.inf, 2, 5, 10, 30, 60, 100, np.inf]
DEPTH_LABELS = ["<2", "2-5", "5-10", "10-30", "30-60", "60-100", ">100"]

# Scalable covariates correlated against the regional residual (section 11).
CORR_FEATURES = [
    "elev_above_str1_m",
    "elev_above_str5_m",
    "elev_above_str7_m",
    "log1p_dist_str1_m",
    "log1p_dist_str5_m",
    "log1p_dist_str7_m",
    "slope_m_m",
    "local_relief_300m_m",
    "log1p_drainage_km2",
    "aridity_index",
    "reitz_effective_recharge",
    "reitz_total_recharge",
    "reitz_et",
    "wolock_bfi",
]

# Allowed first-pass residual-model features (section 12.2). Deliberately EXCLUDES
# ma_dtw_m / zell_sanford_dtw / fan_dtw (benchmarks or pre-built WT products),
# coordinates, dem_m (coordinate shortcut), and any target/residual column.
STATIC_RESIDUAL_FEATURES = [
    "fac_dtw_m",
    "fac_wte_m",
    "elev_above_str1_m",
    "elev_above_str5_m",
    "elev_above_str7_m",
    "log1p_dist_str1_m",
    "log1p_dist_str5_m",
    "log1p_dist_str7_m",
    "slope_m_m",
    "local_relief_300m_m",
    "log1p_drainage_km2",
    "aridity_index",
    "precip_mm_yr",
    "pet_mm_yr",
    "vpd_kpa",
    "tmax_c",
    "reitz_effective_recharge",
    "reitz_total_recharge",
    "reitz_et",
    "wolock_bfi",
]

# Features that must never reach the residual model (asserted in tests).
DISALLOWED_RESIDUAL_FEATURES = {
    "ma_dtw_m",
    "zell_sanford_dtw",
    "fan_dtw",
    "lon",
    "lat",
    "x_5070",
    "y_5070",
    "dem_m",
    "target_dtw_m",
    "target_wte_m",
    "fac_residual_dtw_m",
    "fac_residual_wte_m",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# CLI + config
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", default=None, help="override output directory")
    ap.add_argument(
        "--group-col", default=None, help="spatial CV block (def block_20km)"
    )
    ap.add_argument(
        "--label-mode",
        default=None,
        choices=["water_table", "training"],
        help="water_table (default): point_type==well & is_water_table_label; "
        "training: is_training_label==true (sparser sensitivity check)",
    )
    ap.add_argument("--methods", default=None, help="comma-separated families: idw,rbf")
    ap.add_argument("--idw-k", default=None, help="comma-separated k (def 16,32,64)")
    ap.add_argument("--idw-power", default=None, help="comma-separated power (def 2.0)")
    ap.add_argument(
        "--rbf-kernel",
        default=None,
        help="comma-separated kernels (def thin_plate_spline)",
    )
    ap.add_argument(
        "--rbf-smoothing",
        default=None,
        help="comma-separated smoothing on km coords (def 25,100,400)",
    )
    ap.add_argument(
        "--residual-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="fit the first residual WTE model (HistGradientBoosting)",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug-limit", type=int, default=None, help="cap model rows")
    return ap.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def default_out_dir(cfg: dict) -> Path:
    return Path(cfg["paths"]["out_dir"]).parent / "regional_prior"


def _resolve(cli_val, cfg_rp: dict, key: str, default):
    if cli_val is not None:
        return cli_val
    if key in cfg_rp:
        return cfg_rp[key]
    return default


def _csv_floats(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if str(x).strip()]


def _csv_ints(s: str) -> list[int]:
    return [int(float(x)) for x in str(s).split(",") if str(x).strip()]


def _csv_strs(s) -> list[str]:
    if isinstance(s, (list, tuple)):
        return [str(x) for x in s]
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _power_tag(power: float) -> str:
    """2.0 -> '2', 1.5 -> '15', 2.5 -> '25', 3.0 -> '3'."""
    if float(power).is_integer():
        return str(int(power))
    return str(power).replace(".", "")


def _parse_power_tag(tag: str) -> float:
    """Inverse of _power_tag for documented one-decimal powers."""
    if len(tag) <= 1:
        return float(tag)
    return float(tag) / 10.0


def _idw_name(k: int, power: float) -> str:
    return f"idw_k{k}_p{_power_tag(power)}"


def _kernel_tag(kernel: str) -> str:
    return "tps" if kernel == "thin_plate_spline" else kernel


def _rbf_name(kernel: str, smoothing: float) -> str:
    return f"rbf_{_kernel_tag(kernel)}_s{int(smoothing)}"


def parse_method_name(name: str) -> MethodSpec | None:
    """Lenient parse of documented method names (config ``methods`` list)."""
    if name.startswith("idw_"):
        try:
            _, ktok, ptok = name.split("_", 2)
            k = int(ktok.lstrip("k"))
            power = _parse_power_tag(ptok.lstrip("p"))
        except (ValueError, IndexError):
            return None
        return MethodSpec(_idw_name(k, power), "idw", {"k": k, "power": power})
    if name.startswith("rbf_"):
        parts = name.split("_")
        if len(parts) < 3:
            return None
        kernel = "thin_plate_spline" if parts[1] == "tps" else parts[1]
        stok = parts[-1]
        for prefix in ("smooth", "s"):
            if stok.startswith(prefix):
                stok = stok[len(prefix) :]
                break
        try:
            smoothing = float(stok)
        except ValueError:
            return None
        return MethodSpec(
            _rbf_name(kernel, smoothing),
            "rbf",
            {"kernel": kernel, "smoothing": smoothing},
        )
    return None


def build_method_specs(args: argparse.Namespace, cfg_rp: dict) -> list[MethodSpec]:
    """Method ladder from config names if present, else from family grids."""
    if "methods" in cfg_rp and args.methods is None:
        specs, bad = [], []
        for name in _csv_strs(cfg_rp["methods"]):
            spec = parse_method_name(name)
            (specs if spec else bad).append(spec if spec else name)
        if bad:
            print(f"WARNING: unparseable config method names skipped: {bad}")
        if specs:
            return specs

    families = _csv_strs(_resolve(args.methods, cfg_rp, "families", "idw,rbf"))
    idw_k = _csv_ints(_resolve(args.idw_k, cfg_rp, "idw_k", "16,32,64"))
    idw_power = _csv_floats(_resolve(args.idw_power, cfg_rp, "idw_power", "2.0"))
    rbf_kernel = _csv_strs(
        _resolve(args.rbf_kernel, cfg_rp, "rbf_kernel", "thin_plate_spline")
    )
    rbf_smooth = _csv_floats(
        _resolve(args.rbf_smoothing, cfg_rp, "rbf_smoothing", "25,100,400")
    )

    specs: list[MethodSpec] = []
    if "idw" in families:
        for k in idw_k:
            for p in idw_power:
                specs.append(MethodSpec(_idw_name(k, p), "idw", {"k": k, "power": p}))
    if "rbf" in families:
        for kern in rbf_kernel:
            for sm in rbf_smooth:
                specs.append(
                    MethodSpec(
                        _rbf_name(kern, sm), "rbf", {"kernel": kern, "smoothing": sm}
                    )
                )
    return specs


# --------------------------------------------------------------------------- #
# data selection + derived columns
# --------------------------------------------------------------------------- #
def load_points(cfg: dict) -> gpd.GeoDataFrame:
    table = Path(cfg["covariate_table"]["out"])
    if not table.exists():
        raise SystemExit(f"covariate table not found: {table}")
    return gpd.read_parquet(table)


def select_water_table_wells(
    gdf: gpd.GeoDataFrame, label_mode: str, group_col: str
) -> gpd.GeoDataFrame:
    """Water-table model rows: finite WTE label, coords, and CV group.

    FAC coverage is NOT required (a valid regional label may sit outside the FAC
    footprint). dem_m must be finite so WTE = dem - dtw is defined.
    """
    if group_col not in gdf.columns:
        raise SystemExit(f"group column {group_col!r} not in covariate table")

    if label_mode == "training":
        sel = gdf["is_training_label"].fillna(False)
    else:
        sel = (gdf["point_type"] == "well") & gdf["is_water_table_label"].fillna(False)
    wells = gdf[sel].copy()

    finite = (
        wells["target_dtw_m"].notna()
        & wells["dem_m"].notna()
        & wells["x_5070"].notna()
        & wells["y_5070"].notna()
        & wells[group_col].notna()
    )
    return wells[finite].copy()


def add_observed_columns(wells: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """obs/fac/ma DTW + WTE columns, weights, and depth bin."""
    wells = wells.copy()
    wells["obs_dtw_m"] = wells["target_dtw_m"].astype("float64")
    wte = wells["target_wte_m"].astype("float64")
    wells["obs_wte_m"] = wte.where(wte.notna(), wells["dem_m"] - wells["obs_dtw_m"])

    fac_wte = wells["fac_wte_m"].astype("float64")
    wells["fac_wte_m"] = fac_wte.where(
        fac_wte.notna(), wells["dem_m"] - wells["fac_dtw_m"]
    )
    if "ma_dtw_m" in wells:
        wells["ma_wte_m"] = wells["dem_m"] - wells["ma_dtw_m"]

    wells["weight_raw"] = wells["weight"].astype("float64")
    wells["weight_score"] = wells["weight_raw"].fillna(0.0)
    wells["weight_model"] = wells["weight_score"].clip(lower=1e-3)
    wells["depth_bin"] = pd.cut(wells["obs_dtw_m"], DEPTH_BINS, labels=DEPTH_LABELS)
    return wells


def _grouped_folds(
    groups: np.ndarray, n_splits_max: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Positional GroupKFold splits; raises ValueError if <2 groups.

    GroupKFold is deterministic in the groups + n_splits, so an inner call on a
    subset of rows reproduces the same partition the outer call would assign --
    this is what lets the residual model reuse the global OOF regional prior for
    its test fold while building train targets from an inner re-fit.
    """
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError(f"need >=2 spatial blocks for CV; got {n_groups}")
    gkf = GroupKFold(n_splits=min(n_splits_max, n_groups))
    return list(gkf.split(np.zeros((len(groups), 1)), groups=groups))


def make_group_folds(
    wells: pd.DataFrame, group_col: str, n_splits_max: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        return _grouped_folds(wells[group_col].to_numpy(), n_splits_max)
    except ValueError as exc:
        raise SystemExit(str(exc))


def fold_assignment(n: int, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    cv_fold = np.full(n, -1, dtype="int64")
    for i, (_, te) in enumerate(folds):
        cv_fold[te] = i
    return cv_fold


# --------------------------------------------------------------------------- #
# interpolators
# --------------------------------------------------------------------------- #
def idw_predict(
    train_xy: np.ndarray,
    train_y: np.ndarray,
    query_xy: np.ndarray,
    train_weight: np.ndarray | None = None,
    k: int = 32,
    power: float = 2.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """k-nearest inverse-distance interpolation with optional sample weights."""
    train_xy = np.asarray(train_xy, dtype="float64")
    train_y = np.asarray(train_y, dtype="float64")
    query_xy = np.asarray(query_xy, dtype="float64")
    n_train = len(train_xy)
    if n_train == 0:
        return np.full(len(query_xy), np.nan)
    if train_weight is None:
        train_weight = np.ones(n_train)
    else:
        train_weight = np.asarray(train_weight, dtype="float64")

    k_eff = min(int(k), n_train)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(train_xy)
    dist, idx = nn.kneighbors(query_xy)  # (n_query, k_eff)
    y_nb = train_y[idx]
    w_nb = train_weight[idx]

    # distance-decay weights; the eps floor lets near-coincident points dominate
    w_dist = 1.0 / np.power(dist + eps, power)
    w = w_dist * w_nb
    sumw = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred = (w * y_nb).sum(axis=1) / sumw

    # fall back to unweighted neighbour mean where weights vanish / go nonfinite
    bad = ~np.isfinite(pred) | (sumw <= 0)
    if bad.any():
        pred[bad] = y_nb[bad].mean(axis=1)

    # exact handling of query points coincident with training points
    coincident = dist <= eps
    has_coin = coincident.any(axis=1)
    if has_coin.any():
        wc = coincident * w_nb
        denomw = wc.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            coin_w = (wc * y_nb).sum(axis=1) / denomw
        denomu = coincident.sum(axis=1)
        coin_u = (coincident * y_nb).sum(axis=1) / np.where(denomu > 0, denomu, 1)
        coin = np.where(denomw > 0, coin_w, coin_u)
        pred[has_coin] = coin[has_coin]
    return pred


def rbf_predict(
    train_xy: np.ndarray,
    train_y: np.ndarray,
    query_xy: np.ndarray,
    kernel: str = "thin_plate_spline",
    smoothing: float = 100.0,
    neighbors: int | None = None,
) -> np.ndarray:
    """scipy RBF interpolation; coordinates scaled m -> km before fitting."""
    train_xy = np.asarray(train_xy, dtype="float64") / 1000.0
    query_xy = np.asarray(query_xy, dtype="float64") / 1000.0
    train_y = np.asarray(train_y, dtype="float64")
    rbf = RBFInterpolator(
        train_xy, train_y, kernel=kernel, smoothing=smoothing, neighbors=neighbors
    )
    return rbf(query_xy)


def regional_fit_predict(
    train_xy: np.ndarray,
    train_y: np.ndarray,
    train_w: np.ndarray,
    query_xy: np.ndarray,
    spec: MethodSpec,
) -> np.ndarray:
    """One regional interpolator fit on (train_xy, train_y) -> query predictions."""
    if spec.family == "idw":
        return idw_predict(
            train_xy,
            train_y,
            query_xy,
            train_weight=train_w,
            k=spec.params["k"],
            power=spec.params["power"],
        )
    if spec.family == "rbf":
        return rbf_predict(
            train_xy,
            train_y,
            query_xy,
            kernel=spec.params["kernel"],
            smoothing=spec.params["smoothing"],
        )
    raise ValueError(f"unknown method family {spec.family!r}")


def crossfit_regional_method(
    wells: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: MethodSpec,
) -> np.ndarray:
    """Out-of-fold regional WTE prediction; each fold fit without its own labels."""
    xy = wells[["x_5070", "y_5070"]].to_numpy("float64")
    y = wells["obs_wte_m"].to_numpy("float64")
    w = wells["weight_model"].to_numpy("float64")
    pred = np.full(len(wells), np.nan)
    for tr, te in folds:
        pred[te] = regional_fit_predict(xy[tr], y[tr], w[tr], xy[te], spec)
    return pred


# --------------------------------------------------------------------------- #
# derived per-method columns
# --------------------------------------------------------------------------- #
def clip_dtw(
    pred_wte: np.ndarray, dem: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dtw_raw, dtw_clipped>=0, wte_clipped consistent with dtw_clipped)."""
    dtw_raw = dem - pred_wte
    dtw_clipped = np.clip(dtw_raw, 0, None)
    wte_clipped = dem - dtw_clipped
    return dtw_raw, dtw_clipped, wte_clipped


def add_regional_method_columns(
    wells: gpd.GeoDataFrame, method_name: str, pred_wte: np.ndarray
) -> None:
    """Write the full diagnostic column family for one method (in place)."""
    m = method_name
    dem = wells["dem_m"].to_numpy("float64")
    obs_wte = wells["obs_wte_m"].to_numpy("float64")
    obs_dtw = wells["obs_dtw_m"].to_numpy("float64")
    fac_wte = wells["fac_wte_m"].to_numpy("float64")
    fac_dtw = wells["fac_dtw_m"].to_numpy("float64")

    dtw_raw, dtw_clip, wte_clip = clip_dtw(pred_wte, dem)
    wells[f"regional_wte_oof__{m}"] = pred_wte
    wells[f"regional_dtw_raw_oof__{m}"] = dtw_raw
    wells[f"regional_dtw_oof__{m}"] = dtw_clip
    wells[f"regional_wte_clipped_oof__{m}"] = wte_clip
    wells[f"regional_head_above_ground_m__{m}"] = np.clip(pred_wte - dem, 0, None)
    wells[f"regional_residual_wte__{m}"] = obs_wte - pred_wte
    wells[f"regional_residual_dtw__{m}"] = dtw_clip - obs_dtw
    wells[f"fac_minus_regional_wte__{m}"] = fac_wte - pred_wte
    wells[f"fac_minus_regional_dtw__{m}"] = fac_dtw - dtw_clip


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _wavg(values: np.ndarray, weights: np.ndarray) -> float:
    sw = weights.sum()
    if not np.isfinite(sw) or sw <= 0:
        return float("nan")
    return float(np.average(values, weights=weights))


def score_dtw_product(
    pred_dtw: np.ndarray,
    obs_dtw: np.ndarray,
    pred_wte: np.ndarray,
    obs_wte: np.ndarray,
    weights: np.ndarray,
) -> dict:
    """DTW error metrics + WTE-sign diagnostics on the finite subset."""
    pred_dtw = np.asarray(pred_dtw, "float64")
    obs_dtw = np.asarray(obs_dtw, "float64")
    pred_wte = np.asarray(pred_wte, "float64")
    obs_wte = np.asarray(obs_wte, "float64")
    weights = np.asarray(weights, "float64")
    ok = (
        np.isfinite(pred_dtw)
        & np.isfinite(obs_dtw)
        & np.isfinite(pred_wte)
        & np.isfinite(obs_wte)
        & np.isfinite(weights)
    )
    n = int(ok.sum())
    if n == 0:
        return {"n": 0}
    pd_, od_, pw_, ow_, w_ = (
        pred_dtw[ok],
        obs_dtw[ok],
        pred_wte[ok],
        obs_wte[ok],
        weights[ok],
    )
    err = pd_ - od_
    obs_sh, pred_sh = od_ < SHALLOW_M, pd_ < SHALLOW_M
    tp = int((obs_sh & pred_sh).sum())
    recall = tp / max(int(obs_sh.sum()), 1)
    precision = tp / max(int(pred_sh.sum()), 1)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "n": n,
        "mad": float(np.median(np.abs(err))),
        "wmad": _wavg(np.abs(err), w_),
        "bias": float(np.median(err)),
        "mean_bias": _wavg(err, w_),
        "rmse": float(np.sqrt(_wavg(err**2, w_))),
        "shallow_recall": float(recall),
        "shallow_precision": float(precision),
        "shallow_f1": float(f1),
        "n_obs_shallow": int(obs_sh.sum()),
        "n_pred_shallow": int(pred_sh.sum()),
        "wte_bias": float(np.median(pw_ - ow_)),
        "wte_mean_bias": _wavg(pw_ - ow_, w_),
    }


def _score_groups(wells: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks over model rows, in report order."""
    obs = wells["obs_dtw_m"].to_numpy("float64")
    groups = {
        "all_water_table": np.ones(len(wells), bool),
        "common_fac_footprint": wells["fac_dtw_m"].notna().to_numpy(),
        "deep>=30m": obs >= DEEP_M,
        "shallow<5m": obs < SHALLOW_M,
    }
    for lab in DEPTH_LABELS:
        groups[f"depth {lab}m"] = (wells["depth_bin"] == lab).to_numpy()
    groups["positive_weight"] = wells["weight_score"].to_numpy("float64") > 0
    if "tier" in wells:
        groups["primary_secondary"] = (
            wells["tier"].isin(["primary", "secondary"]).to_numpy()
        )
    return groups


def score_all_products(
    wells: pd.DataFrame, products: list[dict], groups: dict[str, np.ndarray]
) -> pd.DataFrame:
    obs_dtw = wells["obs_dtw_m"].to_numpy("float64")
    obs_wte = wells["obs_wte_m"].to_numpy("float64")
    w = wells["weight_score"].to_numpy("float64")
    rows = []
    for prod in products:
        pdtw = np.asarray(prod["pred_dtw"], "float64")
        pwte = np.asarray(prod["pred_wte"], "float64")
        for gname, mask in groups.items():
            if mask.sum() == 0:
                continue
            metrics = score_dtw_product(
                pdtw[mask], obs_dtw[mask], pwte[mask], obs_wte[mask], w[mask]
            )
            if metrics.get("n", 0) == 0:
                continue
            rows.append(
                {
                    "product": prod["name"],
                    "benchmark_only": prod.get("benchmark_only", False),
                    "group": gname,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_products(
    wells: pd.DataFrame, method_names: list[str], extra: list[dict] | None = None
) -> list[dict]:
    """FAC, Ma (benchmark_only), each RegionalWTE method, plus optional extras."""
    dem = wells["dem_m"].to_numpy("float64")
    products = [
        {
            "name": "FAC",
            "pred_dtw": wells["fac_dtw_m"].to_numpy("float64"),
            "pred_wte": wells["fac_wte_m"].to_numpy("float64"),
        }
    ]
    if "ma_dtw_m" in wells:
        products.append(
            {
                "name": "Ma",
                "benchmark_only": True,
                "pred_dtw": wells["ma_dtw_m"].to_numpy("float64"),
                "pred_wte": (dem - wells["ma_dtw_m"]).to_numpy("float64"),
            }
        )
    for m in method_names:
        products.append(
            {
                "name": f"RegionalWTE[{m}]",
                "pred_dtw": wells[f"regional_dtw_oof__{m}"].to_numpy("float64"),
                "pred_wte": wells[f"regional_wte_clipped_oof__{m}"].to_numpy("float64"),
            }
        )
    return products + (extra or [])


def method_summary(score: pd.DataFrame) -> pd.DataFrame:
    """Compact per-product leaderboard on the two headline groups."""
    cols = [
        "mad",
        "wmad",
        "bias",
        "mean_bias",
        "rmse",
        "shallow_recall",
        "shallow_precision",
    ]
    rows = []
    for prod, sub in score.groupby("product", sort=False):
        row = {"product": prod, "benchmark_only": bool(sub["benchmark_only"].iloc[0])}
        for gname, tag in [("all_water_table", "all"), ("common_fac_footprint", "fac")]:
            g = sub[sub.group == gname]
            if g.empty:
                continue
            row[f"n_{tag}"] = int(g["n"].iloc[0])
            for c in cols:
                row[f"{c}_{tag}"] = float(g[c].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# residual diagnostics + feature correlations
# --------------------------------------------------------------------------- #
def build_residual_diagnostics(
    wells: gpd.GeoDataFrame, method_names: list[str], group_col: str
) -> pd.DataFrame:
    base = [
        "canonical_id",
        "tier",
        "weight_score",
        "confinement_class",
        "target_dtw_m",
        "obs_wte_m",
        "dem_m",
        "fac_dtw_m",
        "fac_wte_m",
        "ma_dtw_m",
        "depth_bin",
        "cv_fold",
        group_col,
        "x_5070",
        "y_5070",
    ]
    per_method = []
    for m in method_names:
        per_method += [
            f"regional_wte_oof__{m}",
            f"regional_dtw_oof__{m}",
            f"regional_residual_wte__{m}",
            f"regional_residual_dtw__{m}",
            f"fac_minus_regional_wte__{m}",
            f"fac_minus_regional_dtw__{m}",
        ]
    cols = [c for c in base + per_method + CORR_FEATURES if c in wells.columns]
    return pd.DataFrame(wells[cols].copy())


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    ok = a.notna() & b.notna()
    if ok.sum() < 10:
        return float("nan"), int(ok.sum())
    rho, _ = spearmanr(a[ok], b[ok])
    return float(rho), int(ok.sum())


def feature_correlations(wells: pd.DataFrame, method_names: list[str]) -> pd.DataFrame:
    rows = []
    for m in method_names:
        res = wells[f"regional_residual_wte__{m}"]
        fmr = wells[f"fac_minus_regional_wte__{m}"]
        for feat in CORR_FEATURES:
            if feat not in wells:
                continue
            r_res, n = _spearman(wells[feat], res)
            r_abs, _ = _spearman(wells[feat], res.abs())
            r_fmr, _ = _spearman(wells[feat], fmr)
            rows.append(
                {
                    "feature": feat,
                    "method": m,
                    "n": n,
                    "spearman_rho_vs_regional_residual_wte": round(r_res, 3),
                    "spearman_rho_vs_abs_regional_residual_wte": round(r_abs, 3),
                    "spearman_rho_vs_fac_minus_regional_wte": round(r_fmr, 3),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# residual model
# --------------------------------------------------------------------------- #
def _hgb() -> HistGradientBoostingRegressor:
    try:
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=20,
            l2_regularization=2.0,
            loss="absolute_error",
        )
    except (TypeError, ValueError):  # older sklearn without absolute_error
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=20,
            l2_regularization=2.0,
            loss="squared_error",
        )


def _per_method_residual_feats(m: str) -> set[str]:
    return {
        f"fac_minus_regional_wte__{m}",
        f"fac_minus_regional_dtw__{m}",
        f"regional_dtw_oof__{m}",
        f"regional_head_above_ground_m__{m}",
    }


def _residual_features(method_name: str) -> list[str]:
    m = method_name
    return STATIC_RESIDUAL_FEATURES + sorted(_per_method_residual_feats(m))


def _build_residual_X(
    df: pd.DataFrame, reg_wte: np.ndarray, feats: list[str], m: str
) -> np.ndarray:
    """Feature matrix for `feats`; the per-method columns are recomputed from the
    supplied regional WTE (so train rows use inner-OOF prior, test rows the
    outer-train prior) rather than read from the global-OOF columns."""
    dem = df["dem_m"].to_numpy("float64")
    fac_wte = df["fac_wte_m"].to_numpy("float64")
    fac_dtw = df["fac_dtw_m"].to_numpy("float64")
    _, dtw_clip, _ = clip_dtw(reg_wte, dem)
    derived = {
        f"fac_minus_regional_wte__{m}": fac_wte - reg_wte,
        f"fac_minus_regional_dtw__{m}": fac_dtw - dtw_clip,
        f"regional_dtw_oof__{m}": dtw_clip,
        f"regional_head_above_ground_m__{m}": np.clip(reg_wte - dem, 0, None),
    }
    cols = [derived[f] if f in derived else df[f].to_numpy("float64") for f in feats]
    return np.column_stack(cols)


def fit_residual_model_oof(
    wells: gpd.GeoDataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
    spec: MethodSpec,
    group_col: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Nested-CV residual-WTE model on top of the regional prior + importance.

    Strictly leak-free (no nested-CV leakage). For each outer fold:
      * the held-out regional prior is the global OOF column, which equals a
        fit on the outer-train labels (same fold partition) -- leak-free;
      * the residual target ``obs_wte - regional`` and the per-method features
        for the *train* rows are rebuilt from an INNER-CV regional prior fit
        only inside the outer-train set, so the outer-test labels never reach
        the regressor's training data.
    """
    m = spec.name
    per_method = _per_method_residual_feats(m)
    feats = [f for f in _residual_features(m) if f in per_method or f in wells.columns]
    assert not (set(feats) & DISALLOWED_RESIDUAL_FEATURES), "leaky residual feature"

    obs_wte = wells["obs_wte_m"].to_numpy("float64")
    dem = wells["dem_m"].to_numpy("float64")
    w = wells["weight_model"].to_numpy("float64")
    xy = wells[["x_5070", "y_5070"]].to_numpy("float64")
    reg_global = wells[f"regional_wte_oof__{m}"].to_numpy("float64")  # leak-free OOF

    res_hat = np.full(len(wells), np.nan)
    imp_acc, imp_n = np.zeros(len(feats)), 0
    for tr, te in folds:
        train_df = wells.iloc[tr]
        # inner-OOF regional prior for the train rows (within outer-train only)
        try:
            inner = _grouped_folds(train_df[group_col].to_numpy())
            reg_tr = crossfit_regional_method(train_df, inner, spec)
        except ValueError:  # too few inner blocks: fit-on-train (rare, logged once)
            reg_tr = regional_fit_predict(xy[tr], obs_wte[tr], w[tr], xy[tr], spec)
        reg_te = reg_global[te]  # == fit on outer-train, predict outer-test

        X_tr = _build_residual_X(train_df, reg_tr, feats, m)
        y_tr = obs_wte[tr] - reg_tr
        X_te = _build_residual_X(wells.iloc[te], reg_te, feats, m)
        y_te = obs_wte[te] - reg_te

        ok = np.isfinite(y_tr)  # drop rows whose inner prior failed; X NaNs are fine
        mdl = _hgb()  # HistGBR tolerates NaN features natively
        mdl.fit(X_tr[ok], y_tr[ok], sample_weight=w[tr][ok])
        res_hat[te] = mdl.predict(X_te)
        try:
            pi = permutation_importance(
                mdl, X_te, y_te, sample_weight=w[te], n_repeats=5, random_state=0
            )
            imp_acc += pi.importances_mean
            imp_n += 1
        except Exception:  # importance is best-effort; never blocks OOF preds
            pass

    hybrid_wte = reg_global + res_hat
    hybrid_dtw = np.clip(dem - hybrid_wte, 0, None)
    out = wells.copy()
    out[f"residual_wte_obs__{m}"] = obs_wte - reg_global
    out[f"residual_wte_hat__{m}"] = res_hat
    out[f"hybrid_wte_oof__{m}"] = hybrid_wte
    out[f"hybrid_dtw_oof__{m}"] = hybrid_dtw

    if imp_n > 0:
        imp = pd.DataFrame(
            {
                "feature": feats,
                "importance_mean": imp_acc / imp_n,
                "status": "permutation_oof",
            }
        ).sort_values("importance_mean", ascending=False)
    else:
        imp = pd.DataFrame(
            {"feature": feats, "importance_mean": np.nan, "status": "not_computed"}
        )
    return out, imp


# --------------------------------------------------------------------------- #
# existing-diagnostics join
# --------------------------------------------------------------------------- #
def join_existing_predictions(wells: gpd.GeoDataFrame, cfg: dict) -> list[dict]:
    """Score prior climate-topo regional/hybrid predictions on the model rows."""
    diag = (
        Path(cfg["paths"]["out_dir"]).parent
        / "diagnostics"
        / "wte_point_cv_predictions.parquet"
    )
    if not diag.exists():
        return []
    prev = gpd.read_parquet(diag)
    cols = [c for c in prev.columns if c.startswith(("regional_dtw__", "hybrid_dtw__"))]
    if not cols or "canonical_id" not in prev.columns:
        return []
    prev = prev[["canonical_id"] + cols].drop_duplicates("canonical_id")
    merged = wells[["canonical_id"]].merge(prev, on="canonical_id", how="left")
    dem = wells["dem_m"].to_numpy("float64")
    products = []
    for c in cols:
        dtw = merged[c].to_numpy("float64")
        kind, fset = c.split("__", 1)
        label = "Regional" if kind == "regional_dtw" else "Hybrid"
        products.append(
            {
                "name": f"Existing {label}[{fset}]",
                "pred_dtw": dtw,
                "pred_wte": dem - dtw,
            }
        )
    return products


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def write_run_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    t0 = perf_counter()
    cfg = load_config(Path(args.config))
    cfg_rp = cfg.get("regional_prior", {})

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(_resolve(None, cfg_rp, "out_dir", str(default_out_dir(cfg))))
    )
    group_col = _resolve(args.group_col, cfg_rp, "group_col", "block_20km")
    label_mode = _resolve(args.label_mode, cfg_rp, "label_mode", "water_table")
    do_residual = bool(_resolve(args.residual_model, cfg_rp, "residual_model", False))
    specs = build_method_specs(args, cfg_rp)

    existing = [
        f for f in ("regional_prior_oof_points.parquet",) if (out_dir / f).exists()
    ]
    if existing and not args.overwrite:
        raise SystemExit(f"{out_dir} has outputs; pass --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = load_points(cfg)
    if label_mode == "training":
        n_total = int(gdf["is_training_label"].fillna(False).sum())
    else:
        n_total = int(
            (
                (gdf["point_type"] == "well")
                & gdf["is_water_table_label"].fillna(False)
            ).sum()
        )
    wells = select_water_table_wells(gdf, label_mode, group_col)
    if len(wells) < 10:
        raise SystemExit(f"need >=10 water-table labels; got {len(wells)}")

    wells = add_observed_columns(wells)
    if not np.isfinite(wells["obs_wte_m"].to_numpy("float64")).any():
        raise SystemExit("no finite WTE labels")
    if args.debug_limit:
        wells = wells.iloc[: args.debug_limit].copy()

    wells = wells.reset_index(drop=True)
    folds = make_group_folds(wells, group_col)
    wells["cv_fold"] = fold_assignment(len(wells), folds)
    wells["cv_group"] = wells[group_col]

    n_fac = int(wells["fac_dtw_m"].notna().sum())
    n_pos_w = int((wells["weight_score"] > 0).sum())
    n_blocks = int(wells[group_col].nunique())

    # --- regional prior methods (out-of-fold) ---------------------------------
    completed, failed, nonfinite_counts = [], [], {}
    for spec in specs:
        try:
            pred = crossfit_regional_method(wells, folds, spec)
        except Exception as exc:  # RBF instability etc.: skip, keep going
            failed.append({"method": spec.name, "error": repr(exc)})
            print(f"WARNING: method {spec.name} failed: {exc!r}")
            continue
        n_finite = int(np.isfinite(pred).sum())
        if n_finite < 10:
            failed.append(
                {"method": spec.name, "error": f"only {n_finite} finite predictions"}
            )
            print(
                f"WARNING: method {spec.name} produced {n_finite} finite predictions; skipped"
            )
            continue
        if n_finite < len(wells):
            nonfinite_counts[spec.name] = len(wells) - n_finite
        add_regional_method_columns(wells, spec.name, pred)
        completed.append(spec.name)

    if not completed:
        raise SystemExit("no regional-prior method completed")

    # --- scoring (regional prior + FAC + Ma + existing) -----------------------
    groups = _score_groups(wells)
    products = build_products(
        wells, completed, extra=join_existing_predictions(wells, cfg)
    )
    score = score_all_products(wells, products, groups)
    score.to_csv(out_dir / "regional_prior_scores.csv", index=False)
    summary = method_summary(score)
    summary.to_csv(out_dir / "regional_prior_method_summary.csv", index=False)

    # recommended method: best all_water_table wmad among completed (post-hoc).
    reg_summary = summary[summary["product"].str.startswith("RegionalWTE[")]
    recommended = None
    if (
        not reg_summary.empty
        and "wmad_all" in reg_summary
        and reg_summary["wmad_all"].notna().any()
    ):
        best = reg_summary.loc[reg_summary["wmad_all"].idxmin(), "product"]
        recommended = best[len("RegionalWTE[") : -1]

    # --- residual diagnostics + feature correlations --------------------------
    build_residual_diagnostics(wells, completed, group_col).to_csv(
        out_dir / "regional_prior_residual_diagnostics.csv", index=False
    )
    feature_correlations(wells, completed).to_csv(
        out_dir / "regional_prior_feature_correlations.csv", index=False
    )

    # --- per-point OOF parquet (with geometry) --------------------------------
    wells.to_parquet(out_dir / "regional_prior_oof_points.parquet")

    # --- optional residual model ----------------------------------------------
    residual_method = None
    if do_residual and recommended is not None:
        residual_method = recommended
        rec_spec = next(s for s in specs if s.name == recommended)
        res_pts, imp = fit_residual_model_oof(wells, folds, rec_spec, group_col)
        m = residual_method
        res_products = [
            {
                "name": f"RegionalWTE[{m}]",
                "pred_dtw": res_pts[f"regional_dtw_oof__{m}"].to_numpy("float64"),
                "pred_wte": res_pts[f"regional_wte_clipped_oof__{m}"].to_numpy(
                    "float64"
                ),
            },
            {
                "name": f"ResidualHybrid[{m}]",
                "pred_dtw": res_pts[f"hybrid_dtw_oof__{m}"].to_numpy("float64"),
                "pred_wte": (
                    res_pts["dem_m"] - res_pts[f"hybrid_dtw_oof__{m}"]
                ).to_numpy("float64"),
            },
            {
                "name": "FAC",
                "pred_dtw": res_pts["fac_dtw_m"].to_numpy("float64"),
                "pred_wte": res_pts["fac_wte_m"].to_numpy("float64"),
            },
        ]
        if "ma_dtw_m" in res_pts:
            res_products.append(
                {
                    "name": "Ma",
                    "benchmark_only": True,
                    "pred_dtw": res_pts["ma_dtw_m"].to_numpy("float64"),
                    "pred_wte": (res_pts["dem_m"] - res_pts["ma_dtw_m"]).to_numpy(
                        "float64"
                    ),
                }
            )
        res_products += join_existing_predictions(res_pts, cfg)
        res_score = score_all_products(res_pts, res_products, _score_groups(res_pts))
        res_score.to_csv(out_dir / "residual_model_scores.csv", index=False)
        imp.to_csv(out_dir / "residual_model_feature_importance.csv", index=False)
        keep = [
            "canonical_id",
            "cv_fold",
            "obs_wte_m",
            "obs_dtw_m",
            f"regional_wte_oof__{m}",
            f"regional_dtw_oof__{m}",
            f"residual_wte_obs__{m}",
            f"residual_wte_hat__{m}",
            f"hybrid_wte_oof__{m}",
            f"hybrid_dtw_oof__{m}",
            "fac_dtw_m",
            "ma_dtw_m",
            "depth_bin",
            "geometry",
        ]
        res_pts[[c for c in keep if c in res_pts.columns]].to_parquet(
            out_dir / "residual_model_oof_predictions.parquet"
        )

    # --- run metadata ----------------------------------------------------------
    run = {
        "config": str(args.config),
        "table": str(cfg["covariate_table"]["out"]),
        "out_dir": str(out_dir),
        "group_col": group_col,
        "label_mode": label_mode,
        "n_water_table_total": n_total,
        "n_model_rows": int(len(wells)),
        "n_water_table_finite_wte": int(wells["obs_wte_m"].notna().sum()),
        "n_common_fac_footprint": n_fac,
        "n_water_table_positive_weight": n_pos_w,
        "n_blocks": n_blocks,
        "n_cv_folds": len(folds),
        "methods_requested": [s.name for s in specs],
        "methods_completed": completed,
        "methods_failed": failed,
        "methods_partial_nonfinite": nonfinite_counts,
        "recommended_method": recommended,
        "recommended_method_basis": "lowest all_water_table wmad (post-hoc diagnostic)",
        "benchmark_only_products": ["Ma"] if "ma_dtw_m" in wells else [],
        "residual_model": do_residual,
        "residual_model_method": residual_method,
        "residual_model_cv": "nested (inner-OOF train residuals, leak-free)",
        "runtime_s": round(perf_counter() - t0, 1),
    }
    write_run_json(out_dir / "regional_prior_run.json", run)

    # --- console summary -------------------------------------------------------
    print(
        f"model rows: {len(wells)}  (total water-table: {n_total}, "
        f"fac-footprint: {n_fac}, positive-weight: {n_pos_w}, blocks: {n_blocks})"
    )
    print(f"methods completed: {completed}")
    if failed:
        print(f"methods failed: {[f['method'] for f in failed]}")
    show = summary[
        [
            c
            for c in [
                "product",
                "n_all",
                "mad_all",
                "wmad_all",
                "bias_all",
                "shallow_recall_all",
            ]
            if c in summary
        ]
    ]
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(show.to_string(index=False))
    if recommended:
        print(f"recommended (post-hoc): {recommended}")
    print(f"\nwrote regional prior -> {out_dir}")


if __name__ == "__main__":
    main()
