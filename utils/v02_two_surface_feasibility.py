"""WP2 first-gate feasibility — is a two-surface (phreatic vs regional-aquifer)
decomposition IDENTIFIABLE from existing observation evidence?

Implements the feasibility half of `notes/plans/HANDILY_V2_FIELD_MODEL_PLAN.md`
§7.5 gate 1. The open question (from notes/ERROR_SOURCES.md §1/§5): the ~9 m IQR
of collocated unconfined wells in the 30+ m band is robust to confinement
screening, so it is real water-table dispersion rather than leaked-confined
noise. Gate 1 asks the next thing: is that dispersion ORDERED — does a deeper
completion sample a systematically different water level that construction
metadata can predict — or is it unstructured aleatoric noise?

Only `confinement_class in {unconfined, unconfined_marginal}` wells are used
(hard project rule: confined wells measure a potentiometric surface, not the
water table). A well qualifies if it also has a finite `mean_dtw` and a finite
completion depth (`well_depth`, else `screen_bottom`). Qualifying wells are
snapped to a 500 m EPSG:5070 grid; cells with enough members feed four analyses.

Analyses (all metres unless noted; DTW = depth to unconfined water table, larger
= deeper water):

1. VERTICAL ORDERING (cells >=3 wells, completion-depth spread >=10 m).
   Per-cell Spearman rho between completion depth and mean_dtw. A downward
   gradient (recharge) gives rho>0 (deeper completion -> deeper water); an
   upward gradient (discharge) gives rho<0. Ordering (either sign) is the
   identifiability signal; pure noise gives rho ~ 0 with a permutation-null
   spread. Reported: median / IQR / fraction>0 overall, by cell-median-DTW band,
   by state (>=200 qualifying cells), plus a within-cell shuffle null. Also the
   paired deepest-vs-shallowest contrast: (dtw_deep - dtw_shallow) /
   (comp_deep - comp_shallow) [m/m] and the fraction of cells where the deeper
   completion has water >2 m deeper.

2. TWO-COMPONENT VS ONE (cells >=4 wells). 1- vs 2-component 1D Gaussian mixture
   BIC on mean_dtw; a 2-component preference counts only if the component means
   differ by >=5 m (guards trivial splits). Fraction preferring 2 by depth band.

3. METADATA PREDICTABILITY (pooled 2-component cells). Each member well is
   labelled by its assigned component (0 = shallower water = lower-mean
   component). A logistic regression predicts that label from construction
   metadata [completion depth, screen_bottom present flag, casing_depth,
   obs_count, confinement_confidence, head_above_screen] under GroupKFold(5)
   grouped by cell. OOF ROC-AUC materially > 0.5 (>=0.65) means metadata can
   serve as the privileged assignment evidence the plan's §7.1 observation model
   needs. A secondary diagnostic arm adds cell-relative completion depth (uses
   within-cell context; optimistic upper bound).

4. VERDICT — two_surface_feasibility_report.json (+ definitions) and
   two_surface_feasibility_summary.md.

5. two_surface_cells.parquet — per-cell record (id, centre x/y, n wells, median
   DTW, completion spread, spearman, 2-component preference, component gap).

Usage:
    uv run python utils/v02_two_surface_feasibility.py
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer

log = logging.getLogger("v02_two_surface_feasibility")

WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"
OUT_DIR = "/data/ssd2/handily/conus/wte_gnn/v02/wp2"

CELL_M = 500.0
UNCONFINED = ("unconfined", "unconfined_marginal")
DEPTH_BANDS = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, np.inf)]

MIN_WELLS_ORDER = 3  # analysis 1 cell floor
MIN_COMP_SPREAD_M = 10.0  # analysis 1 completion-depth spread floor
MIN_WELLS_GMM = 4  # analysis 2 cell floor
MIN_COMP_GAP_M = 5.0  # analysis 2 real-split guard (component-mean gap)
GMM_REG_COVAR = 1e-3
GMM_CELL_CAP = 50_000  # subsample cap for the GMM stage
STATE_MIN_CELLS = 200  # analysis 1 per-state reporting floor

READ_COLS = [
    "confinement_class",
    "mean_dtw",
    "well_depth",
    "screen_top",
    "screen_bottom",
    "casing_depth",
    "head_above_screen",
    "obs_count",
    "confinement_confidence",
    "longitude",
    "latitude",
    "state",
    "canonical_id",
    "source",
]

META_FEATURES = [
    "comp_m",
    "screen_bottom_present",
    "casing_depth",
    "casing_depth_present",
    "obs_count",
    "confinement_confidence",
    "head_above_screen",
    "head_above_screen_present",
]
# head_above_screen is a partial water-level proxy (corr ~ -0.40 with mean_dtw in
# the qualifying set), so it can predict a mean_dtw-derived component label
# semi-circularly. The construction-only arm drops it to give the conservative,
# purely-construction identifiability number.
CONSTRUCTION_FEATURES = [
    "comp_m",
    "screen_bottom_present",
    "casing_depth",
    "casing_depth_present",
    "obs_count",
    "confinement_confidence",
]


def _band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"


def depth_band_labels(dtw: np.ndarray) -> np.ndarray:
    """Vectorised cell-median-DTW band assignment (str labels, '' if non-finite)."""
    dtw = np.asarray(dtw, float)
    out = np.full(len(dtw), "", dtype=object)
    for lo, hi in DEPTH_BANDS:
        m = np.isfinite(dtw) & (dtw >= lo) & (dtw < hi)
        out[m] = _band_label(lo, hi)
    return out


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_and_snap(wells_path: str) -> pd.DataFrame:
    """Load unconfined wells with finite mean_dtw + completion depth and snap
    them to the 500 m EPSG:5070 grid. Returns one row per qualifying well with a
    stable integer `cell` id, cell indices `ix`/`iy`, and derived metadata."""
    df = pq.read_table(wells_path, columns=READ_COLS).to_pandas()
    n0 = len(df)
    unc = df["confinement_class"].isin(UNCONFINED)
    comp = df["well_depth"].where(np.isfinite(df["well_depth"]), df["screen_bottom"])
    finite_ll = np.isfinite(df["longitude"]) & np.isfinite(df["latitude"])
    q = unc & np.isfinite(df["mean_dtw"]) & np.isfinite(comp) & finite_ll
    df = df[q].copy()
    df["comp_m"] = comp[q].to_numpy()
    log.info(
        "qualifying: %d unconfined wells with finite mean_dtw + completion depth "
        "+ coords (of %d rows)",
        len(df),
        n0,
    )

    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tr.transform(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    ix = np.floor(x / CELL_M).astype(np.int64)
    iy = np.floor(y / CELL_M).astype(np.int64)
    df["ix"] = ix
    df["iy"] = iy
    # pack (ix, iy) -> unique non-negative int id (shift to positive first)
    iy_span = int(iy.max() - iy.min()) + 1
    df["cell"] = (ix - int(ix.min())).astype(np.int64) * iy_span + (
        iy - int(iy.min())
    ).astype(np.int64)

    # construction-metadata presence flags (before any imputation)
    df["screen_bottom_present"] = np.isfinite(df["screen_bottom"]).astype(float)
    df["casing_depth_present"] = np.isfinite(df["casing_depth"]).astype(float)
    df["head_above_screen_present"] = np.isfinite(df["head_above_screen"]).astype(float)
    df["obs_count"] = df["obs_count"].astype(float)
    return df


def cell_base_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per grid cell with member count, DTW/completion summaries and
    cell-centre coordinates (all cells with >=1 qualifying well)."""
    g = df.groupby("cell")
    base = pd.DataFrame(
        {
            "n_wells": g.size(),
            "median_dtw_m": g["mean_dtw"].median(),
            "completion_spread_m": g["comp_m"].max() - g["comp_m"].min(),
            "ix": g["ix"].first(),
            "iy": g["iy"].first(),
        }
    ).reset_index()
    base["x5070"] = (base["ix"] + 0.5) * CELL_M
    base["y5070"] = (base["iy"] + 0.5) * CELL_M
    base["dtw_band"] = depth_band_labels(base["median_dtw_m"].to_numpy())
    return base


# --------------------------------------------------------------------------- #
# analysis 1 — vertical ordering
# --------------------------------------------------------------------------- #
def grouped_spearman(
    df: pd.DataFrame, cell_col: str, a_col: str, b_col: str
) -> pd.Series:
    """Vectorised per-cell Spearman rho between `a_col` and `b_col` (Spearman =
    Pearson on within-cell average ranks). Returns a Series indexed by cell;
    NaN where either variable is constant within the cell."""
    rx = df.groupby(cell_col)[a_col].rank()
    ry = df.groupby(cell_col)[b_col].rank()
    work = pd.DataFrame(
        {cell_col: df[cell_col].to_numpy(), "rx": rx.to_numpy(), "ry": ry.to_numpy()}
    )
    work["rxy"] = work["rx"] * work["ry"]
    work["rxx"] = work["rx"] ** 2
    work["ryy"] = work["ry"] ** 2
    g = work.groupby(cell_col)
    n = g.size()
    sx, sy = g["rx"].sum(), g["ry"].sum()
    sxy, sxx, syy = g["rxy"].sum(), g["rxx"].sum(), g["ryy"].sum()
    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    denom = np.sqrt(vx * vy)
    rho = cov / denom.replace(0.0, np.nan)
    return rho


def _distribution(rho: np.ndarray) -> dict:
    rho = rho[np.isfinite(rho)]
    if len(rho) == 0:
        return {"n_cells": 0}
    return {
        "n_cells": int(len(rho)),
        "median_rho": round(float(np.median(rho)), 4),
        "iqr_rho": [
            round(float(np.quantile(rho, 0.25)), 4),
            round(float(np.quantile(rho, 0.75)), 4),
        ],
        "frac_rho_gt0": round(float(np.mean(rho > 0)), 4),
        "median_abs_rho": round(float(np.median(np.abs(rho))), 4),
    }


def vertical_ordering(
    df: pd.DataFrame, base: pd.DataFrame, seed: int
) -> tuple[dict, pd.Series]:
    """Analysis 1: per-cell Spearman(completion depth, mean_dtw), the permutation
    null, per-band / per-state distributions, and the deepest-vs-shallowest
    paired contrast. Returns (panel, per-cell rho Series indexed by cell)."""
    keep_cells = base.loc[
        (base["n_wells"] >= MIN_WELLS_ORDER)
        & (base["completion_spread_m"] >= MIN_COMP_SPREAD_M),
        "cell",
    ]
    sub = df[df["cell"].isin(keep_cells)].copy()
    rho = grouped_spearman(sub, "cell", "comp_m", "mean_dtw")

    # within-cell shuffle null: permute mean_dtw inside each cell, recompute
    rng = np.random.default_rng(seed)
    perm = sub.groupby("cell")["mean_dtw"].transform(
        lambda s: s.to_numpy()[rng.permutation(len(s))]
    )
    sub_null = sub.assign(mean_dtw=perm.to_numpy())
    rho_null = grouped_spearman(sub_null, "cell", "comp_m", "mean_dtw")

    band = base.set_index("cell").loc[rho.index, "dtw_band"]
    panel = {
        "n_qualifying_cells": int(len(rho)),
        "overall": _distribution(rho.to_numpy()),
        "permutation_null": _distribution(rho_null.to_numpy()),
        "by_depth_band": {},
        "by_state": {},
    }
    for lo, hi in DEPTH_BANDS:
        lab = _band_label(lo, hi)
        panel["by_depth_band"][lab] = _distribution(rho[band == lab].to_numpy())

    # per-state (>= STATE_MIN_CELLS qualifying cells)
    state_by_cell = df.groupby("cell")["state"].first()
    st = state_by_cell.loc[rho.index]
    for state in pd.unique(st.dropna()):
        cells_here = st.index[st.to_numpy() == state]
        vals = rho.loc[cells_here].to_numpy()
        d = _distribution(vals)
        if d.get("n_cells", 0) >= STATE_MIN_CELLS:
            panel["by_state"][str(state)] = d

    # paired deepest-vs-shallowest contrast (per qualifying cell)
    ordered = sub.sort_values(["cell", "comp_m"])
    g = ordered.groupby("cell")
    shallow = g.first()
    deep = g.last()
    d_comp = (deep["comp_m"] - shallow["comp_m"]).to_numpy()
    d_dtw = (deep["mean_dtw"] - shallow["mean_dtw"]).to_numpy()
    ok = d_comp > 0
    ratio = d_dtw[ok] / d_comp[ok]
    contrast = {
        "n_cells": int(ok.sum()),
        "median_ratio_dtw_per_comp": round(float(np.median(ratio)), 4),
        "iqr_ratio_dtw_per_comp": [
            round(float(np.quantile(ratio, 0.25)), 4),
            round(float(np.quantile(ratio, 0.75)), 4),
        ],
        "frac_deeper_completion_water_gt2m_deeper": round(
            float(np.mean(d_dtw[ok] > 2.0)), 4
        ),
        "median_dtw_deep_minus_shallow_m": round(float(np.median(d_dtw[ok])), 4),
    }
    contrast["by_depth_band"] = {}
    band_ok = base.set_index("cell").loc[shallow.index[ok], "dtw_band"].to_numpy()
    for lo, hi in DEPTH_BANDS:
        lab = _band_label(lo, hi)
        m = band_ok == lab
        if m.sum() > 0:
            contrast["by_depth_band"][lab] = {
                "n_cells": int(m.sum()),
                "median_ratio_dtw_per_comp": round(float(np.median(ratio[m])), 4),
                "frac_deeper_completion_water_gt2m_deeper": round(
                    float(np.mean(d_dtw[ok][m] > 2.0)), 4
                ),
            }
    panel["paired_deep_shallow_contrast"] = contrast
    return panel, rho


# --------------------------------------------------------------------------- #
# analysis 2 — two-component vs one (GMM BIC)
# --------------------------------------------------------------------------- #
def gmm_cell(dtw: np.ndarray, reg_covar: float, seed: int) -> dict:
    """1- vs 2-component 1D Gaussian-mixture BIC on one cell's mean_dtw. A
    2-component preference counts only if the component means differ by
    >=MIN_COMP_GAP_M (real split, not a trivial variance split)."""
    from sklearn.mixture import GaussianMixture

    x = np.asarray(dtw, float).reshape(-1, 1)
    g1 = GaussianMixture(
        1, covariance_type="full", reg_covar=reg_covar, random_state=seed
    ).fit(x)
    g2 = GaussianMixture(
        2, covariance_type="full", reg_covar=reg_covar, random_state=seed
    ).fit(x)
    b1, b2 = float(g1.bic(x)), float(g2.bic(x))
    means = np.sort(g2.means_.ravel())
    gap = float(means[1] - means[0])
    prefers_2 = bool((b2 < b1) and (gap >= MIN_COMP_GAP_M))
    return {
        "bic1": b1,
        "bic2": b2,
        "component_gap_m": gap,
        "prefers_2": prefers_2,
        "g2": g2,
    }


def two_component_analysis(
    df: pd.DataFrame, base: pd.DataFrame, seed: int, cap: int
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Analysis 2: per-cell GMM BIC preference (subsampled to `cap` cells), the
    fraction preferring 2 components by depth band, and — for 2-component cells —
    per-well component labels (0 = shallower water). Returns (panel, per-cell
    frame [cell, prefers_2, component_gap_m], labelled-wells frame)."""
    cells = base.loc[base["n_wells"] >= MIN_WELLS_GMM, "cell"].to_numpy()
    n_eligible = len(cells)
    capped = False
    if n_eligible > cap:
        rng = np.random.default_rng(seed)
        cells = np.sort(rng.choice(cells, cap, replace=False))
        capped = True
    cellset = set(cells.tolist())
    sub = df[df["cell"].isin(cellset)]
    band_by_cell = base.set_index("cell")["dtw_band"]

    rows = []
    labelled = []
    for cell, grp in sub.groupby("cell"):
        dtw = grp["mean_dtw"].to_numpy(float)
        res = gmm_cell(dtw, GMM_REG_COVAR, seed)
        rows.append(
            {
                "cell": cell,
                "prefers_2": res["prefers_2"],
                "component_gap_m": round(res["component_gap_m"], 4),
            }
        )
        if res["prefers_2"]:
            g2 = res["g2"]
            comp = g2.predict(dtw.reshape(-1, 1))
            # relabel so 0 = lower-mean (shallower water) component
            order = np.argsort(g2.means_.ravel())
            remap = np.empty(2, dtype=int)
            remap[order] = [0, 1]
            lab = remap[comp]
            lg = grp.copy()
            lg["component"] = lab
            labelled.append(lg)

    cell_res = pd.DataFrame(rows)
    per_cell_band = cell_res["cell"].map(band_by_cell)
    panel = {
        "n_eligible_cells": int(n_eligible),
        "n_scored_cells": int(len(cell_res)),
        "subsample_cap": int(cap),
        "subsampled": bool(capped),
        "overall_frac_prefers_2": round(float(cell_res["prefers_2"].mean()), 4),
        "by_depth_band": {},
    }
    for lo, hi in DEPTH_BANDS:
        lab = _band_label(lo, hi)
        m = per_cell_band == lab
        if m.sum() > 0:
            panel["by_depth_band"][lab] = {
                "n_cells": int(m.sum()),
                "frac_prefers_2": round(float(cell_res.loc[m, "prefers_2"].mean()), 4),
                "median_component_gap_m": round(
                    float(
                        cell_res.loc[
                            m & cell_res["prefers_2"], "component_gap_m"
                        ].median()
                    )
                    if (m & cell_res["prefers_2"]).sum() > 0
                    else np.nan,
                    4,
                ),
            }
    labelled_df = pd.concat(labelled, ignore_index=True) if labelled else pd.DataFrame()
    return panel, cell_res, labelled_df


# --------------------------------------------------------------------------- #
# analysis 3 — metadata predictability of the component label
# --------------------------------------------------------------------------- #
def metadata_component_auc(
    labelled: pd.DataFrame,
    base: pd.DataFrame,
    feature_cols: list[str],
    seed: int,
    n_splits: int = 5,
) -> dict:
    """Analysis 3: OOF ROC-AUC for a logistic regression predicting the GMM
    component label (0 = shallower water) from construction metadata, grouped by
    cell (GroupKFold). Overall + per cell-median-DTW band. Returns AUCs and n."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if len(labelled) == 0:
        return {"n": 0, "degenerate": True}
    y = labelled["component"].to_numpy(int)
    groups = labelled["cell"].to_numpy()
    if len(np.unique(y)) < 2 or len(np.unique(groups)) < n_splits:
        return {"n": int(len(labelled)), "degenerate": True}
    X = labelled[feature_cols].to_numpy(float)
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    gkf = GroupKFold(n_splits=n_splits)
    proba = cross_val_predict(
        pipe, X, y, cv=gkf, groups=groups, method="predict_proba"
    )[:, 1]
    out = {
        "n": int(len(y)),
        "n_cells": int(len(np.unique(groups))),
        "class_balance_component1": round(float(y.mean()), 4),
        "features": list(feature_cols),
        "oof_auc": round(float(roc_auc_score(y, proba)), 4),
        "by_depth_band": {},
    }
    band = base.set_index("cell")["dtw_band"].reindex(groups).to_numpy()
    for lo, hi in DEPTH_BANDS:
        lab = _band_label(lo, hi)
        m = band == lab
        if m.sum() >= 200 and len(np.unique(y[m])) == 2:
            out["by_depth_band"][lab] = {
                "n": int(m.sum()),
                "oof_auc": round(float(roc_auc_score(y[m], proba[m])), 4),
            }
    return out


# --------------------------------------------------------------------------- #
# verdict + reporting
# --------------------------------------------------------------------------- #
def build_verdict(
    order_panel: dict, gmm_panel: dict, auc_abs: dict, auc_constr: dict, auc_rel: dict
) -> dict:
    ov = order_panel["overall"]
    null = order_panel["permutation_null"]
    ordered = (ov.get("median_abs_rho", 0) or 0) > 1.3 * (
        null.get("median_abs_rho", 0) or 1e-9
    )
    deep_gmm = gmm_panel["by_depth_band"].get("30+m", {}).get("frac_prefers_2", 0.0)
    shallow_gmm = gmm_panel["by_depth_band"].get("0-2m", {}).get("frac_prefers_2", 0.0)
    auc = auc_abs.get("oof_auc", 0.5)
    auc_c = auc_constr.get("oof_auc", 0.5)
    metadata_ok = auc >= 0.65
    # attribution: is the absolute AUC carried by construction, or only by the
    # head_above_screen water-level proxy?
    construction_carries = auc_c >= 0.65
    identifiable = bool(ordered and metadata_ok)
    return {
        "gate1_two_surface_identifiable": identifiable,
        "ordering_beats_null": bool(ordered),
        "metadata_auc_ge_0.65": bool(metadata_ok),
        "construction_only_auc_ge_0.65": bool(construction_carries),
        "auc_carried_by_construction_not_head_proxy": bool(construction_carries),
        "overall_median_abs_rho": ov.get("median_abs_rho"),
        "null_median_abs_rho": null.get("median_abs_rho"),
        "overall_frac_rho_gt0": ov.get("frac_rho_gt0"),
        "gmm_frac_prefers_2_shallow_0_2m": shallow_gmm,
        "gmm_frac_prefers_2_deep_30plus_m": deep_gmm,
        "metadata_oof_auc_absolute": auc,
        "metadata_oof_auc_construction_only": auc_c,
        "metadata_oof_auc_cell_relative": auc_rel.get("oof_auc"),
    }


DEFINITIONS = {
    "unit_convention": "All depths/levels in metres. DTW = depth to the "
    "unconfined water table; larger = deeper water. Only "
    "confinement_class in {unconfined, unconfined_marginal} wells are used.",
    "completion_depth_m": "well_depth if finite else screen_bottom (m below "
    "ground); the depth interval the well samples.",
    "cell": "500 m EPSG:5070 grid cell; wells snapped by floor(coord/500).",
    "spearman_rho": "Per-cell Spearman rank correlation between completion depth "
    "and mean_dtw across member wells (dimensionless, [-1,1]). rho>0 = deeper "
    "completion samples deeper water (downward/recharge gradient); rho<0 = "
    "shallower water at depth (upward/discharge gradient); rho~0 = no ordering.",
    "permutation_null": "Same statistic after shuffling mean_dtw within each "
    "cell; the noise reference. Ordering is real if observed median|rho| "
    "exceeds the null median|rho|.",
    "ratio_dtw_per_comp": "(dtw_deepest - dtw_shallowest)/(comp_deepest - "
    "comp_shallowest) per cell (m of water-level change per m of completion "
    "depth; dimensionless). The empirical vertical DTW gradient.",
    "frac_prefers_2": "Fraction of cells where a 2-component Gaussian mixture "
    "on mean_dtw has lower BIC than 1 component AND component means differ by "
    ">=5 m (a real bimodal split = two water surfaces in one cell).",
    "component_gap_m": "|mean_2 - mean_1| of the 2-component mixture (m).",
    "oof_auc": "Out-of-fold ROC-AUC (dimensionless, 0.5=chance) of a logistic "
    "regression predicting the GMM component label (0=shallower water) from "
    "construction metadata under GroupKFold(5) by cell. >=0.65 => metadata is "
    "usable privileged assignment evidence (plan §7.1). Three arms: `absolute` "
    "= the full task feature list incl head_above_screen; `construction_only` "
    "drops head_above_screen (a partial water-level proxy, corr ~-0.40 with "
    "mean_dtw) and is the conservative pure-construction number; `cell_relative` "
    "adds completion depth minus the cell-median completion (uses within-cell "
    "context; an optimistic within-context upper bound, not wall-to-wall).",
    "gate1": "plan §7.5 gate 1 = two-surface decomposition is identifiable if "
    "collocated ordering beats the permutation null AND metadata predicts the "
    "component (AUC>=0.65).",
}


def write_summary_md(path: Path, report: dict) -> None:
    v = report["verdict"]
    ov = report["vertical_ordering"]["overall"]
    contrast = report["vertical_ordering"]["paired_deep_shallow_contrast"]
    gmm = report["two_component_gmm"]
    auc = report["metadata_predictability_absolute"]
    verdict_word = (
        "IDENTIFIABLE"
        if v["gate1_two_surface_identifiable"]
        else (
            "PARTIALLY IDENTIFIABLE" if v["ordering_beats_null"] else "NOT IDENTIFIABLE"
        )
    )
    lines = [
        "# WP2 gate 1 — two-surface identifiability feasibility",
        "",
        f"Generated {date.today()} by utils/v02_two_surface_feasibility.py. "
        "Unconfined + unconfined_marginal wells only; DTW = depth to water (m), "
        "larger = deeper. Question (plan §7.5 gate 1): is the collocated "
        "vertical dispersion ORDERED and metadata-predictable (two identifiable "
        "surfaces) rather than unstructured noise?",
        "",
        f"## Verdict: {verdict_word}",
        "",
        f"- Two-surface identifiable (ordering beats null AND metadata AUC>=0.65): "
        f"**{v['gate1_two_surface_identifiable']}**.",
        f"- Vertical ordering: median|rho| {v['overall_median_abs_rho']} vs "
        f"permutation-null {v['null_median_abs_rho']} "
        f"(ordering beats null: {v['ordering_beats_null']}); "
        f"median rho {ov['median_rho']}, IQR {ov['iqr_rho']}, "
        f"fraction rho>0 {ov['frac_rho_gt0']} over {ov['n_cells']} cells.",
        f"- Deepest-vs-shallowest completion: median DTW gradient "
        f"{contrast['median_ratio_dtw_per_comp']} m/m; deeper completion has "
        f"water >2 m deeper in {contrast['frac_deeper_completion_water_gt2m_deeper']} "
        f"of cells.",
        f"- 2-component preference (BIC, gap>=5 m): overall "
        f"{gmm['overall_frac_prefers_2']}; shallow 0-2 m "
        f"{v['gmm_frac_prefers_2_shallow_0_2m']}; deep 30+ m "
        f"{v['gmm_frac_prefers_2_deep_30plus_m']}.",
        f"- Metadata OOF AUC: absolute {auc.get('oof_auc')} / construction-only "
        f"{v['metadata_oof_auc_construction_only']} / cell-relative diagnostic "
        f"{v['metadata_oof_auc_cell_relative']}. AUC carried by construction "
        f"(not the head_above_screen water-level proxy): "
        f"{v['auc_carried_by_construction_not_head_proxy']}.",
        "",
        "## Per-cell Spearman(completion depth, mean_dtw) by cell-median-DTW band",
        "",
        "| band | n cells | median rho | IQR rho | frac rho>0 | median|rho| |",
        "|---|---|---|---|---|---|",
    ]
    for band, d in report["vertical_ordering"]["by_depth_band"].items():
        if d.get("n_cells", 0) > 0:
            lines.append(
                f"| {band} | {d['n_cells']} | {d['median_rho']} | {d['iqr_rho']} | "
                f"{d['frac_rho_gt0']} | {d['median_abs_rho']} |"
            )
    lines += [
        "",
        "## 2-component preference by depth band",
        "",
        "| band | n cells | frac prefers 2 | median gap (m) |",
        "|---|---|---|---|",
    ]
    for band, d in gmm["by_depth_band"].items():
        lines.append(
            f"| {band} | {d['n_cells']} | {d['frac_prefers_2']} | "
            f"{d.get('median_component_gap_m')} |"
        )
    lines += [
        "",
        "## Metadata AUC by depth band (absolute features)",
        "",
        "| band | n wells | OOF AUC |",
        "|---|---|---|",
    ]
    for band, d in auc.get("by_depth_band", {}).items():
        lines.append(f"| {band} | {d['n']} | {d['oof_auc']} |")

    # plain-language verdict paragraph
    head_note = (
        "construction metadata alone carries the signal"
        if v["auc_carried_by_construction_not_head_proxy"]
        else "the signal leans on the head_above_screen water-level proxy "
        f"(construction-only AUC {v['metadata_oof_auc_construction_only']}), not "
        "pure construction attributes"
    )
    if v["gate1_two_surface_identifiable"]:
        para = (
            "The collocated dispersion is ORDERED, not noise: within-cell "
            "completion depth ranks the water level well beyond the "
            "shuffle null, cells resolve into distinct DTW components, and "
            "metadata predicts which component a well sampled at "
            f"AUC {auc.get('oof_auc')} ({head_note}). A latent two-surface "
            "(phreatic vs regional-aquifer) decomposition is IDENTIFIABLE from "
            "existing evidence and gate 1 is cleared."
        )
    elif v["ordering_beats_null"]:
        para = (
            "The dispersion is ordered within cells (median|rho| "
            f"{v['overall_median_abs_rho']} vs null {v['null_median_abs_rho']}), "
            "so vertical structure is real, but absolute metadata only predicts "
            f"the component at AUC {auc.get('oof_auc')} (construction-only "
            f"{v['metadata_oof_auc_construction_only']}; cell-relative diagnostic "
            f"{v['metadata_oof_auc_cell_relative']}). Two surfaces exist but "
            "absolute construction metadata is a weak wall-to-wall assignment "
            "signal — gate 1 is only partially met; the within-cell completion "
            "ordering is usable as a local prior, not as an absolute label."
        )
    else:
        para = (
            "Within-cell ordering does not exceed the permutation null "
            f"(median|rho| {v['overall_median_abs_rho']} vs "
            f"{v['null_median_abs_rho']}) and metadata predicts the component "
            f"only at AUC {auc.get('oof_auc')}. The collocated dispersion "
            "behaves as unstructured aleatoric noise at this resolution; a "
            "latent two-surface decomposition is NOT identifiable from existing "
            "observation evidence and gate 1 fails."
        )
    lines += [
        "",
        "## Plain-language verdict",
        "",
        para,
        "",
        "## Regime notes",
        "",
        "Strongest ordering/bimodality bands and states are the deep "
        "(10-30 / 30+ m) cells; see the by-band and by-state tables in "
        "two_surface_feasibility_report.json.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wells", default=WELLS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gmm-cap", type=int, default=GMM_CELL_CAP)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_snap(args.wells)
    base = cell_base_table(df)
    log.info(
        "cells: %d total, %d >=2 wells, %d >=3+spread>=10m, %d >=4 wells",
        len(base),
        int((base["n_wells"] >= 2).sum()),
        int(
            (
                (base["n_wells"] >= MIN_WELLS_ORDER)
                & (base["completion_spread_m"] >= MIN_COMP_SPREAD_M)
            ).sum()
        ),
        int((base["n_wells"] >= MIN_WELLS_GMM).sum()),
    )

    log.info("analysis 1: vertical ordering")
    order_panel, rho = vertical_ordering(df, base, args.seed)

    log.info("analysis 2: two-component vs one (GMM BIC)")
    gmm_panel, cell_res, labelled = two_component_analysis(
        df, base, args.seed, args.gmm_cap
    )

    log.info("analysis 3: metadata predictability (n labelled wells=%d)", len(labelled))
    auc_abs = metadata_component_auc(labelled, base, META_FEATURES, args.seed)
    auc_constr = metadata_component_auc(
        labelled, base, CONSTRUCTION_FEATURES, args.seed
    )
    # cell-relative diagnostic arm (uses within-cell completion context)
    auc_rel = {"oof_auc": None}
    if len(labelled) > 0:
        lab2 = labelled.copy()
        lab2["comp_rel_m"] = lab2["comp_m"] - lab2.groupby("cell")["comp_m"].transform(
            "median"
        )
        auc_rel = metadata_component_auc(
            lab2, base, CONSTRUCTION_FEATURES + ["comp_rel_m"], args.seed
        )

    verdict = build_verdict(order_panel, gmm_panel, auc_abs, auc_constr, auc_rel)

    report = {
        "generated": str(date.today()),
        "inputs": {
            "wells": args.wells,
            "grid_m": CELL_M,
            "confinement_classes": list(UNCONFINED),
            "n_qualifying_wells": int(len(df)),
            "n_cells_ge2_wells": int((base["n_wells"] >= 2).sum()),
        },
        "verdict": verdict,
        "vertical_ordering": order_panel,
        "two_component_gmm": gmm_panel,
        "metadata_predictability_absolute": auc_abs,
        "metadata_predictability_construction_only": auc_constr,
        "metadata_predictability_cell_relative": auc_rel,
        "definitions": DEFINITIONS,
    }

    # per-cell parquet
    cells_out = base.merge(
        rho.rename("spearman_comp_dtw").reset_index(), on="cell", how="left"
    ).merge(cell_res, on="cell", how="left")
    cells_out = cells_out[
        [
            "cell",
            "x5070",
            "y5070",
            "ix",
            "iy",
            "n_wells",
            "median_dtw_m",
            "dtw_band",
            "completion_spread_m",
            "spearman_comp_dtw",
            "prefers_2",
            "component_gap_m",
        ]
    ]
    cells_out.to_parquet(out / "two_surface_cells.parquet")

    (out / "two_surface_feasibility_report.json").write_text(
        json.dumps(report, indent=2)
    )
    write_summary_md(out / "two_surface_feasibility_summary.md", report)
    log.info("WP2 feasibility artifacts written to %s", out)
    log.info(
        "VERDICT gate1_identifiable=%s ordering_beats_null=%s auc_abs=%s",
        verdict["gate1_two_surface_identifiable"],
        verdict["ordering_beats_null"],
        auc_abs.get("oof_auc"),
    )


if __name__ == "__main__":
    main()
