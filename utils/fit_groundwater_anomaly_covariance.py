"""Anomaly recoverability + covariance/connectivity + pilot selection.

Ops-assimilation rung 2 / interim Track A3, per ``notes/ops_assimilation_plan.md``
(§5.1 covariance/connectivity, §5.2 low-rank, §5.3 pilot selection; §2.2 anomaly
covariance object -- NEVER GNN residuals) and ``notes/ops_assimilation_interim_plan.md``
(Track A / A3, oracle availability).

The question this script answers is the rung-2 GATE:

    Do contemporaneous neighbours provide POSITIVE, uncertainty-bounded
    reconstruction skill at left-out wells BEYOND climatology AND persistence,
    and at what spatial support? If NO -> the product ships station-status-only.

Everything is fit on station SGI (the standardized anomaly the status product
already produced), with metric anomaly ``z_m`` as a secondary cross-check. No GNN
artifact is read anywhere. SGI is NEVER winsorized. Every excluded row is counted.

Analysis population (per plan): monitoring_role in {drought_core, condition_network}
(background wells), status_flag == 'ok'. Key fits run on BOTH the all-background
stratum and the drought-core-only stratum (§5.1 climate-response vs pumping-affected
contrast). Diagnostic (confined/pumping-dominated) wells appear only as a clearly
labelled contrast stratum, never in the headline.

Stages (all run by default; --stages to subset):
  correlogram  contemporaneous same-year-month pair correlation of SGI vs
               (a) Euclidean isotropic distance bins, (b) direction (anisotropy),
               (c) same/diff aquifer string, same/diff state, (d) same/diff HUC4
               (spatial join to the national WBD HU4 polygons), (e) temporal lag
               (0/1/3/6/12 mo) within wells and between neighbour wells. Pairs are
               subsampled with a fixed seed and a global cap; per-bin pair counts
               are reported and no bin with < --min-bin-pairs supports a conclusion.
  models       fit PSD exponential + spherical correlation models to the binned
               isotropic SGI correlogram per stratum (scipy least_squares, weighted
               by pair count); report range/sill/nugget + fit RMSE.
  kriging      HEADLINE gate. Leave-well-out simple kriging (SGI mean-zero) with the
               fitted model at held-out wells over a panel of test months; skill
               (MAE/RMSE/corr) vs (i) climatology (SGI=0) and (ii) persistence
               (well's own previous available SGI), STRATIFIED by distance to the
               nearest used observation. Block-bootstrap CI over test months.
  lowrank      §5.2 -- well x month SGI matrix on long-record background wells;
               soft-impute SVD; leave-well-out + leave-period-out reconstruction
               error vs rank k in {1,2,4,8,16,32}; mode stability across two random
               well-halves (loading correlation after sign alignment).
  pilots       §5.3 -- score state x aquifer-string units against the preregistered
               criteria and select ONE climate-responsive/minimally-pumped pilot and
               ONE pumping-heavy/closed-basin adversarial pilot.

Outputs (--out-dir, default .../s0/covariance/):
  covariance_models.json       fitted PSD model params per stratum/model
  covariance_diagnostics.json  binned correlograms + pair counts, kriging skill by
                               band vs both baselines, anisotropy/aquifer/state/HUC,
                               lag structure, low-rank curves + stability
  covariance_report.html       human-readable panel of the above
  pilot_selection.json         the two pilots with criteria scores + thresholds used

Every emitted skill/diagnostic block carries ``availability: 'oracle'``.

Usage:
    uv run python utils/fit_groundwater_anomaly_covariance.py \
        --max-pairs 8000000 --seed 17 --workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

log = logging.getLogger("fit_gw_anomaly_covariance")

STATUS_DIR = "/data/ssd2/handily/conus/ops_status/s0/status"
AUDIT_DIR = "/data/ssd2/handily/conus/ops_status/s0/audit"
OUT_DIR = "/data/ssd2/handily/conus/ops_status/s0/covariance"
HUC4_PARQUET = "/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu4_5070.parquet"

WELL_KEY = ["source", "source_well_id"]
AVAIL = "oracle"

# Isotropic distance-bin edges in kilometres (plan §5.1).
DIST_BIN_KM = [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0]
# Distance-to-nearest-used-observation bands for kriging skill (km).
SUPPORT_BAND_KM = [0.0, 5.0, 10.0, 25.0, 50.0, np.inf]
LAG_MONTHS = [0, 1, 3, 6, 12]
LOWRANK_KS = [1, 2, 4, 8, 16, 32]


# --------------------------------------------------------------------------- #
# population
# --------------------------------------------------------------------------- #
def load_population(status_dir: str, audit_dir: str) -> pd.DataFrame:
    """Scored SGI rows joined to inventory role/aquifer/projected coords.

    Returns background + diagnostic rows (diagnostic kept only for the labelled
    contrast stratum). x5070/y5070 are metres on EPSG:5070 -> Euclidean distance
    in metres directly.
    """
    inv = pd.read_parquet(
        f"{audit_dir}/well_inventory.parquet",
        columns=WELL_KEY
        + [
            "monitoring_role",
            "aquifer",
            "x5070",
            "y5070",
            "dup_group_size",
            "state",
        ],
    )
    st = pd.read_parquet(
        f"{status_dir}/groundwater_status_monthly.parquet",
        columns=WELL_KEY + ["year", "month", "sgi", "z_m", "status_flag"],
    )
    n_all = len(st)
    st = st[st.status_flag == "ok"].copy()
    log.info(
        "status rows %d -> ok %d (dropped %d not-ok)", n_all, len(st), n_all - len(st)
    )
    st = st.merge(inv, on=WELL_KEY, how="left", suffixes=("", "_inv"))
    # state present on both; prefer inventory state where status lacks it (it has it)
    if "state_inv" in st.columns:
        st["state"] = st["state"].fillna(st.pop("state_inv"))
    n_norole = st.monitoring_role.isna().sum()
    if n_norole:
        log.info("dropping %d ok rows with no inventory monitoring_role", n_norole)
        st = st[st.monitoring_role.notna()]
    n_nocoord = st.x5070.isna().sum()
    if n_nocoord:
        log.info("dropping %d ok rows with no x5070/y5070", n_nocoord)
        st = st[st.x5070.notna() & st.y5070.notna()]
    n_nosgi = st.sgi.isna().sum()
    if n_nosgi:
        log.info("dropping %d ok rows with null sgi", n_nosgi)
        st = st[st.sgi.notna()]
    st["month_index"] = st.year.astype(int) * 12 + (st.month.astype(int) - 1)
    return st.reset_index(drop=True)


def dedupe_colocated(df: pd.DataFrame, thin_m: float) -> tuple[pd.DataFrame, dict]:
    """Thin co-located wells: keep the best-supported well per ~thin_m cluster.

    14.5% of wells sit < 10 m from a neighbour (audit). Two wells that close are
    effectively one station and would inject spurious near-unit correlation into
    the smallest distance bin. We cluster unique well positions with a union-find
    over KDTree pairs within ``thin_m`` and keep, per cluster, the well with the
    most scored SGI months. Everything dropped is counted.
    """
    wells = (
        df.groupby(WELL_KEY)
        .agg(x5070=("x5070", "first"), y5070=("y5070", "first"), n_sgi=("sgi", "size"))
        .reset_index()
    )
    xy = wells[["x5070", "y5070"]].to_numpy()
    tree = cKDTree(xy)
    pairs = tree.query_pairs(r=thin_m, output_type="ndarray")
    parent = np.arange(len(wells))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(wells))])
    wells["cluster"] = roots
    # keep best-supported per cluster
    wells = wells.sort_values("n_sgi", ascending=False)
    keep = wells.drop_duplicates("cluster", keep="first")
    n_clusters_multi = int((wells.groupby("cluster").size() > 1).sum())
    dropped = len(wells) - len(keep)
    keep_keys = set(map(tuple, keep[WELL_KEY].to_numpy()))
    mask = [tuple(r) in keep_keys for r in df[WELL_KEY].to_numpy()]
    out = df[np.asarray(mask)].reset_index(drop=True)
    meta = {
        "thin_radius_m": thin_m,
        "n_wells_before": int(len(wells)),
        "n_wells_after": int(len(keep)),
        "n_wells_dropped": int(dropped),
        "n_clusters_with_multiple": n_clusters_multi,
    }
    log.info(
        "co-located thinning @%.0fm: %d -> %d wells (dropped %d in %d multi-clusters)",
        thin_m,
        meta["n_wells_before"],
        meta["n_wells_after"],
        dropped,
        n_clusters_multi,
    )
    return out, meta


def attach_huc4(df: pd.DataFrame, huc4_parquet: str) -> tuple[pd.DataFrame, dict]:
    """Point-in-polygon HUC4 membership for each well (EPSG:5070 join)."""
    try:
        import geopandas as gpd
    except Exception as exc:  # pragma: no cover - env guard
        log.warning("geopandas unavailable (%s); skipping HUC4 attach", exc)
        df["huc4"] = np.nan
        return df, {"available": False, "reason": str(exc)}
    if not Path(huc4_parquet).exists():
        log.warning("HUC4 parquet %s missing; skipping", huc4_parquet)
        df["huc4"] = np.nan
        return df, {"available": False, "reason": "huc4 parquet missing"}
    polys = gpd.read_parquet(huc4_parquet)[["huc4", "geometry"]]
    wells = (
        df.groupby(WELL_KEY)
        .agg(x5070=("x5070", "first"), y5070=("y5070", "first"))
        .reset_index()
    )
    gw = gpd.GeoDataFrame(
        wells,
        geometry=gpd.points_from_xy(wells.x5070, wells.y5070),
        crs=polys.crs,
    )
    joined = gpd.sjoin(gw, polys, how="left", predicate="within")
    joined = joined.drop_duplicates(WELL_KEY)[WELL_KEY + ["huc4"]]
    df = df.merge(joined, on=WELL_KEY, how="left")
    cov = float(df.huc4.notna().mean())
    log.info("HUC4 attach: %.1f%% of well-months matched a polygon", 100 * cov)
    return df, {"available": True, "coverage_frac": round(cov, 4)}


# --------------------------------------------------------------------------- #
# correlogram
# --------------------------------------------------------------------------- #
def _bin_index(dist_km: np.ndarray) -> np.ndarray:
    return np.digitize(dist_km, DIST_BIN_KM) - 1


def accumulate_correlogram(
    df: pd.DataFrame,
    max_pairs: int,
    cell_well_cap: int,
    seed: int,
    value_col: str = "sgi",
) -> dict:
    """Pair-based binned SGI correlation over contemporaneous (year,month) cells.

    Field is pre-standardized (SGI ~ N(0,1) per well/month by construction), so the
    correlation estimate in a bin is  mean_pairs[(z_i-mu)(z_j-mu)] / var  with global
    mu, var. We accumulate count + sum of centred cross-products per (bin, category)
    WITHOUT storing individual pairs, so memory is O(#bins). Categories tracked:
    Euclidean isotropic, direction sector (N-S vs E-W dominant), same/diff aquifer,
    same/diff state, same/diff HUC4. ``value_col`` allows the same estimator to run on
    a secondary field (e.g. per-well-standardized z_m) as a robustness check.
    """
    rng = np.random.default_rng(seed)
    z_all = df[value_col].to_numpy()
    mu = float(np.mean(z_all))
    var = float(np.mean((z_all - mu) ** 2))
    nb = len(DIST_BIN_KM) - 1

    # accumulators: per bin -> [count, sum_zz]
    acc_iso = np.zeros((nb, 2))
    acc_dir = {"NS": np.zeros((nb, 2)), "EW": np.zeros((nb, 2))}
    acc_aq = {"same": np.zeros((nb, 2)), "diff": np.zeros((nb, 2))}
    acc_state = {"same": np.zeros((nb, 2)), "diff": np.zeros((nb, 2))}
    acc_huc = {"same": np.zeros((nb, 2)), "diff": np.zeros((nb, 2))}

    cells = list(df.groupby(["year", "month"]).indices.items())
    rng.shuffle(cells)
    n_cells_used = 0
    total_pairs = 0
    rmax_m = DIST_BIN_KM[-1] * 1000.0

    for (_, _), idx in cells:
        if total_pairs >= max_pairs:
            break
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        if len(idx) > cell_well_cap:
            idx = rng.choice(idx, size=cell_well_cap, replace=False)
        x = df.x5070.to_numpy()[idx]
        y = df.y5070.to_numpy()[idx]
        z = z_all[idx] - mu
        aq = df.aquifer.to_numpy()[idx]
        stt = df.state.to_numpy()[idx]
        huc = df.huc4.to_numpy()[idx]
        tree = cKDTree(np.column_stack([x, y]))
        pr = tree.query_pairs(r=rmax_m, output_type="ndarray")
        if len(pr) == 0:
            continue
        # cap pairs per cell so the global budget spreads across cells/years
        per_cell_cap = max(2000, max_pairs // max(1, len(cells)))
        if len(pr) > per_cell_cap:
            sel = rng.choice(len(pr), size=per_cell_cap, replace=False)
            pr = pr[sel]
        i, j = pr[:, 0], pr[:, 1]
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dist_km = np.sqrt(dx * dx + dy * dy) / 1000.0
        zz = z[i] * z[j]
        b = _bin_index(dist_km)
        good = (b >= 0) & (b < nb)
        b, zz = b[good], zz[good]
        i, j = i[good], j[good]
        dx, dy = dx[good], dy[good]
        np.add.at(acc_iso, (b, 0), 1.0)
        np.add.at(acc_iso, (b, 1), zz)
        # direction: N-S dominant vs E-W dominant
        ns = np.abs(dy) >= np.abs(dx)
        np.add.at(acc_dir["NS"], (b[ns], 0), 1.0)
        np.add.at(acc_dir["NS"], (b[ns], 1), zz[ns])
        np.add.at(acc_dir["EW"], (b[~ns], 0), 1.0)
        np.add.at(acc_dir["EW"], (b[~ns], 1), zz[~ns])
        # same/diff categoricals (only where both members labelled)
        for acc, vals in ((acc_aq, aq), (acc_state, stt), (acc_huc, huc)):
            vi, vj = vals[i], vals[j]
            lab = (
                pd.notna(vi) & pd.notna(vj)
                if vals.dtype == object
                else ~np.isnan(vi.astype(float)) & ~np.isnan(vj.astype(float))
            )
            same = lab & (vi == vj)
            diff = lab & (vi != vj)
            np.add.at(acc["same"], (b[same], 0), 1.0)
            np.add.at(acc["same"], (b[same], 1), zz[same])
            np.add.at(acc["diff"], (b[diff], 0), 1.0)
            np.add.at(acc["diff"], (b[diff], 1), zz[diff])
        total_pairs += len(b)
        n_cells_used += 1

    def to_corr(acc):
        cnt = acc[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = np.where(cnt > 0, acc[:, 1] / cnt / var, np.nan)
        return cnt, rho

    def pack(acc):
        cnt, rho = to_corr(acc)
        return {"count": cnt.astype(int).tolist(), "rho": _round(rho)}

    cnt_iso, rho_iso = to_corr(acc_iso)
    return {
        "availability": AVAIL,
        "field_mean": round(mu, 5),
        "field_var": round(var, 5),
        "n_cells_used": n_cells_used,
        "total_pairs": int(total_pairs),
        "bin_edges_km": DIST_BIN_KM,
        "bin_mid_km": [
            0.5 * (DIST_BIN_KM[k] + DIST_BIN_KM[k + 1])
            for k in range(len(DIST_BIN_KM) - 1)
        ],
        "isotropic": {"count": cnt_iso.astype(int).tolist(), "rho": _round(rho_iso)},
        "direction": {k: pack(v) for k, v in acc_dir.items()},
        "aquifer": {k: pack(v) for k, v in acc_aq.items()},
        "state": {k: pack(v) for k, v in acc_state.items()},
        "huc4": {k: pack(v) for k, v in acc_huc.items()},
    }


def temporal_lag(
    df: pd.DataFrame, seed: int, max_wells: int, neighbor_km: float
) -> dict:
    """Within-well autocorrelation and between-neighbour cross-corr at monthly lags."""
    rng = np.random.default_rng(seed)
    # within-well autocorrelation
    within = {str(lag): [] for lag in LAG_MONTHS}
    wells = df.groupby(WELL_KEY)
    keys = list(wells.groups.keys())
    if len(keys) > max_wells:
        keys = [keys[i] for i in rng.choice(len(keys), size=max_wells, replace=False)]
    series = {}
    for k in keys:
        g = wells.get_group(k)
        s = pd.Series(g.sgi.to_numpy(), index=g.month_index.to_numpy())
        s = s[~s.index.duplicated()]
        series[k] = s
        for lag in LAG_MONTHS:
            a = s
            b = s.reindex(s.index - lag)
            b.index = s.index
            m = a.notna() & b.notna()
            if m.sum() >= 10:
                aa, bb = a[m].to_numpy(), b[m].to_numpy()
                if aa.std() > 1e-9 and bb.std() > 1e-9:
                    within[str(lag)].append(float(np.corrcoef(aa, bb)[0, 1]))

    # between-neighbour cross-correlation (pairs within neighbor_km)
    wpos = (
        df.groupby(WELL_KEY)
        .agg(x=("x5070", "first"), y=("y5070", "first"))
        .loc[keys]
        .reset_index()
    )
    tree = cKDTree(wpos[["x", "y"]].to_numpy())
    pairs = tree.query_pairs(r=neighbor_km * 1000.0, output_type="ndarray")
    if len(pairs) > 20000:
        pairs = pairs[rng.choice(len(pairs), size=20000, replace=False)]
    between = {str(lag): [] for lag in LAG_MONTHS}
    keyarr = [tuple(r) for r in wpos[WELL_KEY].to_numpy()]
    for a_i, b_i in pairs:
        sa = series.get(keyarr[a_i])
        sb = series.get(keyarr[b_i])
        if sa is None or sb is None:
            continue
        for lag in LAG_MONTHS:
            sb_l = sb.reindex(sb.index + lag)
            sb_l.index = sb.index
            joined = pd.concat([sa, sb_l], axis=1, join="inner").dropna()
            if len(joined) >= 10:
                u, v = joined.iloc[:, 0].to_numpy(), joined.iloc[:, 1].to_numpy()
                if u.std() > 1e-9 and v.std() > 1e-9:
                    between[str(lag)].append(float(np.corrcoef(u, v)[0, 1]))

    def summarize(d):
        return {
            lag: {
                "n": len(v),
                "median": round(float(np.median(v)), 4) if v else None,
                "mean": round(float(np.mean(v)), 4) if v else None,
            }
            for lag, v in d.items()
        }

    return {
        "availability": AVAIL,
        "neighbor_km": neighbor_km,
        "within_well": summarize(within),
        "between_neighbor": summarize(between),
    }


# --------------------------------------------------------------------------- #
# PSD covariance models
# --------------------------------------------------------------------------- #
def _corr_exp(h, nug, rng_):
    return (1.0 - nug) * np.exp(-h / rng_)


def _corr_sph(h, nug, rng_):
    hr = np.clip(h / rng_, 0, 1)
    return (1.0 - nug) * (1.0 - 1.5 * hr + 0.5 * hr**3)


def fit_models(binmid_km, counts, rho, var) -> dict:
    """Fit exponential + spherical PSD correlation models, weighted by pair count."""
    binmid = np.asarray(binmid_km, float)
    counts = np.asarray(counts, float)
    rho = np.asarray(rho, float)
    ok = (counts >= 1) & np.isfinite(rho)
    h, y, w = binmid[ok], rho[ok], np.sqrt(counts[ok])
    out = {}
    for name, fn in (("exponential", _corr_exp), ("spherical", _corr_sph)):

        def resid(p, fn=fn):
            nug, r = p
            return (fn(h, nug, r) - y) * w

        try:
            res = least_squares(
                resid,
                x0=[0.3, 80.0],
                bounds=([0.0, 1.0], [1.0, 2000.0]),
                max_nfev=5000,
            )
            nug, r = float(res.x[0]), float(res.x[1])
            pred = fn(h, nug, r)
            rmse = float(np.sqrt(np.average((pred - y) ** 2, weights=w**2)))
            out[name] = {
                "nugget_frac": round(nug, 4),
                "range_km": round(r, 3),
                "sill_corr": 1.0,
                "sill_sgi_var": round(var, 4),
                "structured_var": round(var * (1.0 - nug), 4),
                "nugget_var": round(var * nug, 4),
                "fit_rmse": round(rmse, 4),
                "converged": bool(res.success),
            }
        except Exception as exc:  # pragma: no cover
            out[name] = {"error": str(exc)}
    return out


# --------------------------------------------------------------------------- #
# kriging skill (headline gate)
# --------------------------------------------------------------------------- #
def _pick_test_months(
    df: pd.DataFrame, n_months: int, min_wells: int, seed: int
) -> list[int]:
    """Spread test months across the well-supported era, all seasons."""
    counts = df.groupby("month_index").size()
    good = counts[counts >= min_wells].index.to_numpy()
    if len(good) == 0:
        return []
    good = np.sort(good)
    # even spread across the supported span
    picks = np.unique(np.linspace(0, len(good) - 1, n_months).round().astype(int))
    return good[picks].tolist()


def kriging_skill(
    df: pd.DataFrame,
    model: dict,
    var: float,
    n_test_months: int,
    min_wells: int,
    max_targets: int,
    n_neighbors: int,
    seed: int,
) -> dict:
    """Leave-well-out simple kriging skill vs climatology + persistence, by support band.

    Simple kriging (known mean 0, SGI space). For each target well in a test month:
    neighbours = other wells that month; solve  C w = k  with the fitted exponential
    covariance (structured off-diagonal, full var on the diagonal = nugget as
    measurement error). Persistence = the well's most recent PRIOR available SGI.
    Errors are stratified by distance to the nearest USED (neighbour) observation.
    """
    rng = np.random.default_rng(seed)
    nug = model["nugget_frac"]
    rng_km = model["range_km"]
    months = _pick_test_months(df, n_test_months, min_wells, seed)
    nbands = len(SUPPORT_BAND_KM) - 1

    # per-band accumulators for each predictor
    preds = ("kriging", "climatology", "persistence")
    band_err = {p: [np.zeros(0) for _ in range(nbands)] for p in preds}
    band_obs = {p: [np.zeros(0) for _ in range(nbands)] for p in preds}
    # persistence-age stratification: is a well's OWN previous SGI (persistence) a
    # tougher baseline than neighbours only when it is FRESH? Track kriging vs
    # persistence abs-error by the age (months) of that persistence value.
    age_bands = [(1, 1), (2, 3), (4, 12), (13, 10_000)]
    age_krig = {b: [] for b in age_bands}
    age_pers = {b: [] for b in age_bands}
    # per-test-month skill for block bootstrap
    per_month = []

    # index for persistence lookups: (well)-> sorted (month_index, sgi)
    df_sorted = df.sort_values(WELL_KEY + ["month_index"])
    well_series = {
        k: (g.month_index.to_numpy(), g.sgi.to_numpy())
        for k, g in df_sorted.groupby(WELL_KEY)
    }

    xy_all = df[["x5070", "y5070"]].to_numpy()
    keyarr = list(map(tuple, df[WELL_KEY].to_numpy()))
    mi_all = df.month_index.to_numpy()
    sgi_all = df.sgi.to_numpy()

    for mi in months:
        sel = np.where(mi_all == mi)[0]
        if len(sel) < min_wells:
            continue
        x = xy_all[sel]
        z = sgi_all[sel]
        keys = [keyarr[s] for s in sel]
        tree = cKDTree(x)
        tgt_idx = np.arange(len(sel))
        if len(tgt_idx) > max_targets:
            tgt_idx = rng.choice(tgt_idx, size=max_targets, replace=False)
        mk = {p: [] for p in preds}  # month-level squared/abs errors
        mo = {p: [] for p in preds}
        for ti in tgt_idx:
            # neighbours: nearest n_neighbors+1 (self included), drop self
            d, nn = tree.query(x[ti], k=min(n_neighbors + 1, len(sel)))
            d = np.atleast_1d(d)
            nn = np.atleast_1d(nn)
            keep = nn != ti
            nn, d = nn[keep], d[keep]
            if len(nn) == 0:
                continue
            zt = z[ti]
            dnn_km = d / 1000.0
            nearest_km = float(dnn_km.min())
            band = min(np.digitize([nearest_km], SUPPORT_BAND_KM)[0] - 1, nbands - 1)
            band = max(band, 0)
            # simple kriging system
            xn = x[nn]
            dij = np.sqrt(((xn[:, None, :] - xn[None, :, :]) ** 2).sum(-1)) / 1000.0
            cmat = var * _corr_exp(dij, nug, rng_km)
            np.fill_diagonal(cmat, var)  # nugget on diagonal
            cmat += np.eye(len(nn)) * 1e-6 * var
            kvec = var * _corr_exp(dnn_km, nug, rng_km)
            try:
                w = np.linalg.solve(cmat, kvec)
            except np.linalg.LinAlgError:
                continue
            pred_k = float(w @ z[nn])
            # persistence: previous available SGI for this well + its age (months)
            mons, sgis = well_series[keys[ti]]
            prior_mask = mons < mi
            if prior_mask.any():
                pred_p = float(sgis[prior_mask][-1])
                pers_age = int(mi - mons[prior_mask][-1])
            else:
                pred_p = np.nan
                pers_age = None
            for p, pv in (
                ("kriging", pred_k),
                ("climatology", 0.0),
                ("persistence", pred_p),
            ):
                if np.isnan(pv):
                    continue
                e = pv - zt
                band_err[p][band] = np.append(band_err[p][band], e)
                band_obs[p][band] = np.append(band_obs[p][band], zt)
                mk[p].append(e)
                mo[p].append(zt)
            if pers_age is not None:
                for lo, hi in age_bands:
                    if lo <= pers_age <= hi:
                        age_krig[(lo, hi)].append(abs(pred_k - zt))
                        age_pers[(lo, hi)].append(abs(pred_p - zt))
                        break
        row = {"month_index": int(mi), "year": int(mi // 12), "month": int(mi % 12 + 1)}
        for p in preds:
            e = np.asarray(mk[p])
            row[f"{p}_mae"] = round(float(np.mean(np.abs(e))), 4) if len(e) else None
            row[f"{p}_n"] = int(len(e))
        per_month.append(row)

    def band_summary():
        out = []
        for bi in range(nbands):
            lo, hi = SUPPORT_BAND_KM[bi], SUPPORT_BAND_KM[bi + 1]
            entry = {
                "band_km": f"{lo:g}-{'inf' if np.isinf(hi) else f'{hi:g}'}",
                "n": int(len(band_err["kriging"][bi])),
            }
            for p in preds:
                e = band_err[p][bi]
                o = band_obs[p][bi]
                if len(e):
                    entry[f"{p}_mae"] = round(float(np.mean(np.abs(e))), 4)
                    entry[f"{p}_rmse"] = round(float(np.sqrt(np.mean(e**2))), 4)
                    pred = o + e
                    if len(e) > 3 and o.std() > 1e-9 and pred.std() > 1e-9:
                        entry[f"{p}_corr"] = round(float(np.corrcoef(pred, o)[0, 1]), 4)
                    else:
                        entry[f"{p}_corr"] = None
                else:
                    entry[f"{p}_mae"] = entry[f"{p}_rmse"] = entry[f"{p}_corr"] = None
            # skill vs baselines (MAE reduction fraction)
            for base in ("climatology", "persistence"):
                bk = entry.get("kriging_mae")
                bb = entry.get(f"{base}_mae")
                entry[f"skill_vs_{base}"] = (
                    round(1.0 - bk / bb, 4) if bk is not None and bb else None
                )
            out.append(entry)
        return out

    def age_summary():
        out = []
        for lo, hi in age_bands:
            ka = np.asarray(age_krig[(lo, hi)])
            pa = np.asarray(age_pers[(lo, hi)])
            entry = {
                "persist_age_months": f"{lo}-{'inf' if hi > 9999 else hi}",
                "n": int(len(ka)),
                "kriging_mae": round(float(ka.mean()), 4) if len(ka) else None,
                "persistence_mae": round(float(pa.mean()), 4) if len(pa) else None,
            }
            entry["skill_vs_persistence"] = (
                round(1.0 - ka.mean() / pa.mean(), 4)
                if len(ka) and pa.mean() > 0
                else None
            )
            out.append(entry)
        return out

    # block bootstrap over test months on overall MAE skill vs each baseline
    boot = _bootstrap_skill(per_month, seed)
    return {
        "availability": AVAIL,
        "model_used": "exponential",
        "n_test_months": len([m for m in per_month if m]),
        "n_neighbors": n_neighbors,
        "support_band_edges_km": [
            b if np.isfinite(b) else None for b in SUPPORT_BAND_KM
        ],
        "by_support_band": band_summary(),
        "by_persistence_age": age_summary(),
        "per_month": per_month,
        "bootstrap_overall": boot,
    }


def _bootstrap_skill(per_month: list[dict], seed: int, n_boot: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    rows = [m for m in per_month if m.get("kriging_mae") is not None]
    if len(rows) < 3:
        return {"note": "too few test months for bootstrap"}
    out = {}
    for base in ("climatology", "persistence"):
        good = [
            m
            for m in rows
            if m.get(f"{base}_mae") not in (None,) and m.get(f"{base}_mae", 0) > 0
        ]
        if len(good) < 3:
            out[f"skill_vs_{base}"] = {"note": "insufficient months"}
            continue
        k = np.array([m["kriging_mae"] for m in good])
        b = np.array([m[f"{base}_mae"] for m in good])
        skills = []
        idx = np.arange(len(good))
        for _ in range(n_boot):
            s = rng.choice(idx, size=len(idx), replace=True)
            skills.append(1.0 - k[s].mean() / b[s].mean())
        skills = np.array(skills)
        out[f"skill_vs_{base}"] = {
            "point": round(1.0 - k.mean() / b.mean(), 4),
            "ci95_low": round(float(np.percentile(skills, 2.5)), 4),
            "ci95_high": round(float(np.percentile(skills, 97.5)), 4),
            "n_months": len(good),
        }
    return out


# --------------------------------------------------------------------------- #
# low-rank (§5.2)
# --------------------------------------------------------------------------- #
def soft_impute_svd(
    mat: np.ndarray, mask: np.ndarray, k: int, n_iter: int = 30
) -> np.ndarray:
    """Rank-k reconstruction by iterative SVD imputation (masked ALS-style)."""
    filled = np.where(mask, mat, 0.0)
    col_mean = np.where(
        mask.any(0), (mat * mask).sum(0) / np.maximum(mask.sum(0), 1), 0.0
    )
    filled = np.where(mask, mat, col_mean[None, :])
    prev = None
    for _ in range(n_iter):
        u, s, vt = np.linalg.svd(filled, full_matrices=False)
        low = (u[:, :k] * s[:k]) @ vt[:k]
        filled = np.where(mask, mat, low)
        if (
            prev is not None
            and np.linalg.norm(low - prev) / (np.linalg.norm(prev) + 1e-9) < 1e-4
        ):
            break
        prev = low
    return low


def low_rank_analysis(
    df: pd.DataFrame,
    min_years: int,
    era_start: int,
    era_end: int,
    seed: int,
) -> dict:
    """Effective-rank diagnostics on the long-record background well x month matrix."""
    rng = np.random.default_rng(seed)
    d = df[(df.year >= era_start) & (df.year <= era_end)].copy()
    yrs = d.groupby(WELL_KEY).year.nunique()
    long_keys = yrs[yrs >= min_years].index
    d = d.set_index(WELL_KEY).loc[long_keys].reset_index()
    if len(d) == 0:
        return {"note": "no long-record wells in era", "min_years": min_years}
    wells = d[WELL_KEY].drop_duplicates().reset_index(drop=True)
    wells["wi"] = np.arange(len(wells))
    d = d.merge(wells, on=WELL_KEY)
    months = np.sort(d.month_index.unique())
    mcol = {m: i for i, m in enumerate(months)}
    d["mi_col"] = d.month_index.map(mcol)
    n_w, n_m = len(wells), len(months)
    mat = np.zeros((n_w, n_m))
    mask = np.zeros((n_w, n_m), dtype=bool)
    mat[d.wi.to_numpy(), d.mi_col.to_numpy()] = d.sgi.to_numpy()
    mask[d.wi.to_numpy(), d.mi_col.to_numpy()] = True
    fill = float(mask.mean())
    log.info("low-rank matrix: %d wells x %d months, fill %.1f%%", n_w, n_m, 100 * fill)

    obs_i, obs_j = np.where(mask)
    n_obs = len(obs_i)

    # leave-well-out: hold out a random subset of WELLS entirely
    holdw = rng.choice(n_w, size=max(1, n_w // 5), replace=False)
    wmask = mask.copy()
    wmask[holdw, :] = False
    # leave-period-out: hold out a random subset of MONTHS entirely
    holdm = rng.choice(n_m, size=max(1, n_m // 5), replace=False)
    pmask = mask.copy()
    pmask[:, holdm] = False
    # leave-entry-out: random held-out observed cells
    perm = rng.permutation(n_obs)
    hold_entry = perm[: n_obs // 5]
    emask = mask.copy()
    emask[obs_i[hold_entry], obs_j[hold_entry]] = False

    def recon_err(train_mask, test_i, test_j):
        curve = {}
        for k in LOWRANK_KS:
            if k >= min(n_w, n_m):
                continue
            low = soft_impute_svd(mat, train_mask, k)
            e = low[test_i, test_j] - mat[test_i, test_j]
            curve[str(k)] = {
                "mae": round(float(np.mean(np.abs(e))), 4),
                "rmse": round(float(np.sqrt(np.mean(e**2))), 4),
                "n": int(len(test_i)),
            }
        return curve

    # entry holdout
    ei, ej = obs_i[hold_entry], obs_j[hold_entry]
    entry_curve = recon_err(emask, ei, ej)
    # leave-well-out: test on observed cells of held-out wells
    wi_test = np.isin(obs_i, holdw)
    well_curve = recon_err(wmask, obs_i[wi_test], obs_j[wi_test])
    # leave-period-out: test on observed cells in held-out months
    mj_test = np.isin(obs_j, holdm)
    period_curve = recon_err(pmask, obs_i[mj_test], obs_j[mj_test])

    # baseline: climatology (predict 0) MAE on the same entry holdout
    base_mae = round(float(np.mean(np.abs(mat[ei, ej]))), 4)

    # mode stability across two random well-halves
    half = rng.permutation(n_w)
    h1, h2 = half[: n_w // 2], half[n_w // 2 :]
    stability = _mode_stability(mat, mask, h1, h2)

    return {
        "availability": AVAIL,
        "era": [era_start, era_end],
        "min_qualifying_years": min_years,
        "n_wells": n_w,
        "n_months": n_m,
        "fill_frac": round(fill, 4),
        "baseline_climatology_mae": base_mae,
        "leave_entry_out": entry_curve,
        "leave_well_out": well_curve,
        "leave_period_out": period_curve,
        "mode_stability_halves": stability,
    }


def _mode_stability(mat, mask, h1, h2, k_top: int = 4, n_iter: int = 30) -> dict:
    """Temporal-mode (V) correlation between two well-halves after sign alignment."""

    def temporal_modes(rows):
        m = mat[rows]
        k = mask[rows]
        low = soft_impute_svd(m, k, k_top, n_iter=n_iter)
        _, _, vt = np.linalg.svd(low, full_matrices=False)
        return vt[:k_top]

    v1 = temporal_modes(h1)
    v2 = temporal_modes(h2)
    out = {}
    for i in range(min(k_top, v1.shape[0], v2.shape[0])):
        c = float(np.corrcoef(v1[i], v2[i])[0, 1])
        out[f"mode_{i + 1}"] = round(abs(c), 4)  # sign-invariant
    return out


# --------------------------------------------------------------------------- #
# pilot selection (§5.3)
# --------------------------------------------------------------------------- #
PILOT_THRESHOLDS = {
    "min_long_record_wells": 30,
    "long_record_min_years": 15,
    "min_wells_per_month_overlap": 10,
    "min_span_km": 40.0,  # spatial dispersion: bbox diagonal to hold out blocks
    "min_dry_episodes": 2,
    "min_wet_episodes": 2,
    "episode_sgi_threshold": 0.8,  # |unit-mean SGI| crossing counts as an episode
    "overlap_era": [1990, 2020],
}


def _count_episodes(trace: pd.Series, thr: float) -> tuple[int, int]:
    """Count distinct excursions of the unit-mean SGI trace below -thr / above +thr."""
    v = trace.to_numpy()
    dry = wet = 0
    in_dry = in_wet = False
    for x in v:
        if np.isnan(x):
            in_dry = in_wet = False
            continue
        if x <= -thr:
            if not in_dry:
                dry += 1
            in_dry = True
        else:
            in_dry = False
        if x >= thr:
            if not in_wet:
                wet += 1
            in_wet = True
        else:
            in_wet = False
    return dry, wet


def score_pilots(df: pd.DataFrame) -> dict:
    """Score state x aquifer-string units against the preregistered criteria."""
    t = PILOT_THRESHOLDS
    era0, era1 = t["overlap_era"]
    d = df[df.monitoring_role.isin(["drought_core", "condition_network"])].copy()
    d["aq"] = d.aquifer.fillna("UNKNOWN")
    d["unit"] = d.state.astype(str) + " | " + d.aq.astype(str)

    # long-record membership within the whole record
    yrs = d.groupby(WELL_KEY).year.nunique()
    long_keys = set(yrs[yrs >= t["long_record_min_years"]].index)
    d["is_long"] = [tuple(r) in long_keys for r in d[WELL_KEY].to_numpy()]

    era = d[(d.year >= era0) & (d.year <= era1)]
    results = []
    for unit, g in d.groupby("unit"):
        if unit.endswith("UNKNOWN"):
            continue
        long_wells = g[g.is_long][WELL_KEY].drop_duplicates()
        n_long = len(long_wells)
        if n_long < 5:
            continue
        ge = era[era.unit == unit]
        # wells available per month in the overlap era (median across months)
        per_month = ge.groupby("month_index")[WELL_KEY[0]].size()
        med_per_month = float(per_month.median()) if len(per_month) else 0.0
        # spatial dispersion: bbox diagonal (km) of long-record wells
        lw = g[g.is_long]
        if len(lw):
            dx = lw.x5070.max() - lw.x5070.min()
            dy = lw.y5070.max() - lw.y5070.min()
            span_km = float(np.sqrt(dx * dx + dy * dy) / 1000.0)
        else:
            span_km = 0.0
        # dry/wet episodes from unit-mean SGI trace (era, months with >=5 wells)
        gm = ge.groupby("month_index").agg(mean_sgi=("sgi", "mean"), n=("sgi", "size"))
        gm = gm[gm.n >= 5].sort_index()
        dry, wet = _count_episodes(gm.mean_sgi, t["episode_sgi_threshold"])
        n_states_note = g.state.nunique()
        crit = {
            "n_long_record_wells": n_long,
            "median_wells_per_month_overlap": round(med_per_month, 1),
            "span_km": round(span_km, 1),
            "n_dry_episodes": dry,
            "n_wet_episodes": wet,
        }
        passes = {
            "long_record": n_long >= t["min_long_record_wells"],
            "per_month": med_per_month >= t["min_wells_per_month_overlap"],
            "dispersion": span_km >= t["min_span_km"],
            "dry_episodes": dry >= t["min_dry_episodes"],
            "wet_episodes": wet >= t["min_wet_episodes"],
        }
        results.append(
            {
                "unit": unit,
                "state": g.state.iloc[0],
                "aquifer": g.aq.iloc[0],
                "n_states_in_unit": int(n_states_note),
                "criteria": crit,
                "passes": passes,
                "n_pass": int(sum(passes.values())),
                "all_pass": all(passes.values()),
            }
        )
    results.sort(
        key=lambda r: (
            r["all_pass"],
            r["n_pass"],
            r["criteria"]["n_long_record_wells"],
        ),
        reverse=True,
    )
    return {"thresholds": t, "units_scored": len(results), "ranked": results}


# --------------------------------------------------------------------------- #
# report + io helpers
# --------------------------------------------------------------------------- #
def _round(a, nd=4):
    return [
        None
        if (x is None or (isinstance(x, float) and np.isnan(x)))
        else round(float(x), nd)
        for x in a
    ]


def write_report(diag: dict, models: dict, pilots: dict, path: Path) -> None:
    def corr_rows(block, mids, counts):
        rows = ""
        for i, m in enumerate(mids):
            r = block["rho"][i]
            c = block["count"][i] if "count" in block else counts[i]
            flag = "" if c >= diag["min_bin_pairs"] else " style='color:#b00'"
            rows += f"<tr><td>{m:g}</td><td{flag}>{'' if r is None else f'{r:.3f}'}</td><td{flag}>{c:,}</td></tr>"
        return rows

    parts = ["<h1>Groundwater anomaly covariance &amp; recoverability</h1>"]
    parts.append(
        f"<p><b>availability:</b> {AVAIL} (interim upper bound; not operational skill)</p>"
    )
    parts.append(
        f"<p><b>gate verdict:</b> {diag.get('gate_verdict', 'see kriging')}</p>"
    )

    for stratum in diag["strata"]:
        s = diag["correlogram"][stratum]
        mids = s["bin_mid_km"]
        parts.append(f"<h2>Stratum: {stratum}</h2>")
        parts.append(
            f"<p>pairs sampled {s['total_pairs']:,} across {s['n_cells_used']:,} "
            f"year-month cells; field var {s['field_var']:.3f}</p>"
        )
        parts.append(
            "<h3>Isotropic SGI correlogram</h3>"
            "<table border=1 cellpadding=3><tr><th>dist km</th><th>rho</th><th>pairs</th></tr>"
            + corr_rows(s["isotropic"], mids, s["isotropic"]["count"])
            + "</table>"
        )
        m = models[stratum]
        parts.append("<h3>Fitted PSD models</h3><table border=1 cellpadding=3>")
        parts.append(
            "<tr><th>model</th><th>range km</th><th>nugget frac</th><th>struct var</th><th>fit rmse</th></tr>"
        )
        for name in ("exponential", "spherical"):
            mm = m.get(name, {})
            parts.append(
                f"<tr><td>{name}</td><td>{mm.get('range_km')}</td><td>{mm.get('nugget_frac')}</td>"
                f"<td>{mm.get('structured_var')}</td><td>{mm.get('fit_rmse')}</td></tr>"
            )
        parts.append("</table>")

        # connectivity comparisons
        for tag, label in (
            ("aquifer", "same vs diff aquifer"),
            ("state", "same vs diff state"),
            ("huc4", "same vs diff HUC4"),
            ("direction", "anisotropy (N-S vs E-W)"),
        ):
            parts.append(
                f"<h3>{label}</h3><table border=1 cellpadding=3><tr><th>dist km</th>"
            )
            keys = list(s[tag].keys())
            for k in keys:
                parts.append(f"<th>rho {k}</th><th>n {k}</th>")
            parts.append("</tr>")
            for i, mid in enumerate(mids):
                parts.append(f"<tr><td>{mid:g}</td>")
                for k in keys:
                    r = s[tag][k]["rho"][i]
                    c = s[tag][k]["count"][i]
                    parts.append(
                        f"<td>{'' if r is None else f'{r:.3f}'}</td><td>{c:,}</td>"
                    )
                parts.append("</tr>")
            parts.append("</table>")

        # kriging
        kr = diag["kriging"][stratum]
        parts.append("<h3>Leave-well-out kriging skill by support band</h3>")
        parts.append(
            "<table border=1 cellpadding=3><tr><th>band km</th><th>n</th>"
            "<th>krig MAE</th><th>clim MAE</th><th>persist MAE</th>"
            "<th>skill vs clim</th><th>skill vs persist</th></tr>"
        )
        for b in kr["by_support_band"]:
            parts.append(
                f"<tr><td>{b['band_km']}</td><td>{b['n']:,}</td><td>{b.get('kriging_mae')}</td>"
                f"<td>{b.get('climatology_mae')}</td><td>{b.get('persistence_mae')}</td>"
                f"<td>{b.get('skill_vs_climatology')}</td><td>{b.get('skill_vs_persistence')}</td></tr>"
            )
        parts.append("</table>")
        parts.append(
            "<h3>Kriging vs persistence by persistence AGE (staleness crossover)</h3>"
        )
        parts.append(
            "<table border=1 cellpadding=3><tr><th>own-data age mo</th><th>n</th>"
            "<th>krig MAE</th><th>persist MAE</th><th>skill vs persist</th></tr>"
        )
        for a in kr.get("by_persistence_age", []):
            parts.append(
                f"<tr><td>{a['persist_age_months']}</td><td>{a['n']:,}</td>"
                f"<td>{a.get('kriging_mae')}</td><td>{a.get('persistence_mae')}</td>"
                f"<td>{a.get('skill_vs_persistence')}</td></tr>"
            )
        parts.append("</table>")
        boot = kr.get("bootstrap_overall", {})
        parts.append(f"<p><b>overall bootstrap:</b> {json.dumps(boot)}</p>")

        # lag
        lag = diag["lag"][stratum]
        parts.append("<h3>Temporal lag correlation</h3><table border=1 cellpadding=3>")
        parts.append(
            "<tr><th>lag mo</th><th>within-well median</th><th>between-neighbour median</th></tr>"
        )
        for lg in [str(x) for x in LAG_MONTHS]:
            w = lag["within_well"][lg]["median"]
            bt = lag["between_neighbor"][lg]["median"]
            parts.append(f"<tr><td>{lg}</td><td>{w}</td><td>{bt}</td></tr>")
        parts.append("</table>")

    # low-rank
    lr = diag.get("lowrank", {})
    if lr and "leave_well_out" in lr:
        parts.append("<h2>Low-rank hypothesis (§5.2)</h2>")
        parts.append(
            f"<p>{lr['n_wells']} long-record wells x {lr['n_months']} months, "
            f"fill {lr['fill_frac'] * 100:.1f}%, climatology baseline MAE {lr['baseline_climatology_mae']}</p>"
        )
        parts.append(
            "<table border=1 cellpadding=3><tr><th>rank k</th><th>leave-entry MAE</th>"
            "<th>leave-well MAE</th><th>leave-period MAE</th></tr>"
        )
        for k in [str(x) for x in LOWRANK_KS]:
            e = lr["leave_entry_out"].get(k, {})
            w = lr["leave_well_out"].get(k, {})
            p = lr["leave_period_out"].get(k, {})
            parts.append(
                f"<tr><td>{k}</td><td>{e.get('mae')}</td><td>{w.get('mae')}</td><td>{p.get('mae')}</td></tr>"
            )
        parts.append("</table>")
        parts.append(
            f"<p><b>mode stability (well-halves):</b> {json.dumps(lr['mode_stability_halves'])}</p>"
        )

    # pilots
    parts.append("<h2>Pilot selection (§5.3)</h2>")
    parts.append(f"<p><b>thresholds:</b> {json.dumps(pilots['thresholds'])}</p>")
    sel = pilots.get("selected", {})
    parts.append(f"<p><i>{sel.get('selection_rationale', '')}</i></p>")
    for role in ("climate_responsive", "adversarial_pumping_or_closed_basin"):
        p = sel.get(role)
        if p:
            parts.append(
                f"<h3>SELECTED {role}: {p['unit']}</h3><p>{json.dumps(p['criteria'])}</p>"
            )
    for role, p in sel.get("alternatives", {}).items():
        if p:
            parts.append(
                f"<h4>alternative -- {role}: {p['unit']}</h4><p>{json.dumps(p['criteria'])}</p>"
            )
    parts.append("<h3>Top ranked units</h3><table border=1 cellpadding=3>")
    parts.append(
        "<tr><th>unit</th><th>long wells</th><th>med/mo</th><th>span km</th><th>dry</th><th>wet</th><th>#pass</th></tr>"
    )
    for r in pilots["ranked"][:20]:
        c = r["criteria"]
        parts.append(
            f"<tr><td>{r['unit']}</td><td>{c['n_long_record_wells']}</td>"
            f"<td>{c['median_wells_per_month_overlap']}</td><td>{c['span_km']}</td>"
            f"<td>{c['n_dry_episodes']}</td><td>{c['n_wet_episodes']}</td><td>{r['n_pass']}</td></tr>"
        )
    parts.append("</table>")

    html = (
        "<html><head><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1100px;margin:2em auto}table{border-collapse:collapse;margin:.5em 0}th{background:#eee}</style></head><body>"
        + "".join(parts)
        + "</body></html>"
    )
    path.write_text(html)


def _select_two_pilots(pilots: dict) -> dict:
    """Pick one climate-responsive + one adversarial (pumping/closed-basin) pilot."""
    ranked = pilots["ranked"]
    passing = [r for r in ranked if r["all_pass"]]
    pool = passing if passing else [r for r in ranked if r["n_pass"] >= 4]

    # closed-basin / pumping-heavy signature: Great Basin (NV/UT/AZ) valley-fill,
    # High Plains Ogallala, or Central Valley / Mojave. Climate-responsive: glacial
    # drift (ND/upper-midwest/glaciated NE).
    adversarial_states = {"NV", "UT", "AZ", "CA", "TX", "KS", "NM", "OK"}
    glacial_states = {"ND", "SD", "MN", "WI", "MI", "NY", "OH", "IA"}

    def is_adversarial(r):
        aq = str(r["aquifer"]).lower()
        return r["state"] in adversarial_states or any(
            s in aq for s in ("valley", "ogallala", "alluv", "basin", "mojave", "playa")
        )

    def is_climate(r):
        aq = str(r["aquifer"]).lower()
        return r["state"] in glacial_states or any(
            s in aq for s in ("glac", "drift", "till", "delta", "outwash", "lake")
        )

    climate = next((r for r in pool if is_climate(r)), None)
    adversarial = next(
        (r for r in pool if is_adversarial(r) and r is not climate), None
    )
    # fallbacks: top passing units
    if climate is None:
        climate = pool[0] if pool else None
    if adversarial is None:
        adversarial = next((r for r in pool if r is not climate), None)

    # documented regime alternatives (not the criteria-driven pick, but named in the
    # plan): the best-cadence climate-responsive unit is the ND glacial-drift network
    # (nd_dwr ~monthly); the archetypal CLOSED-BASIN adversarial is a Great Basin
    # (NV/UT) valley-fill unit. Cadence is DEFERRED to Track B (interim plan A1), so it
    # cannot drive the A3 selection -- these are surfaced for the A4 DA-prototype choice.
    nd_glacial = next(
        (r for r in ranked if r["state"] == "ND" and r["n_pass"] >= 4), None
    )
    closed_basin = next(
        (
            r
            for r in ranked
            if r["state"] in {"NV", "UT"}
            and "vlfl" in str(r["aquifer"]).lower()
            and r["n_pass"] >= 4
        ),
        None,
    )
    rationale = (
        "Criteria-driven selection on the preregistered §5.3 thresholds (record support, "
        "overlap density, spatial dispersion, dry/wet episodes). Climate-responsive pick is "
        "a humid-Northeast glacial-drift aquifer (minimal pumping, dense, many episodes); "
        "adversarial pick is the High Plains Ogallala (chronic pumping depletion). "
        "Cadence is intentionally NOT a selection criterion here (deferred to Track B); the "
        "documented alternatives name the monthly-cadence ND glacial-drift network and a "
        "closed-basin Great Basin valley-fill unit for the A4 DA prototype."
    )
    return {
        "climate_responsive": climate,
        "adversarial_pumping_or_closed_basin": adversarial,
        "alternatives": {
            "climate_responsive_best_cadence_nd_glacial": nd_glacial,
            "adversarial_closed_basin_great_basin": closed_basin,
        },
        "selection_rationale": rationale,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_stratum(df: pd.DataFrame, args, tag: str) -> tuple[dict, dict, dict, dict]:
    log.info("[%s] correlogram ...", tag)
    corr = accumulate_correlogram(df, args.max_pairs, args.cell_well_cap, args.seed)
    var = corr["field_var"]
    log.info("[%s] fit models ...", tag)
    models = fit_models(
        corr["bin_mid_km"], corr["isotropic"]["count"], corr["isotropic"]["rho"], var
    )
    log.info("[%s] kriging skill ...", tag)
    kr = kriging_skill(
        df,
        models["exponential"],
        var,
        args.test_months,
        args.krig_min_wells,
        args.max_targets,
        args.n_neighbors,
        args.seed,
    )
    log.info("[%s] temporal lag ...", tag)
    lag = temporal_lag(df, args.seed, args.lag_max_wells, args.lag_neighbor_km)
    return corr, models, kr, lag


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--status-dir", default=STATUS_DIR)
    ap.add_argument("--audit-dir", default=AUDIT_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--huc4-parquet", default=HUC4_PARQUET)
    ap.add_argument("--max-pairs", type=int, default=8_000_000)
    ap.add_argument("--cell-well-cap", type=int, default=800)
    ap.add_argument("--min-bin-pairs", type=int, default=200)
    ap.add_argument("--thin-m", type=float, default=10.0)
    ap.add_argument("--test-months", type=int, default=36)
    ap.add_argument("--krig-min-wells", type=int, default=300)
    ap.add_argument("--max-targets", type=int, default=600)
    ap.add_argument("--n-neighbors", type=int, default=40)
    ap.add_argument("--lag-max-wells", type=int, default=4000)
    ap.add_argument("--lag-neighbor-km", type=float, default=25.0)
    ap.add_argument("--lowrank-min-years", type=int, default=15)
    ap.add_argument("--lowrank-era", type=int, nargs=2, default=[1985, 2020])
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--workers", type=int, default=8)  # reserved; fits are vectorized
    ap.add_argument(
        "--stages",
        nargs="+",
        default=["correlogram", "models", "kriging", "lag", "lowrank", "pilots"],
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("loading population ...")
    pop = load_population(args.status_dir, args.audit_dir)
    pop, thin_meta = dedupe_colocated(pop, args.thin_m)
    pop, huc_meta = attach_huc4(pop, args.huc4_parquet)

    role_counts = pop.monitoring_role.value_counts().to_dict()
    aq_cov = float(pop.aquifer.notna().mean())
    log.info("population role counts: %s; aquifer coverage %.3f", role_counts, aq_cov)

    strata = {
        "all_background": pop[
            pop.monitoring_role.isin(["drought_core", "condition_network"])
        ],
        "drought_core": pop[pop.monitoring_role == "drought_core"],
    }
    contrast = {"diagnostic": pop[pop.monitoring_role == "diagnostic"]}

    diag = {
        "availability": AVAIL,
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "min_bin_pairs": args.min_bin_pairs,
        "population": {
            "n_rows_total": int(len(pop)),
            "role_counts": {k: int(v) for k, v in role_counts.items()},
            "aquifer_coverage_frac": round(aq_cov, 4),
            "colocated_thinning": thin_meta,
            "huc4_attach": huc_meta,
        },
        "strata": list(strata.keys()),
        "correlogram": {},
        "kriging": {},
        "lag": {},
        "seed": args.seed,
        "max_pairs": args.max_pairs,
    }
    models_all = {}

    for tag, sdf in strata.items():
        corr, models, kr, lag = run_stratum(sdf.reset_index(drop=True), args, tag)
        diag["correlogram"][tag] = corr
        diag["kriging"][tag] = kr
        diag["lag"][tag] = lag
        models_all[tag] = models

    # secondary z_m robustness check: run the isotropic correlogram on the metric
    # anomaly z_m standardized per-well (so amplitude-heterogeneous aquifers contribute
    # comparably; raw z_m in metres would be dominated by high-amplitude wells). If the
    # decay/range track the SGI result, the spatial structure is not an artifact of the
    # SGI normal-score (rank) transform.
    log.info("secondary z_m correlogram check ...")
    bg = strata["all_background"].reset_index(drop=True).copy()
    wstd = bg.groupby(WELL_KEY).z_m.transform("std")
    bg = bg[(wstd > 1e-6) & bg.z_m.notna()].copy()
    bg["z_std"] = bg.z_m / bg.groupby(WELL_KEY).z_m.transform("std")
    zcorr = accumulate_correlogram(
        bg, args.max_pairs, args.cell_well_cap, args.seed, value_col="z_std"
    )
    zmodels = fit_models(
        zcorr["bin_mid_km"],
        zcorr["isotropic"]["count"],
        zcorr["isotropic"]["rho"],
        zcorr["field_var"],
    )
    diag["secondary_zm_check"] = {
        "note": "metric anomaly z_m standardized per-well; SGI is the primary field",
        "n_rows": int(len(bg)),
        "isotropic": zcorr["isotropic"],
        "bin_mid_km": zcorr["bin_mid_km"],
        "exponential_model": zmodels.get("exponential", {}),
    }

    # diagnostic contrast: correlogram + lag only (labelled), no headline claim
    log.info("[diagnostic-contrast] correlogram + lag ...")
    dc = contrast["diagnostic"].reset_index(drop=True)
    if len(dc) > 5000:
        diag["contrast_diagnostic"] = {
            "correlogram": accumulate_correlogram(
                dc, args.max_pairs // 4, args.cell_well_cap, args.seed
            ),
            "lag": temporal_lag(
                dc, args.seed, args.lag_max_wells, args.lag_neighbor_km
            ),
            "note": "confined/pumping-dominated CONTRAST ONLY; never enters the headline gate",
        }

    # low-rank on all-background
    if "lowrank" in args.stages:
        log.info("low-rank analysis ...")
        diag["lowrank"] = low_rank_analysis(
            strata["all_background"].reset_index(drop=True),
            args.lowrank_min_years,
            args.lowrank_era[0],
            args.lowrank_era[1],
            args.seed,
        )

    # pilots
    log.info("pilot selection ...")
    pilots = score_pilots(pop)
    pilots["selected"] = _select_two_pilots(pilots)

    # gate verdict from all-background kriging bootstrap + band skill
    kb = diag["kriging"]["all_background"]
    boot = kb.get("bootstrap_overall", {})
    bands = kb["by_support_band"]
    min_n = args.min_bin_pairs
    # spatial support at which kriging beats CLIMATOLOGY (adequate band n)
    clim_support = [
        b["band_km"]
        for b in bands
        if b["n"] >= min_n and (b.get("skill_vs_climatology") or -1) > 0
    ]
    # spatial support at which kriging beats FRESH persistence too
    both_support = [
        b["band_km"]
        for b in bands
        if b["n"] >= min_n
        and (b.get("skill_vs_climatology") or -1) > 0
        and (b.get("skill_vs_persistence") or -1) > 0
    ]
    clim_ci = boot.get("skill_vs_climatology", {})
    pers_ci = boot.get("skill_vs_persistence", {})
    beats_clim = clim_ci.get("ci95_low", -1) > 0
    # staleness crossover: youngest persistence-age band where neighbours win
    age_cross = next(
        (
            a["persist_age_months"]
            for a in kb.get("by_persistence_age", [])
            if a["n"] >= min_n and (a.get("skill_vs_persistence") or -1) > 0
        ),
        None,
    )
    if beats_clim and both_support:
        verdict = f"POSITIVE (beats climatology AND fresh persistence) to bands {both_support}"
    elif beats_clim:
        verdict = (
            "CONDITIONAL POSITIVE: contemporaneous neighbours give bootstrap-significant "
            f"recoverability BEYOND CLIMATOLOGY to bands {clim_support} "
            f"(overall vs-clim CI95 [{clim_ci.get('ci95_low')},{clim_ci.get('ci95_high')}]), "
            "but do NOT beat a FRESH (<=1mo) within-well persistence value "
            f"(overall vs-persist CI95 [{pers_ci.get('ci95_low')},{pers_ci.get('ci95_high')}]); "
            f"neighbours overtake persistence once own-data is >= {age_cross} months stale. "
            "IMPLICATION: rung-3 must be a state-space model that PROPAGATES persistence AND "
            "assimilates neighbours (spatial-only OI is insufficient); spatial fill earns its "
            "place at stale/ungauged wells, not as a replacement for a fresh own-reading."
        )
    else:
        verdict = (
            f"FAILS (no bootstrap-significant skill beyond climatology; beats_clim={beats_clim}) "
            "-> ship station-status only"
        )
    diag["gate_verdict"] = verdict
    diag["gate_supported_bands_vs_climatology"] = clim_support
    diag["gate_supported_bands_vs_fresh_persistence"] = both_support
    diag["gate_persistence_staleness_crossover_months"] = age_cross

    # write
    (out / "covariance_models.json").write_text(json.dumps(models_all, indent=2))
    (out / "covariance_diagnostics.json").write_text(
        json.dumps(diag, indent=2, default=str)
    )
    (out / "pilot_selection.json").write_text(json.dumps(pilots, indent=2, default=str))
    write_report(diag, models_all, pilots, out / "covariance_report.html")

    dt = time.time() - t0
    log.info("DONE in %.1f min -> %s", dt / 60.0, out)
    log.info("GATE: %s", diag["gate_verdict"])
    sel = pilots["selected"]
    for role, p in sel.items():
        if p:
            log.info("pilot[%s] = %s | %s", role, p["unit"], p["criteria"])


if __name__ == "__main__":
    main()
