"""Station groundwater-status product (ops-assimilation rung 1 / interim A2).

Builds the defensible observational groundwater-status product from the GWX
per-well hydrographs, per ``notes/ops_assimilation_plan.md`` (§2.1 station-anomaly
math, §3.3 QC, §4 rung-1 deliverables) and ``notes/ops_assimilation_interim_plan.md``
(section "A2" + "Snapshot S0"). Two phases, selectable with ``--phase``:

  consolidate  read each eligible hydrograph -> one harmonized month-year table
               (``groundwater_monthly_heads.parquet``): per (well, year, month) the
               robust monthly head statistic (median), obs support, and QC flags.
               This consolidated parquet is the DURABLE interim input for every
               later rung and insulates downstream work from snapshot churn.
  status       from the monthly table compute, per (well, year, calendar-month):
               leave-year-out (LOYO) climatology, metric anomaly z, bounded-plotting
               -position quantile q, head percentile, SGI, and GW-D* class, with
               NGWMN period-of-record eligibility (>10 distinct qualifying years),
               record support, and staleness. Emits
               ``groundwater_status_monthly.parquet`` and
               ``groundwater_station_climatology.parquet``.
  all          (default) both phases in a single pass over the well files -- LOYO
               scoring is entirely within-well, so consolidate+status fuse cleanly.

Harmonization (binding, from the plan):
  * Head series is h = -dtw (a within-well RELATIVE head; ``dtw`` is the most
    complete column and is in METERS -- verified: each series' mean(dtw) reproduces
    the index ``mean_dtw`` exactly). Low head = low quantile is then natural
    (deep water table -> low percentile -> GW-D4).
  * We never mix columns within a well. Where ``wl_elevation_navd88``/``_ngvd29``
    exists we cross-check corr(-dtw, elevation) and FLAG disagreement (< 0.99); we
    still score on -dtw.
  * NO winsorization anywhere. Plausible extremes are preserved; problems are
    flagged, not clipped.

QC decisions (thresholds justified from the S0 data; see ``status_build_report.json``):
  * is_static is UNPOPULATED in S0 (0.02% True across 283k measurements): it is
    NOT a usable pumping indicator, so we do NOT exclude non-static levels
    (that would drop ~99.98% of data). We COUNT static/non-static and detect
    pumping from ``source_flags`` instead (qualifier "Pumping"). See --exclude-*.
  * Pumping levels (source_flags contains "pumping") are EXCLUDED from the monthly
    median (they are not the static water table) but COUNTED (n_pumping).
  * Source-rejected measurements (approval_status contains "reject") are EXCLUDED
    but COUNTED (n_rejected). This is a source quality reject, not a tail clip.
  * Exact-duplicate timestamps are collapsed to their per-timestamp median first
    (counted), then the monthly median is taken over distinct timestamps.
  * Impossible jumps: consecutive |Δh| > --jump-abs-m within --jump-window-days are
    flagged and counted (n_jump_flag); NOT dropped -- the monthly MEDIAN is already
    spike-robust, and dropping would risk clipping the drought tail.
  * Dry / flowing / negative-dtw (artesian) values are counted and KEPT (real signal).

Leave-year-out (no leakage): for each (well, calendar-month m), the value for year
y is scored ONLY against the OTHER years' monthly values for that same (well, m).
C_i(m) = median(other years); z = h(y,m) - C_i(m). The quantile is a bounded
plotting position q = (b + 0.5*e + 0.5)/(n+1) with b/e = # LOYO values below/equal
and n = LOYO sample size; q is strictly in (0, 1) so SGI = Phi^-1(q) is finite.
Every scored year excludes itself from both the climatology and the empirical
distribution. Scoring is per-well only -- no cross-well information is used.

Every output row carries ``availability='oracle'`` and ``climatology='por_loyo'``.

Usage (smoke test against the live current tree while the S0 freeze runs):
    uv run python utils/build_groundwater_status.py --phase all \
        --limit 2000 --workers 8 \
        --snapshot-root /data/ssd2/gwx/products/current \
        --out-dir /data/ssd2/handily/conus/ops_status/s0/status
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

log = logging.getLogger("build_groundwater_status")

CURRENT_PREFIX = "/data/ssd2/gwx/products/current"
S0_SNAPSHOT = "/data/ssd2/gwx/products/snapshots/s0_20260707"
GWX_INDEX = f"{S0_SNAPSHOT}/wells.geoparquet"
# Unique per-well key. canonical_id is NOT usable as a grouping key: 56.7% of
# index rows sit in multi-member canonical groups with collision buckets up to
# ~1449 members, so grouping on it would pool distinct wells into one series.
WELL_KEY = ["source", "source_well_id"]
WT_CLASSES = ("unconfined", "unconfined_marginal")
HEAD_SOURCE = "neg_dtw"  # h = -dtw, the harmonized within-well relative head

# Index columns needed for candidate selection + descriptor passthrough.
INDEX_COLS = [
    "source",
    "source_well_id",
    "canonical_id",
    "state",
    "longitude",
    "latitude",
    "confinement_class",
    "well_use",
    "well_class",
    "well_depth",
    "screen_top",
    "screen_bottom",
    "obs_count",
    "por_start",
    "por_end",
    "mean_dtw",
    "has_time_series",
    "file_path",
]
# Descriptor columns carried from the index into every product so the monthly
# parquet is fully self-describing (well-constant -> dictionary-compresses to ~0).
DESCRIPTOR_COLS = [
    "canonical_id",
    "source",
    "source_well_id",
    "state",
    "longitude",
    "latitude",
    "confinement_class",
    "well_use",
    "well_class",
    "well_depth",
    "screen_top",
    "screen_bottom",
    "obs_count",
    "por_start",
    "por_end",
    "mean_dtw",
]
# Hydrograph columns (all four S0 sources carry these; missing ones are backfilled).
SERIES_COLS = [
    "dtime",
    "dtw",
    "wl_elevation_navd88",
    "wl_elevation_ngvd29",
    "is_static",
    "method",
    "approval_status",
    "source_flags",
]

# GW-D* evidence classes from head percentile (governing plan §0 table).
GW_BOUNDS = [
    (30.0, "None"),
    (20.0, "GW-D0"),
    (10.0, "GW-D1"),
    (5.0, "GW-D2"),
    (2.0, "GW-D3"),
    (-1.0, "GW-D4"),
]


def classify_gw(pct: float) -> str | None:
    """Map head percentile (0-100) to a GW-D* evidence class; None if not scored."""
    if not np.isfinite(pct):
        return None
    for lo, name in GW_BOUNDS:
        if pct > lo:
            return name
    return "GW-D4"


# --------------------------------------------------------------------------- #
# Phase 1 -- consolidate one well's hydrograph into a month-year table.
# --------------------------------------------------------------------------- #
def consolidate_well(series: pd.DataFrame, rec: dict, params: dict):
    """Return (monthly_df, provenance dict, per-well QC Counter) or None if empty.

    ``monthly_df`` has one row per (year, month) with the median head and QC counts.
    """
    counts: Counter = Counter()
    counts["obs_raw"] = len(series)

    d = series.copy()
    for c in SERIES_COLS:
        if c not in d.columns:
            d[c] = np.nan
    d = d[SERIES_COLS]
    d["dtime"] = pd.to_datetime(d["dtime"], utc=True, errors="coerce")

    valid = d["dtw"].notna() & d["dtime"].notna()
    counts["obs_missing_dtw"] = int((~valid).sum())
    d = d.loc[valid].copy()
    if d.empty:
        counts["wells_no_valid_obs"] = 1
        return None

    dtwv = d["dtw"].to_numpy(dtype="float64")
    head = -dtwv  # harmonized within-well relative head, meters

    # Column-provenance + datum cross-check (uses -dtw vs elevation where present).
    prov: dict = {}
    prov["has_navd88"] = bool(d["wl_elevation_navd88"].notna().any())
    prov["has_ngvd29"] = bool(d["wl_elevation_ngvd29"].notna().any())
    elev_corr = np.nan
    for c in ("wl_elevation_navd88", "wl_elevation_ngvd29"):
        v = d[c].to_numpy(dtype="float64")
        m = np.isfinite(v) & np.isfinite(head)
        if m.sum() >= 5 and np.std(v[m]) > 1e-6 and np.std(head[m]) > 1e-6:
            elev_corr = float(np.corrcoef(head[m], v[m])[0, 1])
            break
    prov["elev_corr"] = elev_corr
    prov["elev_disagree"] = bool(
        np.isfinite(elev_corr) and elev_corr < params["elev_corr_min"]
    )
    prov["n_methods_well"] = int(d["method"].dropna().nunique())
    prov["method_change"] = bool(prov["n_methods_well"] > 1)
    prov["head_source"] = HEAD_SOURCE

    # Measurement flags via cheap vectorized substring tests (no JSON parsing).
    sfl = d["source_flags"].astype("string").str.lower().fillna("")
    appr = d["approval_status"].astype("string").str.lower().fillna("")
    pumping = sfl.str.contains("pumping", regex=False).to_numpy()
    rejected = appr.str.contains("reject", regex=False).to_numpy()
    dry = sfl.str.contains("dry", regex=False).to_numpy()
    flowing = (
        (dtwv < 0)
        | sfl.str.contains("artesian", regex=False).to_numpy()
        | sfl.str.contains("flowing", regex=False).to_numpy()
    )
    static_true = (d["is_static"] == True).to_numpy()  # noqa: E712

    counts["flag_pumping"] = int(pumping.sum())
    counts["flag_rejected"] = int(rejected.sum())
    counts["flag_dry"] = int(dry.sum())
    counts["flag_flowing"] = int(flowing.sum())
    counts["flag_negative_dtw"] = int((dtwv < 0).sum())
    counts["is_static_true"] = int(static_true.sum())
    counts["is_static_false"] = int((~static_true).sum())

    d = d.assign(
        _head=head,
        _pump=pumping,
        _rej=rejected,
        _dry=dry,
        _flow=flowing,
        _static=static_true,
        _year=d["dtime"].dt.year.to_numpy(),
        _month=d["dtime"].dt.month.to_numpy(),
    )

    # Per-month raw + excluded tallies (before exclusions drop rows).
    raw_by_m = (
        d.groupby(["_year", "_month"])
        .agg(
            n_obs_raw=("_head", "size"),
            n_pumping=("_pump", "sum"),
            n_rejected=("_rej", "sum"),
        )
        .reset_index()
    )

    excl = np.zeros(len(d), dtype=bool)
    if params["exclude_pumping"]:
        excl |= pumping
    if params["exclude_rejected"]:
        excl |= rejected
    counts["obs_excluded"] = int(excl.sum())
    kept = d.loc[~excl].sort_values("dtime")
    if kept.empty:
        counts["wells_empty_after_excl"] = 1
        return None

    # Collapse exact-duplicate timestamps to their per-timestamp median.
    n_dup = int(kept["dtime"].duplicated(keep="first").sum())
    counts["obs_dup_collapsed"] = n_dup
    prov["n_dup_collapsed"] = n_dup
    if n_dup:
        kept = (
            kept.groupby("dtime", as_index=False)
            .agg(
                _head=("_head", "median"),
                _static=("_static", "sum"),
                _dry=("_dry", "max"),
                _flow=("_flow", "max"),
                _year=("_year", "first"),
                _month=("_month", "first"),
            )
            .sort_values("dtime")
        )

    # Impossible-jump flag vs the previous distinct measurement.
    t_days = kept["dtime"].values.astype("datetime64[s]").astype("int64") / 86400.0
    hh = kept["_head"].to_numpy(dtype="float64")
    jump = np.zeros(len(kept), dtype=bool)
    if len(kept) > 1:
        dh = np.abs(np.diff(hh))
        dd = np.diff(t_days)
        jump[1:] = (dh > params["jump_abs_m"]) & (dd <= params["jump_window_days"])
    kept = kept.assign(_jump=jump)
    counts["obs_jump_flag"] = int(jump.sum())
    counts["obs_kept"] = len(kept)

    monthly = (
        kept.groupby(["_year", "_month"])
        .agg(
            head_median_m=("_head", "median"),
            n_obs=("_head", "size"),
            n_static=("_static", "sum"),
            n_dry=("_dry", "sum"),
            n_flowing=("_flow", "sum"),
            n_jump_flag=("_jump", "sum"),
        )
        .reset_index()
        .rename(columns={"_year": "year", "_month": "month"})
    )
    monthly = monthly.merge(
        raw_by_m.rename(columns={"_year": "year", "_month": "month"}),
        on=["year", "month"],
        how="left",
    )
    monthly["dtw_median_m"] = -monthly["head_median_m"]
    counts["monthly_rows"] = len(monthly)

    last_obs = kept["dtime"].max()
    prov["last_obs_date"] = last_obs
    staleness = np.nan
    if pd.notna(last_obs):
        staleness = float((params["asof"] - last_obs).total_seconds() / 86400.0)
    prov["staleness_days"] = staleness
    return monthly, prov, counts


def _attach_descriptors(monthly: pd.DataFrame, rec: dict, prov: dict) -> pd.DataFrame:
    """Add well-constant id/descriptor/provenance columns to a monthly frame."""
    for c in DESCRIPTOR_COLS:
        monthly[c] = rec.get(c)
    monthly["head_source"] = prov["head_source"]
    monthly["has_navd88"] = prov["has_navd88"]
    monthly["has_ngvd29"] = prov["has_ngvd29"]
    monthly["elev_corr"] = prov["elev_corr"]
    monthly["elev_disagree"] = prov["elev_disagree"]
    monthly["n_methods_well"] = prov["n_methods_well"]
    monthly["method_change"] = prov["method_change"]
    monthly["last_obs_date"] = prov["last_obs_date"]
    monthly["staleness_days"] = prov["staleness_days"]
    monthly["availability"] = "oracle"
    monthly["climatology"] = "por_loyo"
    return monthly


# --------------------------------------------------------------------------- #
# Phase 2 -- per-well LOYO climatology + status scoring (leakage-free).
# --------------------------------------------------------------------------- #
def score_well(monthly: pd.DataFrame, params: dict):
    """Return (climatology rows list, status rows list) for one well's monthly table.

    ``monthly`` has one row per (year, month) plus well-constant passthrough columns.
    """
    r0 = monthly.iloc[0]
    passthrough = {c: r0.get(c) for c in DESCRIPTOR_COLS}
    for c in (
        "has_navd88",
        "has_ngvd29",
        "elev_corr",
        "elev_disagree",
        "method_change",
        "staleness_days",
        "last_obs_date",
    ):
        passthrough[c] = r0.get(c)

    min_loyo = params["min_loyo"]
    clim_rows: list[dict] = []
    status_rows: list[dict] = []

    for m, g in monthly.groupby("month"):
        years = g["year"].to_numpy()
        heads = g["head_median_m"].to_numpy(dtype="float64")
        n_years = int(np.unique(years).size)
        qualifying = n_years > 10  # NGWMN period-of-record convention

        clim_rows.append(
            {
                "month": int(m),
                "n_years_for_month": n_years,
                "ngwmn_qualifying": qualifying,
                "c_median_m": float(np.median(heads)),
                "c_mean_m": float(np.mean(heads)),
                "c_std_m": float(np.std(heads, ddof=1)) if heads.size > 1 else np.nan,
                "c_iqr_m": float(np.subtract(*np.percentile(heads, [75, 25]))),
                "c_min_m": float(np.min(heads)),
                "c_max_m": float(np.max(heads)),
                "availability": "oracle",
                "climatology": "por_loyo",
                **passthrough,
            }
        )

        n_obs_month = g["n_obs"].to_numpy()
        for i in range(len(g)):
            y = int(years[i])
            h = float(heads[i])
            other = heads[years != y]  # LEAVE-YEAR-OUT sample
            n = int(other.size)
            if n >= 1:
                c_loyo = float(np.median(other))
                z = h - c_loyo
            else:
                c_loyo = np.nan
                z = np.nan
            if n >= min_loyo:
                b = float((other < h).sum())
                e = float((other == h).sum())
                q = (b + 0.5 * e + 0.5) / (n + 1.0)  # bounded in (0, 1)
                pct = 100.0 * q
                sgi = float(norm.ppf(q))
                gw = classify_gw(pct)
                flag = "ok"
            else:
                q = pct = sgi = np.nan
                gw = None
                flag = "insufficient_loyo" if n >= 1 else "no_loyo"
            status_rows.append(
                {
                    "year": y,
                    "month": int(m),
                    "head_median_m": h,
                    "n_obs": int(n_obs_month[i]),
                    "n_years_for_month": n_years,
                    "n_loyo": n,
                    "c_loyo_median_m": c_loyo,
                    "z_m": z,
                    "q": q,
                    "head_percentile": pct,
                    "sgi": sgi,
                    "gw_class": gw,
                    "ngwmn_qualifying": qualifying,
                    "status_flag": flag,
                    "availability": "oracle",
                    "climatology": "por_loyo",
                    **passthrough,
                }
            )
    return clim_rows, status_rows


# --------------------------------------------------------------------------- #
# Workers.
# --------------------------------------------------------------------------- #
def _read_series(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return None
    except Exception as exc:  # count + report; never silently swallow
        raise RuntimeError(f"read_error:{path}:{exc}") from exc


def _worker(chunk, params):
    monthly_parts: list[pd.DataFrame] = []
    clim_rows: list[dict] = []
    status_rows: list[dict] = []
    counts: Counter = Counter()
    do_status = params["phase"] in ("all", "status")
    do_consolidate = params["phase"] in ("all", "consolidate")

    for rec in chunk:
        counts["wells_seen"] += 1
        try:
            series = _read_series(rec["path"])
        except RuntimeError as exc:
            counts["wells_read_error"] += 1
            counts[str(exc).split(":")[0]] += 0  # keep reason visible
            log.warning("%s", exc)
            continue
        if series is None:
            counts["wells_missing_file"] += 1
            continue
        res = consolidate_well(series, rec, params)
        if res is None:
            counts["wells_no_monthly"] += 1
            continue
        monthly, prov, wc = res
        counts.update(wc)
        monthly = _attach_descriptors(monthly, rec, prov)
        if do_consolidate:
            monthly_parts.append(monthly)
        if do_status:
            cr, sr = score_well(monthly, params)
            clim_rows.extend(cr)
            status_rows.extend(sr)
        counts["wells_ok"] += 1

    monthly_df = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else None
    clim_df = pd.DataFrame(clim_rows) if clim_rows else None
    status_df = pd.DataFrame(status_rows) if status_rows else None
    return monthly_df, clim_df, status_df, dict(counts)


def _status_worker(groups, params):
    """Phase-2-only worker: score a list of per-well monthly frames from parquet."""
    clim_rows: list[dict] = []
    status_rows: list[dict] = []
    counts: Counter = Counter()
    for monthly in groups:
        counts["wells_seen"] += 1
        cr, sr = score_well(monthly, params)
        clim_rows.extend(cr)
        status_rows.extend(sr)
        counts["wells_ok"] += 1
    clim_df = pd.DataFrame(clim_rows) if clim_rows else None
    status_df = pd.DataFrame(status_rows) if status_rows else None
    return clim_df, status_df, dict(counts)


# --------------------------------------------------------------------------- #
# Candidate selection.
# --------------------------------------------------------------------------- #
def select_candidates(args) -> tuple[pd.DataFrame, tuple[int, int, int]]:
    df = pd.read_parquet(args.gwx_index, columns=INDEX_COLS)
    n_total = len(df)
    conf = tuple(c for c in args.confinement.split(",") if c)
    cand = df[df["has_time_series"] & df["confinement_class"].isin(conf)].copy()
    n_cand = len(cand)
    span_years = (cand["por_end"] - cand["por_start"]).dt.total_seconds() / (
        365.25 * 86400.0
    )
    cand = cand[
        (cand["obs_count"] >= args.min_obs) & (span_years >= args.min_span_years)
    ].copy()
    n_support = len(cand)
    if args.state:
        cand = cand[cand["state"] == args.state].copy()
    # Rewrite the snapshot prefix so all reads hit the frozen S0 tree.
    fp = cand["file_path"].astype("string")
    cand["path"] = np.where(
        fp.str.startswith(CURRENT_PREFIX),
        fp.str.replace(CURRENT_PREFIX, args.snapshot_root, n=1, regex=False),
        fp,
    )
    if args.limit and args.limit < len(cand):
        cand = cand.sample(args.limit, random_state=42).reset_index(drop=True)
    log.info(
        "candidates: total=%d unconfined+ts=%d >=%dobs&>=%dyr=%d selected=%d",
        n_total,
        n_cand,
        args.min_obs,
        args.min_span_years,
        n_support,
        len(cand),
    )
    return cand, (n_total, n_cand, n_support)


def _chunks(records: list, size: int):
    for i in range(0, len(records), size):
        yield records[i : i + size]


# --------------------------------------------------------------------------- #
# Sanity summary for the report.
# --------------------------------------------------------------------------- #
def _sanity(status: pd.DataFrame) -> dict:
    ok = status[status["status_flag"] == "ok"]
    q = ok[ok["ngwmn_qualifying"]]
    sub = q if len(q) else ok
    out: dict = {
        "n_status_rows": int(len(status)),
        "n_scored_ok": int(len(ok)),
        "n_qualifying_ok": int(len(q)),
    }
    if len(sub):
        sgi = sub["sgi"].to_numpy()
        sgi = sgi[np.isfinite(sgi)]
        pct = sub["head_percentile"].to_numpy()
        pct = pct[np.isfinite(pct)]
        out["sgi_mean"] = float(np.mean(sgi))
        out["sgi_std"] = float(np.std(sgi))
        out["sgi_p05_p50_p95"] = [float(np.percentile(sgi, p)) for p in (5, 50, 95)]
        out["percentile_deciles_frac"] = (
            (np.histogram(pct, bins=np.arange(0, 101, 10))[0] / len(pct))
            .round(4)
            .tolist()
        )
        out["gw_class_counts"] = sub["gw_class"].value_counts(dropna=False).to_dict()
        out["gw_class_counts"] = {
            str(k): int(v) for k, v in out["gw_class_counts"].items()
        }
    return out


def _write_parquet(df: pd.DataFrame, path: Path, sort_cols: list[str]) -> None:
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_parquet(path, index=False)
    log.info("wrote %s (%d rows)", path, len(df))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--phase", choices=["all", "consolidate", "status"], default="all")
    p.add_argument("--gwx-index", default=GWX_INDEX)
    p.add_argument(
        "--snapshot-root",
        default=S0_SNAPSHOT,
        help="root that replaces the %s prefix in file_path "
        "(default the frozen S0 snapshot)" % CURRENT_PREFIX,
    )
    p.add_argument("--out-dir", default="/data/ssd2/handily/conus/ops_status/s0/status")
    p.add_argument("--confinement", default=",".join(WT_CLASSES))
    p.add_argument("--min-obs", type=int, default=24)
    p.add_argument("--min-span-years", type=float, default=10.0)
    p.add_argument("--state", default=None, help="optional 2-letter state filter")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="smoke-test: random-sample N candidate wells (seed 42)",
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=200)
    p.add_argument(
        "--asof",
        default="2026-07-07",
        help="staleness reference date (default S0 snapshot date)",
    )
    p.add_argument(
        "--min-loyo",
        type=int,
        default=3,
        help="min leave-year-out sample size to emit q/SGI/percentile",
    )
    p.add_argument(
        "--jump-abs-m",
        type=float,
        default=10.0,
        help="flag consecutive |Δhead| jumps above this (m); "
        "S0 p99.9 of monthly-scale |Δh| ~6.1 m",
    )
    p.add_argument("--jump-window-days", type=float, default=31.0)
    p.add_argument(
        "--elev-corr-min",
        type=float,
        default=0.99,
        help="flag well if corr(-dtw, wl_elevation) below this",
    )
    p.add_argument(
        "--keep-pumping",
        action="store_true",
        help="do NOT exclude pumping levels from the monthly median",
    )
    p.add_argument(
        "--keep-rejected",
        action="store_true",
        help="do NOT exclude source-rejected measurements",
    )
    p.add_argument(
        "--monthly-parquet",
        default=None,
        help="phase=status: path to an existing monthly parquet "
        "(default <out-dir>/groundwater_monthly_heads.parquet)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    asof = pd.Timestamp(args.asof, tz="UTC")
    params = {
        "phase": args.phase,
        "asof": asof,
        "min_loyo": args.min_loyo,
        "jump_abs_m": args.jump_abs_m,
        "jump_window_days": args.jump_window_days,
        "elev_corr_min": args.elev_corr_min,
        "exclude_pumping": not args.keep_pumping,
        "exclude_rejected": not args.keep_rejected,
    }

    t0 = time.time()
    total = Counter()
    report: dict = {
        "phase": args.phase,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "dtw_unit_finding": (
            "dtw is in METERS: each per-well series mean(dtw) reproduces the index "
            "mean_dtw exactly (e.g. nwis 3.079, nd_dwr 2.004, tx_twdb 44.939); "
            "wl_elevation_navd88/ngvd29 are also meters. Head harmonized as h=-dtw."
        ),
        "is_static_finding": (
            "is_static is UNPOPULATED in S0 (~0.02% True across 283k measurements; "
            "only mt_gwic has any True). It is NOT a pumping indicator, so non-static "
            "levels are NOT excluded (that would drop ~99.98% of data). Pumping is "
            "detected from source_flags ('pumping' qualifier) instead; static/"
            "non-static are counted only."
        ),
        "qc_policy": {
            "exclude_pumping": params["exclude_pumping"],
            "exclude_rejected": params["exclude_rejected"],
            "jump_abs_m": args.jump_abs_m,
            "jump_window_days": args.jump_window_days,
            "elev_corr_min": args.elev_corr_min,
            "no_winsorization": True,
            "dup_timestamps": "collapsed to per-timestamp median before monthly median",
        },
    }

    monthly_path = out_dir / "groundwater_monthly_heads.parquet"
    status_path = out_dir / "groundwater_status_monthly.parquet"
    clim_path = out_dir / "groundwater_station_climatology.parquet"

    if args.phase in ("all", "consolidate"):
        cand, n_stage = select_candidates(args)
        report["stage_counts"] = {
            "index_total": n_stage[0],
            "candidate_unconfined_ts": n_stage[1],
            "min_support_subset": n_stage[2],
            "selected": int(len(cand)),
        }
        records = cand[["path"] + [c for c in DESCRIPTOR_COLS]].to_dict("records")
        chunks = list(_chunks(records, args.chunk_size))
        monthly_parts, clim_parts, status_parts = [], [], []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker, ch, params) for ch in chunks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="wells"):
                m_df, c_df, s_df, cnt = fut.result()
                total.update(cnt)
                if m_df is not None:
                    monthly_parts.append(m_df)
                if c_df is not None:
                    clim_parts.append(c_df)
                if s_df is not None:
                    status_parts.append(s_df)

        if monthly_parts:
            monthly_all = pd.concat(monthly_parts, ignore_index=True)
            _write_parquet(monthly_all, monthly_path, ["canonical_id", "year", "month"])
            report["n_monthly_rows"] = int(len(monthly_all))
        else:
            log.warning("no monthly rows produced")
        if args.phase == "all":
            if status_parts:
                status_all = pd.concat(status_parts, ignore_index=True)
                _write_parquet(
                    status_all, status_path, ["canonical_id", "year", "month"]
                )
                report["sanity"] = _sanity(status_all)
            if clim_parts:
                clim_all = pd.concat(clim_parts, ignore_index=True)
                _write_parquet(clim_all, clim_path, ["canonical_id", "month"])
                report["n_climatology_rows"] = int(len(clim_all))

    elif args.phase == "status":
        src = Path(args.monthly_parquet) if args.monthly_parquet else monthly_path
        if not src.exists():
            raise SystemExit(
                f"monthly parquet not found: {src} (run consolidate first)"
            )
        monthly_all = pd.read_parquet(src)
        report["stage_counts"] = {
            "monthly_rows_in": int(len(monthly_all)),
            "wells_in": int(monthly_all.groupby(WELL_KEY).ngroups),
        }
        groups = [g for _, g in monthly_all.groupby(WELL_KEY, sort=False)]
        chunks = list(_chunks(groups, args.chunk_size))
        clim_parts, status_parts = [], []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_status_worker, ch, params) for ch in chunks]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="score"):
                c_df, s_df, cnt = fut.result()
                total.update(cnt)
                if c_df is not None:
                    clim_parts.append(c_df)
                if s_df is not None:
                    status_parts.append(s_df)
        if status_parts:
            status_all = pd.concat(status_parts, ignore_index=True)
            _write_parquet(status_all, status_path, ["canonical_id", "year", "month"])
            report["sanity"] = _sanity(status_all)
        if clim_parts:
            clim_all = pd.concat(clim_parts, ignore_index=True)
            _write_parquet(clim_all, clim_path, ["canonical_id", "month"])
            report["n_climatology_rows"] = int(len(clim_all))

    runtime = time.time() - t0
    report["runtime_sec"] = round(runtime, 1)
    report["qc_tallies"] = {k: int(v) for k, v in sorted(total.items())}
    n_ok = total.get("wells_ok", 0)
    if n_ok and args.phase in ("all", "consolidate"):
        report["extrapolated_full_runtime_min_53867_wells"] = round(
            runtime / n_ok * 53867 / 60.0, 1
        )
    with open(out_dir / "status_build_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("wrote %s", out_dir / "status_build_report.json")
    log.info("done in %.1fs (%d wells ok)", runtime, n_ok)


if __name__ == "__main__":
    main()
