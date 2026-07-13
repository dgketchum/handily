"""Scientific audit of the GWX groundwater-hydrograph snapshot (interim rung A1).

Governing plan: ``notes/ops_assimilation_plan.md`` §3 ("Replacement rung 0 —
monitoring data contract and operational audit"). Interim scope:
``notes/ops_assimilation_interim_plan.md`` Track A / A1. This is the
retrospective, measurement-time half of the §3.2 audit that snapshot S0 can
answer. It reads a frozen local snapshot only, makes zero GWX API calls, and
never writes inside the gwx tree.

Candidate population (governing-plan §3 selector)::

    candidate = wells[
        wells.has_time_series
        & wells.confinement_class.isin(["unconfined", "unconfined_marginal"])
    ]

188,523 wells on S0. The audit inventories these candidates in full; confined /
likely-confined wells (diagnostic-tier by policy) only get a summary head-count.

For every candidate the script opens its per-well hydrograph parquet, derives
record support (obs, qualifying-year × calendar-month support, POR span),
inter-measurement interval statistics, seasonal coverage, method/approval/datum
composition and transition counts, an ``is_static`` fraction, dry/flag counts
parsed from ``source_flags``, duplicate-group size, and a monitoring role
(drought-core / condition-network / diagnostic per §3.3). It then aggregates by
source, state, and calendar month, computes nearest-neighbour spacing per state
(EPSG:5070 + scipy cKDTree), and writes a versioned inventory parquet plus a
JSON + HTML report.

**Latency is deferred.** Every arrival / issue-time / update-cadence quantity is
emitted as an explicitly ``unavailable_pending_contract`` column (schema final,
values pending the GWX operational data contract) — the report states the
cadence/latency conclusions are DEFERRED under the oracle-availability interim
convention, and draws no cadence inference from this snapshot (its downloads
were skipped).

Snapshot remap: ``--snapshot-root`` rewrites the ``/data/ssd2/gwx/products/
current`` prefix of each ``file_path`` to the given root, so the frozen hardlink
snapshot is read even while ``current`` churns.

Usage::

    uv run python utils/audit_gwx_hydrographs.py \
        --snapshot-root /data/ssd2/gwx/products/current \
        --out-dir /tmp/audit_smoke --workers 8 --limit 2000
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import multiprocessing as mp
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from tqdm import tqdm

log = logging.getLogger("audit_gwx_hydrographs")

CURRENT_PREFIX = "/data/ssd2/gwx/products/current"
DEFAULT_SNAPSHOT_ROOT = "/data/ssd2/gwx/products/snapshots/s0_20260707"
DEFAULT_OUT_DIR = "/data/ssd2/handily/conus/ops_status/s0/audit"
WT_CLASSES = ("unconfined", "unconfined_marginal")

# Columns read from each per-well hydrograph parquet.
SERIES_COLS = [
    "dtime",
    "dtw",
    "is_static",
    "method",
    "approval_status",
    "source_flags",
    "wl_elevation_ngvd29",
    "wl_elevation_navd88",
]

# Index columns pulled once for the candidate frame.
INDEX_COLS = [
    "source",
    "source_well_id",
    "canonical_id",
    "state",
    "longitude",
    "latitude",
    "has_time_series",
    "confinement_class",
    "confinement_source",
    "confinement_confidence",
    "source_confinement",
    "aquifer",
    "aquifer_type_code",
    "well_depth",
    "screen_top",
    "screen_bottom",
    "casing_depth",
    "well_use",
    "well_use_usgwd",
    "well_class",
    "class_confidence",
    "is_active",
    "obs_count",
    "por_start",
    "por_end",
    "file_path",
]

# well_use values that mark an extraction (pumping-affected) well; the water
# level in these is not a clean unconfined-table measurement.
PUMPING_USES = frozenset(
    {
        "irrigation",
        "supply",
        "public_supply",
        "industrial",
        "commercial",
        "livestock",
        "mining",
        "sewage",
    }
)

# NWIS measurement-qualifier substrings (source_flags 'qualifier' field). The
# per-observation ``is_static`` boolean column is unpopulated on S0 (see report),
# so static/pumping/affected status is recovered from source_flags text instead.
PUMP_FLAG_TOKENS = ("Pumping", "Above")
STATIC_FLAG_TOKEN = "Static"
AFFECTED_FLAG_TOKENS = ("GWSWAffected", "GWTideAffected", "ForeignSubstance")

# Monitoring-role rule (governing-plan §3.3), written verbatim into the report so
# the assignment is reproducible from documented proxies.
ROLE_RULE = (
    "All candidates are unconfined/unconfined_marginal (selector guarantee). "
    "A QUALIFYING observation = finite dtw AND not flagged pumping/above "
    "(source_flags qualifier). long_record = >=10 distinct qualifying years in "
    ">=6 calendar months. construction_known = any of {well_depth, screen_top, "
    "screen_bottom, casing_depth} finite. pumping_dominated = well_class=="
    "'pumping' OR well_use in {irrigation,supply,public_supply,industrial,"
    "commercial,livestock,mining,sewage} OR (>50% of valid obs pumping-flagged). "
    "Roles: (1) pumping_dominated -> 'diagnostic'; else (2) well_class=='monitoring' "
    "& long_record & construction_known -> 'drought_core'; else (3) "
    "well_class=='monitoring' -> 'condition_network'; else (4) non-monitoring with "
    "long_record & construction_known -> 'condition_network'; else -> 'diagnostic'."
)

SEASON_MONTHS = {
    "djf": (12, 1, 2),
    "mam": (3, 4, 5),
    "jja": (6, 7, 8),
    "son": (9, 10, 11),
}

# Fields with no value from S0 (measurement-time snapshot, downloads skipped).
# Present-but-pending per the interim data contract (§3.1).
LATENCY_STATUS = "unavailable_pending_contract"

# Keys returned by the per-well worker when a file is absent / unreadable / empty.
_NULL_STATS = {
    "n_obs": 0,
    "n_valid_obs": 0,
    "n_dtw_null": 0,
    "n_dtime_null": 0,
    "n_qualifying_obs": 0,
    "n_static_col_true": 0,
    "is_static_frac": np.nan,
    "file_por_start": pd.NaT,
    "file_por_end": pd.NaT,
    "file_por_span_years": np.nan,
    "n_months_with_gt10yrs": 0,
    "n_months_ge10yr": 0,
    "max_month_year_span": 0,
    "month_year_support": None,
    "gap_days_median": np.nan,
    "gap_days_p90": np.nan,
    "gap_days_max": np.nan,
    "n_gaps": 0,
    "n_djf": 0,
    "n_mam": 0,
    "n_jja": 0,
    "n_son": 0,
    "dominant_method": None,
    "n_distinct_methods": 0,
    "n_method_transitions": 0,
    "dominant_approval": None,
    "n_distinct_approvals": 0,
    "n_approval_transitions": 0,
    "dominant_datum": None,
    "n_distinct_datums": 0,
    "n_datum_transitions": 0,
    "n_wl_elev_navd88": 0,
    "n_wl_elev_ngvd29": 0,
    "n_pump_flagged": 0,
    "n_static_flagged": 0,
    "n_affected_flagged": 0,
    "n_flag_nonnull": 0,
}


def remap_path(file_path: str, snapshot_root: str) -> str:
    """Rewrite the ``current`` prefix of a per-well path to the snapshot root."""
    if file_path.startswith(CURRENT_PREFIX):
        return snapshot_root + file_path[len(CURRENT_PREFIX) :]
    return file_path


def _transitions(values: np.ndarray) -> int:
    """Count positions where a time-ordered categorical differs from its predecessor."""
    if values.size < 2:
        return 0
    return int(np.sum(values[1:] != values[:-1]))


def _derive_well_stats(s: pd.DataFrame) -> dict:
    """Derive all per-well record-support/QC statistics from a hydrograph frame."""
    n_obs = len(s)
    if n_obs == 0:
        return {**_NULL_STATS, "file_status": "empty"}

    s = s.sort_values("dtime")
    dtime = s["dtime"]
    valid_time = dtime.notna().to_numpy()
    n_dtime_null = int((~valid_time).sum())
    dtw = s["dtw"].to_numpy(dtype="float64")
    valid = np.isfinite(dtw)
    n_valid = int(valid.sum())

    raw_flags = s["source_flags"].fillna("").astype(str)
    n_flag_nonnull = int((s["source_flags"].notna()).sum())
    pump = np.zeros(n_obs, dtype=bool)
    for tok in PUMP_FLAG_TOKENS:
        pump |= raw_flags.str.contains(tok, regex=False).to_numpy()
    static_fl = raw_flags.str.contains(STATIC_FLAG_TOKEN, regex=False).to_numpy()
    affected = np.zeros(n_obs, dtype=bool)
    for tok in AFFECTED_FLAG_TOKENS:
        affected |= raw_flags.str.contains(tok, regex=False).to_numpy()

    # Qualifying observation: a real water level, with a timestamp, not
    # explicitly a pumping level. A valid dtw with a NaT dtime (present on
    # ~1.7% of files, chiefly tx_twdb) cannot contribute temporal support and
    # is excluded here — but is counted in n_dtime_null, never dropped silently.
    qual = valid & ~pump & valid_time
    n_qual = int(qual.sum())

    months = dtime.dt.month.to_numpy()
    years = dtime.dt.year.to_numpy()
    qmonths = months[qual]
    qyears = years[qual]
    support = np.zeros(12, dtype="int64")
    for m in range(1, 13):
        yy = qyears[qmonths == m]
        support[m - 1] = np.unique(yy).size if yy.size else 0

    # Inter-measurement gaps over qualifying obs (already time-sorted).
    qtime = dtime[qual]
    if n_qual >= 2:
        diffs = qtime.diff().dropna().dt.total_seconds().to_numpy() / 86400.0
        gap_median = float(np.median(diffs))
        gap_p90 = float(np.percentile(diffs, 90))
        gap_max = float(diffs.max())
        n_gaps = int(diffs.size)
    else:
        gap_median = gap_p90 = gap_max = np.nan
        n_gaps = 0

    seasonal = {
        f"n_{name}": int(np.isin(qmonths, mm).sum())
        for name, mm in SEASON_MONTHS.items()
    }

    method = s["method"].fillna("<NA>").astype(str).to_numpy()
    approval = s["approval_status"].fillna("<NA>").astype(str).to_numpy()
    navd = s["wl_elevation_navd88"].notna().to_numpy()
    ngvd = s["wl_elevation_ngvd29"].notna().to_numpy()
    datum = np.where(navd, "navd88", np.where(ngvd, "ngvd29", "none"))
    datum_nn = datum[datum != "none"]

    is_static_col = s["is_static"].fillna(False).astype(bool).to_numpy()
    n_static_col = int(is_static_col.sum())

    vt = dtime[valid_time]
    if len(vt):
        por_start = vt.min()
        por_end = vt.max()
        por_span_years = (por_end - por_start).total_seconds() / (86400.0 * 365.25)
    else:
        por_start = por_end = pd.NaT
        por_span_years = np.nan

    def _mode(arr: np.ndarray):
        if arr.size == 0:
            return None
        vals, counts = np.unique(arr, return_counts=True)
        return str(vals[int(np.argmax(counts))])

    return {
        "file_status": "ok",
        "n_obs": n_obs,
        "n_valid_obs": n_valid,
        "n_dtw_null": n_obs - n_valid,
        "n_dtime_null": n_dtime_null,
        "n_qualifying_obs": n_qual,
        "n_static_col_true": n_static_col,
        "is_static_frac": n_static_col / n_obs,
        "file_por_start": por_start,
        "file_por_end": por_end,
        "file_por_span_years": float(por_span_years),
        "n_months_with_gt10yrs": int(np.sum(support > 10)),
        "n_months_ge10yr": int(np.sum(support >= 10)),
        "max_month_year_span": int(support.max()),
        "month_year_support": support.tolist(),
        "gap_days_median": gap_median,
        "gap_days_p90": gap_p90,
        "gap_days_max": gap_max,
        "n_gaps": n_gaps,
        **seasonal,
        "dominant_method": _mode(method),
        "n_distinct_methods": int(np.unique(method).size),
        "n_method_transitions": _transitions(method),
        "dominant_approval": _mode(approval),
        "n_distinct_approvals": int(np.unique(approval).size),
        "n_approval_transitions": _transitions(approval),
        "dominant_datum": _mode(datum_nn),
        "n_distinct_datums": int(np.unique(datum_nn).size),
        "n_datum_transitions": _transitions(datum_nn),
        "n_wl_elev_navd88": int(navd.sum()),
        "n_wl_elev_ngvd29": int(ngvd.sum()),
        "n_pump_flagged": int(pump.sum()),
        "n_static_flagged": int(static_fl.sum()),
        "n_affected_flagged": int(affected.sum()),
        "n_flag_nonnull": n_flag_nonnull,
    }


def _process_chunk(chunk: list[tuple[int, str]], snapshot_root: str) -> tuple:
    """Read a chunk of per-well files; return per-well stats + flag-vocab counters.

    Missing/unreadable files are counted (never silently dropped): the well still
    yields a row with null support fields and a ``file_status`` marker.
    """
    rows: list[dict] = []
    key_counter: Counter = Counter()
    n_missing = 0
    n_read_error = 0
    for idx, file_path in chunk:
        fp = remap_path(file_path, snapshot_root)
        if not os.path.exists(fp):
            n_missing += 1
            rows.append({"_idx": idx, **_NULL_STATS, "file_status": "missing"})
            continue
        try:
            s = pd.read_parquet(fp, columns=SERIES_COLS)
        except Exception as exc:  # noqa: BLE001 - counted + reported, not swallowed
            n_read_error += 1
            rows.append(
                {
                    "_idx": idx,
                    **_NULL_STATS,
                    "file_status": f"read_error:{type(exc).__name__}",
                }
            )
            continue
        # Global source_flags key vocabulary (JSON top-level keys).
        _accumulate_flag_keys(s["source_flags"], key_counter)
        stats = _derive_well_stats(s)
        stats["_idx"] = idx
        rows.append(stats)
    return rows, key_counter, n_missing, n_read_error


def _accumulate_flag_keys(flags: pd.Series, counter: Counter) -> None:
    """Tally the JSON top-level keys observed in a source_flags column."""
    for blob in flags.dropna().astype(str):
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            counter["<parse_fail>"] += 1
            continue
        if isinstance(obj, dict):
            for k in obj:
                counter[str(k)] += 1


def load_candidates(
    index_path: str, state: str | None, limit: int | None
) -> pd.DataFrame:
    """Load the candidate (unconfined/marginal + has_time_series) well frame."""
    df = pd.read_parquet(index_path, columns=INDEX_COLS)
    n_total = len(df)
    conf_counts = df["confinement_class"].value_counts(dropna=False).to_dict()
    cand = df[
        df["has_time_series"].fillna(False) & df["confinement_class"].isin(WT_CLASSES)
    ].copy()
    if state:
        cand = cand[cand["state"] == state].copy()
    cand = cand.reset_index(drop=True)
    if limit:
        cand = cand.iloc[:limit].copy()
    cand.attrs["n_total_index"] = n_total
    cand.attrs["confinement_counts"] = {str(k): int(v) for k, v in conf_counts.items()}
    return cand


def assign_roles(inv: pd.DataFrame) -> pd.DataFrame:
    """Compute construction_known, pumping_dominated, long_record, and role."""
    construction_cols = ["well_depth", "screen_top", "screen_bottom", "casing_depth"]
    inv["construction_known"] = inv[construction_cols].notna().any(axis=1)

    valid = inv["n_valid_obs"].to_numpy()
    pump = inv["n_pump_flagged"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        pump_frac = np.where(valid > 0, pump / np.maximum(valid, 1), 0.0)
    inv["pump_flag_frac"] = pump_frac

    well_use = inv["well_use"].fillna("").astype(str)
    inv["pumping_dominated"] = (
        (inv["well_class"] == "pumping")
        | well_use.isin(PUMPING_USES)
        | (pump_frac > 0.5)
    )
    inv["long_record"] = inv["n_months_ge10yr"] >= 6

    is_mon = inv["well_class"] == "monitoring"
    role = np.full(len(inv), "diagnostic", dtype=object)
    drought = (
        (~inv["pumping_dominated"])
        & is_mon
        & inv["long_record"]
        & inv["construction_known"]
    )
    cond_mon = (~inv["pumping_dominated"]) & is_mon & ~drought
    cond_other = (
        (~inv["pumping_dominated"])
        & ~is_mon
        & inv["long_record"]
        & inv["construction_known"]
    )
    role[cond_mon.to_numpy() | cond_other.to_numpy()] = "condition_network"
    role[drought.to_numpy()] = "drought_core"
    inv["monitoring_role"] = role
    return inv


def add_latency_columns(inv: pd.DataFrame) -> pd.DataFrame:
    """Attach the (final-schema, pending-value) arrival/ingestion latency columns."""
    inv["arrival_latency_days_median"] = np.nan
    inv["arrival_latency_days_p90"] = np.nan
    inv["ingestion_latency_days_median"] = np.nan
    inv["latency_status"] = LATENCY_STATUS
    return inv


def nearest_neighbour_spacing(inv: pd.DataFrame) -> dict:
    """Per-state and overall NN spacing among candidate wells (EPSG:5070)."""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tr.transform(inv["longitude"].to_numpy(), inv["latitude"].to_numpy())
    inv["x5070"] = x
    inv["y5070"] = y
    good = np.isfinite(x) & np.isfinite(y)

    def _nn(mask: np.ndarray) -> dict | None:
        pts = np.c_[x[mask], y[mask]]
        if len(pts) < 2:
            return None
        d, _ = cKDTree(pts).query(pts, k=2)
        nn = d[:, 1]
        return {
            "n": int(len(pts)),
            "nn_median_m": float(np.median(nn)),
            "nn_p90_m": float(np.percentile(nn, 90)),
            "nn_min_m": float(nn.min()),
            "frac_nn_lt_10m": float(np.mean(nn < 10.0)),
        }

    out = {"overall": _nn(good)}
    per_state = {}
    for st in sorted(inv.loc[good, "state"].dropna().unique()):
        m = good & (inv["state"] == st).to_numpy()
        r = _nn(m)
        if r:
            per_state[str(st)] = r
    out["per_state"] = per_state
    return out


def _dist_series(s: pd.Series) -> dict:
    """Compact numeric distribution for the JSON report."""
    a = s.to_numpy(dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def build_summary(
    inv: pd.DataFrame,
    cand: pd.DataFrame,
    flag_keys: Counter,
    nn: dict,
    n_missing: int,
    n_read_error: int,
    n_empty: int,
    index_path: str,
    snapshot_root: str,
) -> dict:
    """Assemble the JSON summary of the audit."""
    inv["pass_24obs_10yr"] = (inv["obs_count"] >= 24) & (
        inv["index_por_span_years"] >= 10
    )
    inv["pass_60obs_10yr"] = (inv["obs_count"] >= 60) & (
        inv["index_por_span_years"] >= 10
    )
    inv["ngwmn_eligible_any_month"] = inv["n_months_with_gt10yrs"] >= 1

    def group_block(by: str) -> list[dict]:
        rows = []
        for key, g in inv.groupby(by, dropna=False):
            roles = g["monitoring_role"].value_counts().to_dict()
            rows.append(
                {
                    by: str(key),
                    "n_wells": int(len(g)),
                    "n_pass_24obs_10yr": int(g["pass_24obs_10yr"].sum()),
                    "n_pass_60obs_10yr": int(g["pass_60obs_10yr"].sum()),
                    "n_ngwmn_eligible_month": int(g["ngwmn_eligible_any_month"].sum()),
                    "n_drought_core": int(roles.get("drought_core", 0)),
                    "n_condition_network": int(roles.get("condition_network", 0)),
                    "n_diagnostic": int(roles.get("diagnostic", 0)),
                    "n_pumping_dominated": int(g["pumping_dominated"].sum()),
                    "median_n_months_gt10yr": float(
                        g["n_months_with_gt10yrs"].median()
                    ),
                    "median_obs_count": float(g["obs_count"].median()),
                    "median_gap_days": float(g["gap_days_median"].median()),
                }
            )
        return sorted(rows, key=lambda r: -r["n_wells"])

    # Calendar-month aggregate: distribution, across candidate wells, of the
    # number of distinct qualifying years available in each calendar month, and
    # how many wells clear the NGWMN >10-year eligibility for that month.
    support_mat = np.array(
        [r if r is not None else [0] * 12 for r in inv["month_year_support"]],
        dtype="int64",
    )
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    by_month = []
    for i, name in enumerate(month_names):
        col = support_mat[:, i]
        by_month.append(
            {
                "month": name,
                "n_wells_with_obs": int(np.sum(col > 0)),
                "median_distinct_years": float(np.median(col)),
                "p90_distinct_years": float(np.percentile(col, 90)),
                "n_wells_gt10yr_eligible": int(np.sum(col > 10)),
            }
        )

    role_counts = inv["monitoring_role"].value_counts().to_dict()
    file_status_counts = inv["file_status"].value_counts().to_dict()

    n_valid_static_col = int(inv["n_static_col_true"].sum())
    n_total_obs = int(inv["n_obs"].sum())

    coverage_comment = (
        "Nearest-neighbour spacing is computed among candidate wells only. A "
        f"nonzero fraction with NN<10 m ({100 * nn['overall']['frac_nn_lt_10m']:.1f}% "
        "overall) reflects co-located / duplicate sites (see dup_group_size); "
        "these are spatial redundancy, not independent coverage. Large per-state "
        "median NN distances (see per_state) flag coverage holes where sparse "
        "candidate wells cannot support fine spatial fill. This is a spatial-"
        "dispersion inventory only; the issue cadence is NOT chosen here."
    )

    deferred_note = (
        "DEFERRED (Track B, pending GWX operational data contract): "
        "measurement-to-arrival and arrival-to-ingestion latency, source update "
        "cadence, snapshot diffs, and the issue-cadence choice. Snapshot S0 was a "
        "measurement-time freeze whose source downloads were SKIPPED, so its "
        "timestamps say nothing about source update behaviour. Under the interim "
        "oracle-availability convention every skill/coverage number is an upper "
        "bound; NO update-cadence or latency inference is drawn from this snapshot. "
        "The arrival_latency_* / ingestion_latency_* columns are present with null "
        f"values and latency_status='{LATENCY_STATUS}' — the schema is final."
    )

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "index_path": index_path,
        "snapshot_root": snapshot_root,
        "n_total_index_rows": int(cand.attrs.get("n_total_index", -1)),
        "confinement_counts_all": cand.attrs.get("confinement_counts", {}),
        "n_candidates": int(len(inv)),
        "candidate_selector": (
            "has_time_series & confinement_class in ('unconfined',"
            "'unconfined_marginal')"
        ),
        "file_status_counts": {str(k): int(v) for k, v in file_status_counts.items()},
        "n_files_missing": int(n_missing),
        "n_files_read_error": int(n_read_error),
        "n_files_empty": int(n_empty),
        "total_observations_read": n_total_obs,
        "role_rule": ROLE_RULE,
        "role_counts": {str(k): int(v) for k, v in role_counts.items()},
        "support_thresholds": {
            "n_pass_24obs_10yr": int(inv["pass_24obs_10yr"].sum()),
            "n_pass_60obs_10yr": int(inv["pass_60obs_10yr"].sum()),
            "n_ngwmn_eligible_any_month": int(inv["ngwmn_eligible_any_month"].sum()),
            "definition": (
                "obs_count & POR span from the well index; pass_24obs_10yr = "
                "obs_count>=24 & span>=10yr; pass_60obs_10yr = obs_count>=60 & "
                "span>=10yr; ngwmn_eligible_any_month = >=1 calendar month with "
                ">10 distinct qualifying years."
            ),
        },
        "is_static_column": {
            "n_obs_total": n_total_obs,
            "n_is_static_true": n_valid_static_col,
            "frac_is_static_true": (
                n_valid_static_col / n_total_obs if n_total_obs else float("nan")
            ),
            "note": (
                "The per-observation is_static boolean is effectively unpopulated "
                "on S0 (near-zero True). Static/pumping/affected status is instead "
                "recovered from the NWIS source_flags 'qualifier' field; a "
                "QUALIFYING observation is finite-dtw AND not pumping-flagged."
            ),
        },
        "source_flags_vocabulary": {
            "keys": {str(k): int(v) for k, v in flag_keys.most_common()},
            "note": (
                "Observed JSON top-level keys across candidate hydrographs. "
                "Qualifier tokens Static/Above/Pumping/GWSWAffected/GWTideAffected/"
                "ForeignSubstance drive the pumping/affected tallies. dry/censored "
                "measurements surface primarily as null dtw (n_dtw_null), no 'dry' "
                "literal was observed. A separate ~1.7% of files carry valid dtw "
                "with a NULL timestamp (n_dtime_null, chiefly tx_twdb); these are "
                "excluded from temporal support but retained in the obs count."
            ),
        },
        "flag_tallies": {
            "n_pump_flagged": int(inv["n_pump_flagged"].sum()),
            "n_static_flagged": int(inv["n_static_flagged"].sum()),
            "n_affected_flagged": int(inv["n_affected_flagged"].sum()),
            "n_dtw_null": int(inv["n_dtw_null"].sum()),
            "n_dtime_null": int(inv["n_dtime_null"].sum()),
        },
        "background_vs_pumping": {
            "note": (
                "Background = drought_core + condition_network roles; "
                "pumping-affected = diagnostic role or pumping_dominated wells."
            ),
            "n_background": int(
                inv["monitoring_role"].isin(["drought_core", "condition_network"]).sum()
            ),
            "n_pumping_affected": int(inv["pumping_dominated"].sum()),
        },
        "distributions": {
            "obs_count": _dist_series(inv["obs_count"]),
            "index_por_span_years": _dist_series(inv["index_por_span_years"]),
            "n_months_with_gt10yrs": _dist_series(inv["n_months_with_gt10yrs"]),
            "gap_days_median": _dist_series(inv["gap_days_median"]),
            "gap_days_p90": _dist_series(inv["gap_days_p90"]),
            "dup_group_size": _dist_series(inv["dup_group_size"]),
        },
        "by_source": group_block("source"),
        "by_state": group_block("state"),
        "by_calendar_month": by_month,
        "nearest_neighbour_spacing_m": nn,
        "coverage_hole_comment": coverage_comment,
        "latency_cadence_deferred": deferred_note,
        "availability": "oracle",
    }


def _df_to_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, float_format=lambda v: f"{v:.2f}", border=0)


def write_html_report(summary: dict, out_path: Path) -> None:
    """Render a self-contained HTML report from the JSON summary."""
    css = (
        "body{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#1a1a1a;}"
        "h1{font-size:1.5rem;}h2{font-size:1.15rem;margin-top:1.8rem;"
        "border-bottom:1px solid #ccc;padding-bottom:.2rem;}"
        "table{border-collapse:collapse;font-size:.8rem;margin:.5rem 0;}"
        "th,td{border:1px solid #ddd;padding:3px 7px;text-align:right;}"
        "th{background:#f2f2f2;}td:first-child,th:first-child{text-align:left;}"
        ".note{background:#fff8e1;border-left:4px solid #f6c343;padding:.6rem .9rem;"
        "margin:.6rem 0;font-size:.85rem;}"
        ".defer{background:#fdecea;border-left:4px solid #e57373;padding:.6rem .9rem;"
        "margin:.6rem 0;font-size:.85rem;}code{font-size:.8rem;}"
    )
    by_source = pd.DataFrame(summary["by_source"])
    by_state = pd.DataFrame(summary["by_state"])
    by_month = pd.DataFrame(summary["by_calendar_month"])
    nn_rows = [
        {"state": st, **v}
        for st, v in summary["nearest_neighbour_spacing_m"]["per_state"].items()
    ]
    nn_df = pd.DataFrame(nn_rows) if nn_rows else pd.DataFrame()
    flag_df = pd.DataFrame(
        list(summary["source_flags_vocabulary"]["keys"].items()),
        columns=["source_flags_key", "count"],
    )

    parts = [
        f"<style>{css}</style>",
        "<h1>GWX hydrograph scientific audit (S0, interim rung A1)</h1>",
        f"<p>Generated {summary['generated_utc']} &middot; snapshot_root "
        f"<code>{summary['snapshot_root']}</code> &middot; index "
        f"<code>{summary['index_path']}</code> &middot; availability="
        f"<b>{summary['availability']}</b></p>",
        "<div class='defer'><b>Latency / cadence DEFERRED.</b> "
        f"{summary['latency_cadence_deferred']}</div>",
        "<h2>Population</h2>",
        f"<p>Candidates: <b>{summary['n_candidates']:,}</b> "
        f"(selector <code>{summary['candidate_selector']}</code>) of "
        f"{summary['n_total_index_rows']:,} index rows. "
        f"Observations read: {summary['total_observations_read']:,}. "
        f"file_status={summary['file_status_counts']} "
        f"(missing={summary['n_files_missing']}, "
        f"read_error={summary['n_files_read_error']}, "
        f"empty={summary['n_files_empty']}).</p>",
        "<h2>Monitoring roles</h2>",
        f"<div class='note'><b>Role rule.</b> {summary['role_rule']}</div>",
        f"<p>Role counts: <b>{summary['role_counts']}</b>. "
        f"Background (drought_core+condition) = "
        f"{summary['background_vs_pumping']['n_background']:,}; "
        f"pumping-affected = "
        f"{summary['background_vs_pumping']['n_pumping_affected']:,}.</p>",
        "<h2>Support thresholds</h2>",
        f"<p>{summary['support_thresholds']['definition']}</p>"
        f"<p>&ge;24 obs &amp; &ge;10yr: "
        f"<b>{summary['support_thresholds']['n_pass_24obs_10yr']:,}</b> &middot; "
        f"&ge;60 obs &amp; &ge;10yr: "
        f"<b>{summary['support_thresholds']['n_pass_60obs_10yr']:,}</b> &middot; "
        f"NGWMN-eligible (any month &gt;10yr): "
        f"<b>{summary['support_thresholds']['n_ngwmn_eligible_any_month']:,}</b></p>",
        "<h2>is_static column</h2>",
        f"<div class='note'>{summary['is_static_column']['note']} "
        f"frac_is_static_true="
        f"{summary['is_static_column']['frac_is_static_true']:.5f}</div>",
        "<h2>Per source</h2>",
        _df_to_html(by_source),
        "<h2>Per state</h2>",
        _df_to_html(by_state),
        "<h2>Per calendar month (qualifying-year support)</h2>",
        _df_to_html(by_month),
        "<h2>Nearest-neighbour spacing per state (EPSG:5070)</h2>",
        f"<p>Overall: {summary['nearest_neighbour_spacing_m']['overall']}</p>",
        _df_to_html(nn_df) if not nn_df.empty else "<p>n/a</p>",
        f"<div class='note'>{summary['coverage_hole_comment']}</div>",
        "<h2>source_flags vocabulary</h2>",
        f"<p>{summary['source_flags_vocabulary']['note']}</p>",
        _df_to_html(flag_df),
    ]
    out_path.write_text("\n".join(parts))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot-root", default=DEFAULT_SNAPSHOT_ROOT)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--limit", type=int, default=None, help="smoke-test on first N candidates"
    )
    p.add_argument("--state", default=None, help="restrict to a single state (e.g. MT)")
    p.add_argument(
        "--gwx-index",
        default=None,
        help="well index parquet; default = <snapshot-root>/wells.geoparquet if "
        "present, else the current index",
    )
    p.add_argument("--chunk-size", type=int, default=100)
    args = p.parse_args()

    snapshot_root = args.snapshot_root.rstrip("/")
    if args.gwx_index:
        index_path = args.gwx_index
    else:
        snap_index = os.path.join(snapshot_root, "wells.geoparquet")
        index_path = (
            snap_index
            if os.path.exists(snap_index)
            else (CURRENT_PREFIX + "/wells.geoparquet")
        )
    log.info("Using well index: %s", index_path)
    log.info("Snapshot root (per-well parquets): %s", snapshot_root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand = load_candidates(index_path, args.state, args.limit)
    log.info(
        "Candidates: %d (of %d index rows)%s",
        len(cand),
        cand.attrs.get("n_total_index", -1),
        f" state={args.state}" if args.state else "",
    )

    # Build (row_index, file_path) work items and chunk them.
    items = list(zip(cand.index.tolist(), cand["file_path"].tolist()))
    chunks = [
        items[i : i + args.chunk_size] for i in range(0, len(items), args.chunk_size)
    ]

    all_rows: list[dict] = []
    flag_keys: Counter = Counter()
    n_missing = n_read_error = 0
    worker = functools.partial(_process_chunk, snapshot_root=snapshot_root)

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for rows, kc, nm, nre in tqdm(
                pool.imap_unordered(worker, chunks),
                total=len(chunks),
                desc="wells",
                unit="chunk",
            ):
                all_rows.extend(rows)
                flag_keys.update(kc)
                n_missing += nm
                n_read_error += nre
    else:
        for chunk in tqdm(chunks, desc="wells", unit="chunk"):
            rows, kc, nm, nre = worker(chunk)
            all_rows.extend(rows)
            flag_keys.update(kc)
            n_missing += nm
            n_read_error += nre

    stats = pd.DataFrame(all_rows).set_index("_idx")
    inv = cand.join(stats)

    # Duplicate-group size among candidates (wells sharing a canonical_id).
    dup = inv.groupby("canonical_id")["source"].transform("size")
    inv["dup_group_size"] = dup.fillna(1).astype("int64")

    # Index-derived POR span (matches the plan's long-record subset definition).
    inv["index_por_span_years"] = (
        inv["por_end"] - inv["por_start"]
    ).dt.total_seconds() / (86400.0 * 365.25)

    inv = assign_roles(inv)
    inv = add_latency_columns(inv)
    nn = nearest_neighbour_spacing(inv)

    n_empty = int((inv["file_status"] == "empty").sum())
    summary = build_summary(
        inv,
        cand,
        flag_keys,
        nn,
        n_missing,
        n_read_error,
        n_empty,
        index_path,
        snapshot_root,
    )

    # Inventory parquet — one row per candidate well.
    inv_out = inv.copy()
    inv_path = out_dir / "well_inventory.parquet"
    inv_out.to_parquet(inv_path, index=False)
    log.info("Wrote %s (%d rows, %d cols)", inv_path, len(inv_out), inv_out.shape[1])

    with open(out_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    write_html_report(summary, out_dir / "audit_report.html")
    log.info(
        "Wrote %s and %s",
        out_dir / "audit_summary.json",
        out_dir / "audit_report.html",
    )

    # Console recap.
    print("\n=== GWX hydrograph audit (availability=oracle) ===")
    print(f"candidates              {summary['n_candidates']:,}")
    print(f"observations read       {summary['total_observations_read']:,}")
    print(f"file_status             {summary['file_status_counts']}")
    print(f"role_counts             {summary['role_counts']}")
    print(
        f"pass 24obs&10yr         {summary['support_thresholds']['n_pass_24obs_10yr']:,}"
    )
    print(
        f"pass 60obs&10yr         {summary['support_thresholds']['n_pass_60obs_10yr']:,}"
    )
    print(
        f"NGWMN-eligible month    "
        f"{summary['support_thresholds']['n_ngwmn_eligible_any_month']:,}"
    )
    print(
        f"is_static true frac     {summary['is_static_column']['frac_is_static_true']:.5f}"
    )
    print(
        f"NN overall (m)          {summary['nearest_neighbour_spacing_m']['overall']}"
    )
    print("LATENCY/CADENCE: DEFERRED (unavailable_pending_contract)")


if __name__ == "__main__":
    sys.exit(main())
