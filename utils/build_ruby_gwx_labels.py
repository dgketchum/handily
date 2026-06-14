"""Phase 4 of the Ruby HUC8 pilot: GWX observation package.

Builds a calibration/validation well package for the Ruby FAC10 prior from the
GWX ``current`` build. Three artifacts under ``evidence/gwx/``:

  * ``ruby_wells_raw.parquet``           — every GWX index row in Ruby + 20 km
  * ``ruby_wells_canonical.parquet``     — one row per ``canonical_id`` (co-located
    and cross-source duplicates collapsed) with merged metadata
  * ``ruby_well_observation_labels.parquet`` — canonical well + DTW labels, tier,
    and weight, self-contained for Phase 6 scoring

Why each piece matters
----------------------
* **Canonical collapse.** ~7.5 k index rows in Ruby+20 km map to ~4.9 k
  ``canonical_id`` clusters; co-located / cross-source duplicates must not become
  independent calibration points. We pool every member's per-well time series and
  derive one label per cluster.
* **GWAAMON override.** The index has no explicit GWAAMON network flag and no
  precomputed trend column. Ruby-specific rule: an ``mt_gwic`` well is treated as
  monitoring when it has a multi-point series (``has_time_series``) or
  ``well_use == 'monitoring'`` — GWAAMON is a monitoring time-series source even
  when the generic classifier leaves it ``unknown``.
* **Stale ``file_path``.** Index ``file_path`` points at the dead
  ``products/staging/wells/`` tree; basenames resolve under ``current/wells/``.
  We remap by basename and report how many fail to resolve (never silently drop).
* **Trends.** No ``trend_slope`` column exists, so a Theil-Sen slope is fit per
  well; a trend-adjusted 2022 label is emitted only when the slope CI excludes 0.

Tiering (see ``classify_well``) — diagnostics are excluded from FAC tuning:

  diagnostic : no usable DTW; confined/likely_confined/artesian; deep screen
               (non-phreatic); pumping; or poor location (plss/unknown)
  primary    : good location, not confined/deep/pumping, with a real water-table
               label — monitoring time-series, unconfined, or shallow-unknown
  secondary  : good location static wells with unknown confinement (lower weight)

Usage:
    uv run python utils/build_ruby_gwx_labels.py \
        --out-root /data/ssd2/handily/mt/regional/ruby_huc8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("build_ruby_gwx_labels")

WELLS_INDEX = "/data/ssd2/gwx/products/current/wells.geoparquet"
WELLS_DIR = "/data/ssd2/gwx/products/current/wells"

INDEX_COLUMNS = [
    "source",
    "source_well_id",
    "canonical_id",
    "well_class",
    "class_confidence",
    "well_use",
    "h_accuracy_class",
    "confinement_class",
    "confinement_source",
    "well_depth",
    "screen_top",
    "screen_bottom",
    "casing_depth",
    "mean_dtw",
    "obs_count",
    "por_start",
    "por_end",
    "has_time_series",
    "is_static_only",
    "is_active",
    "aquifer",
    "file_path",
    "geometry",
]

# Horizontal accuracy ranking; "good" location for 10 m FAC tuning is map-grade
# or better. plss / unknown cannot be placed on a 10 m grid -> diagnostic.
ACCURACY_RANK = {"survey_gps": 4, "gps": 3, "map": 2, "plss": 1, "unknown": 0}
GOOD_ACCURACY = {"survey_gps", "gps", "map"}
CONFINED = {"confined", "likely_confined", "artesian"}
UNCONFINED = {"unconfined", "unconfined_marginal"}

# Construction / sanity thresholds (metres).
SHALLOW_M = 30.0  # unknown-confinement wells this shallow read as phreatic
DEEP_SCREEN_M = 60.0  # screen top below this -> measurement not a water table
DEEP_WATER_M = 60.0  # DTW deeper than this -> not the phreatic valley table
ARTESIAN_M = -1.0  # DTW below this -> water above surface (artesian/confined)
WELL_DEPTH_SANE_M = 5000.0  # above this the depth field is a data error -> ignore
SANE_DTW = (-10.0, 500.0)  # plausible DTW range; outside -> suspect

# Tier weights for the Phase 7 weighted objective.
W_MONITORING = 1.0  # monitoring time-series, best evidence
W_PRIMARY = 0.6  # unconfined / shallow-unknown / time-series, good location
W_SECONDARY = 0.25  # static, unknown confinement, good location
W_DIAGNOSTIC = 0.0  # excluded from tuning

SPRING_MONTHS = {4, 5, 6}  # align with S2 spring window (Apr 1 - Jun 15)
FALL_MONTHS = {9, 10}  # align with S2 fall window (Sep 1 - Oct 31)


# --------------------------------------------------------------------------- #
# Spatial query
# --------------------------------------------------------------------------- #
def load_ruby_wells(index_path: str, buffer_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """All GWX index rows whose point falls inside the Ruby+buffer polygon."""
    log.info("Reading GWX index: %s", index_path)
    wells = gpd.read_parquet(index_path, columns=INDEX_COLUMNS)
    poly4326 = buffer_gdf.to_crs(4326).union_all()
    minx, miny, maxx, maxy = poly4326.bounds
    box = wells.cx[minx:maxx, miny:maxy]
    inside = box[box.within(poly4326)].copy().reset_index(drop=True)
    log.info("  %d wells in Ruby+buffer (from %d in bbox)", len(inside), len(box))
    return inside


def remap_well_path(file_path: str | None) -> str | None:
    """Map a (possibly stale staging) index ``file_path`` to the current tree."""
    if not isinstance(file_path, str) or not file_path:
        return None
    cur = os.path.join(WELLS_DIR, os.path.basename(file_path))
    return cur if os.path.exists(cur) else None


# --------------------------------------------------------------------------- #
# Per-well time series -> DTW labels
# --------------------------------------------------------------------------- #
def read_well_series(path: str) -> pd.DataFrame | None:
    cols = ["dtime", "dtw", "method", "is_static"]
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception as exc:  # parquet present but unreadable — surface, don't hide
        log.warning("unreadable per-well parquet %s: %s", path, exc)
        return None
    df = df[np.isfinite(df["dtw"])]
    return df if len(df) else None


def _median(vals: pd.Series) -> float | None:
    vals = vals[np.isfinite(vals)]
    return float(vals.median()) if len(vals) else None


def labels_from_series(series: pd.DataFrame) -> dict:
    """Seasonal / era median DTW labels and a Theil-Sen trend from one well's
    pooled time series. All medians are over finite DTW only."""
    s = series.dropna(subset=["dtw"]).copy()
    if "dtime" in s:
        s["dtime"] = pd.to_datetime(s["dtime"], utc=True, errors="coerce")
    dated = s.dropna(subset=["dtime"])
    has_dates = len(dated) > 0
    year = dated["dtime"].dt.year if has_dates else pd.Series(dtype="int64")
    month = dated["dtime"].dt.month if has_dates else pd.Series(dtype="int64")
    n_years = int(year.nunique()) if has_dates else 0

    out: dict = {
        "n_obs_used": int(len(s)),
        "n_years": n_years,
        "date_min": dated["dtime"].min().date().isoformat() if has_dates else None,
        "date_max": dated["dtime"].max().date().isoformat() if has_dates else None,
        "dtw_all_median": _median(s["dtw"]),
        "dtw_2019_2024_median": (
            _median(dated.loc[year.between(2019, 2024), "dtw"]) if has_dates else None
        ),
        "dtw_spring_median": (
            _median(dated.loc[month.isin(SPRING_MONTHS), "dtw"]) if has_dates else None
        ),
        "dtw_fall_median": (
            _median(dated.loc[month.isin(FALL_MONTHS), "dtw"]) if has_dates else None
        ),
        "dtw_latest5_median": None,
        "dtw_trend_slope_m_yr": None,
        "dtw_trend_adj_2022_m": None,
        "is_timeseries": n_years >= 2,
        "latest_method": None,
        "latest_dtw": None,
        "driller_dtw": None,
    }

    if has_dates:
        ymax = int(year.max())
        out["dtw_latest5_median"] = _median(dated.loc[year >= ymax - 4, "dtw"])
        latest = dated.sort_values("dtime").iloc[-1]
        out["latest_method"] = latest.get("method")
        out["latest_dtw"] = float(latest["dtw"])

    drill = s.loc[s["method"] == "driller_report", "dtw"]
    if len(drill):
        out["driller_dtw"] = float(drill.iloc[0])

    # Theil-Sen trend (robust); significant only when the CI excludes zero and
    # the record spans enough distinct years to be meaningful.
    if has_dates and n_years >= 5:
        dec_yr = (year + (dated["dtime"].dt.dayofyear - 1) / 365.25).to_numpy()
        dtw = dated["dtw"].to_numpy()
        slope, intercept, lo, hi = stats.theilslopes(dtw, dec_yr)
        out["dtw_trend_slope_m_yr"] = float(slope)
        if (lo > 0 and hi > 0) or (lo < 0 and hi < 0):
            out["dtw_trend_adj_2022_m"] = float(intercept + slope * 2022.0)
    return out


def pick_label(meta: dict, lab: dict) -> dict:
    """Choose the single DTW label and record its type / period / suspect flag."""
    suspect = False
    if lab["is_timeseries"]:
        if lab["dtw_2019_2024_median"] is not None:
            value, ltype, period = (
                lab["dtw_2019_2024_median"],
                "ts_2019_2024_median",
                "2019-2024",
            )
        elif lab["dtw_latest5_median"] is not None:
            value, ltype, period = (
                lab["dtw_latest5_median"],
                "ts_latest5_median",
                "latest5",
            )
        else:
            value, ltype, period = lab["dtw_all_median"], "ts_all_median", "all"
    else:
        if lab["driller_dtw"] is not None and (
            meta.get("is_static_only") or lab["n_years"] <= 1
        ):
            value, ltype, period = (
                lab["driller_dtw"],
                "static_driller_report",
                lab["date_min"],
            )
        elif lab["latest_dtw"] is not None:
            value, ltype, period = lab["latest_dtw"], "static_manual", lab["date_max"]
        else:
            value, ltype, period = (
                lab["dtw_all_median"],
                "static_single",
                lab["date_min"],
            )
            if lab["date_min"] is None:
                suspect = True  # undated static value
    if (
        value is None
        or not np.isfinite(value)
        or not (SANE_DTW[0] <= value <= SANE_DTW[1])
    ):
        suspect = True
    return {
        "dtw_label_m": value,
        "observation_label_type": ltype,
        "period_represented": period,
        "suspect": suspect,
    }


# --------------------------------------------------------------------------- #
# Canonical collapse + classification
# --------------------------------------------------------------------------- #
def _first_valid(series: pd.Series):
    vals = series.dropna()
    vals = vals[vals != ""] if vals.dtype == object else vals
    return vals.iloc[0] if len(vals) else None


def _override_class(group: pd.DataFrame) -> str | None:
    """Ruby monitoring override: mt_gwic GWAAMON-style or any monitoring tag."""
    is_mt = (group["source"] == "mt_gwic").any()
    has_ts = bool(group["has_time_series"].any())
    is_monitor = (group["well_use"] == "monitoring").any() or (
        group["well_class"] == "monitoring"
    ).any()
    if (is_mt and (has_ts or is_monitor)) or is_monitor:
        return "monitoring"
    classes = group["well_class"].dropna()
    return str(classes.mode().iloc[0]) if len(classes) else None


def classify_well(meta: dict, label: dict) -> dict:
    """Assign tier, weight, and a human-readable reason."""
    acc = meta.get("h_accuracy_class")
    conf = meta.get("confinement_class")
    override = meta.get("ruby_override_class")
    depth = meta.get("well_depth")
    screen_top = meta.get("screen_top")

    value = label["dtw_label_m"]
    good_loc = acc in GOOD_ACCURACY
    confined = conf in CONFINED
    unconfined = conf in UNCONFINED
    monitoring = override == "monitoring"
    pumping = override == "pumping"
    depth_known = (
        depth is not None and np.isfinite(depth) and 0 < depth <= WELL_DEPTH_SANE_M
    )
    shallow = depth_known and depth <= SHALLOW_M
    deep_screen = (
        screen_top is not None
        and np.isfinite(screen_top)
        and screen_top > DEEP_SCREEN_M
    )
    has_label = value is not None and not label["suspect"]

    def diag(reason: str) -> dict:
        return {
            "tier": "diagnostic",
            "weight": W_DIAGNOSTIC,
            "exclusion_reason": reason,
        }

    if not has_label:
        return diag("no_usable_dtw")
    if confined:
        return diag("confined_aquifer")
    # Physical-plausibility guards: keep only readings that are the phreatic
    # valley water table the FAC prior models (these wells are retained as
    # diagnostics for the Phase 6 per-tier / per-setting comparison).
    if value < ARTESIAN_M:
        return diag("artesian_above_surface")
    if depth_known and value > depth:
        return diag("dtw_below_well_bottom")
    if value > DEEP_WATER_M:
        return diag("deep_water_nonphreatic")
    if deep_screen:
        return diag("deep_screen_nonphreatic")
    if pumping:
        return diag("pumping_well")
    if not good_loc:
        return diag("poor_location")

    if monitoring:
        w = W_MONITORING if meta.get("has_time_series") else W_PRIMARY
        return {"tier": "primary", "weight": w, "exclusion_reason": ""}
    if (
        unconfined
        or (conf in (None, "", "unknown") and shallow)
        or meta.get("has_time_series")
    ):
        return {"tier": "primary", "weight": W_PRIMARY, "exclusion_reason": ""}
    return {"tier": "secondary", "weight": W_SECONDARY, "exclusion_reason": ""}


def collapse_and_label(raw: gpd.GeoDataFrame, workers: int) -> gpd.GeoDataFrame:
    raw = raw.copy()
    raw["_acc_rank"] = raw["h_accuracy_class"].map(ACCURACY_RANK).fillna(0)
    raw["_cur_path"] = raw["file_path"].map(remap_well_path)

    n_paths = int(raw["_cur_path"].notna().sum())
    log.info(
        "Reading %d per-well series (%d index rows lacked a current parquet)",
        n_paths,
        len(raw) - n_paths,
    )
    paths = [p for p in raw["_cur_path"].dropna().unique()]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        series_by_path = dict(zip(paths, ex.map(read_well_series, paths)))

    records = []
    for cid, group in raw.groupby("canonical_id", sort=False):
        group = group.sort_values("_acc_rank", ascending=False)
        rep = group.iloc[0]

        frames = [
            series_by_path[p]
            for p in group["_cur_path"].dropna().unique()
            if series_by_path.get(p) is not None
        ]
        pooled = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=["dtime", "dtw", "method", "is_static"])
        )
        lab = labels_from_series(pooled)
        if lab["n_obs_used"] == 0 and np.isfinite(rep.get("mean_dtw", np.nan)):
            # No readable per-well series; fall back to the index mean (same data,
            # pre-aggregated) and flag it as such rather than dropping the well.
            lab["dtw_all_median"] = float(rep["mean_dtw"])

        meta = {
            "canonical_id": cid,
            "source_ids": ";".join(
                sorted(
                    f"{s}:{i}" for s, i in zip(group["source"], group["source_well_id"])
                )
            ),
            "n_members": int(len(group)),
            "sources": ";".join(sorted(group["source"].unique())),
            "h_accuracy_class": rep["h_accuracy_class"],
            "well_class": _first_valid(group["well_class"]),
            "class_confidence": _first_valid(group["class_confidence"]),
            "well_use": _first_valid(group["well_use"]),
            "ruby_override_class": _override_class(group),
            "confinement_class": rep["confinement_class"],
            "confinement_source": rep["confinement_source"],
            "well_depth": _first_valid(group["well_depth"]),
            "screen_top": _first_valid(group["screen_top"]),
            "screen_bottom": _first_valid(group["screen_bottom"]),
            "casing_depth": _first_valid(group["casing_depth"]),
            "aquifer": _first_valid(group["aquifer"]),
            "has_time_series": bool(group["has_time_series"].any()),
            "is_static_only": bool(group["is_static_only"].all()),
            "mean_dtw": _first_valid(group["mean_dtw"]),
            "geometry": rep.geometry,
        }
        label = pick_label(meta, lab)
        if label["observation_label_type"] == "static_single" and not frames:
            label["observation_label_type"] = "index_mean_dtw"
        tier = classify_well(meta, label)
        records.append({**meta, **lab, **label, **tier})

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=raw.crs)
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", default="/data/ssd2/handily/mt/regional/ruby_huc8")
    p.add_argument("--index", default=WELLS_INDEX)
    p.add_argument(
        "--buffer-fgb",
        default=None,
        help="defaults to <out-root>/boundary/ruby_huc8_buffer_20km.fgb",
    )
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    out_root = Path(args.out_root)
    gwx_dir = out_root / "evidence" / "gwx"
    gwx_dir.mkdir(parents=True, exist_ok=True)
    buffer_fgb = args.buffer_fgb or str(
        out_root / "boundary" / "ruby_huc8_buffer_20km.fgb"
    )

    buffer_gdf = gpd.read_file(buffer_fgb)
    raw = load_ruby_wells(args.index, buffer_gdf)
    raw_path = gwx_dir / "ruby_wells_raw.parquet"
    raw.to_parquet(raw_path)
    log.info("Wrote %s (%d rows)", raw_path, len(raw))

    labels = collapse_and_label(raw, workers=args.workers)

    canonical_cols = [
        "canonical_id",
        "source_ids",
        "n_members",
        "sources",
        "h_accuracy_class",
        "well_class",
        "class_confidence",
        "well_use",
        "ruby_override_class",
        "confinement_class",
        "confinement_source",
        "well_depth",
        "screen_top",
        "screen_bottom",
        "casing_depth",
        "aquifer",
        "has_time_series",
        "is_static_only",
        "mean_dtw",
        "geometry",
    ]
    canonical = labels[canonical_cols].copy()
    canonical_path = gwx_dir / "ruby_wells_canonical.parquet"
    canonical.to_parquet(canonical_path)
    labels_path = gwx_dir / "ruby_well_observation_labels.parquet"
    labels.to_parquet(labels_path)

    tier_counts = labels["tier"].value_counts().to_dict()
    reason_counts = (
        labels.loc[labels["tier"] == "diagnostic", "exclusion_reason"]
        .value_counts()
        .to_dict()
    )
    primary = labels[labels["tier"] == "primary"]
    note = {
        "phase": "4_gwx_observation_package",
        "date": date.today().isoformat(),
        "index": args.index,
        "buffer_fgb": buffer_fgb,
        "raw_well_rows": int(len(raw)),
        "canonical_wells": int(len(labels)),
        "tier_counts": {k: int(v) for k, v in tier_counts.items()},
        "diagnostic_reason_counts": {k: int(v) for k, v in reason_counts.items()},
        "monitoring_override_wells": int(
            (labels["ruby_override_class"] == "monitoring").sum()
        ),
        "timeseries_wells": int(labels["is_timeseries"].sum()),
        "primary_dtw_median_m": (
            float(primary["dtw_label_m"].median()) if len(primary) else None
        ),
        "suspect_wells": int(labels["suspect"].sum()),
        "outputs": {
            "raw": str(raw_path),
            "canonical": str(canonical_path),
            "labels": str(labels_path),
        },
    }
    note_path = out_root / "run_notes" / "gwx_observation_package.json"
    with open(note_path, "w") as f:
        json.dump(note, f, indent=2)
    log.info("Wrote %s and %s", canonical_path, labels_path)
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
