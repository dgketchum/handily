"""Phase 6 of the Ruby HUC8 pilot: FAC vs Ma benchmark comparison.

Compares the Ruby FAC10 DTW prior against the Ma et al. national WTD product
(``/nas/gwx/wtd_states/wtd_montana.tif``) as both a raster surface and against
the held-out GWX well + NHD spring observations. Ma is a benchmark to beat, never
a tuning target.

Runs with whichever rasters are supplied:
  * ``--ma`` alone  -> Ma baseline point metrics (wells + springs). Useful before
    the FAC prior exists.
  * ``--fac-rem`` + ``--ma`` -> full comparison: Ma resampled to the FAC grid,
    difference rasters, shallow-agreement classes, and side-by-side point metrics.

Point metrics (per raster):
  * wells: median residual, median absolute residual, RMSE, bias, broken out by
    tier, confinement class, and (if ``--streams`` given) valley/upland setting;
    shallow-class precision/recall at <2 / <5 / <10 m.
  * springs: exact + buffered-minimum residual, capture rate at 30/60/100 m,
    miss fraction above 2/5/10 m (via score_spring_anchors).

Residual sign convention: ``residual = predicted_dtw - observed_dtw`` (positive =
the raster is too deep). Only primary+secondary wells feed the headline metrics;
diagnostic wells are reported separately so confined/deep failures stay visible.

Usage:
    uv run python utils/compare_fac_ma.py \
        --fac-rem .../rem/ruby_fac10_baseline/fac_head_depth_rem_10m.tif \
        --ma /nas/gwx/wtd_states/wtd_montana.tif \
        --wells .../evidence/gwx/ruby_well_observation_labels.parquet \
        --springs .../evidence/springs/nhd_springs_45800.fgb \
        --streams .../streams_regional.fgb \
        --out-dir .../validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray  # noqa: F401  (registers .rio accessor)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_spring_anchors import score_springs  # noqa: E402

log = logging.getLogger("compare_fac_ma")

MA_DEFAULT = "/nas/gwx/wtd_states/wtd_montana.tif"
SHALLOW_LEVELS = (2.0, 5.0, 10.0)
VALLEY_DIST_M = 500.0  # well within this of a mapped stream -> valley setting


def sample_raster_at_points(raster_path: str, gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Sample band 1 at each point; nodata / non-finite / out-of-bounds -> nan.

    ``rasterio.sample`` returns 0.0 (not nodata) for points outside a raster
    whose nodata is None — which the FAC REM is — so out-of-grid wells would
    otherwise read as a spurious 0 m (surface water) and contaminate the score.
    Points outside the raster bounds are masked to NaN explicitly.
    """
    with rasterio.open(raster_path) as src:
        pts = gdf.to_crs(src.crs)
        xs = np.array([g.x for g in pts.geometry], dtype="float64")
        ys = np.array([g.y for g in pts.geometry], dtype="float64")
        vals = np.array(
            [v[0] for v in src.sample(zip(xs, ys), indexes=1)], dtype="float64"
        )
        nod = src.nodata
        b = src.bounds
    vals[(xs < b.left) | (xs > b.right) | (ys < b.bottom) | (ys > b.top)] = np.nan
    if nod is not None and np.isfinite(nod):
        vals[vals == nod] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    return vals


def _resid_stats(resid: np.ndarray) -> dict:
    r = resid[np.isfinite(resid)]
    if r.size == 0:
        return {
            "n": 0,
            "median_residual_m": None,
            "median_abs_residual_m": None,
            "rmse_m": None,
            "bias_m": None,
        }
    return {
        "n": int(r.size),
        "median_residual_m": float(np.median(r)),
        "median_abs_residual_m": float(np.median(np.abs(r))),
        "rmse_m": float(np.sqrt(np.mean(r**2))),
        "bias_m": float(np.mean(r)),
    }


def valley_upland_flag(
    wells: gpd.GeoDataFrame, streams: gpd.GeoDataFrame, dist_m: float
) -> pd.Series:
    """valley if the well is within ``dist_m`` of a mapped stream, else upland.

    Independent of the prediction (uses the FAC stream network, not the REM)."""
    w = wells.to_crs(5070)
    s = streams.to_crs(5070)
    nearest = gpd.sjoin_nearest(w[["geometry"]], s[["geometry"]], distance_col="_d")
    nearest = nearest[~nearest.index.duplicated(keep="first")]
    d = nearest["_d"].reindex(wells.index)
    return np.where(d <= dist_m, "valley", "upland")


def well_metrics(
    wells: gpd.GeoDataFrame,
    raster_path: str,
    label: str,
    streams: gpd.GeoDataFrame | None,
) -> tuple[list[dict], pd.Series]:
    observed = wells["dtw_label_m"].to_numpy(dtype="float64")
    predicted = sample_raster_at_points(raster_path, wells)
    residual = predicted - observed
    rows: list[dict] = []

    def emit(group_type: str, group: str, mask: np.ndarray) -> None:
        st = _resid_stats(residual[mask])
        for stat, val in st.items():
            rows.append(
                {
                    "raster": label,
                    "target": "wells",
                    "group_type": group_type,
                    "group": group,
                    "statistic": stat,
                    "value": val,
                }
            )

    usable = wells["tier"].isin(["primary", "secondary"]).to_numpy()
    emit("tier", "primary", (wells["tier"] == "primary").to_numpy())
    emit("tier", "secondary", (wells["tier"] == "secondary").to_numpy())
    emit("tier", "primary+secondary", usable)
    emit("tier", "diagnostic", (wells["tier"] == "diagnostic").to_numpy())
    emit("tier", "all", np.ones(len(wells), dtype=bool))

    for conf in sorted(wells.loc[usable, "confinement_class"].dropna().unique()):
        emit(
            "confinement",
            str(conf),
            usable & (wells["confinement_class"] == conf).to_numpy(),
        )

    if streams is not None:
        setting = valley_upland_flag(wells, streams, VALLEY_DIST_M)
        for s in ("valley", "upland"):
            emit("setting", s, usable & (setting == s))

    # Shallow-class precision/recall on usable wells.
    obs_u, pred_u = observed[usable], predicted[usable]
    valid = np.isfinite(obs_u) & np.isfinite(pred_u)
    obs_u, pred_u = obs_u[valid], pred_u[valid]
    for lvl in SHALLOW_LEVELS:
        obs_sh, pred_sh = obs_u < lvl, pred_u < lvl
        tp = int((obs_sh & pred_sh).sum())
        prec = tp / int(pred_sh.sum()) if pred_sh.sum() else None
        rec = tp / int(obs_sh.sum()) if obs_sh.sum() else None
        for stat, val in {
            "precision": prec,
            "recall": rec,
            "n_observed_shallow": int(obs_sh.sum()),
        }.items():
            rows.append(
                {
                    "raster": label,
                    "target": "wells",
                    "group_type": "shallow_class",
                    "group": f"<{int(lvl)}m",
                    "statistic": stat,
                    "value": val,
                }
            )

    return rows, pd.Series(residual, index=wells.index, name=f"residual_{label}")


def spring_rows(label: str, summary: dict) -> list[dict]:
    rows = []
    for r, v in summary["capture_rate"].items():
        rows.append(
            {
                "raster": label,
                "target": "springs",
                "group_type": "capture",
                "group": r,
                "statistic": "capture_rate",
                "value": v,
            }
        )
    for lvl, v in summary["miss_fraction_above"].items():
        rows.append(
            {
                "raster": label,
                "target": "springs",
                "group_type": "miss",
                "group": f">{lvl}",
                "statistic": "miss_fraction",
                "value": v,
            }
        )
    rows.append(
        {
            "raster": label,
            "target": "springs",
            "group_type": "residual",
            "group": "all",
            "statistic": "median_residual_m",
            "value": summary["median_residual_m"],
        }
    )
    return rows


def raster_diffs(fac_rem: str, ma: str, out_dir: Path) -> dict:
    """Resample Ma to the FAC grid and write difference + shallow-agreement rasters."""
    fac = rioxarray.open_rasterio(fac_rem).squeeze("band", drop=True)
    ma_da = rioxarray.open_rasterio(ma).squeeze("band", drop=True)
    ma_on = ma_da.rio.reproject_match(fac)
    ma_on_path = out_dir / "ma_wtd_on_fac_grid.tif"
    ma_on.rio.to_raster(ma_on_path, compress="deflate")

    fac_v = fac.where(np.isfinite(fac))
    ma_v = ma_on.where(np.isfinite(ma_on) & (ma_on < 1e30))
    diff = fac_v - ma_v
    diff.rio.to_raster(out_dir / "fac_minus_ma_wtd.tif", compress="deflate")
    abs(diff).rio.to_raster(
        out_dir / "absolute_difference_fac_ma.tif", compress="deflate"
    )

    for lvl in SHALLOW_LEVELS:
        # 0 neither, 1 fac-only, 2 ma-only, 3 both shallow; 255 = nodata.
        cls = (fac_v < lvl).astype("uint8") + 2 * (ma_v < lvl).astype("uint8")
        cls = cls.where(np.isfinite(fac_v) & np.isfinite(ma_v), 255).astype("uint8")
        # Drop the float nodata (3.4e38) inherited from the FAC grid — it can't be
        # written into a uint8 band — and tag 255 as nodata instead.
        cls = cls.rio.write_nodata(255)
        cls.rio.to_raster(
            out_dir / f"shallow_agreement_{int(lvl)}m.tif", compress="deflate"
        )
    d = diff.values
    d = d[np.isfinite(d)]
    return {
        "ma_on_fac_grid": str(ma_on_path),
        "fac_minus_ma_median_m": float(np.median(d)) if d.size else None,
        "fac_minus_ma_mean_m": float(np.mean(d)) if d.size else None,
        "abs_diff_median_m": float(np.median(np.abs(d))) if d.size else None,
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fac-rem", default=None, help="FAC REM DTW raster (optional)")
    p.add_argument("--ma", default=MA_DEFAULT, help="Ma WTD raster")
    p.add_argument(
        "--wells", required=True, help="ruby_well_observation_labels.parquet"
    )
    p.add_argument("--springs", required=True, help="nhd_springs_45800.fgb")
    p.add_argument(
        "--streams", default=None, help="streams_regional.fgb (valley/upland)"
    )
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wells = gpd.read_parquet(args.wells)
    springs = gpd.read_file(args.springs)
    streams = gpd.read_file(args.streams) if args.streams else None

    rasters = {"ma": args.ma}
    if args.fac_rem:
        rasters = {"fac": args.fac_rem, **rasters}

    # When a FAC REM is given, score FAC and Ma on the SAME footprint: drop wells
    # outside the FAC grid (NaN pred via sample_raster_at_points). Ma is statewide
    # and finite everywhere, so without this FAC scores on its in-grid subset
    # while Ma scores on all wells — an apples-to-oranges comparison.
    if args.fac_rem:
        in_grid = np.isfinite(sample_raster_at_points(args.fac_rem, wells))
        n_drop = int((~in_grid).sum())
        if n_drop:
            log.info(
                "Dropping %d / %d wells outside the FAC REM grid; scoring FAC+Ma "
                "on the %d in-grid wells",
                n_drop,
                len(wells),
                int(in_grid.sum()),
            )
        wells = wells.loc[in_grid].reset_index(drop=True)

    all_rows: list[dict] = []
    well_resids = wells[
        [
            "canonical_id",
            "tier",
            "weight",
            "confinement_class",
            "dtw_label_m",
            "geometry",
        ]
    ].copy()
    spring_out = None
    for label, path in rasters.items():
        log.info("Scoring %s wells + springs: %s", label, path)
        rows, resid = well_metrics(wells, path, label, streams)
        all_rows += rows
        well_resids[f"pred_{label}"] = sample_raster_at_points(path, wells)
        well_resids[f"residual_{label}"] = resid
        s_out, s_sum = score_springs(path, springs)
        all_rows += spring_rows(label, s_sum)
        cols = {c: f"{c}_{label}" for c in ("exact_dtw", "residual_m", "captured")}
        s_keep = s_out.rename(columns=cols)[list(cols.values()) + ["geometry"]]
        spring_out = (
            s_keep
            if spring_out is None
            else spring_out.join(s_keep.drop(columns="geometry"))
        )

    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "score_summary.csv"
    summary.to_csv(summary_path, index=False)
    well_resids.to_file(out_dir / "well_residuals.fgb", driver="FlatGeobuf")
    spring_out.to_file(out_dir / "spring_residuals.fgb", driver="FlatGeobuf")

    # Raster diff products are optional and heavier; a write failure here must
    # never discard the point metrics already persisted above.
    diffs = {}
    if args.fac_rem:
        log.info("Building raster difference products")
        try:
            diffs = raster_diffs(args.fac_rem, args.ma, out_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("raster_diffs failed: %s", e)

    note = {
        "phase": "6_ma_benchmark_comparison",
        "rasters": rasters,
        "streams_for_setting": args.streams,
        "raster_diffs": diffs,
        "headline": _headline(summary),
        "outputs": {
            "score_summary": str(summary_path),
            "well_residuals": str(out_dir / "well_residuals.fgb"),
            "spring_residuals": str(out_dir / "spring_residuals.fgb"),
            **diffs,
        },
    }
    with open(out_dir / "comparison_run.json", "w") as f:
        json.dump(note, f, indent=2)
    log.info("Wrote %s", summary_path)
    print(json.dumps(note["headline"], indent=2))


def _headline(summary: pd.DataFrame) -> dict:
    """Pull the central FAC-vs-Ma numbers from the long summary table."""
    out: dict = {}
    for label in summary["raster"].unique():
        sl = summary[summary["raster"] == label]

        def get(gt, g, st):
            m = sl[(sl.group_type == gt) & (sl.group == g) & (sl.statistic == st)]
            return None if m.empty else m["value"].iloc[0]

        out[label] = {
            "primary_well_mad_m": get("tier", "primary", "median_abs_residual_m"),
            "primary_well_bias_m": get("tier", "primary", "bias_m"),
            "spring_capture_60m": get("capture", "60m", "capture_rate"),
            "shallow_recall_<5m": get("shallow_class", "<5m", "recall"),
            "shallow_precision_<5m": get("shallow_class", "<5m", "precision"),
        }
    return out


if __name__ == "__main__":
    main()
