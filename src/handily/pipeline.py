from __future__ import annotations

import dataclasses
import json
import os
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor

from . import compute, dem, io, nhd as nhd_mod
from .config import HandilyConfig

LOGGER = logging.getLogger("handily.pipeline")


@dataclass
class REMWorkflow:
    config: HandilyConfig
    aoi: Any

    def __post_init__(self) -> None:
        if isinstance(self.aoi, gpd.GeoDataFrame):
            self.aoi_gdf = self.aoi
        else:
            self.aoi_gdf = gpd.GeoDataFrame([{}], geometry=[self.aoi], crs="EPSG:4326")

        self.bounds_wsen = tuple(self.aoi_gdf.to_crs("EPSG:4326").total_bounds.tolist())
        LOGGER.info("REMWorkflow instantiated (bounds_wsen=%s)", self.bounds_wsen)

        self.flowlines = None
        self.ndwi = None

        self.dem = None
        self.streams = None
        self.rem = None
        self.fields = None
        self.stats = None

    def fetch_vectors(
        self,
        cache_flowlines: bool = False,
        overwrite_flowlines_cache: bool = False,
        flowlines_cache_name: str = "flowlines_bounds.fgb",
    ) -> None:
        io.ensure_dir(self.config.out_dir)
        flowlines_cache_path = os.path.join(self.config.out_dir, flowlines_cache_name)
        if (
            cache_flowlines
            and (not overwrite_flowlines_cache)
            and os.path.exists(flowlines_cache_path)
        ):
            io.LOGGER.info("Loading cached flowlines: %s", flowlines_cache_path)
            self.flowlines = gpd.read_file(flowlines_cache_path)
        else:
            self.flowlines = io.get_flowlines_within_aoi(
                self.aoi_gdf, local_flowlines_dir=self.config.flowlines_local_dir
            )
            if cache_flowlines:
                io.LOGGER.info("Caching flowlines to: %s", flowlines_cache_path)
                self.flowlines.to_file(flowlines_cache_path, driver="FlatGeobuf")

        # Look for NDWI in the per-AOI out_dir first, fall back to ndwi_dir
        ndwi_paths = io.ndwi_files_for_bounds(self.config.out_dir, self.bounds_wsen)
        if not ndwi_paths:
            ndwi_paths = io.ndwi_files_for_bounds(
                self.config.ndwi_dir, self.bounds_wsen
            )
        if not ndwi_paths:
            raise ValueError(
                "No NDWI rasters found intersecting bounds; place NDWI GeoTIFFs covering the AOI in out_dir or ndwi_dir."
            )
        self.ndwi = io.open_ndwi_mosaic_from_paths(ndwi_paths, self.bounds_wsen)

    def fetch_dem(
        self,
        target_crs_epsg: int = 5070,
        overwrite: bool = False,
        stac_collection_id: str = "usgs-3dep-1m-opr",
        cache_name: str = "dem_bounds_1m.tif",
    ) -> None:
        io.ensure_dir(self.config.out_dir)
        cache_path = os.path.join(self.config.out_dir, cache_name)
        self.dem = dem.get_dem_for_aoi_via_stac(
            aoi_gdf=self.aoi_gdf,
            stac_dir=os.path.expanduser(self.config.stac_dir),
            target_crs_epsg=int(target_crs_epsg),
            cache_path=cache_path,
            overwrite=overwrite,
            stac_download_cache_dir=os.path.join(self.config.out_dir, "stac_cache"),
            stac_collection_id=stac_collection_id,
            delete_stac_cache=self.config.delete_stac_cache,
        )

    def compute_rem(
        self,
        ndwi_threshold: float = 0.15,
        stats: tuple[str, ...] = ("mean",),
        flowlines_buffer_m: float | None = None,
    ) -> None:
        if self.dem is None:
            raise RuntimeError("DEM not available; call fetch_dem() first.")
        if self.flowlines is None or self.ndwi is None:
            raise RuntimeError("Vectors not available; call fetch_vectors() first.")

        dem_crs = self.dem.rio.crs
        flowlines_dem = self.flowlines.to_crs(dem_crs)
        # Exclude hillside canal FCODEs from the stream mask to prevent false
        # near-zero REM patches on hillside pixels below canals.
        fcode_col = nhd_mod.get_fcode_column(flowlines_dem)
        if fcode_col and nhd_mod.REM_EXCLUDED_FCODES:
            flowlines_for_mask = flowlines_dem[
                ~flowlines_dem[fcode_col].isin(nhd_mod.REM_EXCLUDED_FCODES)
            ].copy()
        else:
            flowlines_for_mask = flowlines_dem
        flowlines_for_mask = compute.filter_disconnected_flowlines(flowlines_for_mask)
        buf_m = (
            flowlines_buffer_m
            if flowlines_buffer_m is not None
            else self.config.flowlines_buffer_m
        )

        if self.config.rem_method == "anisotropic_frame":
            from . import rem_frame

            annotated = compute.propagate_flowline_confirmation(
                flowlines_for_mask,
                self.dem,
                ndwi_da=self.ndwi,
                ndwi_threshold=float(ndwi_threshold),
                flowlines_buffer_m=buf_m,
                max_hops=self.config.rem_propagate_hops,
            )
            reachable = annotated[annotated["reachable_from_seed"]].copy()
            self.streams = compute.rasterize_confirmed_flowlines(reachable, self.dem)
            result = rem_frame.compute_rem_anisotropic_frame(
                self.dem, self.ndwi, reachable, self.config
            )
            self.rem = result.rem_da
        elif self.config.rem_propagate_mask:
            self.streams = compute.build_network_propagated_mask(
                flowlines_for_mask,
                self.dem,
                ndwi_da=self.ndwi,
                ndwi_threshold=float(ndwi_threshold),
                flowlines_buffer_m=buf_m,
                max_hops=self.config.rem_propagate_hops,
            )
            self.rem = compute.compute_rem_edt_smooth(
                self.dem, self.streams, sigma=self.config.rem_smooth_sigma
            )
        else:
            self.streams = compute.build_streams_mask_from_nhd_ndwi(
                flowlines_for_mask,
                self.dem,
                ndwi_da=self.ndwi,
                ndwi_threshold=float(ndwi_threshold),
                flowlines_buffer_m=flowlines_buffer_m,
            )
            self.rem = compute.compute_rem_edt_smooth(
                self.dem, self.streams, sigma=self.config.rem_smooth_sigma
            )
        self.fields = io.load_and_clip_fields(
            self.config.fields_path, self.aoi_gdf, dem_crs
        )
        self.stats = compute.compute_field_rem_stats(self.fields, self.rem, stats=stats)

    def results(self, ndwi_threshold: float | None = None) -> dict[str, Any]:
        return {
            "aoi": self.aoi_gdf,
            "flowlines": self.flowlines,
            "ndwi": self.ndwi,
            "streams": self.streams,
            "rem": self.rem,
            "dem": self.dem,
            "fields": self.fields,
            "fields_stats": self.stats,
            "summary": {
                "total_fields": None if self.stats is None else len(self.stats),
                "ndwi_threshold": None
                if ndwi_threshold is None
                else float(ndwi_threshold),
            },
        }

    def run(
        self,
        ndwi_threshold: float = 0.15,
        stats: tuple[str, ...] = ("mean",),
        cache_flowlines: bool = False,
        overwrite_flowlines_cache: bool = False,
        flowlines_buffer_m: float | None = None,
    ) -> dict[str, Any]:
        self.fetch_vectors(
            cache_flowlines=cache_flowlines,
            overwrite_flowlines_cache=overwrite_flowlines_cache,
        )
        self.fetch_dem()
        self.compute_rem(
            ndwi_threshold=ndwi_threshold,
            stats=stats,
            flowlines_buffer_m=flowlines_buffer_m,
        )
        return self.results(ndwi_threshold=ndwi_threshold)


def batch_fetch_dem(
    aoi_gdf: gpd.GeoDataFrame,
    config: HandilyConfig,
    out_root: str,
    overwrite: bool = False,
) -> list[dict]:
    """Download and cache DEMs for each AOI without computing REM.

    Per-AOI directory: {out_root}/aoi_{aoi_id:04d}/
    Skip logic: if dem_bounds_1m.tif already exists and overwrite=False, skip.
    Returns list of dicts with keys: aoi_id, status ('done'|'skipped'|'error'), out_dir.
    """
    n = len(aoi_gdf)
    results = []

    for pos, (_, row) in enumerate(aoi_gdf.iterrows(), start=1):
        aoi_id = int(row["aoi_id"]) if "aoi_id" in row.index else pos - 1
        aoi_out_dir = os.path.join(out_root, f"aoi_{aoi_id:04d}")
        dem_path = os.path.join(aoi_out_dir, "dem_bounds_1m.tif")

        if os.path.exists(dem_path) and not overwrite:
            LOGGER.info(
                "Skipping AOI %04d (%d/%d): dem_bounds_1m.tif exists", aoi_id, pos, n
            )
            results.append(
                {"aoi_id": aoi_id, "status": "skipped", "out_dir": aoi_out_dir}
            )
            continue

        aoi_config = dataclasses.replace(config, out_dir=aoi_out_dir)
        try:
            aoi_geom = row.geometry
            if config.rem_aoi_buffer_m > 0:
                aoi_proj = gpd.GeoSeries([aoi_geom], crs="EPSG:4326").to_crs(
                    "EPSG:5070"
                )
                aoi_geom = (
                    aoi_proj.buffer(config.rem_aoi_buffer_m).to_crs("EPSG:4326").iloc[0]
                )
            workflow = REMWorkflow(config=aoi_config, aoi=aoi_geom)
            workflow.fetch_dem()
            LOGGER.info("DEM saved AOI %04d (%d/%d)", aoi_id, pos, n)
            results.append({"aoi_id": aoi_id, "status": "done", "out_dir": aoi_out_dir})
        except Exception as exc:
            LOGGER.error("Failed AOI %04d (%d/%d): %s", aoi_id, pos, n, exc)
            results.append(
                {
                    "aoi_id": aoi_id,
                    "status": "error",
                    "error": str(exc),
                    "out_dir": aoi_out_dir,
                }
            )

    done = sum(1 for r in results if r["status"] == "done")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    LOGGER.info(
        "Batch DEM fetch complete: done=%d skipped=%d errors=%d", done, skipped, errors
    )
    return results


def batch_run_rem(
    aoi_gdf: gpd.GeoDataFrame,
    config: HandilyConfig,
    out_root: str,
    ndwi_threshold: float = 0.15,
    flowlines_buffer_m: float | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """Run REMWorkflow for each AOI in aoi_gdf, writing outputs to per-AOI subdirectories.

    Per-AOI directory: {out_root}/aoi_{aoi_id:04d}/
    Skip logic: if rem_bounds.tif already exists and overwrite=False, skip that AOI.
    Returns list of dicts with keys: aoi_id, status ('done'|'skipped'|'error'), out_dir.
    """
    n = len(aoi_gdf)
    results = []

    for pos, (_, row) in enumerate(aoi_gdf.iterrows(), start=1):
        aoi_id = int(row["aoi_id"]) if "aoi_id" in row.index else pos - 1
        aoi_out_dir = os.path.join(out_root, f"aoi_{aoi_id:04d}")
        rem_path = os.path.join(aoi_out_dir, "rem_bounds.tif")

        if os.path.exists(rem_path) and not overwrite:
            LOGGER.info(
                "Skipping AOI %04d (%d/%d): rem_bounds.tif exists", aoi_id, pos, n
            )
            results.append(
                {"aoi_id": aoi_id, "status": "skipped", "out_dir": aoi_out_dir}
            )
            continue

        aoi_config = dataclasses.replace(config, out_dir=aoi_out_dir)

        try:
            t0 = time.monotonic()
            # Buffer the AOI so the REM extends beyond the tile edge
            aoi_geom = row.geometry
            if config.rem_aoi_buffer_m > 0:
                aoi_proj = gpd.GeoSeries([aoi_geom], crs="EPSG:4326").to_crs(
                    "EPSG:5070"
                )
                aoi_geom = (
                    aoi_proj.buffer(config.rem_aoi_buffer_m).to_crs("EPSG:4326").iloc[0]
                )
            workflow = REMWorkflow(config=aoi_config, aoi=aoi_geom)
            result = workflow.run(
                ndwi_threshold=ndwi_threshold,
                cache_flowlines=True,
                flowlines_buffer_m=flowlines_buffer_m,
            )
            result["rem"].rio.to_raster(rem_path)
            result["streams"].rio.to_raster(
                os.path.join(aoi_out_dir, "streams_bounds.tif")
            )
            fields = result.get("fields_stats")
            if fields is None:
                fields = result["fields"]
            if "FID" in fields.columns:
                fields = fields.rename(columns={"FID": "FID_"})
            fields.to_file(
                os.path.join(aoi_out_dir, "fields_bounds.fgb"), driver="FlatGeobuf"
            )
            elapsed = time.monotonic() - t0
            LOGGER.info("Completed AOI %04d (%d/%d) in %.1fs", aoi_id, pos, n, elapsed)
            results.append({"aoi_id": aoi_id, "status": "done", "out_dir": aoi_out_dir})
        except Exception as exc:
            LOGGER.error("Failed AOI %04d (%d/%d): %s", aoi_id, pos, n, exc)
            results.append(
                {
                    "aoi_id": aoi_id,
                    "status": "error",
                    "error": str(exc),
                    "out_dir": aoi_out_dir,
                }
            )

    done = sum(1 for r in results if r["status"] == "done")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] == "error")
    LOGGER.info("Batch complete: done=%d skipped=%d errors=%d", done, skipped, errors)
    return results


def run_experiment(
    experiment_dir: str,
    config: HandilyConfig,
    notes: str,
    experiment_id: str | None = None,
) -> dict:
    """Run a single REM experiment, saving outputs to an auto-incremented subdirectory.

    Loads shared DEM and flowlines from *experiment_dir*, computes stream mask + REM
    with parameters from *config*, and writes outputs + run.json to E{n}/.
    """
    import rioxarray  # noqa: F811

    # --- resolve experiment ID ---
    if experiment_id is None:
        existing = [
            int(m.group(1))
            for d in os.listdir(experiment_dir)
            if (m := re.match(r"^E(\d+)$", d))
            and os.path.isdir(os.path.join(experiment_dir, d))
        ]
        next_n = max(existing) + 1 if existing else 0
        experiment_id = f"E{next_n}"

    run_dir = os.path.join(experiment_dir, experiment_id)
    os.makedirs(run_dir, exist_ok=True)

    # Set up per-experiment log file
    run_log = logging.FileHandler(os.path.join(run_dir, "experiment.log"))
    run_log.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.getLogger().addHandler(run_log)

    LOGGER.info("Experiment %s → %s", experiment_id, run_dir)

    # --- load shared data ---
    dem_path = os.path.join(experiment_dir, "dem_bounds_1m.tif")
    flow_path = os.path.join(experiment_dir, "flowlines_bounds.fgb")
    dem_da = rioxarray.open_rasterio(dem_path).squeeze("band", drop=True)
    flowlines = gpd.read_file(flow_path)

    # derive AOI bounds from DEM for NDWI lookup and field clipping
    from pyproj import Transformer

    dem_crs = dem_da.rio.crs
    xmin, ymin, xmax, ymax = dem_da.rio.bounds()
    transformer = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(xmin, ymin)
    lon_max, lat_max = transformer.transform(xmax, ymax)
    bounds_wsen = (lon_min, lat_min, lon_max, lat_max)

    from shapely.geometry import box

    aoi_geom_4326 = box(lon_min, lat_min, lon_max, lat_max)
    aoi_gdf = gpd.GeoDataFrame([{}], geometry=[aoi_geom_4326], crs="EPSG:4326")

    ndwi_paths = io.ndwi_files_for_bounds(experiment_dir, bounds_wsen)
    if not ndwi_paths:
        ndwi_paths = io.ndwi_files_for_bounds(config.ndwi_dir, bounds_wsen)
    if not ndwi_paths:
        raise ValueError("No NDWI rasters found for experiment bounds")
    ndwi_da = io.open_ndwi_mosaic_from_paths(ndwi_paths, bounds_wsen)

    # --- resolve excluded FCODEs ---
    if config.rem_excluded_fcodes is not None:
        excluded = frozenset(config.rem_excluded_fcodes)
    else:
        excluded = nhd_mod.REM_EXCLUDED_FCODES

    # --- filter flowlines ---
    flowlines_dem = flowlines.to_crs(dem_crs)
    fcode_col = nhd_mod.get_fcode_column(flowlines_dem)
    if fcode_col and excluded:
        flowlines_for_mask = flowlines_dem[
            ~flowlines_dem[fcode_col].isin(excluded)
        ].copy()
    else:
        flowlines_for_mask = flowlines_dem

    # Drop isolated pond/lake-crossing 55800 segments (keep main river chains)
    flowlines_for_mask = compute.filter_disconnected_flowlines(flowlines_for_mask)

    # --- compute ---
    t0 = time.monotonic()

    if config.rem_method == "anisotropic_frame":
        from . import rem_frame

        annotated = compute.propagate_flowline_confirmation(
            flowlines_for_mask,
            dem_da,
            ndwi_da=ndwi_da,
            ndwi_threshold=config.ndwi_threshold,
            flowlines_buffer_m=config.flowlines_buffer_m,
            max_hops=config.rem_propagate_hops,
        )
        reachable = annotated[annotated["reachable_from_seed"]].copy()
        streams = compute.rasterize_confirmed_flowlines(reachable, dem_da)
        result = rem_frame.compute_rem_anisotropic_frame(
            dem_da, ndwi_da, reachable, config
        )
        rem_da = result.rem_da
    elif config.rem_propagate_mask:
        streams = compute.build_network_propagated_mask(
            flowlines_for_mask,
            dem_da,
            ndwi_da=ndwi_da,
            ndwi_threshold=config.ndwi_threshold,
            flowlines_buffer_m=config.flowlines_buffer_m,
            max_hops=config.rem_propagate_hops,
        )
        rem_da = compute.compute_rem_edt_smooth(
            dem_da, streams, sigma=config.rem_smooth_sigma
        )
    else:
        streams = compute.build_streams_mask_from_nhd_ndwi(
            flowlines_for_mask,
            dem_da,
            ndwi_da=ndwi_da,
            ndwi_threshold=config.ndwi_threshold,
            flowlines_buffer_m=config.flowlines_buffer_m,
        )
        rem_da = compute.compute_rem_edt_smooth(
            dem_da, streams, sigma=config.rem_smooth_sigma
        )
    fields = io.load_and_clip_fields(config.fields_path, aoi_gdf, dem_crs)
    fields_stats = compute.compute_field_rem_stats(fields, rem_da, stats=("mean",))

    elapsed = time.monotonic() - t0

    # --- write outputs ---
    rem_da.rio.to_raster(os.path.join(run_dir, "rem_bounds.tif"))
    streams.rio.to_raster(os.path.join(run_dir, "streams_bounds.tif"))
    if "FID" in fields_stats.columns:
        fields_stats = fields_stats.rename(columns={"FID": "FID_"})
    fields_stats.to_file(
        os.path.join(run_dir, "fields_bounds.fgb"), driver="FlatGeobuf"
    )

    # --- write run.json ---
    rem_means = fields_stats["rem_mean"].dropna()
    summary = {
        "n_stream_cells": int(np.asarray(streams.data).astype(bool).sum()),
        "n_fields": len(fields_stats),
        "rem_mean_stats": {
            "min": round(float(rem_means.min()), 2) if len(rem_means) else None,
            "mean": round(float(rem_means.mean()), 2) if len(rem_means) else None,
            "median": round(float(rem_means.median()), 2) if len(rem_means) else None,
            "max": round(float(rem_means.max()), 2) if len(rem_means) else None,
        },
    }
    if config.rem_propagate_mask:
        summary["n_seed_features"] = streams.attrs.get("propagation_n_seeds")
        summary["n_confirmed_features"] = streams.attrs.get("propagation_n_confirmed")
        summary["n_propagated_features"] = streams.attrs.get("propagation_n_propagated")

    import subprocess

    try:
        git_hash = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        git_hash = None

    params = {
        "ndwi_threshold": config.ndwi_threshold,
        "flowlines_buffer_m": config.flowlines_buffer_m,
        "rem_smooth_sigma": config.rem_smooth_sigma,
        "rem_excluded_fcodes": sorted(excluded),
        "rem_propagate_mask": config.rem_propagate_mask,
        "rem_propagate_hops": config.rem_propagate_hops,
        "rem_method": config.rem_method,
    }
    if config.rem_method == "anisotropic_frame":
        params.update(
            {
                "rem_snap_w_elev": config.rem_snap_w_elev,
                "rem_snap_w_water": config.rem_snap_w_water,
                "rem_snap_w_prior": config.rem_snap_w_prior,
                "rem_snap_w_transition": config.rem_snap_w_transition,
                "rem_water_support_mode": config.rem_water_support_mode,
                "rem_support_corridor_half_width_m": config.rem_support_corridor_half_width_m,
                "rem_support_corridor_half_length_m": config.rem_support_corridor_half_length_m,
                "rem_min_station_water_hit_fraction": config.rem_min_station_water_hit_fraction,
                "rem_max_consecutive_no_water_m": config.rem_max_consecutive_no_water_m,
                "rem_max_mean_snap_offset_m": config.rem_max_mean_snap_offset_m,
                "rem_min_seeded_fraction": config.rem_min_seeded_fraction,
                "rem_snap_max_offset_m": config.rem_snap_max_offset_m,
                "rem_frame_station_spacing_m": config.rem_frame_station_spacing_m,
                "rem_frame_smoothing_m": config.rem_frame_smoothing_m,
                "rem_zero_mode": config.rem_zero_mode,
            }
        )
    run_meta = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_hash,
        "notes": notes,
        "parameters": params,
        "timing_s": round(elapsed, 1),
        "summary": summary,
    }
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    LOGGER.info("Experiment %s complete (%.1fs)", experiment_id, elapsed)
    logging.getLogger().removeHandler(run_log)
    run_log.close()
    return {
        "experiment_id": experiment_id,
        "run_dir": run_dir,
        "timing_s": round(elapsed, 1),
        "summary": summary,
    }
