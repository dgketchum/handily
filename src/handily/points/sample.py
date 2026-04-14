import logging
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
from rasterio.transform import rowcol
from shapely.geometry import Point

from handily.config import HandilyConfig
from handily.io import (
    aoi_from_bounds,
    ensure_dir,
    get_flowlines_within_aoi,
    load_and_clip_fields,
)
from handily.nhd import classify_flowlines, filter_flowlines_for_stratification

LOGGER = logging.getLogger("handily.points.sample")

GROUP_PRIORITY = [
    "riparian",
    "field_interior",
    "low_rem",
    "field_edge",
    "high_rem_control",
    "base",
]


def _points_out_dir(config: HandilyConfig) -> str:
    if config.points_out_dir:
        return config.points_out_dir
    return os.path.join(config.out_dir, "points")


def _load_rem_da(config: HandilyConfig):
    rem_path = os.path.join(config.out_dir, "rem_bounds.tif")
    if not os.path.exists(rem_path):
        raise FileNotFoundError(f"REM raster not found: {rem_path}")

    rem_da = rxr.open_rasterio(rem_path)
    if "band" in rem_da.dims:
        rem_da = rem_da.squeeze("band", drop=True)
    return rem_da


def _load_fields(
    config: HandilyConfig, aoi_gdf: gpd.GeoDataFrame, target_crs
) -> gpd.GeoDataFrame:
    fields_path = os.path.join(config.out_dir, "fields_bounds.fgb")
    if os.path.exists(fields_path):
        LOGGER.info("Loading AOI fields from %s", fields_path)
        fields = gpd.read_file(fields_path)
        return fields.to_crs(target_crs)

    LOGGER.info("Loading and clipping fields from source dataset")
    fields = load_and_clip_fields(config.fields_path, aoi_gdf, target_crs)
    return fields


def _load_flowlines(
    config: HandilyConfig, aoi_gdf: gpd.GeoDataFrame, target_crs
) -> gpd.GeoDataFrame:
    flowlines_path = os.path.join(config.out_dir, "flowlines_bounds.fgb")
    if os.path.exists(flowlines_path):
        LOGGER.info("Loading AOI flowlines from %s", flowlines_path)
        flowlines = gpd.read_file(flowlines_path)
        return flowlines.to_crs(target_crs)

    LOGGER.info("Fetching flowlines for AOI")
    flowlines = get_flowlines_within_aoi(
        aoi_gdf, local_flowlines_dir=config.flowlines_local_dir
    )
    return flowlines.to_crs(target_crs)


def _build_candidate_grid(
    aoi_gdf: gpd.GeoDataFrame,
    target_crs,
    spacing_m: float,
) -> gpd.GeoDataFrame:
    aoi_proj = aoi_gdf.to_crs(target_crs)
    geom = aoi_proj.geometry.unary_union
    minx, miny, maxx, maxy = geom.bounds

    xs = np.arange(minx + spacing_m / 2.0, maxx, spacing_m)
    ys = np.arange(miny + spacing_m / 2.0, maxy, spacing_m)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("AOI is smaller than candidate point spacing")

    points = [Point(x, y) for x in xs for y in ys]
    candidates = gpd.GeoDataFrame(geometry=gpd.GeoSeries(points, crs=target_crs))
    mask = candidates.geometry.within(geom) | candidates.geometry.touches(geom)
    candidates = candidates.loc[mask].copy()
    candidates = candidates.reset_index(drop=True)
    candidates["candidate_id"] = candidates.index.astype(int)
    candidates["x"] = candidates.geometry.x
    candidates["y"] = candidates.geometry.y
    return candidates


def _sample_rem_values(points: gpd.GeoDataFrame, rem_da) -> gpd.GeoDataFrame:
    transform = rem_da.rio.transform()
    rem_arr = np.asarray(rem_da.data)
    if np.ma.isMaskedArray(rem_arr):
        rem_arr = rem_arr.filled(np.nan)

    xs = points.geometry.x.to_numpy()
    ys = points.geometry.y.to_numpy()
    rows, cols = rowcol(transform, xs, ys)
    rows = np.asarray(rows)
    cols = np.asarray(cols)

    values = np.full(len(points), np.nan, dtype="float64")
    valid = (
        (rows >= 0)
        & (cols >= 0)
        & (rows < rem_arr.shape[0])
        & (cols < rem_arr.shape[1])
    )
    values[valid] = rem_arr[rows[valid], cols[valid]]

    out = points.copy()
    out["rem_at_sample"] = values
    return out


def _assign_stream_context(
    points: gpd.GeoDataFrame, flowlines: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    flowlines = classify_flowlines(flowlines)
    flowlines = filter_flowlines_for_stratification(flowlines)

    out = points.copy()
    if flowlines.empty:
        out["nearest_stream_type"] = "none"
        out["stream_distance"] = np.inf
        out["stream_context_at_sample"] = "none"
        return out

    join = gpd.sjoin_nearest(
        out[["candidate_id", "geometry"]],
        flowlines[["stream_category", "geometry"]],
        how="left",
        distance_col="stream_distance",
    )
    join = (
        join.sort_values("candidate_id")
        .drop_duplicates(subset=["candidate_id"], keep="first")
        .reset_index(drop=True)
    )
    out["nearest_stream_type"] = join["stream_category"].fillna("none")
    out["stream_distance"] = join["stream_distance"].fillna(np.inf)
    out["stream_context_at_sample"] = out["nearest_stream_type"]
    return out


def _assign_field_context(
    points: gpd.GeoDataFrame, fields: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    out = points.copy()
    if fields.empty:
        out["in_irrigated_lands"] = False
        out["dist_field_edge_m"] = np.inf
        return out

    fields = fields.to_crs(out.crs)
    join = gpd.sjoin(
        out[["candidate_id", "geometry"]],
        fields[["geometry"]],
        how="left",
        predicate="within",
    )
    join = (
        join.sort_values("candidate_id")
        .drop_duplicates(subset=["candidate_id"], keep="first")
        .reset_index(drop=True)
    )
    out["in_irrigated_lands"] = join["index_right"].notna().to_numpy()

    boundary = fields.geometry.boundary.unary_union
    if boundary.is_empty:
        out["dist_field_edge_m"] = np.inf
    else:
        out["dist_field_edge_m"] = out.geometry.distance(boundary)
    return out


def _generate_field_interior_points(
    fields: gpd.GeoDataFrame,
    candidate_id_offset: int,
    target_crs,
) -> gpd.GeoDataFrame:
    """Return one representative point per field polygon.

    Uses ``representative_point()`` which is guaranteed to lie inside the
    polygon, unlike the centroid which may fall outside for concave shapes.
    """
    if fields.empty:
        return gpd.GeoDataFrame()

    fields_proj = fields.to_crs(target_crs)
    rep_points = fields_proj.geometry.representative_point()

    pts = gpd.GeoDataFrame(geometry=rep_points, crs=target_crs).reset_index(drop=True)
    pts["candidate_id"] = candidate_id_offset + pts.index.astype(int)
    pts["x"] = pts.geometry.x
    pts["y"] = pts.geometry.y
    return pts


def build_sample_masks(
    points: gpd.GeoDataFrame, config: HandilyConfig
) -> gpd.GeoDataFrame:
    out = points.copy()
    valid_rem = np.isfinite(out["rem_at_sample"])
    out["has_valid_rem"] = valid_rem
    out["is_low_rem_target"] = valid_rem & (
        out["rem_at_sample"] < float(config.points_low_rem_threshold_m)
    )
    out["is_riparian_target"] = valid_rem & (
        out["nearest_stream_type"].isin(["perennial", "intermittent", "managed"])
        & (out["stream_distance"] <= float(config.points_riparian_buffer_m))
    )
    out["is_field_edge_target"] = valid_rem & (
        out["dist_field_edge_m"] <= float(config.points_field_edge_buffer_m)
    )
    out["is_high_rem_control"] = valid_rem & (
        out["rem_at_sample"] > float(config.points_high_rem_threshold_m)
    )
    out["is_valid_base"] = valid_rem
    return out


def sample_points_from_mask(
    candidates: gpd.GeoDataFrame,
    mask_col: str,
    n_points: int,
    rng: np.random.Generator,
    selected_ids: set[int],
    sample_group: str,
) -> gpd.GeoDataFrame:
    eligible = candidates[
        candidates[mask_col] & ~candidates["candidate_id"].isin(selected_ids)
    ].copy()
    eligible_count = len(eligible)

    if eligible_count == 0 or int(n_points) <= 0:
        return eligible.iloc[0:0].copy()

    n_take = min(int(n_points), eligible_count)
    chosen = rng.choice(eligible["candidate_id"].to_numpy(), size=n_take, replace=False)
    sampled = eligible[eligible["candidate_id"].isin(chosen)].copy()
    sampled["sample_group"] = sample_group
    sampled["sampling_weight"] = float(eligible_count) / float(n_take)
    sampled = sampled.sort_values("candidate_id").reset_index(drop=True)
    return sampled


def merge_sample_groups(groups: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    non_empty = [g for g in groups if len(g) > 0]
    if not non_empty:
        raise ValueError("No points were sampled from any group")
    merged = pd.concat(non_empty, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=non_empty[0].crs)


def deduplicate_sample_points(
    points: gpd.GeoDataFrame,
    group_priority: list[str] | None = None,
) -> gpd.GeoDataFrame:
    if group_priority is None:
        group_priority = GROUP_PRIORITY
    priority_map = {name: idx for idx, name in enumerate(group_priority)}
    out = points.copy()
    out["__priority__"] = (
        out["sample_group"].map(priority_map).fillna(len(priority_map))
    )
    out = out.sort_values(["candidate_id", "__priority__"]).drop_duplicates(
        subset=["candidate_id"], keep="first"
    )
    out = out.drop(columns="__priority__").reset_index(drop=True)
    return out


def assign_point_ids(points: gpd.GeoDataFrame, aoi_id: str | int) -> gpd.GeoDataFrame:
    out = points.copy().reset_index(drop=True)
    out["point_id"] = [f"{aoi_id}_{idx:06d}" for idx in range(len(out))]
    return out


def write_sample_points(points: gpd.GeoDataFrame, out_dir: str) -> dict[str, str]:
    ensure_dir(out_dir)
    fgb_path = os.path.join(out_dir, "points.fgb")
    parquet_path = os.path.join(out_dir, "points.parquet")
    points.to_file(fgb_path, driver="FlatGeobuf")
    points.to_parquet(parquet_path)
    return {"fgb": fgb_path, "parquet": parquet_path}


def build_aoi_sample_points(
    aoi_gdf: gpd.GeoDataFrame,
    fields: gpd.GeoDataFrame,
    rem_da,
    flowlines: gpd.GeoDataFrame,
    config: HandilyConfig,
    aoi_id: str | int = "aoi",
) -> gpd.GeoDataFrame:
    spacing_m = max(
        float(config.points_candidate_spacing_m),
        float(config.points_min_spacing_m or 0.0),
    )
    rng = np.random.default_rng(int(config.points_seed))

    LOGGER.info("Building candidate grid (spacing=%.1f m)", spacing_m)
    candidates = _build_candidate_grid(aoi_gdf, rem_da.rio.crs, spacing_m=spacing_m)
    LOGGER.info("Candidate points: %d", len(candidates))

    candidates = _sample_rem_values(candidates, rem_da)
    candidates = _assign_stream_context(candidates, flowlines)
    candidates = _assign_field_context(candidates, fields)
    candidates = build_sample_masks(candidates, config)

    selected_ids: set[int] = set()
    groups = []
    group_specs = [
        ("riparian", "is_riparian_target", config.points_n_riparian),
        ("low_rem", "is_low_rem_target", config.points_n_low_rem),
        ("field_edge", "is_field_edge_target", config.points_n_field_edge),
        ("high_rem_control", "is_high_rem_control", config.points_n_high_rem_control),
        ("base", "is_valid_base", config.points_n_base),
    ]

    for sample_group, mask_col, n_points in group_specs:
        sampled = sample_points_from_mask(
            candidates,
            mask_col=mask_col,
            n_points=n_points,
            rng=rng,
            selected_ids=selected_ids,
            sample_group=sample_group,
        )
        if len(sampled) < int(n_points):
            LOGGER.info(
                "Sample group '%s': requested=%d, available=%d, selected=%d",
                sample_group,
                int(n_points),
                int(
                    (
                        candidates[mask_col]
                        & ~candidates["candidate_id"].isin(selected_ids)
                    ).sum()
                ),
                len(sampled),
            )
        else:
            LOGGER.info("Sample group '%s': selected=%d", sample_group, len(sampled))
        selected_ids.update(sampled["candidate_id"].tolist())
        groups.append(sampled)

    # Add one representative point per irrigated field — guaranteed field coverage.
    if not fields.empty:
        fi = _generate_field_interior_points(
            fields, candidate_id_offset=len(candidates), target_crs=rem_da.rio.crs
        )
        fi = _sample_rem_values(fi, rem_da)
        n_before = len(fi)
        fi = fi.dropna(subset=["rem_at_sample"]).reset_index(drop=True)
        if len(fi) < n_before:
            LOGGER.info(
                "Field interior: dropped %d/%d points with no REM value",
                n_before - len(fi),
                n_before,
            )
        fi = _assign_stream_context(fi, flowlines)
        fi = _assign_field_context(fi, fields)
        fi = build_sample_masks(fi, config)
        fi["sample_group"] = "field_interior"
        fi["sampling_weight"] = 1.0
        groups.append(fi)
        LOGGER.info("Field interior points: %d (one per field polygon)", len(fi))

    points = merge_sample_groups(groups)
    points = deduplicate_sample_points(points)
    points = assign_point_ids(points, aoi_id=aoi_id)

    keep_cols = [
        "point_id",
        "candidate_id",
        "geometry",
        "x",
        "y",
        "sample_group",
        "rem_at_sample",
        "stream_context_at_sample",
        "nearest_stream_type",
        "stream_distance",
        "in_irrigated_lands",
        "dist_field_edge_m",
        "sampling_weight",
        "is_low_rem_target",
        "is_riparian_target",
        "is_field_edge_target",
        "is_high_rem_control",
    ]
    points = points[keep_cols].copy()
    points["aoi_id"] = str(aoi_id)
    points["sample_seed"] = int(config.points_seed)
    points = points[
        [
            "point_id",
            "aoi_id",
            "geometry",
            "x",
            "y",
            "sample_group",
            "sample_seed",
            "in_irrigated_lands",
            "rem_at_sample",
            "stream_context_at_sample",
            "nearest_stream_type",
            "stream_distance",
            "dist_field_edge_m",
            "sampling_weight",
            "is_low_rem_target",
            "is_riparian_target",
            "is_field_edge_target",
            "is_high_rem_control",
            "candidate_id",
        ]
    ]
    return points


def sample_points_from_config(config: HandilyConfig) -> dict:
    if not config.bounds:
        raise ValueError("Config must include bounds for AOI-scoped point sampling")

    aoi_gdf = aoi_from_bounds(tuple(config.bounds))
    rem_da = _load_rem_da(config)
    fields = _load_fields(config, aoi_gdf, rem_da.rio.crs)
    flowlines = _load_flowlines(config, aoi_gdf, rem_da.rio.crs)

    aoi_id = "bounds"
    points = build_aoi_sample_points(
        aoi_gdf=aoi_gdf,
        fields=fields,
        rem_da=rem_da,
        flowlines=flowlines,
        config=config,
        aoi_id=aoi_id,
    )
    out_dir = _points_out_dir(config)
    paths = write_sample_points(points, out_dir)

    result = {
        "out_dir": out_dir,
        "n_points": len(points),
        "group_counts": points["sample_group"].value_counts().to_dict(),
        "paths": paths,
    }
    return result


# ========================= EOF =======================================================================================
