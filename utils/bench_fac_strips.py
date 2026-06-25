"""Benchmark + baseline metrics for FAC strip generation and the sparse burn.

Phase-0 guardrail for the strip-optimization work: run ``generate_fac_strips``
on a synthetic network (default) or on a real config's DEM/streams (``--config``),
time Stage 3 and the sparse burn, and write a JSON metrics file so later changes
can be checked output-for-output and wall-clock against a pinned baseline.

Set ``HANDILY_STRIP_PROFILE=1`` to also print accumulated per-call boundary and
stream-hit timing (and STRtree candidate vs accepted counts) from inside the
strip generator.

Usage:
  uv run python utils/bench_fac_strips.py
  uv run python utils/bench_fac_strips.py --workers 4 --out /tmp/bench.json
  uv run python utils/bench_fac_strips.py --config configs/rem/mt_0009.toml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from shapely.geometry import LineString

from handily.rem_fac import (
    build_orientation_field,
    generate_fac_strips,
    rasterize_sparse_sections_20m,
)


def _synthetic_dem(size_m: float, res_m: float) -> xr.DataArray:
    """A gently tilted plane so the orientation field is well defined."""
    n = int(round(size_m / res_m))
    x = np.arange(n, dtype=np.float64) * res_m + res_m / 2.0
    y = (np.arange(n, dtype=np.float64) * res_m + res_m / 2.0)[::-1]
    xx, yy = np.meshgrid(x, y)
    # tilt down toward +x and +(-y) so down_x/down_y are nonzero everywhere
    arr = 1000.0 + 0.02 * xx + 0.01 * (yy.max() - yy)
    da = xr.DataArray(arr, coords={"y": y, "x": x}, dims=("y", "x"), name="dem")
    da = da.rio.write_crs("EPSG:5070")
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    return da


def _synthetic_streams(size_m: float, n_reaches: int) -> gpd.GeoDataFrame:
    """A fan of roughly parallel reaches so strips find both neighbors and edges."""
    margin = size_m * 0.08
    xs = np.linspace(margin, size_m - margin, n_reaches)
    geoms, rids = [], []
    for i, xc in enumerate(xs):
        geoms.append(LineString([(xc, margin), (xc, size_m - margin)]))
        rids.append(i)
    return gpd.GeoDataFrame(
        {
            "reach_id": rids,
            "stream_id": rids,
            "strahler": [2] * len(rids),
            "geometry": geoms,
        },
        geometry="geometry",
        crs="EPSG:5070",
    )


def _strip_metrics(strips: gpd.GeoDataFrame, max_crossing_strip_m: float) -> dict:
    dist = strips["dist_m"].to_numpy(dtype=np.float64)
    finite = dist[np.isfinite(dist)]
    qs = (
        {f"p{int(q * 100)}": float(np.quantile(finite, q)) for q in (0.5, 0.9, 0.99)}
        if finite.size
        else {}
    )
    by_type = {
        t: int((strips["hit_type"] == t).sum())
        for t in ("interreach", "naked", "edge", "halo")
    }
    long_strips = (
        int((dist > max_crossing_strip_m).sum()) if max_crossing_strip_m > 0 else 0
    )
    return {
        "total": int(len(strips)),
        "by_hit_type": by_type,
        "long_strips_over_max_crossing": long_strips,
        "dist_m_max": float(np.nanmax(dist)) if finite.size else 0.0,
        "dist_m_quantiles": qs,
    }


def _run(streams, dem_da, field, params: dict) -> dict:
    params = dict(params)
    burn_res = float(params.pop("burn_res_m", 10.0))

    t0 = perf_counter()
    strips = generate_fac_strips(streams, dem_da, field=field, **params)
    stage3_s = perf_counter() - t0

    t1 = perf_counter()
    ws_da, count_da = rasterize_sparse_sections_20m(strips, dem_da, res_m=burn_res)
    burn_s = perf_counter() - t1

    cov = float(np.isfinite(ws_da.values).mean())
    metrics = _strip_metrics(strips, float(params.get("max_crossing_strip_m", 0.0)))
    metrics["sparse_burn_coverage_fraction"] = cov
    metrics["stage3_seconds"] = stage3_s
    metrics["sparse_burn_seconds"] = burn_s
    return metrics


def _bench_synthetic(args) -> dict:
    dem_da = _synthetic_dem(args.size_m, args.res_m)
    streams = _synthetic_streams(args.size_m, args.n_reaches)
    field = build_orientation_field(dem_da, coarse_res_m=20.0, smooth_sigma_m=100.0)
    params = dict(
        station_spacing_m=args.station_spacing_m,
        tangent_step_m=10.0,
        min_hit_dist_m=5.0,
        halo_n=args.halo_n,
        workers=args.workers,
        naked_fill_m=args.naked_fill_m,
        max_crossing_strip_m=args.max_crossing_strip_m,
        burn_res_m=args.res_m,
    )
    metrics = _run(streams, dem_da, field, params)
    return {
        "mode": "synthetic",
        "inputs": {
            "size_m": args.size_m,
            "res_m": args.res_m,
            "n_reaches": args.n_reaches,
        },
        "params": params,
        "metrics": metrics,
    }


def _bench_config(args) -> dict:
    from handily.rem_fac_config import FacRemConfig

    cfg = FacRemConfig.from_toml(args.config)
    dem_da = rioxarray.open_rasterio(cfg.dem_path).squeeze("band", drop=True)
    dem_da = dem_da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    dem_da = dem_da.where(dem_da > -1e5)
    streams = gpd.read_file(cfg.streams_path)
    if cfg.min_strahler > 0 and "strahler" in streams.columns:
        streams = streams.loc[streams["strahler"] >= cfg.min_strahler].copy()
    if "reach_id" not in streams.columns:
        streams["reach_id"] = np.arange(len(streams), dtype=np.int64)

    field = build_orientation_field(
        dem_da, coarse_res_m=cfg.coarse_res_m, smooth_sigma_m=cfg.smooth_sigma_m
    )
    params = dict(
        station_spacing_m=cfg.station_spacing_m,
        tangent_step_m=cfg.tangent_step_m,
        min_hit_dist_m=cfg.min_hit_dist_m,
        halo_n=cfg.halo_n,
        workers=args.workers if args.workers is not None else cfg.workers,
        naked_fill_m=cfg.naked_fill_m,
        max_crossing_strip_m=cfg.max_crossing_strip_m,
        burn_res_m=cfg.burn_res_m,
    )
    metrics = _run(streams, dem_da, field, params)
    return {
        "mode": "config",
        "inputs": {
            "config": str(args.config),
            "dem_path": cfg.dem_path,
            "streams_path": cfg.streams_path,
            "n_reaches": int(len(streams)),
        },
        "params": params,
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None, help="real FacRem TOML")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--out", type=Path, default=None, help="JSON metrics output path")
    # synthetic knobs
    p.add_argument("--size-m", type=float, default=3000.0)
    p.add_argument("--res-m", type=float, default=10.0)
    p.add_argument("--n-reaches", type=int, default=12)
    p.add_argument("--station-spacing-m", type=float, default=50.0)
    p.add_argument("--halo-n", type=int, default=8)
    p.add_argument("--naked-fill-m", type=float, default=0.0)
    p.add_argument("--max-crossing-strip-m", type=float, default=800.0)
    args = p.parse_args(argv)

    result = _bench_config(args) if args.config is not None else _bench_synthetic(args)

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
