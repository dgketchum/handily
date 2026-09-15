"""Sample handily's water-table rasters at a set of station/point locations.

Reads any OGR point layer (FlatGeobuf, shapefile, GeoPackage ...), reprojects the
points into each raster's CRS before sampling, and writes a new FlatGeobuf that
carries every original attribute plus the sampled columns.

Two distinct "missing" conventions are preserved rather than collapsed:

* ``handily_dtw_m`` / ``handily_wte_m`` -- the GNN render (``gnn_dtw_10m``,
  ``gnn_wte_10m``) covers the modelled footprint and stores -9999 outside it. A
  point landing there is genuinely off the modelled footprint; it comes through
  as null with its ``*_nodata`` flag set to True, never as -9999.
* ``facrem_dtw_m`` -- the FAC-REM channel-strip solve is only *defined* near
  channels and stores NaN everywhere else; roughly two thirds of a state is
  legitimately outside it. Those cells come through as null with
  ``facrem_dtw_offnetwork`` True. That is expected coverage, not a defect.

Layers are named on the command line as ``name=path`` so the added column names
are explicit and carry units (metres) in the name.

Usage:
    uv run python utils/sample_stations_dtw.py \
        --points /data/ssd2/handily/mt/mesonet/mt_mesonet_stations.fgb \
        --out /data/ssd2/handily/mt/mesonet/mt_mesonet_handily_dtw.fgb \
        --raster handily_dtw_m=<...>/gnn_dtw_10m_mt.vrt \
        --raster handily_wte_m=<...>/gnn_wte_10m_mt.vrt \
        --raster facrem_dtw_m=<...>/fac_rem_dtw_mt.vrt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

# FAC-REM writes NaN (not a sentinel) outside the channel-strip footprint, so its
# missing cells get their own flag name -- "off network", not "nodata".
OFFNETWORK_LAYERS = {"facrem_dtw_m"}


def sample_raster(gdf: gpd.GeoDataFrame, raster: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, missing_mask) for ``gdf`` sampled at ``raster``.

    Points are reprojected into the raster CRS first; nodata and NaN are both
    reported through ``missing_mask`` and returned as NaN in ``values``.
    """
    with rasterio.open(raster) as ds:
        pts = gdf.to_crs(ds.crs)
        coords = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
        vals = np.array([v[0] for v in ds.sample(coords, indexes=1)], dtype="float64")
        nodata = ds.nodata
    missing = ~np.isfinite(vals)
    if nodata is not None:
        missing |= np.isclose(vals, nodata)
    vals = np.where(missing, np.nan, vals)
    return vals, missing


def build(points: Path, out: Path, rasters: dict[str, Path]) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(points)
    if gdf.crs is None:
        raise ValueError(f"{points} has no CRS; cannot reproject for sampling")
    print(f"{len(gdf)} points from {points} ({gdf.crs})")

    for name, raster in rasters.items():
        vals, missing = sample_raster(gdf, raster)
        gdf[name] = vals
        flag = (
            f"{name.removesuffix('_m')}_offnetwork"
            if name in OFFNETWORK_LAYERS
            else f"{name.removesuffix('_m')}_nodata"
        )
        gdf[flag] = missing
        finite = vals[~missing]
        print(
            f"  {name}: {len(finite)} sampled, {int(missing.sum())} missing ({flag}); "
            f"median {np.median(finite):.2f} m, min {finite.min():.2f} m, "
            f"max {finite.max():.2f} m"
            if finite.size
            else f"  {name}: 0 sampled, {int(missing.sum())} missing ({flag})"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="FlatGeobuf")
    print(f"wrote {len(gdf)} points x {len(gdf.columns)} columns -> {out}")
    return gdf


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--points", type=Path, required=True, help="input point layer")
    p.add_argument("--out", type=Path, required=True, help="output FlatGeobuf")
    p.add_argument(
        "--raster",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="output column name (include units) = raster path; repeatable",
    )
    args = p.parse_args(argv)
    rasters = {}
    for spec in args.raster:
        if "=" not in spec:
            p.error(f"--raster expects NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        rasters[name] = Path(path)
    args.rasters = rasters
    return args


def main(argv):
    args = parse_args(argv)
    build(args.points, args.out, args.rasters)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
