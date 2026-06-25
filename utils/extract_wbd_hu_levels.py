"""Extract national WBD HU polygons (HU2..HU12) to EPSG:5070 geoparquet.

The USGS Watershed Boundary Dataset National GDB
(``WBD_National_GDB.gdb``, from The National Map staged products) carries every
hydrologic-unit level as a separate layer (``WBDHU2`` .. ``WBDHU16``) in NAD83
geographic (EPSG:4269). This reprojects the requested levels to EPSG:5070
(CONUS Albers -- the grid every handily well/raster lives on) and writes one
compact geoparquet per level, keyed by the ``huc{level}`` code:

    wbdhu2_5070.parquet  wbdhu4_5070.parquet  ...  wbdhu12_5070.parquet

These are the canonical national HUC polygons for point-in-polygon CV blocking
(HUC12-blocked folds) and any HUC join. Robust to layer/field-name casing.

    uv run python utils/extract_wbd_hu_levels.py \\
        --gdb /nas/hydrography/HUC_Boundaries/wbd_national/WBD_National_GDB.gdb \\
        --out-dir /nas/hydrography/HUC_Boundaries/wbd_national
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import pyogrio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_wbd_hu_levels")

DST_EPSG = 5070
KEEP_EXTRA = ("name", "areasqkm", "states", "hutype")


def _find_layer(gdb: str, level: int) -> str:
    """The GDB layer for a HU level, matched case-insensitively (e.g. WBDHU12)."""
    want = f"wbdhu{level}"
    layers = [name for name, _ in pyogrio.list_layers(gdb)]
    for name in layers:
        if name.lower() == want:
            return name
    raise SystemExit(f"no layer {want!r} in {gdb}; layers present: {layers}")


def _huc_col(gdf: gpd.GeoDataFrame, level: int) -> str:
    want = f"huc{level}"
    for c in gdf.columns:
        if c.lower() == want:
            return c
    raise SystemExit(f"no {want!r} column; columns: {list(gdf.columns)}")


def extract_level(gdb: str, out_dir: Path, level: int, dst_epsg: int) -> Path:
    layer = _find_layer(gdb, level)
    gdf = gpd.read_file(gdb, layer=layer, engine="pyogrio")
    huc_col = _huc_col(gdf, level)
    keep = [huc_col] + [c for c in gdf.columns if c.lower() in KEEP_EXTRA]
    gdf = gdf[keep + ["geometry"]].copy()
    gdf = gdf.rename(columns={huc_col: f"huc{level}"})
    if gdf.crs is None:
        raise SystemExit(f"{layer} has no CRS; cannot reproject")
    gdf = gdf.to_crs(dst_epsg)
    out = out_dir / f"wbdhu{level}_{dst_epsg}.parquet"
    gdf.to_parquet(out)
    log.info(
        "HU%d: %d polygons from %s (%s) -> %s",
        level,
        len(gdf),
        layer,
        gdf.crs.to_string(),
        out.name,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gdb", required=True, help="WBD_National_GDB.gdb path")
    ap.add_argument("--out-dir", default="/nas/hydrography/HUC_Boundaries/wbd_national")
    ap.add_argument("--levels", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--dst-epsg", type=int, default=DST_EPSG)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for level in args.levels:
        extract_level(args.gdb, out_dir, level, args.dst_epsg)
    log.info("done -> %s", out_dir)


if __name__ == "__main__":
    main()
