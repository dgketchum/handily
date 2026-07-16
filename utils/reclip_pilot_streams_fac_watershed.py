"""One-off: re-clip each pilot basin's streams to the FAC-delineated watershed.

The pilot basins were built with the full-window (halo-included) FAC network.
This applies ``regional_fac.clip_streams_to_fac_watershed`` -- the same clip the
canonical build now performs at stream generation -- to the already-built basins
without a WhiteboxTools re-run, using the retained ``d8_pointer.tif`` /
``streams_10m.tif`` / ``flow_accumulation.tif``.

A HUC8 pour point sits upstream of where the FAC mainstem finishes assembling, so
clipping to the HUC polygon severs below-pour-point confluences and leaves the
network fragmented. Clipping to the dominant outlet's contributing area (the
largest connected D8 stream-cell component) keeps the trunk + all confluences and
collapses the basin to ~1 connected component. This reports the reach-graph
component count before/after as the connectivity gate, and overwrites
``streams_regional.fgb`` in place (safe -- re-derivable from the retained
``streams_raw.shp`` / rasters).

Run --dry-run first to see before/after numbers without overwriting.

  uv run python utils/reclip_pilot_streams_fac_watershed.py --dry-run
  uv run python utils/reclip_pilot_streams_fac_watershed.py
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import geopandas as gpd

from handily.regional_fac import clip_streams_to_fac_watershed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reclip_pilot_fac_watershed")

ROOT = Path("/data/ssd2/handily/huc8")
BASINS = [
    "nm_rio_grande_abq",
    "nv_upper_humboldt",
    "mt_big_hole",
    "mt_beaverhead",
    "mt_ruby",
]


def _node(xy, precision: int = 3) -> tuple[float, float]:
    return (round(xy[0], precision), round(xy[1], precision))


def component_stats(gdf: gpd.GeoDataFrame) -> tuple[int, float]:
    """Return ``(n_components, pct_in_largest)`` for the undirected reach graph.

    Reaches sharing a quantized endpoint node are unioned. This is the
    coordinate-node-matching connectivity the GNN reach graph consumes (distinct
    from the authoritative D8 cell graph the clip uses).
    """
    n = len(gdf)
    if n == 0:
        return 0, 0.0
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    node_reaches: dict[tuple[float, float], list[int]] = {}
    for idx, geom in enumerate(gdf.geometry.values):
        coords = list(geom.coords)
        for end in (_node(coords[0]), _node(coords[-1])):
            node_reaches.setdefault(end, []).append(idx)
    for reaches in node_reaches.values():
        first = reaches[0]
        for other in reaches[1:]:
            union(first, other)

    roots = Counter(find(i) for i in range(n))
    largest = max(roots.values())
    return len(roots), 100.0 * largest / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--basins", nargs="*", default=BASINS)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="report before/after counts only; do not overwrite",
    )
    args = ap.parse_args()

    root = Path(args.root)
    for basin in args.basins:
        d = root / basin
        streams_fp = d / "streams_regional.fgb"
        d8_fp = d / "d8_pointer.tif"
        streams_ras_fp = d / "streams_10m.tif"
        fac_fp = d / "flow_accumulation.tif"
        basin_fp = d / "basin_boundary.fgb"
        missing = [
            p.name
            for p in (streams_fp, d8_fp, streams_ras_fp, fac_fp)
            if not p.exists()
        ]
        if missing:
            log.warning("%s: missing %s, skipping", basin, ", ".join(missing))
            continue

        streams = gpd.read_file(streams_fp)
        poly = None
        if basin_fp.exists():
            poly = gpd.read_file(basin_fp).to_crs(streams.crs).geometry.union_all()

        nc0, pct0 = component_stats(streams)
        clipped = clip_streams_to_fac_watershed(
            streams, d8_fp, streams_ras_fp, fac_fp, basin_poly=poly
        )
        nc1, pct1 = component_stats(clipped)
        log.info(
            "%-20s reaches %5d->%5d | components %4d->%4d | largest %5.1f%%->%5.1f%%",
            basin,
            len(streams),
            len(clipped),
            nc0,
            nc1,
            pct0,
            pct1,
        )
        if not args.dry_run:
            clipped.to_file(streams_fp, driver="FlatGeobuf")
            log.info("  overwrote %s", streams_fp)


if __name__ == "__main__":
    main()
