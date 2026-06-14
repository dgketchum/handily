"""Phase 1 of the Ruby HUC8 pilot: domain extraction.

Carves the Ruby River HUC8 (``10020003``) working domain out of the statewide
WBD and NHD layers and lays down the regional workspace:

  * ``boundary/ruby_huc8.fgb``          — unbuffered HUC8 polygon (EPSG:5070)
  * ``boundary/ruby_huc8_buffer_20km.fgb`` — 20 km analysis buffer (EPSG:5070),
    used for GWX well queries, springs near the edge, and edge-safe rasters
  * ``evidence/springs/nhd_springs_45800.fgb`` — NHD spring points
    (``fcode == 45800``) clipped to the unbuffered HUC8

The springs are *candidate* zero-DTW anchors, not hard truth — Phase 5 scores
them with local buffers and visual QA before any are promoted to constraints.

A ``run_notes/domain_extraction.json`` note records input counts, area, bounds,
source paths, and the run date for reproducibility.

Usage:
    uv run python utils/build_ruby_domain.py \
        --out-root /data/ssd2/handily/mt/regional/ruby_huc8
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import geopandas as gpd

log = logging.getLogger("build_ruby_domain")

WBD_DEFAULT = "/nas/boundaries/wbd/NHD_H_Montana_State_Shape/Shape/WBDHU8.shp"
NHDPOINT_DEFAULT = "/nas/boundaries/wbd/NHD_H_Montana_State_Shape/Shape/NHDPoint.shp"
HUC8_DEFAULT = "10020003"
SPRING_FCODE = 45800
WORK_CRS = 5070
BUFFER_KM = 20.0


def _col(gdf: gpd.GeoDataFrame, name: str) -> str:
    """Return the actual column matching ``name`` case-insensitively."""
    for c in gdf.columns:
        if c.lower() == name.lower():
            return c
    raise KeyError(f"no '{name}' column in {list(gdf.columns)}")


def extract_domain(
    out_root: Path,
    wbd_path: str,
    nhdpoint_path: str,
    huc8: str,
    buffer_km: float,
) -> dict:
    boundary_dir = out_root / "boundary"
    springs_dir = out_root / "evidence" / "springs"
    notes_dir = out_root / "run_notes"
    for d in (boundary_dir, springs_dir, notes_dir):
        d.mkdir(parents=True, exist_ok=True)

    log.info("Reading WBD: %s", wbd_path)
    wbd = gpd.read_file(wbd_path)
    hcol, ncol = _col(wbd, "huc8"), _col(wbd, "name")
    ruby = wbd[wbd[hcol] == huc8].copy()
    if len(ruby) != 1:
        raise ValueError(f"expected exactly 1 HUC8 {huc8}, got {len(ruby)}")
    name = str(ruby[ncol].iloc[0])
    ruby = ruby.to_crs(WORK_CRS)
    area_km2 = float(ruby.area.sum() / 1e6)
    bounds_wgs84 = [float(v) for v in ruby.to_crs(4326).total_bounds]
    log.info("  %s HUC8: %.2f km2", name, area_km2)

    boundary_path = boundary_dir / "ruby_huc8.fgb"
    ruby.to_file(boundary_path, driver="FlatGeobuf")

    buffered = ruby.copy()
    buffered["geometry"] = ruby.buffer(buffer_km * 1000.0)
    buffer_path = boundary_dir / f"ruby_huc8_buffer_{int(buffer_km)}km.fgb"
    buffered.to_file(buffer_path, driver="FlatGeobuf")
    log.info("  wrote boundary + %.0f km buffer", buffer_km)

    log.info("Reading NHD points: %s", nhdpoint_path)
    nhd = gpd.read_file(nhdpoint_path).to_crs(WORK_CRS)
    fcol = _col(nhd, "fcode")
    springs = nhd[nhd[fcol] == SPRING_FCODE].copy()
    ruby_geom = ruby.union_all()
    inside = springs[springs.within(ruby_geom)].copy()
    springs_path = springs_dir / "nhd_springs_45800.fgb"
    inside.to_file(springs_path, driver="FlatGeobuf")
    log.info("  %d springs (fcode=%d) inside %s", len(inside), SPRING_FCODE, name)

    note = {
        "phase": "1_domain_extraction",
        "date": date.today().isoformat(),
        "huc8": huc8,
        "name": name,
        "area_km2": round(area_km2, 2),
        "bounds_wgs84": bounds_wgs84,
        "work_crs": f"EPSG:{WORK_CRS}",
        "buffer_km": buffer_km,
        "spring_fcode": SPRING_FCODE,
        "spring_count_inside_huc8": int(len(inside)),
        "sources": {"wbd": wbd_path, "nhdpoint": nhdpoint_path},
        "outputs": {
            "boundary": str(boundary_path),
            "boundary_buffer": str(buffer_path),
            "springs": str(springs_path),
        },
    }
    note_path = notes_dir / "domain_extraction.json"
    with open(note_path, "w") as f:
        json.dump(note, f, indent=2)
    log.info("Wrote run note: %s", note_path)
    return note


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-root",
        default="/data/ssd2/handily/mt/regional/ruby_huc8",
        help="Ruby regional workspace root",
    )
    p.add_argument("--wbd-path", default=WBD_DEFAULT)
    p.add_argument("--nhdpoint-path", default=NHDPOINT_DEFAULT)
    p.add_argument("--huc8", default=HUC8_DEFAULT)
    p.add_argument("--buffer-km", type=float, default=BUFFER_KM)
    args = p.parse_args()
    note = extract_domain(
        Path(args.out_root),
        args.wbd_path,
        args.nhdpoint_path,
        args.huc8,
        args.buffer_km,
    )
    print(json.dumps(note, indent=2))


if __name__ == "__main__":
    main()
