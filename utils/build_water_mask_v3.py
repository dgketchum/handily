"""CONUS screened surface-water mask (V3) on the canonical 100 m EPSG:5070 lattice.

The GNN's only surface-water evidence has been JRC GSW occurrence keyed to a
>=90 %-occurrence block list. That selector is coverage-starved (7 blocks in
pilot HUC8 16040103 vs ~726 real river-corridor cells) AND, taken naively at
occ>=50, snow/terrain-shadow contaminated (54 % of occ>=90 pixels in the pilot
sat at 1,671-3,098 m elevation with a local water density near zero).

**V3 recipe (the probe's winning mask, notes/NAIP_WATER_EVIDENCE_PLAN.md section 11):**

    water = JRC occurrence >= 50 %  AND  (>= 200 occ>=50 pixels within 1 km)

The density clause is the target-blind contiguity screen that removes the
high-elevation false positives. There is deliberately NO ``fac_rem_dtw <= X``
clause: V2 carried one and it censored exactly the deep-error corridor cells
the flatten exists to fix (a validity test built from the surface being
corrected), making 2 of the pilot's 7 evidence blocks WORSE.

Output grid = the canonical CONUS 100 m EPSG:5070 lattice (origin
``X0, Y0 = -2_540_000, 3_258_000``; the same grid the 100 m DEM and every
render lattice snap to), so a mask lookup is a direct cell read at train time
(bundle query rows), at render time (lattice cells) and on the point path.

A 100 m cell is masked when ANY of its ~11 covering 30 m JRC pixels passes the
screen (the same any-overlap semantics the pilot probe used, so the pilot cell
counts are comparable). The density statistic is evaluated per 100 m cell
(exact disk sum of the 30 m-pixel counts within 1 km), not per 30 m pixel --
a ~100 m quantisation of a 1 km-scale statistic, validated against the probe's
exact per-pixel KD-tree mask on the pilot basin (``--pilot-check``).

Outputs (``--out-dir``, default /nas/handily/covariates/water_mask_v3):
  water_mask_v3_100m_5070.tif    uint8 0/1 mask on the canonical lattice
  water_mask_v3_counts_100m.tif  uint16 count of occ>=50 30 m pixels per cell
  water_mask_v3_cells.parquet    x5070/y5070 centers of BOUNDARY masked cells --
                                 the kNN / pseudo-label pool, drop-in for
                                 permanent_water_blocks.parquet. Interior cells
                                 are dropped: exact for nearest-water distances
                                 and the only way a CONUS pool that includes the
                                 ocean stays tractable (see mask_boundary and the
                                 build_manifest note).
  build_manifest.json            recipe + stats

Usage:
    uv run python utils/build_water_mask_v3.py
    uv run python utils/build_water_mask_v3.py --pilot-check 16040103
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_water_mask_v3")

GSW_OCC_DIR = Path("/nas/hydrography/gsw/occurrence")
DEM_100M = "/data/ssd2/handily/conus/covariates/elev48i0100a.tif"
OUT_DIR_DEFAULT = Path("/nas/handily/covariates/water_mask_v3")
MASK_TIF = "water_mask_v3_100m_5070.tif"
COUNT_TIF = "water_mask_v3_counts_100m.tif"
CELLS_PARQUET = "water_mask_v3_cells.parquet"

OCC_MIN_PCT = 50.0  # JRC occurrence threshold (% of valid months inundated)
DENSITY_RADIUS_M = 1000.0  # contiguity screen radius
DENSITY_MIN_PX = 200  # min occ>=50 30 m pixels inside that radius
RES = 100.0
# one-cell shoreline taper of the inference-time FAC flatten (water_flatten_factor)
RAMP_FACTOR = 0.5

_MASK_CACHE: dict[str, tuple] = {}


def canonical_grid(dem_path: str = DEM_100M) -> tuple:
    """(transform, width, height) of the canonical CONUS 100 m EPSG:5070 lattice."""
    with rasterio.open(dem_path) as ds:
        if ds.crs is None or ds.crs.to_epsg() != 5070:
            raise SystemExit(f"DEM not EPSG:5070: {dem_path}")
        if abs(ds.transform.a - RES) > 1e-9:
            raise SystemExit(f"DEM is not {RES:g} m: {dem_path}")
        return ds.transform, ds.width, ds.height


def occ_pixel_counts(
    transform, width: int, height: int, chunk: tuple[int, int] = (1024, 8192)
) -> np.ndarray:
    """Count of JRC occ>=OCC_MIN_PCT 30 m pixels falling in each 100 m lattice cell.

    Streams every 10-degree GSW tile in windows (the 40k x 40k uint8 rasters never
    load whole), converts the qualifying pixel centers to EPSG:5070 and bincounts
    them into the lattice sub-window the chunk actually touches (a full-lattice
    bincount would allocate 12 GB per chunk). JRC occurrence is uint8 0-100 with
    UNTAGGED fill above 100, so validity is ``occ <= 100``.
    """
    tiles = sorted(GSW_OCC_DIR.glob("*.tif"))
    if not tiles:
        raise SystemExit(f"no GSW occurrence tiles in {GSW_OCC_DIR}")
    counts = np.zeros((height, width), dtype="int32")
    tf = Transformer.from_crs(4326, 5070, always_xy=True)
    n_px_total = 0
    ch_r, ch_c = chunk
    for tp in tiles:
        n_tile = 0
        with rasterio.open(tp) as ds:
            t = ds.transform
            for r0 in range(0, ds.height, ch_r):
                nrows = min(ch_r, ds.height - r0)
                for c0 in range(0, ds.width, ch_c):
                    ncols = min(ch_c, ds.width - c0)
                    a = ds.read(1, window=Window(c0, r0, ncols, nrows))
                    m = (a <= 100) & (a >= OCC_MIN_PCT)
                    if not m.any():
                        continue
                    rr, cc = np.where(m)
                    lon, lat = t * (c0 + cc + 0.5, r0 + rr + 0.5)
                    x, y = tf.transform(np.asarray(lon), np.asarray(lat))
                    col = np.floor((x - transform.c) / RES).astype("int64")
                    row = np.floor((transform.f - y) / RES).astype("int64")
                    ok = (row >= 0) & (row < height) & (col >= 0) & (col < width)
                    if not ok.any():
                        continue
                    row, col = row[ok], col[ok]
                    r_lo, r_hi = int(row.min()), int(row.max())
                    c_lo, c_hi = int(col.min()), int(col.max())
                    sub_w = c_hi - c_lo + 1
                    sub_h = r_hi - r_lo + 1
                    local = (row - r_lo) * sub_w + (col - c_lo)
                    sub = np.bincount(local, minlength=sub_h * sub_w).reshape(
                        sub_h, sub_w
                    )
                    counts[r_lo : r_hi + 1, c_lo : c_hi + 1] += sub.astype("int32")
                    n_tile += int(len(row))
        n_px_total += n_tile
        log.info(
            "  %s: %d occ>=%g pixels on the CONUS lattice", tp.name, n_tile, OCC_MIN_PCT
        )
    log.info("total occ>=%g 30 m pixels binned: %d", OCC_MIN_PCT, n_px_total)
    return counts


def disk_sum(
    counts: np.ndarray, radius_cells: int, band_rows: int = 2048
) -> np.ndarray:
    """Exact sum of ``counts`` over a disk of ``radius_cells`` around every cell.

    Row-offset horizontal box sums via a per-band cumulative sum: for each dy in
    [-R, R] the disk half-width is ``hx = floor(sqrt(R^2 - dy^2))`` cells, and the
    band's prefix sums turn each row's contribution into two lookups. Processed in
    horizontal bands (with a 2R-row halo) so the CONUS int64 prefix array is never
    materialised whole.
    """
    h, w = counts.shape
    r = int(radius_cells)
    dys = np.arange(-r, r + 1)
    hxs = np.floor(np.sqrt(np.maximum(r * r - dys * dys, 0))).astype("int64")
    out = np.zeros((h, w), dtype="int32")
    for b0 in range(0, h, band_rows):
        b1 = min(b0 + band_rows, h)
        p0, p1 = max(0, b0 - r), min(h, b1 + r)
        band = counts[p0:p1].astype("int64")
        pad = np.zeros((band.shape[0], w + 2 * r), dtype="int64")
        pad[:, r : r + w] = band
        # prefix[:, j] = sum of pad[:, :j]
        prefix = np.zeros((band.shape[0], w + 2 * r + 1), dtype="int64")
        np.cumsum(pad, axis=1, out=prefix[:, 1:])
        acc = np.zeros((b1 - b0, w), dtype="int64")
        for dy, hx in zip(dys, hxs, strict=True):
            src = np.arange(b0, b1) + dy - p0
            valid = (src >= 0) & (src < band.shape[0])
            if not valid.any():
                continue
            # horizontal box sum over pad cols [j+r-hx, j+r+hx] -> prefix difference
            lo, hi = r - hx, r + hx + 1
            rows = src[valid]
            acc[valid] += prefix[rows, hi : hi + w] - prefix[rows, lo : lo + w]
        out[b0:b1] = np.clip(acc, 0, np.iinfo("int32").max).astype("int32")
        log.info("  disk sum rows %d-%d / %d", b0, b1, h)
    return out


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Masked cells having at least one unmasked 8-neighbour (array edge counts).

    The exported cell list is this shell, not the full mask: it gives identical
    nearest-water distances for every off-mask query point (see the manifest note)
    at ~3 % of the row count, which is what makes a CONUS-wide pool -- ocean
    included -- tractable as a kD-tree.
    """
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    interior = np.ones((h, w), dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            interior &= pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return mask & ~interior


def write_uint(path: Path, arr: np.ndarray, transform, dtype: str) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs="EPSG:5070",
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="YES",
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def mask_grid(path: str | Path) -> tuple[object, int, int]:
    """(transform, width, height) of a mask raster, cached per process."""
    key = str(path)
    if key not in _MASK_CACHE:
        with rasterio.open(key) as ds:
            _MASK_CACHE[key] = (ds.transform, ds.width, ds.height)
    return _MASK_CACHE[key]


def sample_water_mask(
    x5070: np.ndarray, y5070: np.ndarray, path: str | Path
) -> np.ndarray:
    """Boolean 'this location is screened surface water' at EPSG:5070 coords.

    Reads only the bounding window of the query points -- the CONUS mask is
    49810x31390, so a full read is 1.6 GB and a per-basin render would pay it on
    every call. Points outside the lattice return False (the mask covers the whole
    CONUS extent; the lattice edge is ocean or Canada/Mexico, where a
    surface-water flag has no consumer).
    """
    t, w, h = mask_grid(path)
    x5070 = np.asarray(x5070, "float64")
    y5070 = np.asarray(y5070, "float64")
    col = np.floor((x5070 - t.c) / t.a).astype("int64")
    row = np.floor((t.f - y5070) / abs(t.e)).astype("int64")
    ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    out = np.zeros(x5070.shape, bool)
    if not ok.any():
        return out
    r_lo, r_hi = int(row[ok].min()), int(row[ok].max())
    c_lo, c_hi = int(col[ok].min()), int(col[ok].max())
    with rasterio.open(str(path)) as ds:
        arr = ds.read(
            1, window=Window(c_lo, r_lo, c_hi - c_lo + 1, r_hi - r_lo + 1)
        ).astype(bool)
    out[ok] = arr[row[ok] - r_lo, col[ok] - c_lo]
    return out


def water_flatten_factor(
    x5070: np.ndarray,
    y5070: np.ndarray,
    path: str | Path,
    ramp: bool = False,
) -> np.ndarray:
    """Multiplier applied to ``fac_rem_dtw_m`` by the inference-time water flatten.

    ``0.0`` on the mask -- the FAC-REM prior is zeroed at verified surface water,
    exactly the raster substitution the frozen-weight probe made (notes/
    NAIP_WATER_EVIDENCE_PLAN.md section 11). ``1.0`` off it, so nothing else in
    the render moves.

    With ``ramp``, an OFF-mask cell that has at least one masked 8-neighbour gets
    ``RAMP_FACTOR`` (one-cell shoreline taper): the hard flatten leaves a ~2 m
    single-cell DTW step at the mask edge, and halving the neighbour's FAC input
    splits that step over two cells. Only the cell's own 100 m mask cell and its
    immediate ring are consulted; there is no distance decay beyond one cell.

    Points outside the lattice get ``1.0`` (untouched), and the read window's own
    edge is treated as unmasked -- the same convention as ``mask_boundary``.
    """
    t, w, h = mask_grid(path)
    x5070 = np.asarray(x5070, "float64")
    y5070 = np.asarray(y5070, "float64")
    col = np.floor((x5070 - t.c) / t.a).astype("int64")
    row = np.floor((t.f - y5070) / abs(t.e)).astype("int64")
    ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    out = np.ones(x5070.shape, "float64")
    if not ok.any():
        return out
    halo = 1 if ramp else 0
    r_lo = max(0, int(row[ok].min()) - halo)
    r_hi = min(h - 1, int(row[ok].max()) + halo)
    c_lo = max(0, int(col[ok].min()) - halo)
    c_hi = min(w - 1, int(col[ok].max()) + halo)
    with rasterio.open(str(path)) as ds:
        arr = ds.read(
            1, window=Window(c_lo, r_lo, c_hi - c_lo + 1, r_hi - r_lo + 1)
        ).astype(bool)
    rr, cc = row[ok] - r_lo, col[ok] - c_lo
    on = arr[rr, cc]
    f = np.where(on, 0.0, 1.0)
    if ramp:
        ah, aw = arr.shape
        pad = np.zeros((ah + 2, aw + 2), bool)
        pad[1:-1, 1:-1] = arr
        nb = np.zeros(arr.shape, bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb |= pad[1 + dy : 1 + dy + ah, 1 + dx : 1 + dx + aw]
        f = np.where(~on & nb[rr, cc], RAMP_FACTOR, f)
    out[ok] = f
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--dem", default=DEM_100M)
    ap.add_argument(
        "--pilot-check",
        default=None,
        help="HUC8 whose probe mask (rem/waterflat_v2/waterflat_masks.npz m_v3) "
        "the CONUS mask is compared against",
    )
    ap.add_argument(
        "--force", action="store_true", help="rebuild even if outputs exist"
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / MASK_TIF
    count_path = out_dir / COUNT_TIF

    transform, width, height = canonical_grid(args.dem)
    log.info(
        "canonical lattice: %dx%d @ %g m, origin (%.0f, %.0f)",
        width,
        height,
        RES,
        transform.c,
        transform.f,
    )

    if count_path.exists() and not args.force:
        log.info("counts exist, loading %s", count_path)
        with rasterio.open(count_path) as ds:
            counts = ds.read(1).astype("int32")
    else:
        counts = occ_pixel_counts(transform, width, height)
        write_uint(count_path, np.clip(counts, 0, 65535), transform, "uint16")
        log.info("wrote %s", count_path)

    radius_cells = int(round(DENSITY_RADIUS_M / RES))
    dens = disk_sum(counts, radius_cells)
    mask = (counts > 0) & (dens >= DENSITY_MIN_PX)
    n_mask = int(mask.sum())
    log.info(
        "V3 mask: %d cells (%.1f km2); occ>=%g cells before screen %d (%.1f km2)",
        n_mask,
        n_mask * RES * RES / 1e6,
        OCC_MIN_PCT,
        int((counts > 0).sum()),
        int((counts > 0).sum()) * RES * RES / 1e6,
    )
    write_uint(mask_path, mask, transform, "uint8")
    log.info("wrote %s", mask_path)

    shore = mask_boundary(mask)
    n_shore = int(shore.sum())
    log.info(
        "mask boundary (kNN pool + pseudo-label pool): %d of %d masked cells "
        "(%.1f %%); interior cells dropped: %d",
        n_shore,
        n_mask,
        100 * n_shore / max(n_mask, 1),
        n_mask - n_shore,
    )
    rows, cols = np.where(shore)
    cells = pd.DataFrame(
        {
            "x5070": transform.c + (cols + 0.5) * RES,
            "y5070": transform.f - (rows + 0.5) * RES,
        }
    )
    cells.to_parquet(out_dir / CELLS_PARQUET)
    log.info("wrote %s (%d rows)", out_dir / CELLS_PARQUET, len(cells))

    (out_dir / "build_manifest.json").write_text(
        json.dumps(
            {
                "recipe": "V3 = JRC GSW occurrence >= 50 % AND >= 200 occ>=50 30 m "
                "pixels within 1 km (no fac clause -- see "
                "notes/NAIP_WATER_EVIDENCE_PLAN.md section 11)",
                "occ_min_pct": OCC_MIN_PCT,
                "density_radius_m": DENSITY_RADIUS_M,
                "density_min_px": DENSITY_MIN_PX,
                "density_units": "count of 30 m JRC occ>=50 pixels inside the radius "
                "(dimensionless count)",
                "grid": {
                    "crs": "EPSG:5070",
                    "res_m": RES,
                    "width": width,
                    "height": height,
                    "origin_x": transform.c,
                    "origin_y": transform.f,
                    "reference": args.dem,
                },
                "cell_semantics": "a 100 m cell is masked when ANY covering 30 m JRC "
                "pixel passes the screen (any-overlap, matching the pilot probe)",
                "gsw_tiles": sorted(p.name for p in GSW_OCC_DIR.glob("*.tif")),
                "n_cells_occ50": int((counts > 0).sum()),
                "n_cells_masked": n_mask,
                "n_cells_boundary": n_shore,
                "area_km2_occ50": float((counts > 0).sum() * RES * RES / 1e6),
                "area_km2_masked": float(n_mask * RES * RES / 1e6),
                "cells_parquet_semantics": "MASK BOUNDARY cells only (a masked cell "
                "with at least one unmasked 8-neighbour). The full mask is ~1.5e8 "
                "cells CONUS -- JRC covers the ocean and the 100 m DEM has no "
                "offshore nodata, so open water is admitted by construction and a "
                "kD-tree over it is intractable. Dropping interior cells is EXACT "
                "for the nearest-water features: for any query point off the mask "
                "the nearest masked cell is necessarily a boundary cell (the "
                "neighbour of an interior cell in the direction of the query is "
                "masked and strictly closer), so distances are unchanged. On-mask "
                "points carry the on_water flag read from the raster, so their "
                "membership does not depend on the pool. The only distortion is a "
                "query point deep inside a wide waterbody, whose distance is to the "
                "shoreline rather than ~0.",
                "outputs": {
                    "mask": str(mask_path),
                    "counts": str(count_path),
                    "cells": str(out_dir / CELLS_PARQUET),
                },
            },
            indent=2,
        )
    )
    log.info("wrote %s", out_dir / "build_manifest.json")

    if args.pilot_check:
        pilot_check(args.pilot_check, mask_path)


def pilot_check(basin: str, mask_path: Path) -> dict:
    """Agreement of the CONUS mask with the pilot probe's exact per-pixel mask.

    The probe screened each 30 m occ>=50 PIXEL by its own KD-tree ball count and
    then took any-overlap to the 100 m lattice; the CONUS build accumulates 30 m
    pixel counts into 100 m cells and screens each CELL by a 1 km disk sum. The
    two differ only in where the 1 km window is centred, so this quantifies the
    quantisation cost on the basin where the flatten was validated.
    """
    npz = Path(f"/data/ssd2/handily/huc8/{basin}/rem/waterflat_v2/waterflat_masks.npz")
    d = np.load(npz)
    qx, qy, ref = d["qx"], d["qy"], d["m_v3"].astype(bool)
    got = sample_water_mask(qx, qy, mask_path)
    both = int((ref & got).sum())
    log.info(
        "pilot %s: probe m_v3 %d cells, CONUS mask %d cells, intersection %d "
        "(recall %.3f, precision %.3f) over %d in-boundary lattice cells; "
        "occ>=50 unscreened %d",
        basin,
        int(ref.sum()),
        int(got.sum()),
        both,
        both / max(int(ref.sum()), 1),
        both / max(int(got.sum()), 1),
        len(qx),
        int(d["m_occ50"].sum()),
    )
    return {
        "basin": basin,
        "n_probe": int(ref.sum()),
        "n_conus": int(got.sum()),
        "n_intersect": both,
    }


if __name__ == "__main__":
    main()
