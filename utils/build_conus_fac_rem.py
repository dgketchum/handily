"""Per-HUC8 CONUS FAC-REM builder (Phase-2 pilot).

Builds a static-seed FAC-REM terrain feature at each well, windowed per HUC8 so
the basin-wide build never has to fit in memory. For each HUC8:

  1. 3DEP 10 m DEM over the HUC8 polygon + halo (regional_fac.download/build).
  2. WhiteboxTools D8 flow accumulation (regional_fac.compute_regional_fac) ->
     flow_accumulation.tif used only for reach orientation + base snap.
  3. NHDPlus V2 flowlines clipped to the halo (COMID-keyed, joined to reach
     attributes: streamorde, nhd_class perenniality, totdasqkm drainage).
  4. NHD open water (NHDWaterbody + NHDArea) rasterized to a 10 m hard-support
     mask aligned to the DEM grid.
  5. rem_fac head-solve with ``seed_from_nhd_class`` (soft seed from NHD
     perenniality, no per-AOI imagery) -> fac_head_depth_rem_10m.tif.
  6. Sample the REM at wells whose huc8 == this HUC8 (the unbuffered footprint)
     -> fac_rem_shards/<huc8>.parquet (canonical_id, fac_rem_dtw_m).

Resumable per HUC8: a HUC8 is skipped only when BOTH its well shard and its 100 m
mosaic-completion marker (mosaic_markers/<huc8>.done) exist. A shard without a
marker (e.g. from a pre-mosaic run, or a run whose mosaic write failed) triggers a
mosaic backfill from the surviving 10 m REM rather than a silent skip; if the 10 m
REM has already been cleaned up, that HUC8 must be rebuilt with --force. rem_fac
runs as a fresh subprocess per HUC8 so a single failure cannot abort the pilot.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import Affine, windows
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from handily import regional_fac

log = logging.getLogger("build_conus_fac_rem")

# --- shared inputs (CONUS-wide, read once) -------------------------------
HUC8_POLYS = "/data/ssd2/handily/conus/wte_gnn/huc8_polys.parquet"
FLOWLINE_GEOM = "/data/ssd2/handily/conus/wte_gnn/nhd_flowline_geom.parquet"
REACH_NODES = "/data/ssd2/handily/conus/wte_gnn/graph/reach_nodes.parquet"
WELLS = "/data/ssd2/handily/conus/wte_gnn/conus_wells_hand.parquet"
NHD_GDB = (
    "/data/ssd2/handily/conus/cache/NHDPlusNationalData/"
    "NHDPlusV21_National_Seamless_Flattened_Lower48.gdb"
)
PROFILE = "/home/dgketchum/code/handily/configs/rem/profiles/conus_fac_rem.toml"
OUT_ROOT = "/data/ssd2/handily/conus/fac_rem"

# --- 100 m CONUS retention grid ------------------------------------------
# Per-HUC8 10 m REM rasters are first-class products but too large to keep
# CONUS-wide, so each is downsampled (average) onto this shared 100 m grid and
# appended to a mosaic. The grid is defined pixel-for-pixel from the DTW stacker
# grid (wte_dtw_100m_5070.tif) so every retained tile co-registers with it.
COMMON_GRID = "/data/ssd2/handily/conus/hydrography90m/wte_dtw_100m_5070.tif"
MOSAIC_CRS = "EPSG:5070"
MOSAIC_TRANSFORM = Affine(100.0, 0.0, -2540000.0, 0.0, -100.0, 3258000.0)
MOSAIC_WIDTH = 49810
MOSAIC_HEIGHT = 31390
MOSAIC_NODATA = -9999.0
# FAC-REM = depth (shallow water-table prior); WS = strip-fill-IDW water-surface
# ELEVATION. Distinct surfaces, named so they can never be confused (see
# notes/FAC_REM_WTE_NAMING_AND_RETENTION_HANDOFF.md).
REM_MOSAIC = "fac_rem_dtw_100m_5070.tif"
WS_MOSAIC = "fac_rem_water_surface_100m_5070.tif"

# Regime-spread pilot HUC8s (verified to exist with unconfined wells).
PILOT = {
    "13030102": "arid_rift_mesilla",
    "13020203": "arid_rift_albuquerque",
    "02080109": "humid_atlantic_coastal",
    "10200101": "montane_mt_missouri_hw",
    "11030010": "high_plains",
    "17010205": "pnw_montane",
}

UNCONFINED = ("unconfined", "unconfined_marginal")

# Per-HUC8 intermediates safe to delete after the shard is written (large,
# fully derivable from inputs). These dominate per-HUC8 disk (3-4 GB each on a
# large HUC8); leaving them in place would blow the CONUS-wide build past disk.
#
# The 10 m REM depth (fac_head_depth_rem_10m.tif) and water surface
# (fac_rem_water_surface_10m.tif) are deliberately NOT here: they are first-class
# model products, not scratch. They are downsampled into the 100 m mosaics
# (_retain_mosaics) and only then deleted, so a mosaic failure never loses them.
_CLEANUP = (
    # rem_fac intermediate rasters (3-4 GB each on large HUC8s)
    "fac_head_depth_sparse_10m.tif",
    "fac_normals_smoothed_dem.tif",
    "fac_normals_cross_sections.fgb",
    # terrain inputs (re-derivable from the shared dem_tiles cache + NHD)
    "dem_10m.tif",
    "flow_accumulation.tif",
    # legacy WBT intermediate names (harmless if absent on this rem_fac variant)
    "dem_10m_filled.tif",
    "d8_pointer.tif",
    "streams_10m.tif",
    "stream_order.tif",
    "streams_raw.shp",
    "streams_raw.shx",
    "streams_raw.dbf",
    "streams_raw.prj",
    "streams_regional.fgb",  # WBT-vectorized streams (unused; we use NHD flowlines)
    "dem_10m.vrt",
    "basin_cutline.fgb",
)


def _haloed(poly, halo_m: float):
    """Return (haloed_5070_geom, bbox_wgs84, bbox_nad83) for a 5070 polygon."""
    halo = poly.buffer(halo_m)
    gs = gpd.GeoSeries([halo], crs=5070)
    bbox_wgs84 = tuple(gs.to_crs(4326).total_bounds)
    bbox_nad83 = tuple(gs.to_crs(4269).total_bounds)
    return halo, bbox_wgs84, bbox_nad83


def _build_streams(flow_join: gpd.GeoDataFrame, halo, out_path: Path) -> int:
    """Clip COMID flowlines to the halo and write a rem_fac streams file."""
    idx = flow_join.sindex.query(halo, predicate="intersects")
    sub = flow_join.iloc[idx].copy()
    sub = sub[sub.geometry.notnull() & ~sub.geometry.is_empty]
    if sub.empty:
        raise SystemExit("no NHD flowlines intersect the HUC8 halo")

    cols = {
        "stream_id": sub["comid"].astype("int64"),
        "reach_id": sub["comid"].astype("int64"),
        "strahler": sub["streamorde"].fillna(0).astype("int64"),
        "length_m": (sub["lengthkm"].fillna(0.0) * 1000.0).astype(float),
        "nhd_class": sub["nhd_class"].fillna("other").astype(str),
        "geometry": sub.geometry.values,
    }
    # Prefer authoritative NHD drainage, but only if complete for this window —
    # a partial NaN drainage column would corrupt the head-solve sag targets, so
    # fall back to windowed FAC (omit the column) rather than silently patch.
    n_da_nan = int(sub["totdasqkm"].isna().sum())
    if n_da_nan == 0:
        cols["drainage_km2"] = sub["totdasqkm"].astype(float)
    else:
        log.warning(
            "  %d/%d reaches lack totdasqkm -> using windowed FAC drainage",
            n_da_nan,
            len(sub),
        )
    streams = gpd.GeoDataFrame(cols, geometry="geometry", crs=5070)
    streams.to_file(out_path, driver="FlatGeobuf")
    return len(streams)


def _build_support_mask(dem_path: Path, bbox_nad83, out_path: Path) -> int:
    """Rasterize NHD open water to a 10 m binary mask aligned to the DEM grid."""
    with rasterio.open(dem_path) as d:
        transform = d.transform
        shape = (d.height, d.width)
        crs = d.crs

    parts = []
    for layer in ("NHDWaterbody", "NHDArea"):
        try:
            g = gpd.read_file(NHD_GDB, layer=layer, bbox=bbox_nad83)
        except Exception as e:  # noqa: BLE001 - missing/empty bbox read is non-fatal
            log.warning("  %s read failed (%s); skipping layer", layer, e)
            continue
        if len(g):
            parts.append(g.to_crs(5070)[["geometry"]])

    mask = np.zeros(shape, dtype="uint8")
    n_feat = 0
    if parts:
        water = pd.concat(parts, ignore_index=True)
        water = water[water.geometry.notnull() & ~water.geometry.is_empty]
        n_feat = len(water)
        if n_feat:
            mask = rasterize(
                ((geom, 1) for geom in water.geometry),
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype="uint8",
            )
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(mask, 1)
    return n_feat


def _write_config(workdir: Path, dem: Path, streams: Path, fac: Path, support: Path):
    cfg = workdir / "fac_rem.toml"
    cfg.write_text(
        f'profile = "{PROFILE}"\n\n'
        "[paths]\n"
        f'dem_path = "{dem}"\n'
        f'streams_path = "{streams}"\n'
        f'fac_path = "{fac}"\n'
        f'support_path = "{support}"\n'
        f'out_dir = "{workdir}"\n'
    )
    return cfg


def _sample_wells(rem_path: Path, wells_huc8: pd.DataFrame) -> pd.DataFrame:
    coords = list(zip(wells_huc8["x5070"], wells_huc8["y5070"]))
    with rasterio.open(rem_path) as r:
        nodata = r.nodata
        vals = np.array([v[0] for v in r.sample(coords)], dtype=float)
    if nodata is not None:
        vals[vals == nodata] = np.nan
    vals[~np.isfinite(vals)] = np.nan
    return pd.DataFrame(
        {
            "canonical_id": wells_huc8["canonical_id"].to_numpy(),
            "huc8": wells_huc8["huc8"].to_numpy(),
            "fac_rem_dtw_m": vals,
        }
    )


def _mosaic_profile() -> dict:
    """rasterio profile for a 100 m CONUS mosaic on the common grid."""
    return {
        "driver": "GTiff",
        "height": MOSAIC_HEIGHT,
        "width": MOSAIC_WIDTH,
        "count": 1,
        "dtype": "float32",
        "crs": MOSAIC_CRS,
        "transform": MOSAIC_TRANSFORM,
        "nodata": MOSAIC_NODATA,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "lzw",
        "BIGTIFF": "YES",
    }


def _ensure_mosaic(path: Path) -> None:
    """Create an all-nodata 100 m mosaic on the common grid if it is absent.

    Initialised explicitly to nodata (not sparse) so that any HUC8 never built
    reads back as nodata rather than a spurious 0 depth.
    """
    if path.exists():
        return
    prof = _mosaic_profile()
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **prof) as dst:
        block = int(prof["blockysize"])
        fill = np.full((block, dst.width), MOSAIC_NODATA, dtype="float32")
        for r0 in range(0, dst.height, block):
            h = min(block, dst.height - r0)
            dst.write(fill[:h], 1, window=Window(0, r0, dst.width, h))


def _grid_window(bounds, transform, width: int, height: int) -> Window | None:
    """Integer, in-bounds mosaic window covering *bounds* (5070)."""
    minx, miny, maxx, maxy = bounds
    win = windows.from_bounds(minx, miny, maxx, maxy, transform)
    col0 = max(0, int(np.floor(win.col_off)))
    row0 = max(0, int(np.floor(win.row_off)))
    col1 = min(width, int(np.ceil(win.col_off + win.width)))
    row1 = min(height, int(np.ceil(win.row_off + win.height)))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def _append_to_mosaic(src_path: Path, poly, mosaic_path: Path) -> int:
    """Downsample a 10 m raster to the 100 m grid and write the unbuffered HUC8
    footprint into the CONUS mosaic in place. Returns cells written.

    The 10 m REM is built over a halo; only cells inside the unbuffered HUC8
    polygon are written so adjacent HUC8s never overwrite each other's footprint.
    """
    _ensure_mosaic(mosaic_path)
    with rasterio.open(mosaic_path, "r+") as dst:
        win = _grid_window(poly.bounds, dst.transform, dst.width, dst.height)
        if win is None:
            return 0
        win_transform = windows.transform(win, dst.transform)
        h, w = int(win.height), int(win.width)

        downsampled = np.full((h, w), MOSAIC_NODATA, dtype="float32")
        with rasterio.open(src_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=downsampled,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=win_transform,
                dst_crs=dst.crs,
                dst_nodata=MOSAIC_NODATA,
                resampling=Resampling.average,
            )
        keep = rasterize(
            [(poly, 1)],
            out_shape=(h, w),
            transform=win_transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        new = keep & (downsampled != MOSAIC_NODATA) & np.isfinite(downsampled)
        block = dst.read(1, window=win)
        # Clear the whole unbuffered footprint first, then write finite cells.
        # Footprints partition CONUS (unbuffered HUC8 polygons don't overlap), so
        # this never wipes a neighbour; it does ensure a rebuild whose new tile is
        # nodata over a previously-finite cell clears the stale value instead of
        # leaving it behind.
        block[keep] = MOSAIC_NODATA
        block[new] = downsampled[new]
        dst.write(block, 1, window=win)
        return int(new.sum())


def _mosaic_marker(out_root: Path, huc8) -> Path:
    """Sentinel written once a HUC8's 10 m REM/WS are downsampled into the 100 m
    mosaics. Its presence (alongside the shard) is the resume signal that the
    durable mosaic retention is complete, not just the well shard."""
    return out_root / "mosaic_markers" / f"{huc8}.done"


def _retain_mosaics(huc8, rem_path: Path, ws_path: Path, poly, out_root: Path) -> bool:
    """Downsample the 10 m REM (depth) and water surface (elevation) into the
    100 m CONUS mosaics. Returns True only if the REM mosaic write succeeded —
    the gate for deleting the 10 m rasters.
    """
    try:
        n = _append_to_mosaic(rem_path, poly, out_root / REM_MOSAIC)
        log.info("[%s] +%d cells -> %s", huc8, n, REM_MOSAIC)
        if ws_path.exists():
            nw = _append_to_mosaic(ws_path, poly, out_root / WS_MOSAIC)
            log.info("[%s] +%d cells -> %s", huc8, nw, WS_MOSAIC)
        return True
    except Exception as e:  # noqa: BLE001 - keep the 10 m REM if mosaicking fails
        log.exception("[%s] 100 m mosaic append failed: %s", huc8, e)
        return False


def finalize_cog(out_root: Path) -> None:
    """Convert the incremental 100 m mosaics to true COGs (overviews + COG IFD
    layout) as ``*_cog.tif``. Run once after a CONUS batch finishes -- the
    per-HUC8 mosaics are plain tiled BigTIFFs because a COG cannot be appended to
    incrementally.
    """
    for name in (REM_MOSAIC, WS_MOSAIC):
        src = out_root / name
        if not src.exists():
            log.info("no %s to finalize -> skip", name)
            continue
        dst = src.with_name(f"{src.stem}_cog.tif")
        cmd = [
            "gdal_translate",
            str(src),
            str(dst),
            "-of",
            "COG",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "RESAMPLING=AVERAGE",
            "-co",
            "BIGTIFF=YES",
        ]
        log.info("finalizing COG: %s -> %s", src.name, dst.name)
        subprocess.run(cmd, check=True)
        log.info("wrote %s", dst)


def build_one_huc8(huc8, label, poly, flow_join, wells_unique, args) -> bool:
    out_root = Path(args.out_root)
    shard_dir = out_root / "fac_rem_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"{huc8}.parquet"
    workdir = out_root / huc8
    marker = _mosaic_marker(out_root, huc8)

    if shard.exists() and not args.force:
        if marker.exists():
            log.info("[%s %s] shard + mosaic retained -> skip", huc8, label)
            return True
        # Shard predates the 100 m mosaics (pre-mosaic run) or a prior mosaic
        # write failed. Backfill from the surviving 10 m rasters if present rather
        # than report "built" with no mosaic cells; if the 10 m REM is already
        # cleaned up, the only recovery is a --force rebuild.
        rem_path = workdir / "fac_head_depth_rem_10m.tif"
        ws_path = workdir / "fac_rem_water_surface_10m.tif"
        if rem_path.exists():
            if _retain_mosaics(huc8, rem_path, ws_path, poly, out_root):
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                if not args.keep_intermediates:
                    rem_path.unlink(missing_ok=True)
                    ws_path.unlink(missing_ok=True)
                log.info("[%s %s] backfilled 100 m mosaics from 10 m REM", huc8, label)
            else:
                log.error(
                    "[%s %s] mosaic backfill failed; keeping 10 m REM", huc8, label
                )
            return True
        log.warning(
            "[%s %s] shard exists but mosaics never retained and 10 m REM is gone "
            "-> rerun with --force to rebuild and retain",
            huc8,
            label,
        )
        return True

    workdir.mkdir(parents=True, exist_ok=True)
    dem_tiles = out_root / "dem_tiles"  # shared 3DEP cache across HUC8s
    log.info("[%s %s] building", huc8, label)

    halo, bbox_wgs84, bbox_nad83 = _haloed(poly, args.halo_km * 1000.0)

    # 1-2. DEM + FAC
    tiles = regional_fac.download_3dep_10m_tiles(bbox_wgs84, dem_tiles)
    if not tiles:
        log.warning("[%s] no 3DEP tiles (border/ocean) -> skip", huc8)
        return False
    dem_path = workdir / "dem_10m.tif"
    regional_fac.build_regional_dem(
        tiles,
        gpd.GeoDataFrame({"geometry": [poly]}, crs=5070),
        dem_path,
        target_crs_epsg=5070,
        buffer_m=args.halo_km * 1000.0,
    )
    regional_fac.compute_regional_fac(dem_path, workdir, max_procs=args.workers)
    fac_path = workdir / "flow_accumulation.tif"

    # 3-4. NHD streams + open-water support mask
    streams_path = workdir / "streams.fgb"
    n_streams = _build_streams(flow_join, halo, streams_path)
    support_path = workdir / "nhd_water_support.tif"
    n_water = _build_support_mask(dem_path, bbox_nad83, support_path)
    log.info("[%s] %d reaches, %d NHD water features", huc8, n_streams, n_water)

    # 5. rem_fac head-solve (fresh subprocess; isolated env via venv python)
    cfg = _write_config(workdir, dem_path, streams_path, fac_path, support_path)
    run_log = workdir / "rem_fac.log"
    with open(run_log, "w") as fh:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "handily.rem_fac",
                "--config",
                str(cfg),
                # The strip layer is a burn intermediate, not a product, and is
                # cleaned up anyway -- skip materializing tens of millions of
                # LineStrings to disk at CONUS scale.
                "--no-strip-debug",
            ],
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        log.error("[%s] rem_fac failed (rc=%d); see %s", huc8, proc.returncode, run_log)
        return False

    rem_path = workdir / "fac_head_depth_rem_10m.tif"
    if not rem_path.exists():
        log.error("[%s] rem_fac produced no REM raster", huc8)
        return False

    # 6. sample wells in the unbuffered footprint (huc8 assignment) -> shard
    wells_huc8 = wells_unique[wells_unique["huc8"] == huc8]
    out = _sample_wells(rem_path, wells_huc8)
    finite = int(out["fac_rem_dtw_m"].notna().sum())
    out.to_parquet(shard, index=False)
    log.info(
        "[%s] sampled %d wells (%d finite, %.0f%%) -> %s",
        huc8,
        len(out),
        finite,
        100.0 * finite / max(len(out), 1),
        shard.name,
    )

    # Retained durable products: the 10 m REM (depth) and water surface
    # (elevation) are first-class outputs but too large to keep CONUS-wide.
    # Downsample them into the 100 m mosaics on the common grid, then drop the
    # 10 m rasters -- but only after the mosaic write succeeds, so a mosaic
    # failure never silently loses them.
    ws_path = workdir / "fac_rem_water_surface_10m.tif"
    mosaic_ok = _retain_mosaics(huc8, rem_path, ws_path, poly, out_root)
    if mosaic_ok:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()

    if not args.keep_intermediates:
        for name in _CLEANUP:
            (workdir / name).unlink(missing_ok=True)
        if mosaic_ok:
            rem_path.unlink(missing_ok=True)
            ws_path.unlink(missing_ok=True)
        else:
            log.warning(
                "[%s] keeping 10 m REM/water surface in %s (mosaic write failed)",
                huc8,
                workdir,
            )
    return True


def _panel(resid: np.ndarray) -> dict:
    r = resid[np.isfinite(resid)]
    if not len(r):
        return {"n": 0}
    return {
        "n": int(len(r)),
        "mad": float(np.median(np.abs(r))),
        "bias": float(np.mean(r)),
        "medR": float(np.median(r)),
        "rmse": float(np.sqrt(np.mean(r**2))),
    }


def validate(args) -> None:
    """Per-HUC8 + pooled panel: fac_rem vs hand vs janssen against mean_dtw."""
    out_root = Path(args.out_root)
    shards = sorted((out_root / "fac_rem_shards").glob("*.parquet"))
    if not shards:
        raise SystemExit("no shards to validate")
    fac = pd.concat((pd.read_parquet(s) for s in shards), ignore_index=True)
    w = pd.read_parquet(
        WELLS,
        columns=[
            "canonical_id",
            "huc8",
            "confinement_class",
            "well_class",
            "source",
            "mean_dtw",
            "hand_m",
            "janssen_dtw",
        ],
    ).drop_duplicates("canonical_id")
    df = fac.drop(columns=["huc8"]).merge(w, on="canonical_id", how="inner")
    # water-table wells only; exclude NWIS/NGWMN (benchmark leakage)
    df = df[df["confinement_class"].isin(UNCONFINED)]
    df = df[~df["source"].astype(str).str.lower().isin(("nwis", "ngwmn"))]
    bands = [(0, 2), (2, 5), (5, 10), (10, 30), (30, 1e9)]

    def show(sub: pd.DataFrame, title: str) -> None:
        # common footprint: rows where all three predictors are finite
        m = sub[["fac_rem_dtw_m", "hand_m", "janssen_dtw", "mean_dtw"]].notna().all(1)
        cf = sub[m]
        log.info("=== %s (n=%d, common-fp n=%d) ===", title, len(sub), len(cf))
        for name, col in (
            ("fac_rem", "fac_rem_dtw_m"),
            ("hand", "hand_m"),
            ("janssen", "janssen_dtw"),
        ):
            p = _panel((cf[col] - cf["mean_dtw"]).to_numpy())
            line = " ".join(
                f"{p.get(k, float('nan')):.2f}" if k != "n" else f"n={p['n']}"
                for k in ("n", "mad", "bias", "medR", "rmse")
            )
            db = " ".join(
                f"{lo}-{hi if hi < 1e8 else '+'}m:"
                f"{_panel((g[col] - g['mean_dtw']).to_numpy()).get('mad', float('nan')):.1f}"
                for (lo, hi) in bands
                for g in [cf[(cf.mean_dtw >= lo) & (cf.mean_dtw < hi)]]
            )
            log.info("  %-8s %s | %s", name, line, db)
        # shallow skill for every predictor so the GO/NO-GO gate (fac_rem vs HAND
        # <5m recall) reads directly off the panel.
        for thr in (2, 5, 10):
            obs_mask = cf.mean_dtw < thr
            obs = int(obs_mask.sum())
            for name, col in (
                ("fac_rem", "fac_rem_dtw_m"),
                ("hand", "hand_m"),
                ("janssen", "janssen_dtw"),
            ):
                pred_mask = cf[col] < thr
                tp = int((pred_mask & obs_mask).sum())
                pred = int(pred_mask.sum())
                log.info(
                    "  %-8s shallow<%dm: recall=%.3f precision=%.3f (obs=%d)",
                    name,
                    thr,
                    tp / max(obs, 1),
                    tp / max(pred, 1),
                    obs,
                )

    for huc8 in sorted(df["huc8"].unique()):
        show(df[df["huc8"] == huc8], f"HUC8 {huc8} {PILOT.get(huc8, '')}")
    show(df, "POOLED (all pilot HUC8s)")


def _validate_halo(halo_km: float) -> None:
    """Warn loudly if the halo is too thin for the active profile's reach.

    A strip can travel up to ``max_crossing_strip_m`` to find a neighbor stream,
    and the sparse burn is then IDW-filled out to ``idw_radius_m``. If the halo
    is narrower than their sum, an interior cell near the unbuffered HUC8 edge can
    draw on streams/strips that were clipped away at the halo boundary, producing
    a seam in the 100 m mosaic. This is advisory (some border HUC8s legitimately
    run thin), not a hard stop.
    """
    # The profile defines no [paths], so it cannot be instantiated as a full
    # FacRemConfig; read the two values we need straight from the (profile-aware)
    # raw TOML, falling back to the dataclass defaults if a key is absent.
    from handily.rem_fac_config import _load_toml_with_profile

    raw, _ = _load_toml_with_profile(Path(PROFILE))
    max_crossing = float(raw.get("strips", {}).get("max_crossing_strip_m", 0.0))
    idw_radius = float(raw.get("raster", {}).get("idw_radius_m", 200.0))
    needed_m = max_crossing + idw_radius
    halo_m = halo_km * 1000.0
    if halo_m < needed_m:
        log.warning(
            "HALO TOO THIN: halo_km=%.1f (%.0f m) < max_crossing_strip_m"
            " (%.0f) + idw_radius_m (%.0f) = %.0f m. Cells near the HUC8 edge"
            " may seam in the mosaic; raise --halo-km to >= %.1f.",
            halo_km,
            halo_m,
            max_crossing,
            idw_radius,
            needed_m,
            needed_m / 1000.0,
        )
    else:
        log.info(
            "halo OK: %.0f m >= max_crossing_strip_m (%.0f) + idw_radius_m (%.0f)",
            halo_m,
            max_crossing,
            idw_radius,
        )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--huc8", nargs="*", help="HUC8 codes (default: pilot set)")
    p.add_argument(
        "--huc8-list",
        help="file with one HUC8 code per line (for CONUS-wide batches)",
    )
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--halo-km", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--keep-intermediates", action="store_true")
    p.add_argument("--force", action="store_true", help="rebuild even if shard exists")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the post-build validation panel (for long batches)",
    )
    p.add_argument(
        "--finalize-cog",
        action="store_true",
        help="convert the incremental 100m mosaics to true COGs and exit",
    )
    args = p.parse_args(argv)

    if args.finalize_cog:
        finalize_cog(Path(args.out_root))
        return

    if args.validate_only:
        validate(args)
        return

    _validate_halo(args.halo_km)

    if args.huc8_list:
        listed = [
            ln.strip()
            for ln in Path(args.huc8_list).read_text().splitlines()
            if ln.strip() and not ln.startswith("#")
        ]
        huc8s = listed + list(args.huc8 or [])
    else:
        huc8s = args.huc8 or list(PILOT)
    log.info("target HUC8s: %d", len(huc8s))

    polys = gpd.read_parquet(HUC8_POLYS).set_index("huc8")
    log.info("loading flowline geometry + reach attributes ...")
    flow = gpd.read_parquet(FLOWLINE_GEOM)
    rn = pd.read_parquet(
        REACH_NODES,
        columns=["comid", "streamorde", "nhd_class", "totdasqkm", "lengthkm"],
    )
    flow_join = flow.merge(rn, on="comid", how="left")
    wells_unique = pd.read_parquet(
        WELLS, columns=["canonical_id", "huc8", "x5070", "y5070"]
    ).drop_duplicates("canonical_id")
    log.info("flowlines=%d wells=%d", len(flow_join), len(wells_unique))

    ok = 0
    for huc8 in huc8s:
        if huc8 not in polys.index:
            log.warning("HUC8 %s not in polygons -> skip", huc8)
            continue
        label = PILOT.get(huc8, "custom")
        try:
            ok += bool(
                build_one_huc8(
                    huc8,
                    label,
                    polys.loc[huc8, "geometry"],
                    flow_join,
                    wells_unique,
                    args,
                )
            )
        except Exception as e:  # noqa: BLE001 - keep the pilot going past one failure
            log.exception("[%s] build failed: %s", huc8, e)
    log.info("built %d/%d HUC8s", ok, len(huc8s))
    if ok and not args.no_validate:
        validate(args)


if __name__ == "__main__":
    main()
