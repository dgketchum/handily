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

Resumable per HUC8 (skips a HUC8 whose shard already exists). rem_fac runs as a
fresh subprocess per HUC8 so a single failure cannot abort the pilot.
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
from rasterio.features import rasterize

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

# WBT intermediates safe to delete after a successful build (large, derivable).
_CLEANUP = (
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


def build_one_huc8(huc8, label, poly, flow_join, wells_unique, args) -> bool:
    out_root = Path(args.out_root)
    shard_dir = out_root / "fac_rem_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_dir / f"{huc8}.parquet"
    if shard.exists() and not args.force:
        log.info("[%s %s] shard exists -> skip", huc8, label)
        return True

    workdir = out_root / huc8
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
            [sys.executable, "-m", "handily.rem_fac", "--config", str(cfg)],
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

    if not args.keep_intermediates:
        for name in _CLEANUP:
            (workdir / name).unlink(missing_ok=True)
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
        for thr in (2, 5, 10):
            tp = int(((cf.fac_rem_dtw_m < thr) & (cf.mean_dtw < thr)).sum())
            obs = int((cf.mean_dtw < thr).sum())
            pred = int((cf.fac_rem_dtw_m < thr).sum())
            log.info(
                "  fac_rem shallow<%dm: recall=%.3f precision=%.3f (obs=%d)",
                thr,
                tp / max(obs, 1),
                tp / max(pred, 1),
                obs,
            )

    for huc8 in sorted(df["huc8"].unique()):
        show(df[df["huc8"] == huc8], f"HUC8 {huc8} {PILOT.get(huc8, '')}")
    show(df, "POOLED (all pilot HUC8s)")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--huc8", nargs="*", help="HUC8 codes (default: pilot set)")
    p.add_argument("--out-root", default=OUT_ROOT)
    p.add_argument("--halo-km", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--keep-intermediates", action="store_true")
    p.add_argument("--force", action="store_true", help="rebuild even if shard exists")
    p.add_argument("--validate-only", action="store_true")
    args = p.parse_args(argv)

    if args.validate_only:
        validate(args)
        return

    huc8s = args.huc8 or list(PILOT)
    log.info("pilot HUC8s: %s", huc8s)

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
    if ok:
        validate(args)


if __name__ == "__main__":
    main()
