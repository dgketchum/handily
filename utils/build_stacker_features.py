"""Assemble the leakage-free stacker feature table at gwx unconfined wells.

The downstream model (level-1 stacker) combines two level-0 DTW predictors:

  * ConusWTE  -- the frozen regional water-table ELEVATION surface
    (`coarse_wte_frozen.tif`); DTW = land_surface_elev_m - WTE(x,y). It is fit on a
    SACRIFICIAL 1/5 single-obs well partition that is RETIRED via the manifest
    (`wte_frozen_retired_wells.parquet`). Because the stacker never trains/evals on a
    retired well, ConusWTE is a pre-built exogenous feature -- no cross-fitting.
  * ConusFAC  -- terrain-following FAC-REM HAND depth (`fac_rem_shards/*.parquet`,
    canonical_id -> fac_rem_dtw_m). Target-blind (never sees a well label), so it is
    a clean feature anywhere it exists. It is currently a 6-HUC8 PILOT, not CONUS-
    wide, so most rows have fac_rem_dtw_m = NaN; `fac_covered` flags the trainable
    subset.

Plus the regime gate (`coarse_wte_frozen_support.tif`, distance to nearest fit well:
large => ConusWTE extrapolated => lean on ConusFAC), regime discriminators that let
the stacker tell artifact-deep (2 km WTE sagging below a perched well) from genuinely
deep tables -- elev_above_coarse_m, slope_deg, tri_100m, dist_to_stream_m,
log_drainage_area -- and HUC8/4/2 codes for spatial-block CV (autocorrelation makes
random CV optimistic).

Leakage discipline enforced here: unconfined/unconfined_marginal only, dedup by
canonical_id (wells.geoparquet has multiple source rows per well), and hard-exclude
every retired canonical_id. Target = mean_dtw (observed depth to the unconfined
table). nwis/ngwmn are KEPT (source-agnostic model development); dropping them is a
separate Ma-fairness experiment, not a default.

Usage:
    uv run python utils/build_stacker_features.py \
        --out /data/ssd2/handily/conus/stacker/wte_fac_features.parquet
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

log = logging.getLogger("build_stacker_features")

HYDRO = "/data/ssd2/handily/conus/hydrography90m"
WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"
MANIFEST = f"{HYDRO}/wte_frozen_retired_wells.parquet"
WTE_FROZEN = f"{HYDRO}/coarse_wte_frozen.tif"
WTE_SUPPORT = f"{HYDRO}/coarse_wte_frozen_support.tif"
FAC_SHARD_DIR = "/data/ssd2/handily/conus/fac_rem/fac_rem_shards"
HUC8_POLYS = "/data/ssd2/handily/conus/wte_gnn/huc8_polys.parquet"
WT_CLASSES = ("unconfined", "unconfined_marginal")

# Regime discriminators (artifact-deep vs real-deep). coarse_surface is an exact
# grid-match to the frozen WTE (2 km); the rest are full-CONUS 100 m EPSG:5070.
# slope/TRI/dist-to-stream are precomputed once (see commands.sh) from the shared DEM
# and the Hydrography90m channel mask; accumulation already exists.
COARSE_SURFACE = f"{HYDRO}/coarse_surface.tif"
ACCUM = f"{HYDRO}/accumulation_conus_100m_5070.tif"
COV_DIR = "/data/ssd2/handily/conus/covariates"
SLOPE = f"{COV_DIR}/slope_deg.tif"
TRI = f"{COV_DIR}/tri_100m.tif"
DIST_STREAM = f"{COV_DIR}/dist_to_stream_m.tif"
# Aridity climatology (gridMET, EPSG:4326 -> sample at lon/lat, NOT x5070/y5070).
# The real-deep discriminator: arid basins (low AI = P/PET) carry genuinely deep
# regional tables, so the stacker must NOT pull them shallow the way it does the
# high-relief artifact-deep wells.
GRIDMET_AI = f"{COV_DIR}/gridmet_aridity_index.tif"
GRIDMET_P = f"{COV_DIR}/gridmet_mean_annual_precip_mm.tif"
# ETRM water-balance fluxes (CONUS 5070 1 km, 2020-2024 mean -> sample at x5070/y5070).
# The flux test of the real-deep hypothesis static aridity failed: recharge integrates
# P-ET-runoff, so low recharge flags genuinely deep regional tables (no water reaching
# the table) while ETa flags where water is actually used (shallow/accessible table).
ETRM_DIR = "/nas/etrm/conus/recharge_250m"
ETRM_RECHARGE = f"{ETRM_DIR}/conus_mean_recharge_2020-2024_albers1km.tif"
ETRM_ETA = f"{ETRM_DIR}/conus_mean_eta_2020-2024_albers1km.tif"
ETRM_RUNOFF = f"{ETRM_DIR}/conus_mean_runoff_2020-2024_albers1km.tif"


def sample_coarse(path: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Nearest-cell sample of a single-band EPSG:5070 raster at (x, y) in 5070 metres.

    Nearest (not bilinear) on purpose: cannot bleed the nodata fill into a valid
    sample near the domain edge. Out-of-grid and nodata -> NaN (not fabricated). The
    band is read in its native dtype and only the sampled points are cast to float64,
    so a full-CONUS 100 m raster (~6 GB float32) is not doubled to float64 in memory.
    """
    with rasterio.open(path) as src:
        arr = src.read(1)
        t = src.transform
        nd = src.nodata
    h, w = arr.shape
    col = np.floor((x - t.c) / t.a).astype(np.int64)
    row = np.floor((y - t.f) / t.e).astype(np.int64)
    ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    v = np.full(x.shape, np.nan, dtype=np.float64)
    v[ok] = arr[row[ok], col[ok]].astype(np.float64)
    if nd is not None:
        v[v == nd] = np.nan
    return v


def load_clean_wells(
    wells_path: str, manifest_path: str, wt_classes=WT_CLASSES
) -> pd.DataFrame:
    """Unconfined wells, deduped by canonical_id, with retired wells hard-excluded."""
    cols = [
        "canonical_id",
        "source",
        "confinement_class",
        "obs_count",
        "longitude",
        "latitude",
        "mean_dtw",
        "land_surface_elev_m",
    ]
    df = pd.read_parquet(wells_path, columns=cols)
    df = df[df.confinement_class.isin(wt_classes)]
    df = df[
        df.mean_dtw.notna()
        & df.land_surface_elev_m.notna()
        & df.longitude.notna()
        & df.latitude.notna()
    ]
    df = df.drop_duplicates("canonical_id")
    retired = set(pd.read_parquet(manifest_path)["canonical_id"])
    n_before = len(df)
    df = df[~df.canonical_id.isin(retired)]
    log.info(
        "clean wells: %d unconfined (deduped) - %d retired = %d",
        n_before,
        n_before - len(df),
        len(df),
    )
    return df.reset_index(drop=True)


def join_huc(df: pd.DataFrame, pts: gpd.GeoDataFrame, polys_path: str) -> pd.DataFrame:
    """Spatial-join HUC8 (and derive HUC4/HUC2) onto df via point-in-polygon."""
    polys = gpd.read_parquet(polys_path)[["huc8", "geometry"]]
    j = gpd.sjoin(pts.to_crs(polys.crs), polys, how="left", predicate="within")
    # A point on a shared edge can match >1 poly -> keep the first per well.
    j = j[~j.index.duplicated(keep="first")].reindex(df.index)
    huc8 = j["huc8"].astype("string").to_numpy()
    df["huc8"] = huc8
    df["huc4"] = pd.Series(huc8, index=df.index).str[:4]
    df["huc2"] = pd.Series(huc8, index=df.index).str[:2]
    n_huc = int(pd.Series(huc8).notna().sum())
    log.info(
        "HUC8 joined for %d / %d wells (%d outside CONUS HUC8 polys)",
        n_huc,
        len(df),
        len(df) - n_huc,
    )
    return df


def join_fac(df: pd.DataFrame, shard_dir: str) -> pd.DataFrame:
    """Left-merge fac_rem_dtw_m (ConusFAC pilot) on canonical_id."""
    shards = sorted(glob.glob(f"{shard_dir}/*.parquet"))
    if not shards:
        log.warning("no ConusFAC shards in %s; fac_rem_dtw_m all NaN", shard_dir)
        df["fac_rem_dtw_m"] = np.nan
        return df
    fac = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    fac = fac.drop_duplicates("canonical_id")[["canonical_id", "fac_rem_dtw_m"]]
    df = df.merge(fac, on="canonical_id", how="left")
    n_cov = int(df.fac_rem_dtw_m.notna().sum())
    log.info(
        "ConusFAC (%d-HUC8 pilot): %d / %d clean wells covered (%.1f%%)",
        len(shards),
        n_cov,
        len(df),
        100.0 * n_cov / max(len(df), 1),
    )
    return df


COVARIATES = (
    "elev_above_coarse_m",
    "slope_deg",
    "tri_100m",
    "dist_to_stream_m",
    "log_drainage_area",
    "aridity_index",
    "mean_annual_precip_mm",
    "etrm_recharge_mm",
    "etrm_eta_mm",
    "etrm_runoff_mm",
)


def add_covariates(
    df: pd.DataFrame,
    *,
    coarse_surface: str = COARSE_SURFACE,
    slope: str = SLOPE,
    tri: str = TRI,
    dist_stream: str = DIST_STREAM,
    accum: str = ACCUM,
    gridmet_ai: str = GRIDMET_AI,
    gridmet_p: str = GRIDMET_P,
    etrm_recharge: str = ETRM_RECHARGE,
    etrm_eta: str = ETRM_ETA,
    etrm_runoff: str = ETRM_RUNOFF,
) -> pd.DataFrame:
    """Sample the regime discriminators that separate artifact-deep from real-deep.

    - elev_above_coarse_m: well land surface minus its 2 km cell mean. Large positive
      => the well is perched above the cell, exactly where the 2 km WTE sags and
      fabricates depth (the artifact-deep flag).
    - slope_deg, tri_100m: local steepness / ruggedness (montane terrace regime where
      a smooth surface structurally cannot track relief).
    - dist_to_stream_m: near-stream (FAC/HAND regime) vs basin (WTE regime).
    - log_drainage_area: valley-axis vs upland position.
    - aridity_index, mean_annual_precip_mm: the REAL-deep discriminator. Relief flags
      artifact-deep; aridity flags genuinely deep regional tables (low AI = P/PET =>
      arid basin => deep, must NOT be pulled shallow like a high-relief artifact).

    The 5070 rasters are sampled at x5070/y5070; the gridMET rasters are EPSG:4326 so
    they are sampled at the wells' lon/lat (sample_coarse is CRS-agnostic -- it indexes
    by the raster's own transform, so the query units must match the raster CRS).

    Off-footprint / nodata stays NaN (HGB handles missing natively; no imputation).
    """
    x = df.x5070.to_numpy()
    y = df.y5070.to_numpy()
    lon = df.longitude.to_numpy()
    lat = df.latitude.to_numpy()
    df["elev_above_coarse_m"] = df.land_surface_elev_m.to_numpy() - sample_coarse(
        coarse_surface, x, y
    )
    df["slope_deg"] = sample_coarse(slope, x, y)
    df["tri_100m"] = sample_coarse(tri, x, y)
    df["dist_to_stream_m"] = sample_coarse(dist_stream, x, y)
    df["log_drainage_area"] = np.log1p(np.abs(sample_coarse(accum, x, y)))
    # gridMET aridity is EPSG:4326 -> sample at lon/lat (NOT 5070 metres).
    df["aridity_index"] = sample_coarse(gridmet_ai, lon, lat)
    df["mean_annual_precip_mm"] = sample_coarse(gridmet_p, lon, lat)
    # ETRM fluxes are EPSG:5070 1 km -> sample at x5070/y5070 (nodata -9999 -> NaN).
    df["etrm_recharge_mm"] = sample_coarse(etrm_recharge, x, y)
    df["etrm_eta_mm"] = sample_coarse(etrm_eta, x, y)
    df["etrm_runoff_mm"] = sample_coarse(etrm_runoff, x, y)
    for c in COVARIATES:
        n = int(df[c].notna().sum())
        log.info(
            "covariate %-20s covered %d / %d (%.1f%%)",
            c,
            n,
            len(df),
            100.0 * n / max(len(df), 1),
        )
    return df


def build(
    out_path: str,
    *,
    wells_path: str = WELLS,
    manifest_path: str = MANIFEST,
    wte_frozen: str = WTE_FROZEN,
    wte_support: str = WTE_SUPPORT,
    fac_shard_dir: str = FAC_SHARD_DIR,
    huc8_polys: str = HUC8_POLYS,
) -> None:
    df = load_clean_wells(wells_path, manifest_path)

    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    x, y = tr.transform(df.longitude.to_numpy(), df.latitude.to_numpy())
    df["x5070"], df["y5070"] = np.asarray(x), np.asarray(y)

    # Level-0 predictor 1: ConusWTE DTW = land surface - regional WTE elevation.
    wte_elev = sample_coarse(wte_frozen, df.x5070.to_numpy(), df.y5070.to_numpy())
    df["wte_surface_elev_m"] = wte_elev
    df["wte_dtw"] = df.land_surface_elev_m.to_numpy() - wte_elev
    # Regime gate: distance to nearest fit well (m).
    df["wte_support_dist_m"] = sample_coarse(
        wte_support, df.x5070.to_numpy(), df.y5070.to_numpy()
    )

    pts = gpd.GeoDataFrame(
        df[["canonical_id"]],
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs=4326,
    )
    df = join_huc(df, pts, huc8_polys)

    # Level-0 predictor 2: ConusFAC HAND depth (target-blind; pilot coverage).
    df = join_fac(df, fac_shard_dir)

    # Regime discriminators (separate artifact-deep WTE sag from genuinely deep table).
    df = add_covariates(df)

    cols = [
        "canonical_id",
        "source",
        "confinement_class",
        "obs_count",
        "longitude",
        "latitude",
        "x5070",
        "y5070",
        "huc8",
        "huc4",
        "huc2",
        "mean_dtw",
        "land_surface_elev_m",
        "wte_surface_elev_m",
        "wte_dtw",
        "wte_support_dist_m",
        "fac_rem_dtw_m",
        "elev_above_coarse_m",
        "slope_deg",
        "tri_100m",
        "dist_to_stream_m",
        "log_drainage_area",
        "aridity_index",
        "mean_annual_precip_mm",
        "etrm_recharge_mm",
        "etrm_eta_mm",
        "etrm_runoff_mm",
    ]
    out = df[cols].copy()
    out["fac_covered"] = out.fac_rem_dtw_m.notna()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    log.info(
        "wrote %d-row stacker feature table -> %s (%d ConusFAC-covered, trainable now)",
        len(out),
        out_path,
        int(out.fac_covered.sum()),
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wells", default=WELLS)
    p.add_argument("--manifest", default=MANIFEST)
    p.add_argument("--wte-frozen", default=WTE_FROZEN)
    p.add_argument("--wte-support", default=WTE_SUPPORT)
    p.add_argument("--fac-shard-dir", default=FAC_SHARD_DIR)
    p.add_argument("--huc8-polys", default=HUC8_POLYS)
    p.add_argument(
        "--out", default="/data/ssd2/handily/conus/stacker/wte_fac_features.parquet"
    )
    args = p.parse_args(argv)
    build(
        args.out,
        wells_path=args.wells,
        manifest_path=args.manifest,
        wte_frozen=args.wte_frozen,
        wte_support=args.wte_support,
        fac_shard_dir=args.fac_shard_dir,
        huc8_polys=args.huc8_polys,
    )


if __name__ == "__main__":
    main()
