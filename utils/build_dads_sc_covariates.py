"""Derive SC-ready covariate rasters from the dads HTD 1 km static stack.

The dads project (~/code/dads) built PRISM-style terrain, GRASS r.sun clear-sky
irradiance (365 DOY bands) and 5-season Landsat climatologies on the full-CONUS
"HTD" 1 km EPSG:5070 grid. This script reduces the two big stacks to small
static rasters the spatial-context sampler can read band-wise, and converts the
categorical facet-orientation classes to sin/cos:

  derived/rsun_seasonal_htd_1km.tif   6 bands  DJF/MAM/JJA/SON/annual mean + DJF/JJA ratio
  derived/landsat_indices_htd_1km.tif 10 bands ndvi_p0..p4, ndvi_amp, ndmi_p2, mndwi_p2,
                                               b10_p2_k, b10_amp_k
  derived/facet_sincos_htd_1km.tif    4 bands  facet sin/cos at 12 km and 36 km

Landsat band scaling follows dads: reflectance x10000 (cancels in ratios), B10
brightness temperature x100 (divided out here -> Kelvin). Facet classes follow
dads join_prism_terrain_to_table.py: 0=flat -> sin=cos=0, 1..8 = N..NW clockwise.

Design/plan: notes/SC_COVARIATE_UPGRADE.md.

Usage:
    uv run python utils/build_dads_sc_covariates.py \
        --bank /nas/handily/covariates/dads_htd_1km
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_dads_sc_covariates")

# season -> day-of-year membership (non-leap; band b of the rsun stack = DOY b)
SEASON_DOYS = {
    "djf": list(range(335, 366)) + list(range(1, 60)),
    "mam": list(range(60, 152)),
    "jja": list(range(152, 244)),
    "son": list(range(244, 335)),
}

LANDSAT_PERIODS = 5  # p0..p4 seasonal composites
LANDSAT_BANDS = ["b2", "b3", "b4", "b5", "b6", "b7", "b10"]

# dads facet orientation class -> geographic azimuth degrees (0 = N, clockwise)
FACET_ORIENT_DEG = {
    1: 0.0,
    2: 45.0,
    3: 90.0,
    4: 135.0,
    5: 180.0,
    6: 225.0,
    7: 270.0,
    8: 315.0,
}
# raster band index (1-based) per facet smoothing scale, from the band descriptions
FACET_BAND_12KM = 2
FACET_BAND_36KM = 4


def _profile(src: rasterio.DatasetReader, count: int) -> dict:
    prof = src.profile.copy()
    prof.update(
        count=count,
        dtype="float32",
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    return prof


def _read_band(src: rasterio.DatasetReader, band: int) -> np.ndarray:
    arr = src.read(band, masked=True)
    return np.ma.filled(arr.astype("float64"), np.nan)


def build_rsun_seasonal(rsun_tif: Path, out_tif: Path) -> None:
    """Seasonal/annual clear-sky GHI means + DJF/JJA ratio from the 365-DOY stack.

    Accumulates one band at a time (the stack is ~28 GB decompressed); per-pixel
    nanmean via finite-sum + finite-count so off-domain nodata stays NaN.
    """
    with rasterio.open(rsun_tif) as src:
        if src.count != 365:
            raise SystemExit(
                f"{rsun_tif} has {src.count} bands; expected 365 DOY bands"
            )
        shape = (src.height, src.width)
        sums = {s: np.zeros(shape) for s in SEASON_DOYS}
        cnts = {s: np.zeros(shape, dtype="int32") for s in SEASON_DOYS}
        doy_season = {d: s for s, ds in SEASON_DOYS.items() for d in ds}
        for doy in range(1, 366):
            arr = _read_band(src, doy)
            fin = np.isfinite(arr)
            s = doy_season[doy]
            sums[s][fin] += arr[fin]
            cnts[s][fin] += 1
            if doy % 60 == 0:
                log.info("rsun: accumulated %d/365 DOY bands", doy)
        prof = _profile(src, 6)
    means = {}
    for s in SEASON_DOYS:
        with np.errstate(invalid="ignore"):
            means[s] = np.where(cnts[s] > 0, sums[s] / cnts[s], np.nan)
    total = sum(sums.values())
    n_tot = sum(cnts.values())
    with np.errstate(invalid="ignore"):
        ann = np.where(n_tot > 0, total / n_tot, np.nan)
        ratio = np.where(means["jja"] > 0, means["djf"] / means["jja"], np.nan)
    names = [
        "rsun_djf",
        "rsun_mam",
        "rsun_jja",
        "rsun_son",
        "rsun_ann",
        "rsun_djf_jja_ratio",
    ]
    stack = [means["djf"], means["mam"], means["jja"], means["son"], ann, ratio]
    with rasterio.open(out_tif, "w", **prof) as dst:
        for i, (name, arr) in enumerate(zip(names, stack), start=1):
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, name)
    log.info("wrote %s (%s)", out_tif, ", ".join(names))


def build_landsat_indices(landsat_tif: Path, out_tif: Path) -> None:
    """Seasonal NDVI (+amplitude), summer NDMI/MNDWI and B10 thermal from the climatology.

    Summer = period 2 (dads' representative-period convention). Reflectance scale
    (x10000) cancels in the normalized ratios; B10 (x100) is converted to Kelvin.
    """
    with rasterio.open(landsat_tif) as src:
        desc = {d: i + 1 for i, d in enumerate(src.descriptions)}
        expected = [f"p{p}_{b}" for p in range(LANDSAT_PERIODS) for b in LANDSAT_BANDS]
        missing = [d for d in expected if d not in desc]
        if missing:
            raise SystemExit(f"{landsat_tif} lacks expected bands: {missing[:5]} ...")

        def band(p: int, b: str) -> np.ndarray:
            return _read_band(src, desc[f"p{p}_{b}"])

        ndvi = []
        for p in range(LANDSAT_PERIODS):
            nir, red = band(p, "b5"), band(p, "b4")
            with np.errstate(invalid="ignore", divide="ignore"):
                nd = (nir - red) / (nir + red)
            ndvi.append(np.where(np.isfinite(nd), nd, np.nan))
        ndvi_stack = np.stack(ndvi)
        with np.errstate(invalid="ignore"):
            ndvi_amp = np.nanmax(ndvi_stack, axis=0) - np.nanmin(ndvi_stack, axis=0)
        nir2, swir1_2, green2 = band(2, "b5"), band(2, "b6"), band(2, "b3")
        with np.errstate(invalid="ignore", divide="ignore"):
            ndmi_p2 = (nir2 - swir1_2) / (nir2 + swir1_2)
            mndwi_p2 = (green2 - swir1_2) / (green2 + swir1_2)
        b10 = np.stack([band(p, "b10") for p in range(LANDSAT_PERIODS)]) / 100.0
        b10_p2 = b10[2]
        with np.errstate(invalid="ignore"):
            b10_amp = np.nanmax(b10, axis=0) - np.nanmin(b10, axis=0)
        prof = _profile(src, 10)
    names = [
        *[f"ndvi_p{p}" for p in range(LANDSAT_PERIODS)],
        "ndvi_amp",
        "ndmi_p2",
        "mndwi_p2",
        "b10_p2_k",
        "b10_amp_k",
    ]
    stack = [*ndvi, ndvi_amp, ndmi_p2, mndwi_p2, b10_p2, b10_amp]
    with rasterio.open(out_tif, "w", **prof) as dst:
        for i, (name, arr) in enumerate(zip(names, stack), start=1):
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, name)
    log.info("wrote %s (%s)", out_tif, ", ".join(names))


def build_facet_sincos(orient_tif: Path, out_tif: Path) -> None:
    """Facet-orientation classes -> sin/cos at the 12 km and 36 km scales."""
    with rasterio.open(orient_tif) as src:
        prof = _profile(src, 4)
        out = []
        for b, scale in ((FACET_BAND_12KM, "12km"), (FACET_BAND_36KM, "36km")):
            cls = src.read(b)
            deg = np.full(cls.shape, np.nan)
            for c, d in FACET_ORIENT_DEG.items():
                deg[cls == c] = d
            rad = np.radians(deg)
            # flat (class 0) and any unmapped class -> 0/0, matching dads
            out.append(
                (f"facet_sin_{scale}", np.where(np.isfinite(rad), np.sin(rad), 0.0))
            )
            out.append(
                (f"facet_cos_{scale}", np.where(np.isfinite(rad), np.cos(rad), 0.0))
            )
    with rasterio.open(out_tif, "w", **prof) as dst:
        for i, (name, arr) in enumerate(out, start=1):
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, name)
    log.info("wrote %s (%s)", out_tif, ", ".join(n for n, _ in out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bank",
        default="/nas/handily/covariates/dads_htd_1km",
        help="handily-side copy of the dads HTD 1 km stack (see notes/SC_COVARIATE_UPGRADE.md)",
    )
    args = ap.parse_args()
    bank = Path(args.bank)
    derived = bank / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    build_facet_sincos(
        bank / "prism_facet_orientation_htd_1km.tif",
        derived / "facet_sincos_htd_1km.tif",
    )
    build_landsat_indices(
        bank / "landsat_htd_1km.tif", derived / "landsat_indices_htd_1km.tif"
    )
    build_rsun_seasonal(
        bank / "rsun_htd_1km.tif", derived / "rsun_seasonal_htd_1km.tif"
    )


if __name__ == "__main__":
    main()
