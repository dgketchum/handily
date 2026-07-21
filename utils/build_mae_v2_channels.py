"""Derive + warp the non-100 m v2 MAE payload channels onto the canonical lattice.

The v2 patch manifest (configs/mae/manifest_v2.json) requires every channel on the
canonical 100 m EPSG:5070 lattice (build_mae_patches.load_channel_full asserts it).
Most of the roster already conforms; this script builds the rest:

  1. Landsat seasonal indices (10): derived at native 1 km from the 35-band 1987-2025
     climatology /nas/handily/covariates/dads_htd_1km/landsat_htd_1km.tif
     (5 periods x b2,b3,b4,b5,b6,b7,b10) -> ndvi per period, ndvi amplitude,
     ndmi/mndwi at p2 (May-midJul peak), b10 at p2 + b10 amplitude. The native
     10-band product is kept at dads_htd_1km/landsat_indices_htd_1km.tif.
  2. PRISM 1991-2020 annual normals (11), 800 m -> 100 m.
  3. ETRM recharge + ETa (2), 250 m -> 100 m.
  4. Aridity index P/PET (1), 1 km -> 100 m.

All warps are bilinear (continuous fields), float32, nodata -9999, DEFLATE, written to
/nas/handily/covariates/mae_v2_100m/<channel>.tif on the exact reference grid.

    uv run python utils/build_mae_v2_channels.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_mae_patches import LATTICE_ORIGIN, REF_H, REF_W, RES_M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_mae_v2_channels")

OUT_DIR = Path("/nas/handily/covariates/mae_v2_100m")
REF_TRANSFORM = Affine(RES_M, 0.0, LATTICE_ORIGIN[0], 0.0, -RES_M, LATTICE_ORIGIN[1])
NODATA = -9999.0

LANDSAT_35 = "/nas/handily/covariates/dads_htd_1km/landsat_htd_1km.tif"
LANDSAT_IDX_NATIVE = "/nas/handily/covariates/dads_htd_1km/landsat_indices_htd_1km.tif"

# (out channel name, source path, band)
WARP_SOURCES: list[tuple[str, str, int]] = [
    (
        "etrm_recharge_mm_yr",
        "/nas/handily/covariates/ecohydro/etrm_mean_recharge_mm_yr.tif",
        1,
    ),
    ("etrm_eta_mm_yr", "/nas/handily/covariates/ecohydro/etrm_eta_mm_yr.tif", 1),
    ("aridity_index", "/nas/handily/covariates/ecohydro/aridity_index.tif", 1),
    (
        "prism_ppt_mm",
        "/nas/handily/covariates/climate/prism_ppt_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_tmean_c",
        "/nas/handily/covariates/climate/prism_tmean_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_tmax_c",
        "/nas/handily/covariates/climate/prism_tmax_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_tmin_c",
        "/nas/handily/covariates/climate/prism_tmin_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_tdmean_c",
        "/nas/handily/covariates/climate/prism_tdmean_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_vpdmin_hpa",
        "/nas/handily/covariates/climate/prism_vpdmin_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_vpdmax_hpa",
        "/nas/handily/covariates/climate/prism_vpdmax_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_soltotal_mj",
        "/nas/handily/covariates/climate/prism_soltotal_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_solclear_mj",
        "/nas/handily/covariates/climate/prism_solclear_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_soltrans",
        "/nas/handily/covariates/climate/prism_soltrans_annual_norm_9120.tif",
        1,
    ),
    (
        "prism_solslope_mj",
        "/nas/handily/covariates/climate/prism_solslope_annual_norm_9120.tif",
        1,
    ),
]

INDEX_NAMES = [
    "lst_ndvi_p0",
    "lst_ndvi_p1",
    "lst_ndvi_p2",
    "lst_ndvi_p3",
    "lst_ndvi_p4",
    "lst_ndvi_amp",
    "lst_ndmi_p2",
    "lst_mndwi_p2",
    "lst_b10_p2_k",
    "lst_b10_amp_k",
]


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized difference (a-b)/(a+b); 0-denominator -> NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out


def compute_landsat_indices(stack: np.ndarray, names: list[str]) -> np.ndarray:
    """[35,H,W] period-band climatology -> [10,H,W] index stack (INDEX_NAMES order).

    Band names are '<pP>_<bB>'; periods p0..p4, bands b2..b7,b10. NDVI=(b5-b4)/(b5+b4),
    NDMI=(b5-b6)/(b5+b6), MNDWI=(b3-b6)/(b3+b6); amplitudes = max-min across periods."""
    bi = {n: i for i, n in enumerate(names)}
    ndvi = np.stack(
        [_nd(stack[bi[f"p{p}_b5"]], stack[bi[f"p{p}_b4"]]) for p in range(5)]
    )
    b10 = np.stack([stack[bi[f"p{p}_b10"]] for p in range(5)])
    with np.errstate(invalid="ignore"):
        ndvi_amp = np.nanmax(ndvi, axis=0) - np.nanmin(ndvi, axis=0)
        b10_amp = np.nanmax(b10, axis=0) - np.nanmin(b10, axis=0)
    return np.stack(
        [
            ndvi[0],
            ndvi[1],
            ndvi[2],
            ndvi[3],
            ndvi[4],
            ndvi_amp,
            _nd(stack[bi["p2_b5"]], stack[bi["p2_b6"]]),
            _nd(stack[bi["p2_b3"]], stack[bi["p2_b6"]]),
            b10[2],
            b10_amp,
        ]
    ).astype("float32")


def build_landsat_indices() -> None:
    with rasterio.open(LANDSAT_35) as src:
        stack = src.read().astype("float32")
        names = list(src.descriptions)
        profile = src.profile
    idx = compute_landsat_indices(stack, names)
    for i, name in enumerate(INDEX_NAMES):
        v = idx[i][np.isfinite(idx[i])]
        log.info(
            "%s: valid %.1f%% min %.3f med %.3f max %.3f",
            name,
            100 * v.size / idx[i].size,
            v.min(),
            np.median(v),
            v.max(),
        )
    out = np.nan_to_num(idx, nan=NODATA)
    profile.update(
        count=len(INDEX_NAMES), dtype="float32", nodata=NODATA, compress="deflate"
    )
    with rasterio.open(LANDSAT_IDX_NATIVE, "w", **profile) as dst:
        dst.write(out)
        for i, name in enumerate(INDEX_NAMES):
            dst.set_band_description(i + 1, name)
    log.info("wrote native 10-band index stack -> %s", LANDSAT_IDX_NATIVE)


def warp_to_lattice(name: str, src_path: str, band: int) -> None:
    out = OUT_DIR / f"{name}.tif"
    if out.exists():
        log.info("%s exists, skipping", out)
        return
    profile = dict(
        driver="GTiff",
        dtype="float32",
        count=1,
        width=REF_W,
        height=REF_H,
        crs="EPSG:5070",
        transform=REF_TRANSFORM,
        nodata=NODATA,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )
    tmp = out.with_suffix(".tmp.tif")
    with rasterio.open(src_path) as s:
        with (
            WarpedVRT(
                s,
                crs="EPSG:5070",
                transform=REF_TRANSFORM,
                width=REF_W,
                height=REF_H,
                resampling=Resampling.bilinear,
                src_nodata=s.nodata,
                nodata=NODATA,
            ) as vrt,
            rasterio.open(tmp, "w", **profile) as dst,
        ):
            for r0 in range(0, REF_H, 1024):
                h = min(1024, REF_H - r0)
                win = Window(0, r0, REF_W, h)
                dst.write(vrt.read(band, window=win).astype("float32"), 1, window=win)
    tmp.rename(out)
    log.info("warped %s band %d -> %s", src_path, band, out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not Path(LANDSAT_IDX_NATIVE).exists():
        build_landsat_indices()
    else:
        log.info("%s exists, skipping derive", LANDSAT_IDX_NATIVE)
    jobs = WARP_SOURCES + [
        (name, LANDSAT_IDX_NATIVE, i + 1) for i, name in enumerate(INDEX_NAMES)
    ]
    for name, src, band in jobs:
        warp_to_lattice(name, src, band)
    log.info("done: %d channels -> %s", len(jobs), OUT_DIR)


if __name__ == "__main__":
    main()
