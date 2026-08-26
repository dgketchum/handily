"""Derive terrain covariates for the CONUS DTW stacker from the 100 m DEM.

All outputs land on the canonical CONUS grid (EPSG:5070, 100 m, 49810x31390,
origin -2540000/3258000, nodata -9999) since the source DEM already is on it.

Products -> /nas/handily/covariates/terrain/:
  twi.tif                 topographic wetness index, ln(SCA / tan(slope))
  northness.tif           cos(aspect)   (flat -> 0)
  eastness.tif            sin(aspect)   (flat -> 0)
  geomorphons.tif         10-class landform (WhiteboxTools)
  profile_curvature.tif   WhiteboxTools
  plan_curvature.tif      WhiteboxTools
  tpi_500m/2km/5km/10km   elevation deviation from mean at 4 scales (WBT DevFromMeanElev)

Multi-scale terrain-position family (the deep-bench / valley-vs-terrace signal the
GNN residual is organized by; see notes deep-bench work):
  haf_500m/2km/5km/10km   height above local floor = z - focal_min(z, w) at 4 scales
  twi_500m/2km/5km/10km   multi-scale TWI: ln(SCA / tan(slope_w)) where slope_w is the
                          slope of the DEM smoothed to scale w while the true D8 upslope
                          area (SCA) is retained -- broad valleys read wet at coarse
                          scale, benches stay dry, which single-scale TWI collapses.

Scales (window widths on the 100 m grid): 500 m=5px, 2 km=21px, 5 km=51px, 10 km=101px.

TWI reuses the precomputed D8 accumulation (signed Hydrography90m convention ->
abs()) and slope, so no flow routing is rerun. Slope is floored at 0.1 deg on
flats (a standard, physically-motivated treatment of zero-gradient cells, not a
nodata patch) so tan(slope) never divides by zero.
"""

import os
import time
import argparse
import subprocess

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import minimum_filter, uniform_filter

DEM = "/data/ssd2/handily/conus/covariates/elev48i0100a.tif"
ACCUM = "/data/ssd2/handily/conus/hydrography90m/accumulation_conus_100m_5070.tif"
SLOPE = "/data/ssd2/handily/conus/covariates/slope_deg.tif"
OUT_DIR = "/nas/handily/covariates/terrain"
CELLSIZE = 100.0
NODATA = -9999.0
SLOPE_FLOOR_DEG = 0.1  # floor zero-gradient cells (flats) before tan()
BLOCK_ROWS = 1024

# Multi-scale terrain family: name -> odd window width in cells (100 m grid).
# 500 m..10 km spans the local terrace step to the regional basin floor.
SCALES = {"500m": 5, "2km": 21, "5km": 51, "10km": 101}

PROFILE = dict(
    driver="GTiff",
    dtype="float32",
    count=1,
    nodata=NODATA,
    compress="deflate",
    predictor=2,
    tiled=True,
    blockxsize=256,
    blockysize=256,
    BIGTIFF="YES",
)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_haf(scale_name, w):
    """Height above local floor: z - focal_min(z, w x w), windowed with halo.

    Invalid cells are set to +inf before the min so they never win the floor
    (we want the minimum over VALID cells in the window, not a nodata patch);
    the output stays nodata wherever the center DEM cell is nodata. Result is
    clipped at 0 (a cell cannot sit below its own neighborhood minimum).
    """
    out_path = os.path.join(OUT_DIR, f"haf_{scale_name}.tif")
    halo = (w - 1) // 2
    with rasterio.open(DEM) as ds:
        nod = ds.nodata
        H, W = ds.height, ds.width
        prof = dict(PROFILE, width=W, height=H, crs=ds.crs, transform=ds.transform)
        with rasterio.open(out_path, "w", **prof) as dst:
            for r0 in range(0, H, BLOCK_ROWS):
                nr = min(BLOCK_ROWS, H - r0)
                rs, re = max(0, r0 - halo), min(H, r0 + nr + halo)
                z = ds.read(1, window=Window(0, rs, W, re - rs)).astype("float32")
                m = z != nod
                zf = np.where(m, z, np.inf)
                fmin = minimum_filter(zf, size=w, mode="nearest")
                top = r0 - rs
                z_c, m_c, fmin_c = (
                    z[top : top + nr],
                    m[top : top + nr],
                    fmin[top : top + nr],
                )
                out = np.full((nr, W), NODATA, "float32")
                valid = m_c & np.isfinite(fmin_c)
                out[valid] = np.maximum(z_c[valid] - fmin_c[valid], 0.0).astype(
                    "float32"
                )
                dst.write(out, 1, window=Window(0, r0, W, nr))
                if (r0 // BLOCK_ROWS) % 8 == 0:
                    log(f"  haf_{scale_name} rows {r0}/{H}")


def build_twi_scale(scale_name, w):
    """Multi-scale TWI: ln( (|acc|+1)*cell / tan(slope_w) ).

    slope_w is the slope of the DEM smoothed (nan-aware box filter) to scale w, so
    a broad valley reads low-gradient (wet) at coarse scale even where the local
    gradient is steep; the true D8 upslope area (SCA) is retained unchanged. Slope
    floored at 0.1 deg on flats. Windowed with halo for the smoothing kernel.
    """
    out_path = os.path.join(OUT_DIR, f"twi_{scale_name}.tif")
    halo = (w - 1) // 2
    floor = np.radians(SLOPE_FLOOR_DEG)
    with rasterio.open(DEM) as dem_ds, rasterio.open(ACCUM) as acc_ds:
        dnod, anod = dem_ds.nodata, acc_ds.nodata
        H, W = dem_ds.height, dem_ds.width
        prof = dict(
            PROFILE, width=W, height=H, crs=dem_ds.crs, transform=dem_ds.transform
        )
        with rasterio.open(out_path, "w", **prof) as dst:
            for r0 in range(0, H, BLOCK_ROWS):
                nr = min(BLOCK_ROWS, H - r0)
                rs, re = max(0, r0 - halo), min(H, r0 + nr + halo)
                z = dem_ds.read(1, window=Window(0, rs, W, re - rs)).astype("float64")
                m = z != dnod
                # nan-aware box smoothing: average only valid cells in the window
                num = uniform_filter(np.where(m, z, 0.0), size=w, mode="nearest")
                den = uniform_filter(m.astype("float64"), size=w, mode="nearest")
                zf = np.where(den > 0, num / den, np.nan)
                gy, gx = np.gradient(zf, CELLSIZE)
                slope = np.arctan(np.hypot(gx, gy))  # radians, from smoothed DEM
                top = r0 - rs
                slope_c, m_c = slope[top : top + nr], m[top : top + nr]
                acc = acc_ds.read(1, window=Window(0, r0, W, nr)).astype("float64")
                av = (acc != anod) & np.isfinite(acc)
                valid = m_c & av & np.isfinite(slope_c)
                sca = (np.abs(acc) + 1.0) * CELLSIZE
                tan_b = np.tan(np.maximum(slope_c, floor))
                out = np.full((nr, W), NODATA, "float32")
                out[valid] = np.log(sca[valid] / tan_b[valid]).astype("float32")
                dst.write(out, 1, window=Window(0, r0, W, nr))
                if (r0 // BLOCK_ROWS) % 8 == 0:
                    log(f"  twi_{scale_name} rows {r0}/{H}")


def build_twi(out_path):
    """TWI = ln( (|acc|+1) * cellsize / tan(slope) ), windowed."""
    with rasterio.open(ACCUM) as acc_ds, rasterio.open(SLOPE) as slp_ds:
        acc_nd = acc_ds.nodata
        slp_nd = slp_ds.nodata
        prof = dict(
            PROFILE,
            width=acc_ds.width,
            height=acc_ds.height,
            crs=acc_ds.crs,
            transform=acc_ds.transform,
        )
        floor_rad = np.radians(SLOPE_FLOOR_DEG)
        with rasterio.open(out_path, "w", **prof) as dst:
            h = acc_ds.height
            for r0 in range(0, h, BLOCK_ROWS):
                nr = min(BLOCK_ROWS, h - r0)
                win = Window(0, r0, acc_ds.width, nr)
                acc = acc_ds.read(1, window=win)
                slp = slp_ds.read(1, window=win)
                valid = (
                    (acc != acc_nd)
                    & (slp != slp_nd)
                    & np.isfinite(acc)
                    & np.isfinite(slp)
                )
                sca = (np.abs(acc) + 1.0) * CELLSIZE
                tan_b = np.tan(np.maximum(np.radians(slp), floor_rad))
                out = np.full(acc.shape, NODATA, dtype="float32")
                out[valid] = np.log(sca[valid] / tan_b[valid]).astype("float32")
                dst.write(out, 1, window=win)
                if (r0 // BLOCK_ROWS) % 8 == 0:
                    log(f"  twi rows {r0}/{h}")


def build_aspect_derivatives(north_path, east_path, tmp_aspect):
    """gdaldem aspect (deg cw from N, flat=-9999) -> northness/eastness."""
    if not os.path.exists(tmp_aspect):
        log("  running gdaldem aspect")
        subprocess.run(
            [
                "gdaldem",
                "aspect",
                DEM,
                tmp_aspect,
                "-compute_edges",
                "-co",
                "COMPRESS=DEFLATE",
                "-co",
                "BIGTIFF=YES",
            ],
            check=True,
        )
    with rasterio.open(tmp_aspect) as a_ds:
        a_nd = a_ds.nodata
        prof = dict(
            PROFILE,
            width=a_ds.width,
            height=a_ds.height,
            crs=a_ds.crs,
            transform=a_ds.transform,
        )
        with (
            rasterio.open(north_path, "w", **prof) as ndst,
            rasterio.open(east_path, "w", **prof) as edst,
        ):
            h = a_ds.height
            for r0 in range(0, h, BLOCK_ROWS):
                nr = min(BLOCK_ROWS, h - r0)
                win = Window(0, r0, a_ds.width, nr)
                asp = a_ds.read(1, window=win)
                valid = np.isfinite(asp) & (asp != a_nd)
                flat = valid & (asp < 0)  # gdaldem flat marker
                slope_cells = valid & (asp >= 0)
                rad = np.radians(asp)
                north = np.full(asp.shape, NODATA, dtype="float32")
                east = np.full(asp.shape, NODATA, dtype="float32")
                north[slope_cells] = np.cos(rad[slope_cells]).astype("float32")
                east[slope_cells] = np.sin(rad[slope_cells]).astype("float32")
                north[flat] = 0.0
                east[flat] = 0.0
                ndst.write(north, 1, window=win)
                edst.write(east, 1, window=win)


def build_tpi_scales(skip_existing=True):
    """Multi-scale TPI = WBT DevFromMeanElev (standardized elevation deviation) at the
    4 SCALES. Skips scales already on disk (tpi_2km/tpi_10km predate the 4-scale set)."""
    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(True)
    wbt.set_compress_rasters(True)
    for sc, w in SCALES.items():
        out = os.path.join(OUT_DIR, f"tpi_{sc}.tif")
        if skip_existing and os.path.exists(out):
            log(f"WBT tpi_{sc} exists, skip")
            continue
        try:
            t0 = time.time()
            log(f"WBT tpi_{sc} start (filter={w})")
            wbt.dev_from_mean_elev(DEM, out, filterx=w, filtery=w)
            log(f"WBT tpi_{sc} done in {time.time() - t0:.0f}s")
        except (
            Exception
        ) as e:  # log tool failure, keep going (operational, not data patch)
            log(f"WBT tpi_{sc} FAILED: {e}")


def run_wbt():
    import whitebox

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(True)
    wbt.set_compress_rasters(True)
    steps = [
        (
            "geomorphons",
            lambda: wbt.geomorphons(
                DEM,
                os.path.join(OUT_DIR, "geomorphons.tif"),
                search=50,
                threshold=0.0,
                forms=True,
            ),
        ),
        (
            "profile_curvature",
            lambda: wbt.profile_curvature(
                DEM, os.path.join(OUT_DIR, "profile_curvature.tif"), log=False
            ),
        ),
        (
            "plan_curvature",
            lambda: wbt.plan_curvature(
                DEM, os.path.join(OUT_DIR, "plan_curvature.tif"), log=False
            ),
        ),
    ]
    for name, fn in steps:
        try:
            t0 = time.time()
            log(f"WBT {name} start")
            fn()
            log(f"WBT {name} done in {time.time() - t0:.0f}s")
        except (
            Exception
        ) as e:  # log tool failure, keep going (operational, not data patch)
            log(f"WBT {name} FAILED: {e}")
    build_tpi_scales()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-wbt", action="store_true")
    ap.add_argument(
        "--only", default="", help="comma list: twi,aspect,wbt,haf,twi_multiscale,tpi"
    )
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    only = set(s for s in a.only.split(",") if s)

    if (
        "tpi" in only
    ):  # standalone (skip-if-exists); WBT DevFromMeanElev, no heavy layers
        build_tpi_scales()

    if not only or "twi" in only:
        try:
            t0 = time.time()
            log("TWI start")
            build_twi(os.path.join(OUT_DIR, "twi.tif"))
            log(f"TWI done in {time.time() - t0:.0f}s")
        except Exception as e:
            log(f"TWI FAILED: {e}")

    if not only or "aspect" in only:
        try:
            t0 = time.time()
            log("aspect/northness/eastness start")
            build_aspect_derivatives(
                os.path.join(OUT_DIR, "northness.tif"),
                os.path.join(OUT_DIR, "eastness.tif"),
                os.path.join(OUT_DIR, "_aspect_tmp.tif"),
            )
            log(f"aspect derivatives done in {time.time() - t0:.0f}s")
        except Exception as e:
            log(f"aspect FAILED: {e}")

    if not only or "haf" in only:
        for sc, w in SCALES.items():
            try:
                t0 = time.time()
                log(f"HAF {sc} start")
                build_haf(sc, w)
                log(f"HAF {sc} done in {time.time() - t0:.0f}s")
            except Exception as e:
                log(f"HAF {sc} FAILED: {e}")

    if not only or "twi_multiscale" in only:
        for sc, w in SCALES.items():
            try:
                t0 = time.time()
                log(f"TWI {sc} start")
                build_twi_scale(sc, w)
                log(f"TWI {sc} done in {time.time() - t0:.0f}s")
            except Exception as e:
                log(f"TWI {sc} FAILED: {e}")

    if (not only or "wbt" in only) and not a.skip_wbt:
        run_wbt()

    log("ALL TERRAIN STEPS COMPLETE")


if __name__ == "__main__":
    main()
