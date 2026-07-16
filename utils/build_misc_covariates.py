"""Second processing pass: GLiM lithology, GLHYMPS permeability/porosity, and
drainage density. Outputs onto the canonical CONUS grid (EPSG:5070, 100 m,
49810x31390). Skip-if-exists; each step logs failures without aborting the rest.
"""

import os
import csv
import time
import subprocess

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

DL = "/nas/handily/covariates/_download/ex"
COV = "/nas/handily/covariates"
TE = ["-2540000", "119000", "2441000", "3258000"]
TR = ["100", "100"]
CRS = "EPSG:5070"
CO = [
    "-co",
    "COMPRESS=DEFLATE",
    "-co",
    "PREDICTOR=2",
    "-co",
    "TILED=YES",
    "-co",
    "BIGTIFF=YES",
]
CHANNEL = "/data/ssd2/handily/conus/hydrography90m/channel_mask_100m_5070.tif"

# GLiM 'xx' lithology class -> integer code + description (standard GLiM legend)
GLIM = {
    "su": (1, "unconsolidated sediments"),
    "ss": (2, "siliciclastic sedimentary"),
    "sm": (3, "mixed sedimentary"),
    "sc": (4, "carbonate sedimentary"),
    "py": (5, "pyroclastics"),
    "ev": (6, "evaporites"),
    "pa": (7, "acid plutonic"),
    "pb": (8, "basic plutonic"),
    "pi": (9, "intermediate plutonic"),
    "va": (10, "acid volcanic"),
    "vb": (11, "basic volcanic"),
    "vi": (12, "intermediate volcanic"),
    "mt": (13, "metamorphics"),
    "ig": (14, "ice/glaciers"),
    "wb": (15, "water bodies"),
    "nd": (0, "no data"),
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def do_glim():
    log("GLIM lithology")
    out = f"{COV}/geology/lithology_glim.tif"
    if os.path.exists(out):
        log(f"  skip (exists): {out}")
        return
    gdb = f"{DL}/glim/LiMW_GIS 2015.gdb"
    gpkg = f"{DL}/_glim_5070.gpkg"
    # reproject keeping the xx class string (OGRSQL has no CASE; burn per-class
    # instead). Rebuild unconditionally — a stale/empty gpkg from an interrupted
    # run must not be reused (the output-tif skip above already guards rework).
    if os.path.exists(gpkg):
        os.remove(gpkg)
    run(
        [
            "ogr2ogr",
            "-t_srs",
            CRS,
            "-nlt",
            "PROMOTE_TO_MULTI",
            "-nln",
            "data",
            "-select",
            "xx",
            "-skipfailures",
            gpkg,
            gdb,
            "GLiM_export",
        ]
    )
    tmp = out + ".tmp.tif"
    first = True
    for k, (code, _desc) in sorted(GLIM.items(), key=lambda x: x[1][0]):
        if code == 0:
            continue
        cmd = [
            "gdal_rasterize",
            "-l",
            "data",
            "-where",
            f"xx='{k}'",
            "-burn",
            str(code),
        ]
        if first:
            cmd += [
                "-te",
                *TE,
                "-tr",
                *TR,
                "-a_nodata",
                "0",
                "-init",
                "0",
                "-ot",
                "Byte",
                *CO,
            ]
            first = False
        run(cmd + [gpkg, tmp])
        log(f"  burned {k}={code}")
    os.replace(tmp, out)
    with open(f"{COV}/geology/lithology_glim_legend.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "glim_class", "description"])
        for k, (c, d) in sorted(GLIM.items(), key=lambda x: x[1][0]):
            w.writerow([c, k, d])
    log(f"  wrote {out}")


def do_glhymps():
    log("GLHYMPS permeability/porosity")
    shp = f"{DL}/glhymps/GLHYMPS.shp"
    gpkg = f"{DL}/_glhymps_5070.gpkg"
    perm = f"{COV}/geology/permeability_logk_x100.tif"
    poro = f"{COV}/geology/porosity_x100.tif"
    if os.path.exists(perm) and os.path.exists(poro):
        log("  skip (exists)")
        return
    if not os.path.exists(gpkg):
        run(
            [
                "ogr2ogr",
                "-t_srs",
                CRS,
                "-nlt",
                "PROMOTE_TO_MULTI",
                "-nln",
                "data",
                "-dialect",
                "OGRSQL",
                "-sql",
                "SELECT logK_Ferr_ AS logk, Porosity_x AS poro FROM GLHYMPS",
                "-skipfailures",
                gpkg,
                shp,
            ]
        )
    for field, out, dt in (("logk", perm, "Int32"), ("poro", poro, "Int16")):
        if os.path.exists(out):
            continue
        tmp = out + ".tmp.tif"
        run(
            [
                "gdal_rasterize",
                "-l",
                "data",
                "-a",
                field,
                "-te",
                *TE,
                "-tr",
                *TR,
                "-a_nodata",
                "-32768",
                "-init",
                "-32768",
                "-ot",
                dt,
                *CO,
                gpkg,
                tmp,
            ]
        )
        os.replace(tmp, out)
        log(f"  wrote {out}  (NOTE: value = true x100; divide by 100)")


def do_drainage_density(win=21):
    """Fraction of valid land cells that are channel, in a win x win window."""
    log("drainage density")
    out = f"{COV}/hydrography/drainage_density_2km.tif"
    if os.path.exists(out):
        log(f"  skip (exists): {out}")
        return
    with rasterio.open(CHANNEL) as ds:
        nd = ds.nodata
        cm = ds.read(1)
        prof = dict(ds.profile)
    valid = (cm != nd) if nd is not None else np.ones_like(cm, bool)
    chan = ((cm == 1) & valid).astype("float32")
    validf = valid.astype("float32")
    num = uniform_filter(chan, size=win, mode="constant", cval=0.0)
    den = uniform_filter(validf, size=win, mode="constant", cval=0.0)
    dens = np.full(cm.shape, -9999.0, dtype="float32")
    ok = den > 0
    dens[ok] = (num[ok] / den[ok]).astype("float32")
    dens[~valid] = -9999.0
    prof.update(
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
        predictor=2,
        tiled=True,
        BIGTIFF="YES",
    )
    tmp = out + ".tmp.tif"
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(dens, 1)
    os.replace(tmp, out)
    log(f"  wrote {out}")


def main():
    for fn in (do_glim, do_glhymps, do_drainage_density):
        try:
            fn()
        except Exception as e:
            log(f"{fn.__name__} FAILED: {e}")
    log("MISC COVARIATE PASS COMPLETE")


if __name__ == "__main__":
    main()
