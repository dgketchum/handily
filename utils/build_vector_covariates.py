"""Rasterize / warp the downloaded geology + context + hydrography + irrigation
datasets onto the canonical CONUS grid (EPSG:5070, 100 m, 49810x31390).

Categorical vectors are reprojected to 5070 then burned by an integer code field
(legend CSV written alongside). Source rasters are warped to the grid.

Inputs under /nas/handily/covariates/_download/ex/ ; outputs to the category dirs.
Each step is skip-if-exists and logs failures without aborting the rest.
"""

import os
import csv
import time
import subprocess

import geopandas as gpd

DL = "/nas/handily/covariates/_download/ex"
COV = "/nas/handily/covariates"
TE = ["-2540000", "119000", "2441000", "3258000"]  # xmin ymin xmax ymax (5070)
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def reproject(src, dst_gpkg, sql=None, layer="data"):
    if os.path.exists(dst_gpkg):
        os.remove(dst_gpkg)
    cmd = [
        "ogr2ogr",
        "-t_srs",
        CRS,
        "-nlt",
        "PROMOTE_TO_MULTI",
        "-nln",
        layer,
        dst_gpkg,
        src,
    ]
    if sql:
        cmd += ["-dialect", "OGRSQL", "-sql", sql]
    run(cmd)
    return dst_gpkg


def rasterize(gpkg, field, out, dtype="Int32", nodata=0, layer="data"):
    if os.path.exists(out):
        log(f"  skip (exists): {out}")
        return
    tmp = out + ".tmp.tif"
    run(
        [
            "gdal_rasterize",
            "-l",
            layer,
            "-a",
            field,
            "-te",
            *TE,
            "-tr",
            *TR,
            "-a_nodata",
            str(nodata),
            "-ot",
            dtype,
            "-init",
            str(nodata),
            *CO,
            gpkg,
            tmp,
        ]
    )
    os.replace(tmp, out)
    log(f"  wrote {out}")


def warp(src, out, resample, src_nodata=None, dst_nodata=-9999, dtype="Float32"):
    if os.path.exists(out):
        log(f"  skip (exists): {out}")
        return
    tmp = out + ".tmp.tif"
    cmd = [
        "gdalwarp",
        "-t_srs",
        CRS,
        "-te",
        *TE,
        "-tr",
        *TR,
        "-r",
        resample,
        "-ot",
        dtype,
        "-dstnodata",
        str(dst_nodata),
        "-overwrite",
        *CO,
    ]
    if src_nodata is not None:
        cmd += ["-srcnodata", str(src_nodata)]
    cmd += [src, tmp]
    run(cmd)
    os.replace(tmp, out)
    log(f"  wrote {out}")


def legend(shp, code_field, name_fields, out_csv, cast_int=False):
    gdf = gpd.read_file(shp, columns=[code_field, *name_fields], ignore_geometry=True)
    seen = {}
    for _, row in gdf.iterrows():
        c = row[code_field]
        if cast_int:
            try:
                c = int(c)
            except (ValueError, TypeError):
                continue
        seen[c] = tuple(row[f] for f in name_fields)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", *name_fields])
        for c in sorted(seen, key=lambda x: (x is None, x)):
            w.writerow([c, *seen[c]])
    log(f"  legend {out_csv} ({len(seen)} classes)")


def do_aquifers():
    log("AQUIFERS")
    shp = f"{DL}/aquifers_us/us_aquifers.shp"
    g = reproject(
        shp,
        f"{DL}/_aquifers_5070.gpkg",
        sql="SELECT AQ_CODE, AQ_NAME, ROCK_TYPE, ROCK_NAME FROM us_aquifers",
    )
    rasterize(g, "AQ_CODE", f"{COV}/geology/aquifer_code.tif", "Int32")
    rasterize(g, "ROCK_TYPE", f"{COV}/geology/aquifer_rocktype.tif", "Int16")
    legend(shp, "AQ_CODE", ["AQ_NAME"], f"{COV}/geology/aquifer_code_legend.csv")
    legend(
        shp, "ROCK_TYPE", ["ROCK_NAME"], f"{COV}/geology/aquifer_rocktype_legend.csv"
    )


def do_ecoregions():
    log("ECOREGIONS L3")
    shp = f"{DL}/us_eco_l3/us_eco_l3.shp"
    g = reproject(
        shp,
        f"{DL}/_ecol3_5070.gpkg",
        sql="SELECT CAST(US_L3CODE AS integer) AS l3code, US_L3NAME FROM us_eco_l3",
    )
    rasterize(g, "l3code", f"{COV}/context/ecoregion_l3_code.tif", "Int16")
    legend(
        shp,
        "US_L3CODE",
        ["US_L3NAME"],
        f"{COV}/context/ecoregion_l3_legend.csv",
        cast_int=True,
    )


def do_physio():
    log("PHYSIOGRAPHY (Fenneman)")
    shp = f"{DL}/physio_shp/physio.shp"
    g = reproject(
        shp,
        f"{DL}/_physio_5070.gpkg",
        sql="SELECT PROVCODE, PROVINCE, DIVISION, SECTION FROM physio",
    )
    rasterize(g, "PROVCODE", f"{COV}/context/physio_province_code.tif", "Int16")
    legend(
        shp,
        "PROVCODE",
        ["PROVINCE", "DIVISION"],
        f"{COV}/context/physio_province_legend.csv",
    )


def do_karst():
    log("KARST")
    out = f"{COV}/geology/karst_type.tif"
    if os.path.exists(out):
        log(f"  skip (exists): {out}")
        return
    base = f"{DL}/USKarstMap/Shapefiles/Continguous48"
    # code: burned in order so later (higher-priority) overwrites earlier
    layers = [
        ("Piping48", 5),
        ("Volcanics48", 4),
        ("SandstoneKarst48", 3),
        ("Evaporites48", 2),
        ("Carbonates48", 1),
    ]
    tmp = out + ".tmp.tif"
    first = True
    for name, code in layers:
        shp = f"{base}/{name}.shp"
        if not os.path.exists(shp):
            log(f"  karst layer missing: {name}")
            continue
        g = reproject(shp, f"{DL}/_karst_{name}.gpkg")
        if first:
            run(
                [
                    "gdal_rasterize",
                    "-l",
                    "data",
                    "-burn",
                    str(code),
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
                    g,
                    tmp,
                ]
            )
            first = False
        else:
            run(["gdal_rasterize", "-l", "data", "-burn", str(code), g, tmp])
        log(f"  burned {name}={code}")
    os.replace(tmp, out)
    log(
        f"  wrote {out}  (1=carbonate 2=evaporite 3=sandstone 4=volcanic 5=piping 0=none)"
    )


def do_mirad():
    log("IRRIGATION (MIrAD-US 2017 250 m)")
    src = f"{DL}/mirad250m_17v4/mirad250m_17v4/mirad250_17v4.tif"
    warp(
        src,
        f"{COV}/anthropogenic/irrigation_2017.tif",
        "near",
        src_nodata=255,
        dst_nodata=255,
        dtype="Byte",
    )


def do_bfi():
    log("BASEFLOW INDEX (BFI48 1 km)")
    src = f"{DL}/bfi48grd/bfi48grd"
    warp(
        src,
        f"{COV}/hydrography/bfi.tif",
        "bilinear",
        src_nodata=255,
        dst_nodata=-9999,
        dtype="Float32",
    )


def main():
    for fn in (do_aquifers, do_ecoregions, do_physio, do_karst, do_mirad, do_bfi):
        try:
            fn()
        except Exception as e:
            log(f"{fn.__name__} FAILED: {e}")
    log("VECTOR/RASTER COVARIATE PASS COMPLETE")


if __name__ == "__main__":
    main()
