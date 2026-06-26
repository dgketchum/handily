"""Relief-aware IDW regional water-table ELEVATION on the coarse grid.

The naive WTE IDW (``build_wte_regional_prior.idw_predict`` /
``build_conus_graph_inputs.crossfit_idw``) interpolates absolute water-table
elevation by HORIZONTAL distance only. In high relief that smears a valley well's
head onto an upland well -- water tables tens of metres above ground, an
RMSE 80-138 m catastrophic tail. The fix is one knob: lift the interpolation into
``(x, y, vw * z_surf)`` so only wells at SIMILAR ground elevation inform a cell.

Diagnosed + tuned 2026-06-25 (see ``notes/GNN_PROGRESS.md`` and
``notes/CONUS_DTW_STACKER.md``). At the FAC-footprint wells, vw=100 (horizontal
metres per vertical metre) cuts leave-fold-out MAD 8.45 -> 6.18 and RMSE
80 -> 54, with NO harm in flat terrain (the penalty is a no-op where dz~0) and the
deep-band (30 m+) optimum at vw~100. As vw->inf this provably becomes depth-space
IDW; vw=100 is the smooth-but-terrain-aware middle that also beats the biharmonic
``coarse_wte_frozen.tif`` on the tail (RMSE 54 vs 138).

Output is written on the SAME coarse grid as ``coarse_surface.tif`` /
``coarse_wte_frozen.tif`` so the existing render path reproduces per-AOI 10 m WTE:

    uv run python utils/build_wte_surface.py --render-only \\
        --coarse-wte <this output> --dem <aoi 10 m DEM> --out-dir <aoi>/...

Discipline (CLAUDE.md): water-table wells only (confinement_class in
{unconfined, unconfined_marginal}); Ma is never an input. ``--retired`` restricts
the fit population to the frozen surface's retirement manifest so the gridded base
carries the SAME leak-safety as ``coarse_wte_frozen.tif``.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

log = logging.getLogger("build_wte_idw_grid")

HYDRO = "/data/ssd2/handily/conus/hydrography90m"
WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"
COARSE_SURFACE = f"{HYDRO}/coarse_surface.tif"
WT_CLASSES = ("unconfined", "unconfined_marginal")
OUT_NODATA = -9999.0


def load_wells(wells_path: str, retired_path: str | None) -> tuple[np.ndarray, ...]:
    """Water-table wells with x5070, y5070, land-surface elevation, mean WTE."""
    cols = [
        "canonical_id",
        "confinement_class",
        "mean_wte_m",
        "land_surface_elev_m",
        "longitude",
        "latitude",
    ]
    df = pd.read_parquet(wells_path, columns=cols)
    df = df[df.confinement_class.isin(WT_CLASSES)]
    df = df[df.mean_wte_m.notna() & df.longitude.notna() & df.latitude.notna()]
    df = df.drop_duplicates("canonical_id")
    if retired_path:
        ret = set(pd.read_parquet(retired_path)["canonical_id"])
        df = df[df.canonical_id.isin(ret)]
        log.info("restricted to %d retired fit-set wells", len(df))
    # land_surface_elev_m is the vertical coordinate of the relief lift; it must be
    # present. Verified 0%% NaN on the retired set -- fail loudly if that changes
    # rather than silently dropping wells (CLAUDE.md).
    n_bad = int(df.land_surface_elev_m.isna().sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} fit wells lack land_surface_elev_m; investigate before gridding"
        )
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    x, y = tr.transform(df.longitude.to_numpy(), df.latitude.to_numpy())
    return (
        np.asarray(x),
        np.asarray(y),
        df.land_surface_elev_m.to_numpy("float64"),
        df.mean_wte_m.to_numpy("float64"),
    )


def grid_idw(
    coarse_surface: str,
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    wte: np.ndarray,
    vw: float,
    k: int,
    power: float,
    block: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Relief-aware IDW of well WTE onto every valid cell of the coarse grid."""
    with rasterio.open(coarse_surface) as ds:
        zc = ds.read(1).astype("float64")
        prof = ds.profile
        nod = ds.nodata
        transform = ds.transform
    valid = np.isfinite(zc)
    if nod is not None:
        valid &= zc != nod
    rows, cols = np.nonzero(valid)
    # cell-centre coordinates in 5070
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = zc[rows, cols]
    log.info("valid land cells: %d", len(rows))

    tree3 = cKDTree(np.column_stack([wx, wy, vw * wz]))
    tree2 = cKDTree(np.column_stack([wx, wy]))
    kk = min(k, len(wx))
    pred = np.empty(len(rows), dtype="float64")
    support = np.empty(len(rows), dtype="float64")
    for s in range(0, len(rows), block):
        e = min(s + block, len(rows))
        q3 = np.column_stack([xs[s:e], ys[s:e], vw * zs[s:e]])
        d, idx = tree3.query(q3, k=kk)
        if kk == 1:
            d, idx = d[:, None], idx[:, None]
        w = 1.0 / np.maximum(d, 1.0) ** power
        pred[s:e] = (w * wte[idx]).sum(1) / w.sum(1)
        support[s:e], _ = tree2.query(np.column_stack([xs[s:e], ys[s:e]]), k=1)
        log.info("  gridded %d / %d", e, len(rows))

    out = np.full(zc.shape, OUT_NODATA, dtype="float32")
    out[rows, cols] = pred.astype("float32")
    sup = np.full(zc.shape, OUT_NODATA, dtype="float32")
    sup[rows, cols] = support.astype("float32")
    prof.update(dtype="float32", nodata=OUT_NODATA, count=1, compress="LZW")
    return out, sup, prof


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wells", default=WELLS)
    p.add_argument(
        "--retired",
        default=f"{HYDRO}/wte_frozen_retired_wells.parquet",
        help="retirement manifest; restricts fit to the frozen surface's well set "
        "(pass '' to fit on all water-table wells)",
    )
    p.add_argument("--coarse-surface", default=COARSE_SURFACE)
    p.add_argument("--out", default=f"{HYDRO}/coarse_wte_idw_relief.tif")
    p.add_argument(
        "--support-out", default=f"{HYDRO}/coarse_wte_idw_relief_support.tif"
    )
    p.add_argument(
        "--vw", type=float, default=100.0, help="vertical weight (h-m per v-m)"
    )
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--power", type=float, default=2.0)
    args = p.parse_args(argv)

    retired = args.retired or None
    wx, wy, wz, wte = load_wells(args.wells, retired)
    log.info(
        "fit wells: %d  vw=%.0f k=%d power=%.1f", len(wx), args.vw, args.k, args.power
    )
    out, sup, prof = grid_idw(
        args.coarse_surface, wx, wy, wz, wte, args.vw, args.k, args.power
    )
    with rasterio.open(args.out, "w", **prof) as dst:
        dst.write(out, 1)
    log.info("wrote relief-IDW coarse WTE -> %s", args.out)
    with rasterio.open(args.support_out, "w", **prof) as dst:
        dst.write(sup, 1)
    log.info("wrote fit-well support distance (m) -> %s", args.support_out)


if __name__ == "__main__":
    main()
