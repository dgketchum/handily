"""Smooth water-table ELEVATION surface from drainage anchors (step 4).

See notes/WTE_ELEVATION_SURFACE.md. The water-table head is a smooth (subdued-
topography) potential field that equals the land surface at the drainage network.
Given the channel anchors from step 3 (utils/build_wte_anchors.py), solve a
Laplace/Poisson Dirichlet problem for the head between anchors, then render
DTW = land_surface - head.

Pipeline (tractable: solve coarse where the head is smooth, render fine):
  1. gdalwarp -r average: DEM -> coarse mean surface (domain + nodata);
     anchor_head -> coarse anchor head (channel stage). Same coarse grid.
  2. solve_head(): sparse 5-point Laplacian, Dirichlet at anchors, Neumann at the
     domain edge, RHS = -mound_c (c>=0 mounds the table between streams, the one
     hydraulic knob; c=0 = pure harmonic interpolation = baseline). Solved with CG.
     Connected components WITHOUT an anchor are left NaN -- the solve has no
     information there, so it must not fabricate a head (project rule on missing
     data). Coarse default 2 km: regional head varies over tens of km, so this is
     the physical smoothing scale and keeps the system small enough for CG.
  3. gdalwarp -r bilinear: coarse head -> 100 m, aligned to the DEM grid.
  4. windowed finalize: reimpose head=surface on the 100 m channel anchors, clip
     head <= surface (water table at/below ground except daylighting at streams),
     DTW = surface - head. Outputs wte_surface_100m + wte_dtw_100m (EPSG:5070).

Run:
  uv run python utils/build_wte_surface.py --coarse-m 2000 --mound-c 0
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

import numpy as np
import rasterio
import scipy.ndimage as ndi
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

log = logging.getLogger("build_wte_surface")

HYDRO_DIR = "/data/ssd2/handily/conus/hydrography90m"
ANCHOR_PATH = f"{HYDRO_DIR}/anchor_head_100m_5070.tif"
DEM_PATH = "/data/ssd1/streamflow-ml-data/conus-dem/data/elev48i0100a.tif"
OUT_NODATA = -9999.0

# 4-connectivity matches the 5-point stencil.
_CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def solve_head(
    domain: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_val: np.ndarray,
    mound_c: float = 0.0,
    rtol: float = 1e-6,
    maxiter: int = 50000,
) -> np.ndarray:
    """Laplace/Poisson Dirichlet solve on a 2D grid.

    nabla^2 h = -mound_c on free cells; h = anchor_val on anchors; no-flux
    (Neumann) where a neighbor leaves the domain. Returns head float64, NaN
    outside the domain and in connected components that contain no anchor (no
    boundary information -> not fabricated).
    """
    h, w = domain.shape
    lbl, _ = ndi.label(domain, structure=_CONN4)
    anchored = np.unique(lbl[anchor_mask & domain])
    anchored = anchored[anchored > 0]
    solvable = np.isin(lbl, anchored)
    free = solvable & ~anchor_mask

    head = np.full((h, w), np.nan, dtype=np.float64)
    head[anchor_mask & solvable] = anchor_val[anchor_mask & solvable]
    k = int(free.sum())
    if k == 0:
        return head

    idx = np.full((h, w), -1, dtype=np.int64)
    idx[free] = np.arange(k)
    fr, fc = np.nonzero(free)

    deg = np.zeros(k, dtype=np.float64)
    b = np.full(k, float(mound_c), dtype=np.float64)
    off_r: list[np.ndarray] = []
    off_c: list[np.ndarray] = []
    for dr, dc in _DIRS:
        nr, nc = fr + dr, fc + dc
        inb = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        nrc, ncc = np.clip(nr, 0, h - 1), np.clip(nc, 0, w - 1)
        nbr_solv = inb & solvable[nrc, ncc]  # coupled neighbor (else no-flux)
        deg += nbr_solv
        is_anchor = nbr_solv & anchor_mask[nrc, ncc]
        is_free = nbr_solv & ~anchor_mask[nrc, ncc]
        b[is_anchor] += anchor_val[nrc[is_anchor], ncc[is_anchor]]
        loc = np.nonzero(is_free)[0]
        off_r.append(loc)
        off_c.append(idx[nrc[loc], ncc[loc]])

    rows = np.concatenate([np.arange(k), *off_r])
    cols = np.concatenate([np.arange(k), *off_c])
    data = np.concatenate([deg, *[-np.ones(len(r)) for r in off_r]])
    a = coo_matrix((data, (rows, cols)), shape=(k, k)).tocsr()

    x, info = cg(a, b, rtol=rtol, maxiter=maxiter, M=diags(1.0 / deg))
    if info != 0:
        raise SystemExit(f"CG did not converge (info={info}, k={k})")
    head[free] = x
    return head


def _warp(src: str, dst: str, wkt: str, bounds, res: float, method: str) -> None:
    left, bottom, right, top = bounds
    cmd = [
        "gdalwarp",
        "-t_srs",
        wkt,
        "-te",
        str(left),
        str(bottom),
        str(right),
        str(top),
        "-tr",
        str(res),
        str(res),
        "-r",
        method,
        "-co",
        "COMPRESS=LZW",
        "-co",
        "TILED=YES",
        "-co",
        "BIGTIFF=YES",
        "-multi",
        "--config",
        "GDAL_CACHEMAX",
        "2048",
        "-overwrite",
        src,
        dst,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _finalize(
    dem_path: str,
    anchor_grid_path: str,
    wte_grid_path: str,
    surf_path: Path,
    dtw_path: Path,
    dem_nodata: float,
    stripe_rows: int,
    reimpose: bool = True,
) -> None:
    """Windowed render: clip head <= surface, DTW = surface - head.

    With reimpose=True also pins head=surface at channel anchors (daylighting,
    for the physics head). The data-fitted regional WTE sets reimpose=False -- its
    one-sided stream cap already places the table, which may sit BELOW the streambed
    on arid losing reaches, so DTW must not be forced to 0 at every stream.
    dem / anchor / wte rasters must all be on the same (DEM) grid.
    """
    n_valid = 0
    dtw_sum = 0.0
    with (
        rasterio.open(dem_path) as dem_src,
        rasterio.open(anchor_grid_path) as anc_src,
        rasterio.open(wte_grid_path) as wte_src,
    ):
        prof = dem_src.profile | dict(
            dtype="float32",
            nodata=OUT_NODATA,
            count=1,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            bigtiff="yes",
        )
        a_nd = anc_src.nodata
        w_nd = wte_src.nodata
        height, width = dem_src.height, dem_src.width
        with (
            rasterio.open(surf_path, "w", **prof) as s_dst,
            rasterio.open(dtw_path, "w", **prof) as d_dst,
        ):
            for r0 in range(0, height, stripe_rows):
                r1 = min(r0 + stripe_rows, height)
                win = ((r0, r1), (0, width))
                surface = dem_src.read(1, window=win).astype(np.float64)
                anc = anc_src.read(1, window=win).astype(np.float64)
                wte = wte_src.read(1, window=win).astype(np.float64)

                land = surface != dem_nodata
                has_head = land & (wte != w_nd) & np.isfinite(wte)
                is_anchor = land & (anc != a_nd) & np.isfinite(anc)

                head_f = np.full(surface.shape, np.nan)
                head_f[has_head] = np.minimum(wte[has_head], surface[has_head])
                if reimpose:
                    head_f[is_anchor] = surface[is_anchor]  # daylight at streams

                valid = np.isfinite(head_f)
                head_w = np.where(valid, head_f, OUT_NODATA).astype(np.float32)
                dtw = np.where(valid, surface - head_f, OUT_NODATA).astype(np.float32)
                s_dst.write(head_w, 1, window=win)
                d_dst.write(dtw, 1, window=win)
                n_valid += int(valid.sum())
                dtw_sum += float(np.where(valid, surface - head_f, 0.0).sum())
                log.info("finalize rows %d-%d / %d", r0, r1, height)
    log.info("=== WTE surface done ===")
    log.info("valid cells: %d   mean DTW: %.2f m", n_valid, dtw_sum / max(n_valid, 1))
    log.info("wrote %s", surf_path)
    log.info("wrote %s", dtw_path)


def build(
    anchor_path: str,
    dem_path: str,
    out_dir: str,
    coarse_m: float,
    mound_c: float,
    stripe_rows: int = 2048,
    render_only: bool = False,
    coarse_wte_path: str | None = None,
    reimpose: bool = True,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wkt_path = out / "target_5070.wkt"
    if not wkt_path.exists():
        wkt_path.write_text(
            subprocess.run(
                ["gdalsrsinfo", "-o", "wkt", dem_path],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    wkt = str(wkt_path)

    with rasterio.open(dem_path) as d:
        bounds = (d.bounds.left, d.bounds.bottom, d.bounds.right, d.bounds.top)
        dem_nodata = d.nodata
        dem_res = abs(d.transform.a)
    res_tag = int(round(dem_res))

    # The coarse head is solved ONCE over the full (CONUS) DEM; --render-only
    # reuses it to render DTW against a finer DEM over any sub-window, rather than
    # re-solving in a window (which would impose wrong no-flux boundaries).
    coarse_wte = Path(coarse_wte_path) if coarse_wte_path else out / "coarse_wte.tif"
    if render_only:
        if not coarse_wte.exists():
            raise SystemExit(
                f"--render-only needs an existing coarse head: {coarse_wte}"
            )
        log.info("render-only: reusing coarse head %s", coarse_wte)
    else:
        coarse_surf = out / "coarse_surface.tif"
        coarse_anchor = out / "coarse_anchor.tif"
        log.info("coarse warps @ %.0f m ...", coarse_m)
        _warp(dem_path, str(coarse_surf), wkt, bounds, coarse_m, "average")
        _warp(anchor_path, str(coarse_anchor), wkt, bounds, coarse_m, "average")

        with rasterio.open(coarse_surf) as s, rasterio.open(coarse_anchor) as a:
            surf = s.read(1).astype(np.float64)
            anchor = a.read(1).astype(np.float64)
            coarse_profile = s.profile
            s_nodata = s.nodata
            a_nodata = a.nodata
        domain = np.isfinite(surf) & (surf != s_nodata)
        anchor_mask = domain & np.isfinite(anchor) & (anchor != a_nodata)
        log.info(
            "coarse grid %s: domain %d cells, anchors %d",
            domain.shape,
            int(domain.sum()),
            int(anchor_mask.sum()),
        )

        head = solve_head(domain, anchor_mask, anchor, mound_c=mound_c)
        solved = np.isfinite(head)
        log.info(
            "solved head on %d / %d domain cells (%.1f%%); %d unanchored -> nodata",
            int(solved.sum()),
            int(domain.sum()),
            100.0 * solved.sum() / max(int(domain.sum()), 1),
            int(domain.sum() - solved.sum()),
        )

        cw_profile = coarse_profile | dict(dtype="float32", nodata=OUT_NODATA, count=1)
        head_out = np.where(solved, head, OUT_NODATA).astype(np.float32)
        with rasterio.open(coarse_wte, "w", **cw_profile) as dst:
            dst.write(head_out, 1)

    wte_up = out / f"wte_upsampled_{res_tag}m.tif"
    log.info("upsample coarse head -> %d m ...", res_tag)
    _warp(str(coarse_wte), str(wte_up), wkt, bounds, dem_res, "bilinear")

    # Anchors must be on the DEM grid for the finalize. The native CONUS 100 m
    # anchor already is; a finer render resamples it (near, value-faithful).
    if render_only:
        anchor_grid = out / f"anchor_render_{res_tag}m.tif"
        _warp(anchor_path, str(anchor_grid), wkt, bounds, dem_res, "near")
        anchor_grid_path = str(anchor_grid)
    else:
        anchor_grid_path = anchor_path

    surf_path = out / f"wte_surface_{res_tag}m_5070.tif"
    dtw_path = out / f"wte_dtw_{res_tag}m_5070.tif"
    _finalize(
        dem_path,
        anchor_grid_path,
        str(wte_up),
        surf_path,
        dtw_path,
        dem_nodata,
        stripe_rows,
        reimpose=reimpose,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor", default=ANCHOR_PATH)
    p.add_argument("--dem", default=DEM_PATH)
    p.add_argument("--out-dir", default=HYDRO_DIR)
    p.add_argument("--coarse-m", type=float, default=2000.0)
    p.add_argument("--mound-c", type=float, default=0.0, help=">=0 recharge mound")
    p.add_argument("--stripe-rows", type=int, default=2048)
    p.add_argument(
        "--render-only",
        action="store_true",
        help="skip the solve; render DTW from --coarse-wte against --dem (e.g. a 10 m AOI)",
    )
    p.add_argument(
        "--coarse-wte",
        default=None,
        help="solved coarse head to reuse in --render-only (default <out-dir>/coarse_wte.tif)",
    )
    p.add_argument(
        "--no-reimpose",
        action="store_true",
        help="do not pin head=surface at stream anchors (for the data-fitted regional WTE)",
    )
    args = p.parse_args(argv)
    build(
        args.anchor,
        args.dem,
        args.out_dir,
        args.coarse_m,
        args.mound_c,
        args.stripe_rows,
        render_only=args.render_only,
        coarse_wte_path=args.coarse_wte,
        reimpose=not args.no_reimpose,
    )


if __name__ == "__main__":
    main()
