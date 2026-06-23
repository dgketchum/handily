"""CONUS regional water-table ELEVATION by robust, stream-capped smooth fit.

See notes/WTE_ELEVATION_SURFACE.md. The pure-physics harmonic head (step 4)
under-mounds in high relief (Ruby +108 m): a flat inter-stream table is wrong where
the real table rises into the mountains. The fix is data-driven -- fit a smooth
regional surface to the unconfined well water-table elevations themselves, letting
the wells lift the table into the uplands, while two physical constraints keep it
honest:

  1. ROBUST data fidelity (Huber IRLS) -- deep and shallow wells that depart from
     the regional trend (perched, mislabeled, bad records) are statistical outliers
     and get down-weighted. Confined/unconfined separation is gwx's job; we trust
     `confinement_class` and only reject trend outliers.
  2. STREAM CAP -- the regional table stays at or below major-stream stage at
     stream cells (a one-sided upper bound: arid losing reaches may sit below the
     streambed where wells justify it, but the table never domes above the
     drainage in a valley).
  3. SMOOTHNESS -- lambda * ||nabla^2 S||^2 (biharmonic) gives the regional trend
     and interpolates where wells are sparse.

Objective:  min_S  sum_w  rho_huber(S(x_w) - wte_w)  +  lambda * ||nabla^2 S||^2
            s.t.   S <= z_stream  at major-stream cells.

Solved by IRLS: each iteration is a weighted sparse least-squares (biharmonic
smoothness + diagonal data/cap weights), solved with CG. The cap is an active set
(weight on only where S currently exceeds the stream stage).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import scipy.ndimage as ndi
from pyproj import Transformer
from scipy.sparse import coo_matrix, diags, eye
from scipy.sparse.linalg import cg

log = logging.getLogger("build_wte_regional")

HYDRO_DIR = "/data/ssd2/handily/conus/hydrography90m"
COARSE_SURFACE = f"{HYDRO_DIR}/coarse_surface.tif"  # step-4 2 km mean land surface
COARSE_ANCHOR = f"{HYDRO_DIR}/coarse_anchor.tif"  # step-4 2 km stream stage (cap)
WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"
WT_CLASSES = ("unconfined", "unconfined_marginal")
# Source-agnostic by default: nwis/ngwmn are valid unconfined water-table
# observations and belong in fundamental model development. Excluding them is a
# Ma/Janssen *benchmark-fairness* choice scoped to a head-to-head eval, NOT a
# constraint on this surface fit. Pass --exclude-sources to opt in per experiment.
OUT_NODATA = -9999.0

# 4-connectivity matches the 5-point Laplacian stencil.
_CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def build_laplacian(mask: np.ndarray):
    """Sparse 5-point Laplacian over the True cells of ``mask`` (Neumann edges).

    Returns (L, idx) where L is (N, N) with (L @ s)_i ~ nabla^2 s at cell i
    (diagonal -deg_i, +1 per in-mask neighbor), and idx maps grid -> row index
    (-1 off-mask). Row-major cell order.
    """
    h, w = mask.shape
    n = int(mask.sum())
    idx = np.full((h, w), -1, dtype=np.int64)
    idx[mask] = np.arange(n)
    mr, mc = np.nonzero(mask)

    deg = np.zeros(n, dtype=np.float64)
    off_r: list[np.ndarray] = []
    off_c: list[np.ndarray] = []
    for dr, dc in _DIRS:
        nr, nc = mr + dr, mc + dc
        inb = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        nrc, ncc = np.clip(nr, 0, h - 1), np.clip(nc, 0, w - 1)
        nbr = inb & mask[nrc, ncc]
        deg += nbr
        loc = np.nonzero(nbr)[0]
        off_r.append(loc)
        off_c.append(idx[nrc[loc], ncc[loc]])

    rows = np.concatenate([np.arange(n), *off_r])
    cols = np.concatenate([np.arange(n), *off_c])
    data = np.concatenate([-deg, *[np.ones(len(r)) for r in off_r]])
    lap = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return lap, idx


def _robust_weight(resid: np.ndarray, scale: float) -> np.ndarray:
    """Cauchy/Lorentzian IRLS weight: 1 / (1 + (r/scale)^2).

    Redescending (decays ~ (scale/r)^2), so gross deep/shallow outliers are
    effectively rejected, while staying smooth/stable (no hard Tukey cutoff).
    """
    return 1.0 / (1.0 + (resid / scale) ** 2)


def robust_smooth_fit(
    domain: np.ndarray,
    well_mean: np.ndarray,
    well_n: np.ndarray,
    cap_value: np.ndarray,
    cap_mask: np.ndarray,
    *,
    lam: float = 1.0,
    robust_scale: float = 10.0,
    cap_weight: float = 1.0e3,
    n_iter: int = 8,
    rtol: float = 1e-7,
    maxiter: int = 50000,
) -> np.ndarray:
    """Robust, stream-capped membrane-smooth fit of a regional surface.

    domain      : bool (H,W) cells in the solve domain.
    well_mean   : float (H,W) per-cell aggregated well WTE (nan/ignored where none).
    well_n      : float (H,W) well count per cell (base data weight; 0 = no well).
    cap_value   : float (H,W) major-stream stage (upper bound; ignored off cap cells).
    cap_mask    : bool (H,W) major-stream cells (subset of domain), one-sided S<=cap.
    lam         : smoothness weight on ||grad S||^2 (membrane / graph-Laplacian).
    robust_scale: residual scale (m); wells far beyond it are rejected as outliers.
    cap_weight  : penalty strength when the cap is active (S>cap).

    Returns S float64, NaN outside domain and in components with no data (well/cap).
    """
    h, w = domain.shape
    has_well = domain & (well_n > 0)
    data_cell = has_well | (domain & cap_mask)
    # Keep only connected components that carry at least one observation/constraint.
    lbl, _ = ndi.label(domain, structure=_CONN4)
    keep_lbls = np.unique(lbl[data_cell])
    keep_lbls = keep_lbls[keep_lbls > 0]
    solvable = np.isin(lbl, keep_lbls)

    out = np.full((h, w), np.nan, dtype=np.float64)
    n = int(solvable.sum())
    if n == 0:
        return out

    lap, idx = build_laplacian(solvable)
    # Membrane smoother: penalty lam*||grad S||^2 -> graph Laplacian K = -L (PSD,
    # deg on the diagonal). A biharmonic (thin-plate, L^T L) curvature penalty is
    # smoother but its (L/h)^4 conditioning is intractable for CG without multigrid
    # (pyamg unavailable); the membrane operator is the same one the harmonic solve
    # converges on quickly. C0 (kinks at data) vs C1 -- fine for a regional surface.
    k = (-lap).tocsr()
    # tiny ridge keeps the system non-singular if a component has only cap cells
    reg = lam * k + 1e-9 * eye(n, format="csr")
    reg_diag = lam * k.diagonal() + 1e-9  # for the Jacobi preconditioner

    cells = solvable
    wn = np.where(has_well[cells], well_n[cells], 0.0).astype(np.float64)
    ym = np.where(has_well[cells], well_mean[cells], 0.0).astype(np.float64)
    cm = (cap_mask & solvable)[cells]
    cv = np.where(cm, np.nan_to_num(cap_value[cells]), 0.0).astype(np.float64)

    def solve(weight_well: np.ndarray, cap_active: np.ndarray, x0):
        w_well = weight_well
        w_cap = np.where(cap_active, cap_weight, 0.0)
        a = reg + diags(w_well + w_cap)
        b = w_well * ym + w_cap * cv
        m = diags(1.0 / (reg_diag + w_well + w_cap))  # Jacobi preconditioner
        s, info = cg(a, b, rtol=rtol, maxiter=maxiter, x0=x0, M=m)
        if info != 0:
            raise SystemExit(f"CG did not converge (info={info}, n={n})")
        return s

    # init: non-robust smoothing, cap inactive
    s = solve(wn, np.zeros(n, dtype=bool), None)
    for _ in range(n_iter):
        resid = s - ym
        wh = _robust_weight(resid, robust_scale)
        weight_well = np.where(wn > 0, wn * wh, 0.0)
        cap_active = cm & (s > cv)
        s = solve(weight_well, cap_active, s)

    out[cells] = s
    return out


def aggregate_wells(
    wells_path: str,
    transform,
    shape: tuple[int, int],
    *,
    wt_classes=WT_CLASSES,
    exclude_sources=(),
    holdout_bboxes=None,
    weight_cap: float = 10.0,
    single_obs_only: bool = False,
    sample_frac: float | None = None,
    seed: int = 0,
):
    """Bin unconfined well WTEs onto the coarse grid.

    Returns (well_mean, well_weight, well_count, fit_ids, n_used, n_holdout):
      well_mean   per-cell median of mean_wte_m (within-cell robustness),
      well_weight min(count, weight_cap) data weight (dense clusters don't dominate),
      well_count  raw per-cell fit-well count (uncapped; for the support raster),
      fit_ids     canonical_id of every well whose label informed the surface
                  (the retirement manifest -- exactly the in-grid, sampled,
                  non-held-out wells), so downstream can hard-exclude them forever.

    single_obs_only keeps only obs_count==1 wells (the abundant driller-log bulk,
    cheap to retire). sample_frac draws a deterministic random fraction (seed) as
    the sacrificial fit set -- the rest stay available for the downstream stacker.
    holdout_bboxes (list of (xmin,ymin,xmax,ymax) in grid CRS) additionally drops
    wells inside any window from the fit (leakage-free regional eval in one fit).
    """
    h, w = shape
    cols = [
        "longitude",
        "latitude",
        "mean_wte_m",
        "confinement_class",
        "source",
        "obs_count",
        "canonical_id",
    ]
    df = pd.read_parquet(wells_path, columns=cols)
    df = df[df.confinement_class.isin(wt_classes)]
    if exclude_sources:
        df = df[~df.source.isin(exclude_sources)]
    df = df[df.mean_wte_m.notna() & df.longitude.notna() & df.latitude.notna()]
    if single_obs_only:
        df = df[df.obs_count == 1]
    # One row per physical well: wells.geoparquet carries multiple source records per
    # canonical_id. Dedup BEFORE sampling so each well is sampled/weighted once and
    # the manifest is a clean per-well retirement list (downstream excludes by
    # canonical_id, so any retained sibling of a fit well is dropped too).
    df = df.drop_duplicates("canonical_id")
    if sample_frac is not None:
        df = df.sample(frac=sample_frac, random_state=seed)

    tr = Transformer.from_crs(4326, 5070, always_xy=True)
    x, y = tr.transform(df.longitude.to_numpy(), df.latitude.to_numpy())
    x, y = np.asarray(x), np.asarray(y)
    wte = df.mean_wte_m.to_numpy()
    cid = df.canonical_id.to_numpy()

    n_holdout = 0
    if holdout_bboxes:
        drop = np.zeros(x.shape, dtype=bool)
        for x0, y0, x1, y1 in holdout_bboxes:
            drop |= (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        keep = ~drop
        n_holdout = int(drop.sum())
        x, y, wte, cid = x[keep], y[keep], wte[keep], cid[keep]

    col = np.floor((x - transform.c) / transform.a).astype(np.int64)
    row = np.floor((y - transform.f) / transform.e).astype(np.int64)
    ing = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    row, col, wte, cid = row[ing], col[ing], wte[ing], cid[ing]

    cell = row * w + col
    agg = (
        pd.DataFrame({"cell": cell, "wte": wte})
        .groupby("cell")["wte"]
        .agg(["median", "size"])
    )
    well_mean = np.full((h, w), np.nan, dtype=np.float64)
    well_weight = np.zeros((h, w), dtype=np.float64)
    well_count = np.zeros((h, w), dtype=np.float64)
    cr = agg.index.to_numpy() // w
    cc = agg.index.to_numpy() % w
    well_mean[cr, cc] = agg["median"].to_numpy()
    well_weight[cr, cc] = np.minimum(agg["size"].to_numpy(), weight_cap)
    well_count[cr, cc] = agg["size"].to_numpy()
    return well_mean, well_weight, well_count, cid, int(ing.sum()), n_holdout


def build(
    coarse_surface: str,
    coarse_anchor: str,
    wells_path: str,
    out_path: str,
    *,
    lam: float,
    robust_scale: float,
    cap_weight: float,
    n_iter: int,
    holdout_bboxes=None,
    exclude_sources=(),
    single_obs_only: bool = False,
    sample_frac: float | None = None,
    seed: int = 0,
    support_out: str | None = None,
    retire_manifest_out: str | None = None,
) -> None:
    with rasterio.open(coarse_surface) as s:
        surf = s.read(1).astype(np.float64)
        transform = s.transform
        profile = s.profile
        s_nd = s.nodata
    with rasterio.open(coarse_anchor) as a:
        cap_value = a.read(1).astype(np.float64)
        a_nd = a.nodata

    domain = np.isfinite(surf) & (surf != s_nd)
    cap_mask = domain & np.isfinite(cap_value) & (cap_value != a_nd)
    log.info(
        "coarse grid %s: domain %d, stream-cap cells %d",
        domain.shape,
        int(domain.sum()),
        int(cap_mask.sum()),
    )

    well_mean, well_weight, well_count, fit_ids, n_used, n_hold = aggregate_wells(
        wells_path,
        transform,
        domain.shape,
        exclude_sources=exclude_sources,
        holdout_bboxes=holdout_bboxes,
        single_obs_only=single_obs_only,
        sample_frac=sample_frac,
        seed=seed,
    )
    occupied = int((well_weight > 0).sum())
    log.info(
        "wells: %d binned into %d coarse cells (%d held out for eval); fit set = %d",
        n_used,
        occupied,
        n_hold,
        len(fit_ids),
    )

    # Persist the retirement manifest FIRST: these canonical_ids saw their label go
    # into the surface and must be hard-excluded from every downstream train/eval.
    if retire_manifest_out:
        Path(retire_manifest_out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"canonical_id": fit_ids}).to_parquet(
            retire_manifest_out, index=False
        )
        log.info("retired %d fit-set wells -> %s", len(fit_ids), retire_manifest_out)

    s_fit = robust_smooth_fit(
        domain,
        well_mean,
        well_weight,
        cap_value,
        cap_mask,
        lam=lam,
        robust_scale=robust_scale,
        cap_weight=cap_weight,
        n_iter=n_iter,
    )
    solved = np.isfinite(s_fit)
    capped = int((cap_mask & solved & (s_fit > cap_value - 1e-6)).sum())
    log.info(
        "fit on %d / %d domain cells; %d at the stream cap",
        int(solved.sum()),
        int(domain.sum()),
        capped,
    )

    prof = profile | dict(dtype="float32", nodata=OUT_NODATA, count=1)
    out = np.where(solved, s_fit, OUT_NODATA).astype(np.float32)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)
    log.info("wrote coarse regional WTE -> %s", out_path)

    # Support raster: distance (m) to the nearest fit-well cell. Large = the surface
    # is extrapolated with no local well constraint, so the stacker should lean on
    # ConusFAC there; small = WTE is well-supported. Parameter-free regime signal.
    if support_out:
        pix = abs(transform.a)
        dist = ndi.distance_transform_edt(well_count == 0) * pix
        sup = np.where(solved, dist, OUT_NODATA).astype(np.float32)
        with rasterio.open(support_out, "w", **prof) as dst:
            dst.write(sup, 1)
        log.info("wrote fit-well support distance (m) -> %s", support_out)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coarse-surface", default=COARSE_SURFACE)
    p.add_argument("--coarse-anchor", default=COARSE_ANCHOR)
    p.add_argument("--wells", default=WELLS)
    p.add_argument("--out", default=f"{HYDRO_DIR}/coarse_wte_regional.tif")
    p.add_argument("--lam", type=float, default=1.0, help="smoothness weight")
    p.add_argument("--robust-scale", type=float, default=10.0, help="outlier scale (m)")
    p.add_argument("--cap-weight", type=float, default=1.0e3)
    p.add_argument("--n-iter", type=int, default=8)
    p.add_argument(
        "--exclude-sources",
        default="",
        help="comma-list of well sources to drop from the fit (default: none -- "
        "source-agnostic). Use e.g. nwis,ngwmn only for a Ma-fairness experiment.",
    )
    p.add_argument(
        "--single-obs-only",
        action="store_true",
        help="keep only obs_count==1 wells (the abundant driller-log bulk) for the "
        "fit, so the scarce multi-obs/monitoring wells stay for the stacker",
    )
    p.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="fit on a deterministic random fraction (e.g. 0.2); the sampled wells "
        "are the SACRIFICIAL set -- retire them via --retire-manifest-out and the "
        "surface becomes a leakage-free, pre-built exogenous feature",
    )
    p.add_argument("--seed", type=int, default=0, help="sampling seed (reproducible)")
    p.add_argument(
        "--support-out",
        default=None,
        help="write a fit-well support raster (distance in m to nearest fit-well "
        "cell) for the stacker's regime gate",
    )
    p.add_argument(
        "--retire-manifest-out",
        default=None,
        help="write the canonical_id of every well that informed the surface "
        "(parquet); downstream MUST hard-exclude these forever",
    )
    p.add_argument(
        "--holdout-bbox",
        type=float,
        nargs=4,
        action="append",
        default=None,
        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        help="EPSG:5070 bbox of wells to EXCLUDE from the fit (leakage-free eval); "
        "repeatable to hold out several disjoint eval windows in one fit",
    )
    args = p.parse_args(argv)
    build(
        args.coarse_surface,
        args.coarse_anchor,
        args.wells,
        args.out,
        lam=args.lam,
        robust_scale=args.robust_scale,
        cap_weight=args.cap_weight,
        n_iter=args.n_iter,
        holdout_bboxes=[tuple(b) for b in args.holdout_bbox]
        if args.holdout_bbox
        else None,
        exclude_sources=tuple(s for s in args.exclude_sources.split(",") if s),
        single_obs_only=args.single_obs_only,
        sample_frac=args.sample_frac,
        seed=args.seed,
        support_out=args.support_out,
        retire_manifest_out=args.retire_manifest_out,
    )


if __name__ == "__main__":
    main()
