"""GLR: gaining/losing-reach hydraulic screen for the E6 shoreline pseudo-labels.

E6 (notes/E6_SHORELINE_RING.md) established that the land-side DTW=0 shoreline
label IS learnable and fixes the 0-2 m over-deepening, but is regionally bimodal:
plausible in humid basins (ring ground ~ nearby water-table altitude) yet grossly
false in arid basins (perched / losing / reservoir-fed lakes whose water table sits
tens of metres below ground). A single global shore weight cannot separate the two,
so the arid-false fraction injects a shallow bias and fires the deep-tail NO-GO.

GLR adds the hydraulic screen E6 deferred: the label is admitted only where the
observed water table itself supports a shore water table at/near ground. Because that
consults observed WTE, it is a TARGET-DERIVED screen and MUST be cross-fit by fold,
exactly like the relief-IDW prior R and the deep-well datum
(docs/inference_leakage_prevention.md; the 2026-07-13 crossfit-R fix).

Screen (pre-registered, notes/plans/GLR_PLAN.md):
  * per ring point, per fold f: evidence pool = fold f's TRAINING wells (the wells
    with cv_fold != f -- the SAME leave-one-fold-out pool crossfit_idw uses for R),
    confinement_class in {unconfined, unconfined_marginal}, within D = 5 km;
  * d_glr = median over the K <= 5 nearest such wells of (ring z_surf - well WTE) [m]
    (positive = ring ground above the local water-table altitude);
  * PASS iff n_wells >= 1 AND d_glr <= +5 m (no lower bound: WT above land surface is
    a discharge zone where DTW=0 remains plausible);
  * conservative default -- no well evidence => NO label (absence of wells is not
    evidence of connection).

Leakage rule realized here: fold f's pass bit `glr_pass_fold_{f}` is computed ONLY
from wells with cv_fold != f, so a held-out well (cv_fold == f) can never influence
which labels fold f trains on. `glr_pass_full` (all wells, no fold split) is emitted
for DIAGNOSTICS / DEPLOYMENT ONLY -- never used to weight a CV-fold's training loss.

The label VALUE stays 0 m (from the target-blind NHD+GSW ring evidence); the wells
decide only WHETHER the DTW=0 claim stands, never its magnitude (setting the value
from well WTE would be well-IDW, already in the model as R/anchors).

    uv run python utils/build_glr_labels.py \
        --ring /data/ssd2/handily/conus/wte_gnn/shoreline/shoreline_ring_points.parquet \
        --bundle /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2s \
        --out-dir /data/ssd2/handily/conus/wte_gnn/glr
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_glr_labels")

# --- screen constants (pre-registered) -------------------------------------------
SEARCH_RADIUS_M = 5000.0  # D: evidence pool radius
K_NEAREST = 5  # K: median over the K nearest wells within D
PASS_MAX_D_GLR_M = 5.0  # PASS iff d_glr <= this (ring ground <= 5 m above local WT)
UNCONFINED = ("unconfined", "unconfined_marginal")
# canonical 100 m EPSG:5070 lattice (same grid build_shoreline_points snaps to); the
# stable join key between a ring point and its is_shore_pseudo bundle query node.
LAT_X0, LAT_Y0, LAT_RES = -2540000.0, 3258000.0, 100.0
# HUC2 regime split for the survivor audit (E6 label_audit.txt regimes).
HUMID_HUC2 = {f"{h:02d}" for h in range(1, 9)}  # 01-08 (medians 0.5-5 m)
ARID_HUC2 = {"12", "13", "14", "15", "16"}  # arid west (medians +14..+38 m)


def cell_ids(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Canonical 100 m lattice integer cell ids for EPSG:5070 points (join key)."""
    col = np.floor((np.asarray(x, "float64") - LAT_X0) / LAT_RES).astype("int64")
    row = np.floor((LAT_Y0 - np.asarray(y, "float64")) / LAT_RES).astype("int64")
    return col * 100_000_000 + row


def screen_pool(
    ring_xy: np.ndarray,
    z_surf: np.ndarray,
    well_xy: np.ndarray,
    well_wte: np.ndarray,
) -> dict:
    """Run the GLR screen for one evidence pool of wells.

    Returns per-ring arrays: d_glr (m; NaN if no well within D), n_wells (0..K count
    among the K nearest that lie within D), dist_nearest (m to the single nearest pool
    well, uncapped), and pass (bool). Vectorized over all rings.
    """
    k = min(K_NEAREST, len(well_xy))
    tree = cKDTree(well_xy)
    dist, idx = tree.query(ring_xy, k=k)
    if k == 1:
        dist, idx = dist[:, None], idx[:, None]
    within = dist <= SEARCH_RADIUS_M  # (n_ring, k) mask of neighbors inside D
    n_wells = within.sum(axis=1).astype("int64")
    diff = z_surf[:, None] - well_wte[idx]  # ring ground - well WT altitude
    diff_masked = np.where(within, diff, np.nan)
    # rings with no well within D are all-NaN rows -> nanmedian warns then returns NaN;
    # that NaN IS the conservative "no evidence => no label" outcome (not masked data),
    # so the warning is expected and suppressed for a clean funnel log.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d_glr = np.nanmedian(diff_masked, axis=1)  # NaN where no neighbor within D
    d_glr = np.where(n_wells >= 1, d_glr, np.nan)
    passed = (n_wells >= 1) & np.isfinite(d_glr) & (d_glr <= PASS_MAX_D_GLR_M)
    return {
        "d_glr": d_glr,
        "n_wells": n_wells,
        "dist_nearest": dist[:, 0],
        "pass": passed,
    }


def load_well_pool(bundle: Path) -> pd.DataFrame:
    """Real monitoring wells with cv_fold + WTE: the screening evidence pool.

    Real = not water-pseudo and not shore-pseudo; confinement in {unconfined,
    unconfined_marginal}; finite observed WTE. These are the exact wells crossfit_idw
    builds the leave-one-fold-out prior R from, so a per-fold pool cv_fold != f mirrors
    the model's fold contract.
    """
    qn = pd.read_parquet(
        bundle / "query_nodes.parquet",
        columns=[
            "x5070",
            "y5070",
            "cv_fold",
            "is_water_pseudo",
            "is_shore_pseudo",
            "wte_obs_m",
            "confinement_class",
        ],
    )
    real = (
        ~qn["is_water_pseudo"].astype(bool)
        & ~qn["is_shore_pseudo"].astype(bool)
        & qn["confinement_class"].isin(UNCONFINED)
        & np.isfinite(qn["wte_obs_m"].to_numpy("float64"))
    )
    pool = qn[real][["x5070", "y5070", "cv_fold", "wte_obs_m"]].reset_index(drop=True)
    pool["cv_fold"] = pool["cv_fold"].astype("int64")
    log.info(
        "screening pool: %d real unconfined wells; per-fold counts %s",
        len(pool),
        pool["cv_fold"].value_counts().sort_index().to_dict(),
    )
    return pool


def build_labels(ring: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Per-fold cross-fit GLR pass bits + the full-pool diagnostic, keyed by cell_id."""
    ring_xy = ring[["x5070", "y5070"]].to_numpy("float64")
    z_surf = ring["z_surf"].to_numpy("float64")
    folds = sorted(pool["cv_fold"].unique().tolist())
    out = pd.DataFrame(
        {
            "cell_id": cell_ids(ring_xy[:, 0], ring_xy[:, 1]),
            "x5070": ring["x5070"].to_numpy("float64"),
            "y5070": ring["y5070"].to_numpy("float64"),
            "huc8": ring["huc8"].astype(str).to_numpy(),
            "huc2": ring["huc8"].astype(str).str[:2].to_numpy(),
            "z_surf": z_surf,
        }
    )
    pool_xy = pool[["x5070", "y5070"]].to_numpy("float64")
    pool_wte = pool["wte_obs_m"].to_numpy("float64")
    pool_fold = pool["cv_fold"].to_numpy("int64")
    for f in folds:
        # leave-one-fold-out: fold f's TRAINING wells are those with cv_fold != f.
        tr = pool_fold != f
        r = screen_pool(ring_xy, z_surf, pool_xy[tr], pool_wte[tr])
        out[f"glr_pass_fold_{f}"] = r["pass"]
        out[f"n_wells_fold_{f}"] = r["n_wells"]
        out[f"d_glr_fold_{f}"] = r["d_glr"]
        out[f"dist_nearest_fold_{f}"] = r["dist_nearest"]
        log.info(
            "fold %d: pool=%d wells -> %d/%d rings PASS (d_glr<=%.0fm within %.0fkm)",
            f,
            int(tr.sum()),
            int(r["pass"].sum()),
            len(out),
            PASS_MAX_D_GLR_M,
            SEARCH_RADIUS_M / 1000.0,
        )
    # full-pool diagnostic (all wells; deployment screen, NEVER a CV-fold weight)
    rf = screen_pool(ring_xy, z_surf, pool_xy, pool_wte)
    out["glr_pass_full"] = rf["pass"]
    out["n_wells_full"] = rf["n_wells"]
    out["d_glr_full"] = rf["d_glr"]
    out["dist_nearest_full"] = rf["dist_nearest"]
    log.info(
        "full pool (%d wells): %d/%d rings PASS",
        len(pool),
        int(rf["pass"].sum()),
        len(out),
    )
    return out, folds


def write_funnel(
    out: pd.DataFrame, folds: list, pool: pd.DataFrame, path: Path
) -> None:
    """Funnel + fold-awareness proof + humid/arid survivor split + d_glr distributions."""

    def _q(a: np.ndarray) -> str:
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return "n=0"
        p = np.percentile(a, [5, 25, 50, 75, 95])
        return (
            f"n={len(a)} mean={a.mean():+.2f} median={np.median(a):+.2f} "
            f"p5={p[0]:+.2f} p25={p[1]:+.2f} p50={p[2]:+.2f} p75={p[3]:+.2f} "
            f"p95={p[4]:+.2f}"
        )

    n_ring = len(out)
    fold_counts = {f: int(out[f"glr_pass_fold_{f}"].sum()) for f in folds}
    lines = [
        "GLR (gaining/losing-reach) cross-fit shoreline label screen -- funnel + audit",
        f"screen: median(ring z_surf - well WTE) over K<={K_NEAREST} nearest unconfined",
        f"        training wells within D={SEARCH_RADIUS_M / 1000:.0f} km; PASS iff "
        f"n>=1 AND d_glr <= +{PASS_MAX_D_GLR_M:.0f} m",
        f"rings: {n_ring}; screening pool: {len(pool)} real unconfined wells",
        "",
        "PER-FOLD PASS COUNTS (cross-fit; pool = wells with cv_fold != f) -- these MUST",
        "differ across folds (fold-awareness proof; a global mask would be constant):",
    ]
    for f in folds:
        lines.append(
            f"  fold {f}: {fold_counts[f]:6d} PASS  "
            f"({100.0 * fold_counts[f] / n_ring:.2f}% of rings)"
        )
    fc = np.array(list(fold_counts.values()))
    lines += [
        f"  spread: min={fc.min()} max={fc.max()} range={fc.max() - fc.min()} "
        f"(range>0 => masks are genuinely fold-aware)",
        f"  full-pool PASS: {int(out['glr_pass_full'].sum())} "
        f"({100.0 * out['glr_pass_full'].mean():.2f}%)",
        "",
        "SURVIVOR REGIME SPLIT (full-pool passing rings, by HUC2):",
    ]
    surv = out[out["glr_pass_full"]]
    huc2 = surv["huc2"].to_numpy()
    n_humid = int(np.isin(huc2, list(HUMID_HUC2)).sum())
    n_arid = int(np.isin(huc2, list(ARID_HUC2)).sum())
    lines += [
        f"  humid (HUC2 01-08): {n_humid} ({100.0 * n_humid / max(len(surv), 1):.1f}%)",
        f"  arid  (HUC2 12-16): {n_arid} ({100.0 * n_arid / max(len(surv), 1):.1f}%)",
        f"  other:              {len(surv) - n_humid - n_arid}",
        "  by HUC2 (passing count / total rings in HUC2):",
    ]
    for h in sorted(out["huc2"].unique()):
        tot = int((out["huc2"] == h).sum())
        pas = int(((out["huc2"] == h) & out["glr_pass_full"]).sum())
        lines.append(f"    {h}: {pas:5d} / {tot:5d}")
    lines += [
        "",
        "d_glr DISTRIBUTION (m; ring z_surf - nearest well WT altitude), full pool:",
        f"  all rings with evidence: {_q(out['d_glr_full'].to_numpy('float64'))}",
        f"  passing rings:           {_q(surv['d_glr_full'].to_numpy('float64'))}",
        "  d_glr by HUC2 (all rings with evidence within D):",
    ]
    for h in sorted(out["huc2"].unique()):
        a = out.loc[out["huc2"] == h, "d_glr_full"].to_numpy("float64")
        a = a[np.isfinite(a)]
        if len(a):
            lines.append(f"    {h}: {_q(a)}")
    path.write_text("\n".join(lines) + "\n")
    log.info("wrote funnel/audit -> %s", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ring",
        default="/data/ssd2/handily/conus/wte_gnn/shoreline/shoreline_ring_points.parquet",
    )
    ap.add_argument(
        "--bundle",
        default="/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2s",
    )
    ap.add_argument("--out-dir", default="/data/ssd2/handily/conus/wte_gnn/glr")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ring = pd.read_parquet(args.ring)
    if not np.isfinite(ring["z_surf"].to_numpy("float64")).all():
        raise SystemExit("ring z_surf has non-finite values (unexpected)")
    log.info("ring points: %d", len(ring))
    pool = load_well_pool(Path(args.bundle))

    out, folds = build_labels(ring, pool)
    # every ring cell must be unique (one is_shore_pseudo node per lattice cell)
    if out["cell_id"].duplicated().any():
        raise SystemExit("duplicate cell_id among ring points (lattice-join not 1:1)")

    out_path = out_dir / "glr_labels.parquet"
    out.to_parquet(out_path, index=False)
    log.info("wrote %d GLR labels -> %s", len(out), out_path)

    write_funnel(out, folds, pool, out_dir / "glr_funnel.txt")


if __name__ == "__main__":
    main()
