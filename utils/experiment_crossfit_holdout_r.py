"""Per-well crossfit holdout for the regional R (NV fix round, item A, option 1).

Recomputes the bundle's OOF R (``regional_wte_idw_oof_m``: relief-lifted kNN IDW
of observed WTE, k / power / vw from the manifest) under per-well holdouts
instead of the archived leave-fold-out, and scores R - wte_obs at the real
bundle wells. Measurement only: no bundle rebuild, no retrain.

Holdout arms (when predicting at well i, exclude every pool well j with):
  * ``H0``      cv_fold[j] == cv_fold[i]          (archived leave-fold-out; replication)
  * ``H1``      cv_unit[j] == cv_unit[i]          (leave-HUC12-unit-out)
  * ``H2_<D>km`` horizontal 5070 distance <= D km (leave-disk-out)
A well never sees itself (row identity is excluded in every arm).

Pools:
  * ``P0`` = base real rows (the manifest ``r_contract`` pool; no ``nv_ndwr``)
  * ``P1`` = P0 U ``nv_ndwr`` rows (MEASUREMENT of admitting NDWR wells as R
    sources; not a decision)

kNN: cKDTree on ``B._relief_coords`` over the pool, query k + margin, drop
excluded neighbours, keep the first k survivors (distance order), IDW
1 / max(d, 1)^power in lifted space. Rows with < k survivors are re-queried with
a larger K until k survive or the pool is exhausted (then what is left is used;
counts reported).

Usage:
  uv run python utils/experiment_crossfit_holdout_r.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_conus_graph_inputs as B  # noqa: E402
from experiment_sink_pseudo_r import (  # noqa: E402
    DEPTH_BANDS,
    HUC8_ROOT,
    NEG_BASINS,
    PV_BBOX,
    PV_BLOCK,
    PV_FOLD,
    PV_RADIUS_M,
    SHIPPED_ARM,
    md_table,
    nv_basins,
    panel,
)

BAND_LABELS = ["0-2", "2-5", "5-10", "10-30", "30+"]


def knn_idw_holdout(
    pcoords: np.ndarray,
    pval: np.ndarray,
    qcoords: np.ndarray,
    excluded,
    k: int,
    power: float,
    margin: int,
    chunk: int = 20000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """IDW over the first k non-excluded kNN neighbours of each query.

    ``excluded(qrows, idx)`` -> bool (len(qrows), K) mask of pool neighbours to drop.
    Returns pred, neighbour idx (n, k; -1 = missing), weights (n, k; 0 = missing),
    and escalation stats.
    """
    tree = cKDTree(pcoords)
    npool = len(pcoords)
    nq = len(qcoords)
    pred = np.full(nq, np.nan)
    nidx = np.full((nq, k), -1, dtype="int64")
    nw = np.zeros((nq, k))
    todo = np.arange(nq)
    K = min(k + margin, npool)
    stats = {"n_escalated": 0, "max_K": K, "n_short": 0, "min_survivors": k}
    first = True
    while len(todo):
        left = []
        for s in range(0, len(todo), chunk):
            rows = todo[s : s + chunk]
            dist, idx = tree.query(qcoords[rows], k=K)
            if K == 1:
                dist, idx = dist[:, None], idx[:, None]
            keep = ~excluded(rows, idx)
            nkeep = keep.sum(1)
            ok = (nkeep >= k) | (K >= npool)
            # only rows with >= k survivors (or an exhausted pool) are finalized
            keep, dist, idx = keep[ok], dist[ok], idx[ok]
            sel = keep & (np.cumsum(keep, 1) <= k)
            pos = np.argsort(~sel, axis=1, kind="stable")[:, :k]
            valid = np.take_along_axis(sel, pos, 1)
            d = np.take_along_axis(dist, pos, 1)
            ii = np.take_along_axis(idx, pos, 1)
            w = np.where(valid, 1.0 / np.maximum(d, 1.0) ** power, 0.0)
            v = np.where(valid, pval[ii], 0.0)
            r_ok = rows[ok]
            pred[r_ok] = (w * v).sum(1) / w.sum(1)
            nidx[r_ok] = np.where(valid, ii, -1)
            nw[r_ok] = w / w.sum(1, keepdims=True)
            nsurv = valid.sum(1)
            short = nsurv < k
            stats["n_short"] += int(short.sum())
            if short.any():
                stats["min_survivors"] = min(
                    stats["min_survivors"], int(nsurv[short].min())
                )
            left.append(rows[~ok])
        todo = np.concatenate(left) if left else np.array([], dtype="int64")
        if len(todo):
            if first:
                stats["n_escalated"] = int(len(todo))
                first = False
            K = min(K * 4, npool)
            stats["max_K"] = K
    return pred, nidx, nw, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--bundle",
        default="/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold",
    )
    ap.add_argument(
        "--out-dir", default="/data/ssd2/handily/nv/regional/crossfit_holdout_r/"
    )
    ap.add_argument("--disks-km", default="2,5,10")
    ap.add_argument("--margin", type=int, default=96)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = Path(args.bundle)
    man = json.loads((bundle / "graph_manifest.json").read_text())
    k, p, vw = man["idw_k"], man["idw_power"], man["r_relief_vw"]
    src_name = man["appended_wells"]["source_name"]
    print(f"manifest: k={k} power={p} vw={vw} folds={man['cv_scheme']}")

    qn = pd.read_parquet(bundle / "query_nodes.parquet")
    for c in ("is_shore_pseudo", "is_swl_aux"):
        if c in qn.columns:
            raise SystemExit(f"bundle has {c}; extend the real-row rule before use")
    real = qn[~qn["is_water_pseudo"].to_numpy(bool)].reset_index(drop=True)
    is_app = real["source"].astype(str).eq(src_name).to_numpy()
    xy = real[["x5070", "y5070"]].to_numpy("float64")
    z = real["z_surf_well_m"].to_numpy("float64")
    wte = real["wte_obs_m"].to_numpy("float64")
    fold = real["cv_fold"].to_numpy("int64")
    unit = real["cv_unit"].astype(str).to_numpy()
    arch = real["regional_wte_idw_oof_m"].to_numpy("float64")
    for name, a in (("wte_obs_m", wte), ("z_surf_well_m", z), ("R_arch", arch)):
        if not np.isfinite(a).all():
            raise SystemExit(f"non-finite {name} in real rows; investigate")
    n_blk = int(pd.Series(unit).str.startswith("blk_").sum())
    print(
        f"real rows {len(real):,}; base pool {int((~is_app).sum()):,}; {src_name} "
        f"{int(is_app.sum()):,}; cv_unit HUC12 codes {len(real) - n_blk:,}, "
        f"blk_ fallback {n_blk}"
    )
    coords = B._relief_coords(xy, z, vw)

    # --- groups (same definitions as experiment_sink_pseudo_r) ---------------
    nv = nv_basins()
    hu = real["huc8"].astype(str).to_numpy()
    is_nv = np.isin(hu, nv)
    is_mon = (real["well_class"] == "monitoring").to_numpy()
    with rasterio.open(
        HUC8_ROOT / "16040105" / "gnn" / SHIPPED_ARM / "gnn_dtw_100m.tif"
    ) as src:
        dtw_pv = src.read(1, masked=True)
        tpv = src.transform
    prow, pcol = np.nonzero((dtw_pv < 0).filled(False))
    px = tpv.c + (pcol + 0.5) * tpv.a
    py = tpv.f + (prow + 0.5) * tpv.e
    plon, plat = Transformer.from_crs(5070, 4326, always_xy=True).transform(px, py)
    inb = (
        (plon >= PV_BBOX[0])
        & (plon <= PV_BBOX[2])
        & (plat >= PV_BBOX[1])
        & (plat <= PV_BBOX[3])
    )
    d_patch, _ = cKDTree(np.c_[px[inb], py[inb]]).query(xy)
    is_pv = d_patch <= PV_RADIUS_M
    is_pvblk = (real["block_40km"].astype(str) == PV_BLOCK).to_numpy() & (
        fold == PV_FOLD
    )
    is_neg = np.isin(hu, NEG_BASINS)
    to5070 = Transformer.from_crs(4326, 5070, always_xy=True)
    bx, by = to5070.transform(
        [PV_BBOX[0], PV_BBOX[2], PV_BBOX[2], PV_BBOX[0]],
        [PV_BBOX[1], PV_BBOX[1], PV_BBOX[3], PV_BBOX[3]],
    )
    pv_poly = Polygon(zip(bx, by))
    pv_in_pool_P0 = int((is_pv & ~is_app).sum())
    print(
        f"NV basins {len(nv)}; NV wells {int(is_nv.sum()):,}; PV patch group "
        f"{int(is_pv.sum())} (base-pool members {pv_in_pool_P0}); fold5/{PV_BLOCK} "
        f"{int(is_pvblk.sum())}; neg basins wells {int(is_neg.sum())}"
    )
    groups = {
        "pv_12km_of_neg_patch": is_pv,
        "pv_fold5_block_-44_55": is_pvblk,
        "neg_dtw_14_basins": is_neg,
        "nv_all": is_nv,
        "nv_monitoring": is_nv & is_mon,
        "nv_ndwr": is_nv & is_app,
        "conus_monitoring_34503": is_mon,
    }

    # --- arms -------------------------------------------------------------------
    disks = [float(v) for v in args.disks_km.split(",")]
    pools = {"P0": ~is_app, "P1": np.ones(len(real), bool)}
    holdouts = ["H0", "H1"] + [f"H2_{d:g}km" for d in disks]

    def make_excl(h: str, pidx: np.ndarray):
        pf, pu = fold[pidx], unit[pidx]
        pxy = xy[pidx]

        def excl(rows, idx):
            m = pidx[idx] == rows[:, None]
            if h == "H0":
                m |= pf[idx] == fold[rows][:, None]
            elif h == "H1":
                m |= pu[idx] == unit[rows][:, None]
            else:
                dmax = float(h.split("_")[1][:-2]) * 1000.0
                hd = np.hypot(
                    pxy[idx, 0] - xy[rows, 0][:, None],
                    pxy[idx, 1] - xy[rows, 1][:, None],
                )
                m |= hd <= dmax
            return m

        return excl

    arms: dict[str, np.ndarray] = {"archived": arch}
    esc = []
    diag = []
    allq = np.arange(len(real))
    for pname, pmask in pools.items():
        pidx = np.where(pmask)[0]
        for h in holdouts:
            arm = f"{h}_{pname}"
            r, nidx, nw, st = knn_idw_holdout(
                coords[pidx],
                wte[pidx],
                coords,
                make_excl(h, pidx),
                k,
                p,
                args.margin,
            )
            if not np.isfinite(r).all():
                raise SystemExit(f"{arm}: non-finite R; investigate")
            arms[arm] = r
            esc.append({"arm": arm, "pool_n": len(pidx), **st})
            print(f"{arm}: pool {len(pidx):,} {st}")
            # mechanism diagnostic: neighbour distance + local weight share
            for g, gm in (
                ("pv_12km_of_neg_patch", is_pv),
                ("pv_fold5_block", is_pvblk),
                ("nv_all", is_nv),
                ("conus_monitoring", is_mon),
            ):
                qi = allq[gm]
                ii = nidx[qi]
                valid = ii >= 0
                gi = pidx[np.where(valid, ii, 0)]
                hd = np.hypot(
                    xy[gi, 0] - xy[qi, 0][:, None], xy[gi, 1] - xy[qi, 1][:, None]
                )
                hd = np.where(valid, hd, np.inf)
                w = nw[qi]
                share12 = (w * (hd <= PV_RADIUS_M)).sum(1)
                ndwr_share = (w * (is_app[gi] & valid)).sum(1)
                diag.append(
                    {
                        "arm": arm,
                        "group": g,
                        "n": int(gm.sum()),
                        "nearest_used_km_median": float(np.median(hd.min(1)) / 1e3),
                        "used_hdist_km_median": float(np.median(hd[valid]) / 1e3),
                        "wshare_within_12km_mean": float(share12.mean()),
                        "wshare_within_12km_median": float(np.median(share12)),
                        "wshare_ndwr_sources_mean": float(ndwr_share.mean()),
                        "used_in_pv_bbox_poly_frac": float(
                            np.mean(
                                gpd.points_from_xy(
                                    xy[gi[valid], 0], xy[gi[valid], 1]
                                ).within(pv_poly)
                            )
                        ),
                    }
                )

    # --- replication ------------------------------------------------------------
    r_cf = B.crossfit_idw(xy, wte, fold, k, p, z=z, vw=vw, pool=~is_app)
    rep = {}
    for name, r in (
        ("H0_P0_vs_archived", arms["H0_P0"]),
        ("crossfit_vs_archived", r_cf),
    ):
        d = np.abs(r - arch)
        rep[name] = {
            "all_median_m": float(np.median(d)),
            "all_p99_m": float(np.percentile(d, 99)),
            "all_max_m": float(d.max()),
            "nv_max_m": float(d[is_nv].max()),
        }
    print("replication:", json.dumps(rep))
    (out_dir / "replication.json").write_text(json.dumps(rep, indent=1))
    if rep["H0_P0_vs_archived"]["all_max_m"] > 1e-3:
        raise SystemExit("H0/P0 replication FAILED (max |dR| > 1e-3 m); invalid")

    # --- panels -----------------------------------------------------------------
    rows = []
    for arm, r in arms.items():
        res = r - wte
        for g, m in groups.items():
            rows.append({"arm": arm, "group": g, **panel(res[m])})
    pan = pd.DataFrame(rows)
    pan.to_csv(out_dir / "panel_groups.csv", index=False)

    band = pd.cut(
        real["mean_dtw"].to_numpy("float64"),
        DEPTH_BANDS,
        right=False,
        labels=BAND_LABELS,
    )
    if pd.isna(band).any():
        raise SystemExit(
            "wells outside depth bands (negative/NaN mean_dtw); investigate"
        )
    rows = []
    for arm, r in arms.items():
        res = r - wte
        for scope, sm in (("nv_all", is_nv), ("conus_monitoring", is_mon)):
            for b in BAND_LABELS:
                rows.append(
                    {
                        "arm": arm,
                        "scope": scope,
                        "obs_dtw_band_m": b,
                        **panel(res[sm & (band == b)]),
                    }
                )
    dep = pd.DataFrame(rows)
    dep.to_csv(out_dir / "panel_depth.csv", index=False)

    rows = []
    for arm, r in arms.items():
        res = r - wte
        for b in sorted(set(hu[is_nv])):
            rows.append({"arm": arm, "huc8": b, **panel(res[is_nv & (hu == b)])})
    hb = pd.DataFrame(rows)
    hb.to_csv(out_dir / "panel_nv_huc8.csv", index=False)

    escdf = pd.DataFrame(esc)
    escdf.to_csv(out_dir / "knn_escalation.csv", index=False)
    dg = pd.DataFrame(diag)
    dg.to_csv(out_dir / "mechanism.csv", index=False)

    keep = is_pv | is_pvblk | is_neg
    pw = real.loc[
        keep,
        [
            "canonical_id",
            "source",
            "huc8",
            "cv_unit",
            "cv_fold",
            "block_40km",
            "x5070",
            "y5070",
            "z_surf_well_m",
            "wte_obs_m",
        ],
    ].copy()
    for arm, r in arms.items():
        pw[f"R_{arm}"] = r[keep]
    pw.to_parquet(out_dir / "per_well_r_pv_neg.parquet")

    # basins moved > 1 m in median |resid| vs archived
    base = hb[hb["arm"] == "archived"].set_index("huc8")["median_abs_m"]
    mv = []
    for arm in arms:
        if arm == "archived":
            continue
        d = hb[hb["arm"] == arm].set_index("huc8")["median_abs_m"] - base
        mv.append(
            {
                "arm": arm,
                "n_basins": len(d),
                "better_gt1m": int((d < -1).sum()),
                "worse_gt1m": int((d > 1).sum()),
            }
        )
    mvdf = pd.DataFrame(mv)

    # --- summary.md --------------------------------------------------------------
    cols = [
        "arm",
        "n",
        "median_abs_m",
        "p90_abs_m",
        "mean_bias_m",
        "median_resid_m",
        "rmse_m",
    ]
    md = [
        "# Per-well crossfit holdout R experiment",
        "",
        f"Bundle `{bundle}`; k {k}, power {p}, relief vw {vw}. resid = R - wte_obs (m), "
        "positive = R above the observed water table. P1 arms admit `nv_ndwr` wells as R "
        "sources: a MEASUREMENT, not a decision.",
        "",
        "## Replication |dR| (m)",
        "",
        "```",
        json.dumps(rep, indent=1),
        "```",
        "",
        "## kNN escalation (rows needing K > k + margin; rows with < k survivors)",
        "",
        md_table(escdf),
        "",
        "## Mechanism: horizontal distance to used pool wells (km); IDW weight share within 12 km and from nv_ndwr sources (dimensionless); used_in_pv_bbox_poly_frac = fraction of used neighbours inside the PV bbox",
        "",
        md_table(dg, fmt=".3f"),
        "",
    ]
    for g in groups:
        md += [f"## {g} (m)", "", md_table(pan[pan["group"] == g][cols]), ""]
    for scope in ("nv_all", "conus_monitoring"):
        t = dep[dep["scope"] == scope]
        md += [
            f"## {scope} by observed DTW band (m): median |resid| / p90 |resid| / median resid",
            "",
        ]
        piv = t.assign(
            cell=t.apply(
                lambda r: (
                    f"{r.median_abs_m:.2f} / {r.p90_abs_m:.2f} / {r.median_resid_m:+.2f} (n {r.n})"
                ),
                axis=1,
            )
        ).pivot(index="arm", columns="obs_dtw_band_m", values="cell")[BAND_LABELS]
        md += [md_table(piv.loc[list(arms)], index=True), ""]
    md += [
        "## NV basins moving > 1 m in median |resid| vs archived",
        "",
        md_table(mvdf),
        "",
    ]
    neg = hb[hb["huc8"].isin(NEG_BASINS)].pivot(
        index="huc8", columns="arm", values="mean_bias_m"
    )[list(arms)]
    md += [
        "## Mean bias (m) by negative-DTW basin",
        "",
        md_table(neg, index=True, fmt=".1f"),
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
