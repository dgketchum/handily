"""Candidate regional WTE base surfaces (R) from NON-monitoring unconfined wells.

The parity arms build R as a leave-one-fold-out relief-kNN-IDW of the 34,503
monitoring-well WTEs, so a held-out fold's labels still reach neighbouring
training wells' R (notes/PARITY_RETRAIN_REVIEW_2026-10-26.md). This script asks
whether a surface built with NO monitoring wells at all -- so no folds, no
crossfit, and one frame for training and render -- is a usable base.

Donor pool: every unconfined / unconfined_marginal well in the HAND table whose
``well_class`` is NOT monitoring (pumping + unknown-class driller records,
~1.7 M). WTE = DEM land surface (the bundle's shared datum) - mean_dtw. For each
monitoring well the ``k`` nearest donors in the parity relief-lifted metric
(x, y, vw*z) give four distance-weighted statistics of donor WTE (weights
1/max(d,1)^power, the parity IDW weights):

    mean    -- the parity construction applied to the pumping pool
    median  -- robust centre
    q25     -- lower quartile (deeper table)
    q75     -- upper quartile (shallower table; drawdown-affected donors fall
               below it, so this is the candidate for a drawn-low pool)

Each is scored at the monitoring wells against the observed WTE and against the
parity R: the uncapped crossfit R from the base bundle, and the 2 m
ground-capped R the parity bundle stores and the model trains over. Residual r = R - wte_obs
in metres; positive = R above the observed table = DTW too shallow. Monitoring
wells never enter the donor pool, so no self-exclusion is needed; ``--exclude-m``
is a horizontal radius for the duplicate-record case (a driller record of the
same physical well) and the count of monitoring wells with a donor inside 1 m
is always reported.

Outputs (``--out-dir``): ``monitoring_wells_r_candidates.parquet`` (every value
per well), ``panel_core.csv``, ``panel_depth.csv``, ``panel_huc2.csv``,
``panel_dist.csv``, ``delta_bootstrap.csv`` (paired HUC8-block bootstrap of the
MAD / RMSE delta vs the parity R), ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pumping_r_core import (  # noqa: E402
    DEM,
    HAND,
    STATS,
    cap_shares,
    load_donors,
    weighted_stats,
)

log = logging.getLogger("pumping_r")

BUNDLE = (
    "/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2_ndwr_ufold_parity"
)
BASE_BUNDLE = "/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2"
DEPTH_EDGES = [0, 2, 5, 10, 30, np.inf]
DEPTH_LABELS = ["0-2", "2-5", "5-10", "10-30", "30+"]
DIST_EDGES_KM = [0, 0.3, 1, 5, np.inf]
DIST_LABELS = ["0-0.3km", "0.3-1km", "1-5km", "5+km"]


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--hand", default=HAND)
    ap.add_argument("--dem", default=DEM)
    ap.add_argument(
        "--bundle", default=BUNDLE, help="parity bundle (monitoring wells + capped R)"
    )
    ap.add_argument(
        "--base-bundle",
        default=BASE_BUNDLE,
        help="base bundle holding the uncapped crossfit R",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=[32],
        help="neighbours per query (one run per k)",
    )
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--vw", type=float, default=100.0, help="relief lift (parity: 100)")
    ap.add_argument(
        "--exclude-m",
        type=float,
        default=0.0,
        help="horizontal donor exclusion radius, m",
    )
    ap.add_argument(
        "--cap-m", type=float, default=2.0, help="ground cap R <= z - cap (parity: 2)"
    )
    ap.add_argument(
        "--pool-classes",
        default="pumping,unknown",
        help="HAND well_class values admitted as donors",
    )
    ap.add_argument(
        "--max-share",
        type=float,
        default=0.0,
        help="cap any one donor's share of a cell's total weight (0 = plain IDW; 0.125 = n_eff >= 8)",
    )
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_monitoring(bundle: str) -> pd.DataFrame:
    qn = pd.read_parquet(f"{bundle}/query_nodes.parquet")
    m = qn.get("is_water_pseudo", pd.Series(False, index=qn.index)).astype(bool)
    base = qn[(~m) & (qn["well_class"] == "monitoring")].copy()
    man = json.load(open(f"{bundle}/graph_manifest.json"))
    app = man.get("appended_wells", {}).get("source_name")
    if app is not None:
        base = base[base["source"] != app]
    need = [
        "x5070",
        "y5070",
        "z_surf_well_m",
        "mean_dtw",
        "wte_obs_m",
        "regional_wte_idw_oof_m",
        "huc8",
        "huc2",
    ]
    miss = [c for c in need if c not in base.columns]
    if miss:
        raise SystemExit(f"bundle query_nodes lacks {miss}")
    base = base[need + ["query_node_idx", "source"]].reset_index(drop=True)
    log.info("monitoring wells from the bundle: %d", len(base))
    return base


def uncapped_reference(base_bundle: str, mon: pd.DataFrame, cap_m: float) -> np.ndarray:
    """The parity bundle stores R already ground-capped (R <= z - cap). The
    uncapped crossfit R lives in the base bundle under the same query_node_idx;
    fetch it and verify that capping it reproduces the parity column exactly."""
    qn = pd.read_parquet(
        f"{base_bundle}/query_nodes.parquet",
        columns=["query_node_idx", "regional_wte_idw_oof_m"],
    ).set_index("query_node_idx")
    r = (
        qn["regional_wte_idw_oof_m"]
        .reindex(mon["query_node_idx"].to_numpy())
        .to_numpy("float64")
    )
    if not np.all(np.isfinite(r)):
        raise SystemExit("base bundle lacks R for some parity monitoring wells")
    z = mon["z_surf_well_m"].to_numpy("float64")
    d = np.max(
        np.abs(
            np.minimum(r, z - cap_m) - mon["regional_wte_idw_oof_m"].to_numpy("float64")
        )
    )
    log.info(
        "base-bundle R capped at z - %.1f m vs parity R: max |diff| %.4f m; cap binds %.3f of wells",
        cap_m,
        d,
        float(np.mean(r > z - cap_m)),
    )
    if d > 1e-3:
        raise SystemExit("parity R is not min(base R, z - cap); check the bundles")
    return r


def panel(r: np.ndarray) -> dict[str, float]:
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {
            "n": 0,
            "mad_m": np.nan,
            "median_r_m": np.nan,
            "bias_m": np.nan,
            "rmse_m": np.nan,
            "p95_abs_m": np.nan,
        }
    return {
        "n": int(r.size),
        "mad_m": float(np.median(np.abs(r))),
        "median_r_m": float(np.median(r)),
        "bias_m": float(np.mean(r)),
        "rmse_m": float(np.sqrt(np.mean(r**2))),
        "p95_abs_m": float(np.percentile(np.abs(r), 95)),
    }


def block_bootstrap(
    a: np.ndarray, b: np.ndarray, blocks: np.ndarray, n: int, seed: int
) -> dict[str, float]:
    """Paired block bootstrap of MAD(b) - MAD(a) and RMSE(b) - MAD(a)-style
    deltas (reference minus candidate; positive = candidate better)."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, blocks = a[ok], b[ok], blocks[ok]
    ub, inv = np.unique(blocks, return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(len(ub))]
    rng = np.random.RandomState(seed)
    d_mad, d_rmse = [], []
    for _ in range(n):
        pick = rng.randint(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        d_mad.append(np.median(np.abs(b[idx])) - np.median(np.abs(a[idx])))
        d_rmse.append(np.sqrt(np.mean(b[idx] ** 2)) - np.sqrt(np.mean(a[idx] ** 2)))
    d_mad, d_rmse = np.array(d_mad), np.array(d_rmse)
    return {
        "n_blocks": int(len(ub)),
        "n_wells": int(ok.sum()),
        "d_mad_m": float(np.median(np.abs(b)) - np.median(np.abs(a))),
        "d_mad_lo": float(np.percentile(d_mad, 2.5)),
        "d_mad_hi": float(np.percentile(d_mad, 97.5)),
        "d_rmse_m": float(np.sqrt(np.mean(b**2)) - np.sqrt(np.mean(a**2))),
        "d_rmse_lo": float(np.percentile(d_rmse, 2.5)),
        "d_rmse_hi": float(np.percentile(d_rmse, 97.5)),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    mon = load_monitoring(args.bundle)
    donors = load_donors(args.hand, args.dem, args.pool_classes.split(","))

    # reference R: uncapped crossfit R from the base bundle, capped R = the parity column
    z = mon["z_surf_well_m"].to_numpy("float64")
    r_par_cap = mon["regional_wte_idw_oof_m"].to_numpy("float64")
    r_par = uncapped_reference(args.base_bundle, mon, args.cap_m)
    wte_obs = mon["wte_obs_m"].to_numpy("float64")

    # donor tree in the parity relief-lifted metric
    dxy = donors[["x5070", "y5070"]].to_numpy("float64")
    dz = donors["z_surf_m"].to_numpy("float64")
    dw = donors["wte_m"].to_numpy("float64")
    tree = cKDTree(np.column_stack([dxy[:, 0], dxy[:, 1], args.vw * dz]))
    qxy = mon[["x5070", "y5070"]].to_numpy("float64")
    q = np.column_stack([qxy[:, 0], qxy[:, 1], args.vw * z])

    # horizontal nearest donor (duplicate-record check), independent of k
    htree = cKDTree(dxy)
    hd, _ = htree.query(qxy, k=1)
    mon["nearest_donor_m"] = hd
    log.info(
        "monitoring wells with a donor within 1 m: %d; within 10 m: %d; within 100 m: %d",
        int((hd < 1).sum()),
        int((hd < 10).sum()),
        int((hd < 100).sum()),
    )

    cols = {}
    for k in args.k:
        kk = k + (16 if args.exclude_m > 0 else 0)
        dist, idx = tree.query(q, k=kk)
        vals = dw[idx]
        geo = np.hypot(
            dxy[idx, 0] - qxy[:, 0][:, None], dxy[idx, 1] - qxy[:, 1][:, None]
        )
        keep = np.ones_like(dist, dtype=bool)
        if args.exclude_m > 0:
            keep &= geo >= args.exclude_m
            keep &= np.cumsum(keep, 1) <= k
        wts = cap_shares(
            np.where(keep, 1.0 / np.maximum(dist, 1.0) ** args.power, 0.0),
            args.max_share,
        )
        st = weighted_stats(vals, wts)
        log.info(
            "k=%d: rows with no surviving donor: %d", k, int((wts.sum(1) <= 0).sum())
        )
        mon[f"nearest_lifted_k{k}_m"] = dist[:, 0]
        mon[f"mean_geo_k{k}_km"] = (
            (geo * keep).sum(1) / np.maximum(keep.sum(1), 1) / 1000.0
        )
        for s in STATS:
            cols[f"{s}_k{k}"] = st[s]
            cols[f"{s}_k{k}_cap"] = np.minimum(st[s], z - args.cap_m)
    for c, v in cols.items():
        mon[f"r_{c}"] = v
    mon["r_parity"] = r_par
    mon["r_parity_cap"] = r_par_cap

    cand = ["parity", "parity_cap"] + list(cols.keys())
    resid = {c: mon[f"r_{c}"].to_numpy("float64") - wte_obs for c in cand}
    for c in cand:
        mon[f"resid_{c}"] = resid[c]
    mon.to_parquet(out / "monitoring_wells_r_candidates.parquet", index=False)

    rows = [{"candidate": c, **panel(resid[c])} for c in cand]
    pd.DataFrame(rows).to_csv(out / "panel_core.csv", index=False)

    dtw = mon["mean_dtw"].to_numpy("float64")
    rows = []
    for lab, lo, hi in zip(DEPTH_LABELS, DEPTH_EDGES[:-1], DEPTH_EDGES[1:]):
        m = (dtw >= lo) & (dtw < hi)
        for c in cand:
            rows.append({"obs_depth_band_m": lab, "candidate": c, **panel(resid[c][m])})
    pd.DataFrame(rows).to_csv(out / "panel_depth.csv", index=False)

    rows = []
    for h2, g in mon.groupby("huc2"):
        for c in cand:
            rows.append(
                {
                    "huc2": h2,
                    "candidate": c,
                    **panel(g[f"resid_{c}"].to_numpy("float64")),
                }
            )
    pd.DataFrame(rows).to_csv(out / "panel_huc2.csv", index=False)

    dk = mon["nearest_donor_m"].to_numpy("float64") / 1000.0
    rows = []
    for lab, lo, hi in zip(DIST_LABELS, DIST_EDGES_KM[:-1], DIST_EDGES_KM[1:]):
        m = (dk >= lo) & (dk < hi)
        for c in cand:
            rows.append(
                {"nearest_donor_band": lab, "candidate": c, **panel(resid[c][m])}
            )
    pd.DataFrame(rows).to_csv(out / "panel_dist.csv", index=False)

    blocks = mon["huc8"].astype(str).to_numpy()
    rows = []
    for ref in ("parity", "parity_cap"):
        for c in cols:
            if c.endswith("_cap") != ref.endswith("_cap"):
                continue
            rows.append(
                {
                    "reference": ref,
                    "candidate": c,
                    **block_bootstrap(
                        resid[c], resid[ref], blocks, args.boot, args.seed
                    ),
                }
            )
    boot = pd.DataFrame(rows)
    boot.to_csv(out / "delta_bootstrap.csv", index=False)

    core = pd.DataFrame([{"candidate": c, **panel(resid[c])} for c in cand])
    with open(out / "report.md", "w") as f:
        f.write(
            "# Pumping-well R candidates vs the parity R at the monitoring wells\n\n"
        )
        f.write(
            f"Bundle `{args.bundle}`; donors = HAND wells with well_class in {args.pool_classes} ({len(donors):,}); "
        )
        f.write(
            f"k = {args.k}, power {args.power}, vw {args.vw}, exclude {args.exclude_m} m, cap {args.cap_m} m, max donor share {args.max_share}.\n\n"
        )
        f.write(
            "Residual r = R - observed WTE, m; positive = R above the observed table (DTW too shallow). "
        )
        f.write(
            "`*_cap` rows apply the parity 2 m ground cap. Delta = MAD(reference) - MAD(candidate), m; positive = candidate better; "
        )
        f.write(f"paired HUC8-block bootstrap CI95, {args.boot} resamples.\n\n")
        f.write(
            f"Monitoring wells with a donor within 1 / 10 / 100 m: {int((hd < 1).sum())} / {int((hd < 10).sum())} / {int((hd < 100).sum())} of {len(mon):,}.\n\n"
        )
        f.write(
            "## Core panel\n\n" + core.to_markdown(index=False, floatfmt=".2f") + "\n\n"
        )
        f.write(
            "## Delta vs parity\n\n"
            + boot.to_markdown(index=False, floatfmt=".2f")
            + "\n\n"
        )
        dep = pd.read_csv(out / "panel_depth.csv")
        piv = dep.pivot(index="candidate", columns="obs_depth_band_m", values="mad_m")[
            DEPTH_LABELS
        ]
        f.write(
            "## MAD (m) by observed depth band\n\n"
            + piv.to_markdown(floatfmt=".2f")
            + "\n\n"
        )
        piv = dep.pivot(
            index="candidate", columns="obs_depth_band_m", values="median_r_m"
        )[DEPTH_LABELS]
        f.write(
            "## Median residual (m) by observed depth band\n\n"
            + piv.to_markdown(floatfmt=".2f")
            + "\n"
        )
    log.info("wrote %s", out / "report.md")
    print(core.to_string(index=False))
    print(boot.to_string(index=False))


if __name__ == "__main__":
    main()
