"""Score handily point-path arms against the New Mexico study wells
(``build_nm_study_well_points.py``) with the project metric panel.

Residual = predicted - observed DTW (m); positive = predicted too deep.

Per study, three footprints: ``indep_clean`` (independent of NWIS and at least
``--min-dist-train-km`` from any training well of the arm's bundle), ``clean``
(any provenance, same distance screen) and ``all``. The ``mesilla_2010`` set
carries water-table ELEVATIONS; its observed DTW is ``z_surf_m - wt_elev_m``
with the 100 m land surface the point path sampled (``dtw_from_dem``).

Predictors: every ``--arm``, Ma, and the regional-model DTW surfaces the
builder sampled (reported on their own coverage; the common footprint for the
paired deltas is finite obs + every arm + Ma). Paired HUC8-block bootstrap
(``score_nv_shallow_panel.BlockBoot``) for the deltas of ``--ref`` against
each other arm and against Ma.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_nv_shallow_panel import (  # noqa: E402
    DEPTH_BANDS,
    DEPTH_LABELS,
    BlockBoot,
    delta_rows,
    log,
    panel_stats,
)

MODEL_COLS = [
    "urgb_dtw_2015_m",
    "mrgb_predev_ss_dtw_m",
    "mesilla_model_ss_dtw_m",
    "abq_model_ss_dtw_m",
]
MIN_N = 10


def join_arm(t: pd.DataFrame, path: str, name: str) -> pd.DataFrame:
    p = pd.read_parquet(path)
    keep = ["x5070", "y5070", "pred_dtw_m", "z_surf_m", "r_wte_m", "fac_rem_dtw_m"]
    keep += [c for c in p.columns if c == "sigma_m" or c.startswith("p_dtw_lt_")]
    p = p[[c for c in keep if c in p.columns]].drop_duplicates(["x5070", "y5070"])
    m = t[["x5070", "y5070"]].merge(p, on=["x5070", "y5070"], how="left")
    if len(m) != len(t):
        raise SystemExit(f"{name}: join changed the row count")
    if not np.isfinite(m["pred_dtw_m"]).any():
        raise SystemExit(f"{name}: no finite predictions joined from {path}")
    m = m.drop(columns=["x5070", "y5070"])
    return m.add_prefix(f"{name}__")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--points",
        default="/data/ssd2/handily/nm/regional/wells/nm_study_wells_points.parquet",
    )
    ap.add_argument(
        "--arm", action="append", required=True, help="NAME=pt_preds.parquet"
    )
    ap.add_argument("--ref", required=True, help="arm NAME under evaluation")
    ap.add_argument("--min-dist-train-km", type=float, default=0.05)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t = pd.read_parquet(args.points)
    arms = dict(a.split("=", 1) for a in args.arm)
    if args.ref not in arms:
        raise SystemExit(f"--ref {args.ref} is not an --arm")
    for name, path in arms.items():
        t = pd.concat([t, join_arm(t, path, name)], axis=1)
        log(f"joined {name}: {path}")
    first = next(iter(arms))
    z = t[f"{first}__z_surf_m"].to_numpy("float64")
    for name in arms:
        zz = t[f"{name}__z_surf_m"].to_numpy("float64")
        both = np.isfinite(z) & np.isfinite(zz)
        if both.any() and np.abs(z[both] - zz[both]).max() > 1e-3:
            raise SystemExit(f"{name}: z_surf differs from {first}")
    t["z_surf_m"] = z

    # elevation-only studies -> DTW from the point path's land surface
    derive = ~np.isfinite(t["obs_dtw_m"]) & np.isfinite(t["wt_elev_m"]) & np.isfinite(z)
    t["dtw_from_dem"] = derive
    t.loc[derive, "obs_dtw_m"] = z[derive] - t.loc[derive, "wt_elev_m"].to_numpy()
    log(
        f"observed DTW derived from land surface - WT elevation at {int(derive.sum())} wells"
    )

    obs = t["obs_dtw_m"].to_numpy("float64")
    ma = t["ma_dtw_m"].to_numpy("float64")
    preds = {n: t[f"{n}__pred_dtw_m"].to_numpy("float64") for n in arms}
    common = np.isfinite(obs) & np.isfinite(ma)
    for p in preds.values():
        common &= np.isfinite(p)
    log(
        f"common footprint (finite obs, Ma, every arm): {int(common.sum())} of {len(t)}"
    )
    t = t[common].reset_index(drop=True)
    obs, ma = obs[common], ma[common]
    preds = {n: p[common] for n, p in preds.items()}
    models = {c: t[c].to_numpy("float64") for c in MODEL_COLS if c in t.columns}
    all_preds = {**preds, "Ma": ma, **models}

    indep = t["independent_of_nwis"].to_numpy(bool)
    clean = t["dist_train_km"].to_numpy("float64") >= args.min_dist_train_km
    study = t["study"].to_numpy(object)
    strata: dict[str, np.ndarray] = {}
    for s in ["all studies"] + sorted(set(study)):
        base = np.ones(len(t), bool) if s == "all studies" else study == s
        strata[f"{s} | all"] = base
        strata[f"{s} | clean"] = base & clean
        strata[f"{s} | indep_clean"] = base & clean & indep
        for (lo, hi), lab in zip(DEPTH_BANDS, DEPTH_LABELS):
            strata[f"{s} | clean | obs {lab}"] = base & clean & (obs >= lo) & (obs < hi)
    strata = {k: m for k, m in strata.items() if m.sum() >= MIN_N}

    panel = []
    for label, m in strata.items():
        for name, p in all_preds.items():
            fin = m & np.isfinite(p)
            row = {"stratum": label, "predictor": name, "n": int(fin.sum())}
            if fin.sum() >= MIN_N:
                row.update(panel_stats(obs[fin], p[fin]))
                row["n_stratum"] = int(m.sum())
            panel.append(row)
    pd.DataFrame(panel).to_csv(out / "panel.csv", index=False)

    blocks = t["huc8"].astype(str).to_numpy()
    boot = BlockBoot(blocks, args.n_boot, args.seed)
    log(f"bootstrap: {args.n_boot} resamples over {boot.n_blocks} HUC8 blocks")
    deltas = []
    for name, p in preds.items():
        if name != args.ref:
            deltas += delta_rows(boot, obs, preds[args.ref], p, strata, args.ref, name)
    deltas += delta_rows(boot, obs, preds[args.ref], ma, strata, args.ref, "Ma")
    for mname, mp in models.items():
        fin = np.isfinite(mp)
        sub = {k: m & fin for k, m in strata.items() if (m & fin).sum() >= MIN_N}
        if sub:
            deltas += delta_rows(boot, obs, preds[args.ref], mp, sub, args.ref, mname)
    pd.DataFrame(deltas).to_csv(out / "deltas.csv", index=False)
    t.to_parquet(out / "sites_scored.parquet", index=False)
    summ = {
        "points": args.points,
        "arms": arms,
        "ref": args.ref,
        "min_dist_train_km": args.min_dist_train_km,
        "n_common": int(len(t)),
        "n_dtw_from_dem": int(t["dtw_from_dem"].sum()),
        "n_blocks": int(boot.n_blocks),
        "strata_n": {k: int(m.sum()) for k, m in strata.items()},
    }
    (out / "run_summary.json").write_text(json.dumps(summ, indent=2))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
