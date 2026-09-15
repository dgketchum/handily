"""CONUS out-of-fold comparison of shallow-lever arms against a reference arm
(tier 1 of ``notes/SHALLOW_LEVER_EXPERIMENTS.md``).

Pairs each arm's ``gnn_oof_predictions.parquet`` with the reference's by well
(real wells only; water and shore pseudo-rows dropped), and reports the full
metric panel (MAD, median residual, mean bias, RMSE, p95 |residual|, all m),
the doctrine depth bands, shallow precision / recall at 2 / 5 / 10 m
(dimensionless), an HUC2 16 (Great Basin) row, and paired block-bootstrap
CI95s on the arm-minus-reference differences (blocks = the bundle's
``cv_unit`` HUC12 fold units, drawn with replacement, the same resampled well
set for both predictors). Sign conventions and the delta definitions are those
of ``utils/score_nv_shallow_panel.py``.

Run::

    uv run --directory /home/dgketchum/code/handily python \
        /home/dgketchum/code/handily/utils/compare_conus_oof_arms.py \
        --ref r1e=/data/ssd2/handily/conus/wte_gnn/gnn_conus_monitoring_water_src_r1e \
        --arm m20=/data/ssd2/handily/conus/wte_gnn/gnn_conus_monitoring_water_src_r1e_m20 \
        --out-dir /data/ssd2/handily/conus/wte_gnn/shallow_levers/tier1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "utils"))

from compare_nv_ma_handily_shape import (  # noqa: E402
    DEPTH_BANDS,
    DEPTH_LABELS,
    SHALLOW_THRESHOLDS,
    panel_stats,
)
from score_nv_shallow_panel import BlockBoot, delta_rows, f1, log, pr_at  # noqa: E402


def load_oof(run_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(run_dir / "gnn_oof_predictions.parquet")
    real = ~df["is_water_pseudo"].to_numpy(bool)
    if "is_shore_pseudo" in df.columns:
        real &= ~df["is_shore_pseudo"].to_numpy(bool)
    df = df[real].reset_index(drop=True)
    if df["canonical_id"].duplicated().any():
        raise SystemExit(f"{run_dir}: canonical_id not unique among real wells")
    return df


def blocks_for(df: pd.DataFrame, graph_dir: Path) -> np.ndarray:
    qn = pd.read_parquet(
        graph_dir / "query_nodes.parquet", columns=["canonical_id", "cv_unit"]
    )
    qn = qn.drop_duplicates("canonical_id").set_index("canonical_id")
    u = qn["cv_unit"].reindex(df["canonical_id"].to_numpy())
    if u.isna().any():
        raise SystemExit(f"{int(u.isna().sum())} wells lack a cv_unit block")
    return u.astype(str).to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="NAME=run_dir of the reference arm")
    ap.add_argument("--arm", action="append", required=True, help="NAME=run_dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ref_name, ref_dir = args.ref.split("=", 1)
    ref = load_oof(Path(ref_dir))
    run = json.loads((Path(ref_dir) / "gnn_run.json").read_text())
    blocks = blocks_for(ref, Path(run["graph_dir"]))
    log(
        f"reference {ref_name}: {len(ref)} real wells, {len(np.unique(blocks))} cv_unit blocks"
    )
    obs = ref["obs_dtw_m"].to_numpy("float64")
    huc2 = ref["huc2"].astype(str).str.zfill(2).to_numpy()
    preds = {ref_name: ref["gnn_dtw_m"].to_numpy("float64")}
    extras = {}
    fold_info = {ref_name: run["folds"]}
    for a in args.arm:
        name, d = a.split("=", 1)
        df = load_oof(Path(d))
        if (
            len(df) != len(ref)
            or (df["canonical_id"].to_numpy() != ref["canonical_id"].to_numpy()).any()
        ):
            df = (
                df.set_index("canonical_id")
                .reindex(ref["canonical_id"].to_numpy())
                .reset_index()
            )
            if df["gnn_dtw_m"].isna().any():
                raise SystemExit(f"{name}: OOF does not cover the reference well set")
        preds[name] = df["gnn_dtw_m"].to_numpy("float64")
        extras[name] = {
            "median_sigma_m": float(np.nanmedian(df["gnn_sigma_m"]))
            if "gnn_sigma_m" in df
            else None,
            "gate_w_mean": {
                c: float(df[c].mean()) for c in df.columns if c.startswith("gate_w_")
            },
            "ordinal_cols": [c for c in df.columns if c.startswith("p_dtw_lt_")],
        }
        fold_info[name] = json.loads((Path(d) / "gnn_run.json").read_text())["folds"]
        log(f"  joined {name}")
    common = np.isfinite(obs)
    for p in preds.values():
        common &= np.isfinite(p)
    obs, huc2, blocks = obs[common], huc2[common], blocks[common]
    preds = {n: p[common] for n, p in preds.items()}
    log(f"  common footprint: {int(common.sum())} wells")

    strata = {"all": np.ones(obs.size, bool), "HUC2 16": huc2 == "16"}
    for (lo, hi), lab in zip(DEPTH_BANDS, DEPTH_LABELS):
        strata[f"obs {lab}"] = (obs >= lo) & (obs < hi)
        strata[f"HUC2 16 obs {lab}"] = (huc2 == "16") & (obs >= lo) & (obs < hi)

    panel, shallow = [], []
    for label, m in strata.items():
        for name, p in preds.items():
            row = {"stratum": label, "predictor": name}
            row.update(
                panel_stats(obs[m], p[m]) if m.sum() >= 30 else {"n": int(m.sum())}
            )
            panel.append(row)
        if label in ("all", "HUC2 16"):
            for thr in SHALLOW_THRESHOLDS:
                for name, p in preds.items():
                    pr, rc = pr_at(obs[m], (p < thr)[m], thr)
                    shallow.append(
                        {
                            "stratum": label,
                            "threshold_m": thr,
                            "predictor": name,
                            "n_obs_shallow": int((obs[m] < thr).sum()),
                            "n_pred_shallow": int((p[m] < thr).sum()),
                            "precision": pr,
                            "recall": rc,
                            "f1": f1(pr, rc),
                        }
                    )
    pd.DataFrame(panel).to_csv(out / "panel.csv", index=False)
    pd.DataFrame(shallow).to_csv(out / "shallow_pr.csv", index=False)

    boot = BlockBoot(blocks, args.n_boot, args.seed)
    deltas = []
    for name, p in preds.items():
        if name == ref_name:
            continue
        deltas += delta_rows(boot, obs, p, preds[ref_name], strata, name, ref_name)
        log(f"  deltas done: {name}")
    pd.DataFrame(deltas).to_csv(out / "deltas.csv", index=False)
    (out / "run_summary.json").write_text(
        json.dumps(
            {
                "ref": ref_name,
                "n_wells": int(obs.size),
                "n_blocks": boot.n_blocks,
                "n_boot": args.n_boot,
                "arm_extras": extras,
                "folds": fold_info,
            },
            indent=2,
        )
    )
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
