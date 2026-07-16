"""Score the edge-gated GNN vs FAC / Ma / Regional / Fusion on GWX wells.

Joins the GNN OOF predictions (``train_wte_gnn.py``) with the tabular-fusion OOF
(``build_wte_regional_prior.py``) on ``canonical_id`` so every predictor is read
on the SAME wells and SAME cross-fit regional base, then scores them with the
shared GWX metric panel -- HEADLINE on the independent (non-NWIS) wells,
SECONDARY on NWIS (Ma flagged contaminated). The GNN earns the graph machinery
only if it clears the Fusion MAD on the independent headline. Ma stays
benchmark-only.

    uv run python utils/score_gnn_gwx.py \\
        --gnn-oof .../hybrid/gwx/gnn/gnn_oof_predictions.parquet \\
        --fusion-oof .../hybrid/gwx/regional_prior/residual_model_oof_predictions.parquet \\
        --labels .../evidence/gwx/gwx_wte_labels.parquet \\
        --streams .../streams_regional.fgb \\
        --out-dir .../hybrid/gwx/gnn_validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import resid_stats, tag_setting  # noqa: E402
from validate_regional_prior_gwx_wells import score_scope  # noqa: E402

log = logging.getLogger("score_gnn_gwx")
NWIS = {"nwis", "ngwmn"}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gnn-oof", required=True, help="gnn_oof_predictions.parquet")
    p.add_argument(
        "--fusion-oof", required=True, help="residual_model_oof_predictions.parquet"
    )
    p.add_argument(
        "--labels", required=True, help="gwx_wte_labels.parquet (well_class)"
    )
    p.add_argument("--streams", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--valley-dist-m", type=float, default=500.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gnn = gpd.read_parquet(args.gnn_oof).to_crs(5070)
    fus = pd.read_parquet(args.fusion_oof)
    m = next(c for c in fus.columns if c.startswith("hybrid_dtw_oof__")).split("__", 1)[
        1
    ]
    log.info("residual method: %s; gnn rows=%d fusion rows=%d", m, len(gnn), len(fus))

    fcols = [
        "canonical_id",
        "fac_dtw_m",
        "ma_dtw_m",
        f"regional_dtw_oof__{m}",
        f"hybrid_dtw_oof__{m}",
    ]
    df = gnn.merge(fus[fcols], on="canonical_id", how="inner")
    if len(df) != len(gnn):
        raise SystemExit(
            f"join lost rows: gnn={len(gnn)} joined={len(df)} (canonical_id mismatch)"
        )

    labels = pd.read_parquet(args.labels, columns=["canonical_id", "well_class"])
    df = df.merge(labels, on="canonical_id", how="left")

    df["mean_dtw"] = df["obs_dtw_m"].astype("float64")
    df["pred_FAC"] = df["fac_dtw_m"].astype("float64")
    df["pred_Ma"] = df["ma_dtw_m"].astype("float64")
    df["pred_Regional"] = df[f"regional_dtw_oof__{m}"].astype("float64")
    df["pred_Fusion"] = df[f"hybrid_dtw_oof__{m}"].astype("float64")
    df["pred_GNN"] = df["gnn_dtw_m"].astype("float64")
    preds = ["FAC", "Ma", "Regional", "Fusion", "GNN"]

    setting, dist = tag_setting(df, args.streams, args.valley_dist_m)
    df["setting"] = setting
    df["dist_stream_m"] = dist
    df["y5070"] = df.geometry.y
    for label in preds:
        df[f"resid_{label}"] = df[f"pred_{label}"] - df["mean_dtw"]

    finite = np.ones(len(df), bool)
    for label in preds:
        finite &= df[f"pred_{label}"].notna().to_numpy()
    cw = df.loc[finite].copy()
    log.info("common-footprint rows (all predictors finite): %d", len(cw))

    is_nwis = cw["source"].isin(NWIS).to_numpy()
    indep_cw = cw.loc[~is_nwis].copy()
    nwis_cw = cw.loc[is_nwis].copy()

    sum_indep = score_scope(indep_cw, preds, args.valley_dist_m)
    sum_indep.to_csv(out_dir / "score_summary_independent.csv", index=False)
    if len(nwis_cw):
        score_scope(nwis_cw, preds, args.valley_dist_m).to_csv(
            out_dir / "score_summary_nwis.csv", index=False
        )

    keep = [
        "source",
        "well_class",
        "mean_dtw",
        "setting",
        "dist_stream_m",
        *[f"pred_{label}" for label in preds],
        *[f"resid_{label}" for label in preds],
        "geometry",
    ]
    cw[keep].to_file(out_dir / "gnn_well_residuals.fgb", driver="FlatGeobuf")

    def headline(sub: pd.DataFrame) -> dict:
        o = sub["mean_dtw"].to_numpy("float64")
        return {
            label: resid_stats(sub[f"pred_{label}"].to_numpy(), o) for label in preds
        }

    run = {
        "gnn_oof": args.gnn_oof,
        "fusion_oof": args.fusion_oof,
        "residual_method": m,
        "n_common_independent": int(len(indep_cw)),
        "n_common_nwis": int(len(nwis_cw)),
        "predictors": preds,
        "scoring": "OOF GNN vs FAC/Ma/Regional/Fusion; Ma benchmark-only; non-NWIS headline",
        "caveats": [
            "Independent set shallow-dominated; deep bands small -> read NWIS panel.",
            "NWIS panel Ma is leakage-inflated (trained on NWIS).",
            "GNN must clear Fusion MAD on the independent headline to earn the graph.",
        ],
        "headline_independent": headline(indep_cw),
        "secondary_nwis_Ma_contaminated": headline(nwis_cw) if len(nwis_cw) else None,
    }
    (out_dir / "gnn_validation_run.json").write_text(json.dumps(run, indent=2))

    print(
        f"\n=== GNN vs FAC/Ma/Regional/Fusion, INDEPENDENT non-NWIS: n={len(indep_cw)} ==="
    )
    sl = sum_indep[sum_indep.group_type == "all"]
    for label in preds:
        r = sl[sl.predictor == label]
        if not r.empty:
            print(
                f"  {label:10} MAD={r.mad_m.iloc[0]:6.2f}  bias={r.bias_m.iloc[0]:+7.2f}  "
                f"med={r.median_residual_m.iloc[0]:+7.2f}  RMSE={r.rmse_m.iloc[0]:7.2f}"
            )
    print("\n  by obs-depth band (MAD):")
    for g in [f"{lo}-{hi}m" for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 30)]] + [
        "30-infm"
    ]:
        sb = sum_indep[(sum_indep.group_type == "obs_depth") & (sum_indep.group == g)]
        if not sb.empty:
            line = f"  {g:10} n={int(sb['n'].iloc[0]):5d}"
            for label in preds:
                rr = sb[sb.predictor == label]
                if not rr.empty:
                    line += f"  {label}={rr.mad_m.iloc[0]:5.2f}"
            print(line)
    print("\n  by setting (MAD):")
    for s in ("valley", "upland"):
        sb = sum_indep[(sum_indep.group_type == "setting") & (sum_indep.group == s)]
        if not sb.empty:
            line = f"  {s:10} n={int(sb['n'].iloc[0]):5d}"
            for label in preds:
                rr = sb[sb.predictor == label]
                if not rr.empty:
                    line += f"  {label}={rr.mad_m.iloc[0]:5.2f}"
            print(line)
    log.info(
        "wrote score_summary_*.csv, gnn_well_residuals.fgb, gnn_validation_run.json -> %s",
        out_dir,
    )


if __name__ == "__main__":
    main()
