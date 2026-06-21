"""Score the hybrid fusion (FAC + regional + covariates) vs Ma on GWX wells.

Reads the leak-free nested-CV OOF predictions written by
``build_wte_regional_prior.py`` (``residual_model_oof_predictions.parquet``),
rejoins ``source`` from the GWX labels parquet via ``canonical_id``, and scores
the fusion / regional / FAC / Ma predictors with the same metric panel the FAC
and regional-prior GWX validators use -- HEADLINE on the independent (non-NWIS)
wells, SECONDARY on NWIS (Ma flagged contaminated). Ma stays benchmark-only.

    uv run python utils/score_fusion_gwx.py \\
        --oof    .../hybrid/gwx/regional_prior/residual_model_oof_predictions.parquet \\
        --labels .../evidence/gwx/gwx_wte_labels.parquet \\
        --streams .../streams_regional.fgb \\
        --out-dir .../hybrid/gwx/fusion_validation
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

log = logging.getLogger("score_fusion_gwx")
NWIS = {"nwis", "ngwmn"}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--oof", required=True, help="residual_model_oof_predictions.parquet"
    )
    p.add_argument(
        "--labels", required=True, help="gwx_wte_labels.parquet (source key)"
    )
    p.add_argument("--streams", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--valley-dist-m", type=float, default=500.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oof = gpd.read_parquet(args.oof).to_crs(5070)
    m = next(c for c in oof.columns if c.startswith("hybrid_dtw_oof__")).split("__", 1)[
        1
    ]
    log.info("residual method: %s; oof rows: %d", m, len(oof))

    labels = pd.read_parquet(
        args.labels, columns=["canonical_id", "source", "well_class"]
    )
    oof = oof.merge(labels, on="canonical_id", how="left")
    if oof["source"].isna().any():
        n = int(oof["source"].isna().sum())
        raise SystemExit(f"{n} oof rows did not rejoin a source via canonical_id")

    oof["mean_dtw"] = oof["obs_dtw_m"].astype("float64")
    oof["pred_FAC"] = oof["fac_dtw_m"].astype("float64")
    oof["pred_Ma"] = oof["ma_dtw_m"].astype("float64")
    oof["pred_Regional"] = oof[f"regional_dtw_oof__{m}"].astype("float64")
    oof["pred_Fusion"] = oof[f"hybrid_dtw_oof__{m}"].astype("float64")
    preds = ["FAC", "Ma", "Regional", "Fusion"]

    setting, dist = tag_setting(oof, args.streams, args.valley_dist_m)
    oof["setting"] = setting
    oof["dist_stream_m"] = dist
    oof["y5070"] = oof.geometry.y
    for label in preds:
        oof[f"resid_{label}"] = oof[f"pred_{label}"] - oof["mean_dtw"]

    finite = np.ones(len(oof), bool)
    for label in preds:
        finite &= oof[f"pred_{label}"].notna().to_numpy()
    cw = oof.loc[finite].copy()
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
    cw[keep].to_file(out_dir / "fusion_well_residuals.fgb", driver="FlatGeobuf")

    def headline(sub: pd.DataFrame) -> dict:
        o = sub["mean_dtw"].to_numpy("float64")
        return {
            label: resid_stats(sub[f"pred_{label}"].to_numpy(), o) for label in preds
        }

    run = {
        "oof": args.oof,
        "residual_method": m,
        "n_common_independent": int(len(indep_cw)),
        "n_common_nwis": int(len(nwis_cw)),
        "predictors": preds,
        "scoring": "nested-CV OOF fusion; Ma benchmark-only; non-NWIS headline",
        "caveats": [
            "Independent set shallow-dominated; deep bands small -> read NWIS panel.",
            "NWIS panel Ma is leakage-inflated (trained on NWIS).",
        ],
        "headline_independent": headline(indep_cw),
        "secondary_nwis_Ma_contaminated": headline(nwis_cw) if len(nwis_cw) else None,
    }
    with open(out_dir / "fusion_validation_run.json", "w") as f:
        json.dump(run, f, indent=2)

    print(
        f"\n=== fusion vs FAC/Ma/regional, INDEPENDENT non-NWIS: n={len(indep_cw)} ==="
    )
    sl = sum_indep[sum_indep.group_type == "all"]
    for label in preds:
        r = sl[sl.predictor == label]
        if not r.empty:
            print(
                f"  {label:10} MAD={r.mad_m.iloc[0]:6.2f}  bias={r.bias_m.iloc[0]:+7.2f}  med={r.median_residual_m.iloc[0]:+7.2f}  RMSE={r.rmse_m.iloc[0]:7.2f}"
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
    log.info(
        "wrote score_summary_*.csv, fusion_well_residuals.fgb, fusion_validation_run.json -> %s",
        out_dir,
    )


if __name__ == "__main__":
    main()
