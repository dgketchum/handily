"""Confinement-restricted FAC-vs-Ma water-table scorecard.

DTW means depth to the *unconfined* water table, so the headline accuracy
comparison must be computed on confirmed water-table wells (confinement_class in
{unconfined, unconfined_marginal}), not on the full tier set that folds in
unknown/confined wells. This reads a ``well_residuals.fgb`` produced by
``compare_fac_ma.py`` (per-well ``pred_fac``/``residual_fac``,
``pred_ma``/``residual_ma``, ``confinement_class``, ``tier``, ``dtw_label_m``)
and emits the standard subset table (CSV).

FAC and Ma are scored on the identical well set: rows where either prediction is
non-finite are dropped, because the FAC REM is only defined within its grid /
IDW corridor — a well outside it cannot be scored for FAC (the documented
out-of-grid handling in compare_fac_ma), not a missing-data bug. The dropped
count is reported.

Residual sign convention (from compare_fac_ma): residual = pred - obs, so a
negative bias means the prediction is too shallow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

UNCONFINED = {"unconfined", "unconfined_marginal"}


def _metrics(df: pd.DataFrame) -> dict:
    rf = df["residual_fac"].to_numpy(float)
    rm = df["residual_ma"].to_numpy(float)
    obs = df["dtw_label_m"].to_numpy(float)
    pf = df["pred_fac"].to_numpy(float)
    pm = df["pred_ma"].to_numpy(float)
    out = {
        "n": len(df),
        "fac_mad": round(float(np.median(np.abs(rf))), 2),
        "fac_bias": round(float(np.mean(rf)), 2),
        "fac_rmse": round(float(np.sqrt(np.mean(rf**2))), 2),
        "ma_mad": round(float(np.median(np.abs(rm))), 2),
        "ma_bias": round(float(np.mean(rm)), 2),
        "ma_rmse": round(float(np.sqrt(np.mean(rm**2))), 2),
    }
    for thr, key in ((5, "5"), (2, "2")):
        m = obs < thr
        n_lt = int(m.sum())
        if key == "5":
            out["n_obs_lt5"] = n_lt
        if n_lt > 0:
            out[f"fac_rec{key}"] = round(float(np.mean(pf[m] < thr)), 2)
            out[f"ma_rec{key}"] = round(float(np.mean(pm[m] < thr)), 2)
        else:
            out[f"fac_rec{key}"] = np.nan
            out[f"ma_rec{key}"] = np.nan
    out["winner_mad"] = "FAC" if out["fac_mad"] < out["ma_mad"] else "Ma"
    return out


def build_scorecard(residuals_path: str, out_path: str) -> pd.DataFrame:
    wr = gpd.read_file(residuals_path)
    n_all = len(wr)
    both = wr["residual_fac"].notna() & wr["residual_ma"].notna()
    df = wr[both].copy()
    print(
        f"{n_all} wells in {residuals_path}; "
        f"{int((~both).sum())} dropped (no FAC or Ma prediction); "
        f"{len(df)} scored on the common set"
    )

    uc = df["confinement_class"].isin(UNCONFINED)
    subsets = [
        ("WATER-TABLE primary (unconf+marg, primary)", uc & (df["tier"] == "primary")),
        ("WATER-TABLE all-tier (unconf+marg)", uc),
        ("  unconfined only", df["confinement_class"] == "unconfined"),
        (
            "  unconfined_marginal only",
            df["confinement_class"] == "unconfined_marginal",
        ),
        ("[ref] primary headline (incl unknown)", df["tier"] == "primary"),
        ("[ref] primary+secondary (usable)", df["tier"].isin(["primary", "secondary"])),
    ]

    rows = []
    for name, mask in subsets:
        sub = df[mask]
        if len(sub) == 0:
            print(f"  (skipped empty subset: {name})")
            continue
        rows.append({"subset": name, **_metrics(sub)})

    cols = [
        "subset",
        "n",
        "fac_mad",
        "fac_bias",
        "fac_rmse",
        "ma_mad",
        "ma_bias",
        "ma_rmse",
        "fac_rec5",
        "ma_rec5",
        "n_obs_lt5",
        "fac_rec2",
        "ma_rec2",
        "winner_mad",
    ]
    table = pd.DataFrame(rows)[cols]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(table.to_string(index=False))
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--residuals", required=True, help="well_residuals.fgb from compare_fac_ma.py"
    )
    ap.add_argument("--out", required=True, help="output scorecard CSV")
    args = ap.parse_args()
    build_scorecard(args.residuals, args.out)


if __name__ == "__main__":
    main()
