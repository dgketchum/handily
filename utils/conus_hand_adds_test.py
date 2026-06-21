"""Decisive test of the GNN premise: does OWP HAND reduce DTW error OVER the
IDW well-interpolation prior, or is it redundant?

Joins the HAND-probe wells (canonical_id, hand_m, mean_dtw) to the CONUS IDW OOF
predictions (idw, janssen, block). Then, blocked-CV (group=40km block) on these
wells, compares:
  M0  IDW alone (the regional prior, already OOF)
  Mj  Janssen (benchmark)
  M1  GBM(idw, hand)           <- does HAND add to the prior?
  M2  GBM(idw, hand, x, y)     <- HAND + smooth spatial trend

If M1 ~ M0, HAND carries no marginal signal over interpolation and the
HAND/flow-graph build is not worth its cost (consistent with the RGA GNN
negative). If M1 << M0, the build has life.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

PROBE = (
    "/tmp/claude-1002/-home-dgketchum-code-handily/"
    "bbbe8296-ffe5-4f10-a7cd-8e7cb124f528/scratchpad/conus_hand_probe.parquet"
)
IDW = (
    "/tmp/claude-1002/-home-dgketchum-code-handily/"
    "bbbe8296-ffe5-4f10-a7cd-8e7cb124f528/scratchpad/idw_oof.parquet"
)


def mad(a, b):
    return float(np.median(np.abs(a - b)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cv_gbm(X, y, groups, folds=5):
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, groups):
        m = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_depth=3,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=1.0,
            min_samples_leaf=40,
            random_state=0,
        )
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def report(name, pred, dtw, lon):
    print(f"\n{name}: MAD={mad(pred, dtw):.2f}  RMSE={rmse(pred, dtw):.1f}")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 30), (30, 1000)]:
        m = (dtw >= lo) & (dtw < hi)
        if m.sum() < 30:
            continue
        print(f"    {lo:>2}-{hi:<4}m n={m.sum():5d}  MAD={mad(pred[m], dtw[m]):6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=PROBE)
    ap.add_argument("--idw", default=IDW)
    args = ap.parse_args()

    hand = pd.read_parquet(args.probe)[["canonical_id", "hand_m", "huc8"]]
    idw = pd.read_parquet(args.idw)
    df = hand.merge(idw, on="canonical_id", how="inner")
    df = df[
        np.isfinite(df["hand_m"]) & np.isfinite(df["idw"]) & np.isfinite(df["mean_dtw"])
    ]
    nwis = df["source"].isin({"nwis", "ngwmn"})
    df = df[~nwis].reset_index(drop=True)
    print(f"joined probe-HAND x IDW-OOF, independent wells: n={len(df):,}")

    dtw = df["mean_dtw"].values
    lon = df["longitude"].values
    groups = df["block"].values

    print("\n==== baselines (these exact wells) ====")
    report("M0  IDW alone", df["idw"].values, dtw, lon)
    report("Mj  Janssen", df["janssen"].values, dtw, lon)

    print("\n==== does HAND add over the IDW prior? ====")
    X1 = df[["idw", "hand_m"]].values
    p1 = cv_gbm(X1, dtw, groups)
    report("M1  GBM(idw, hand)", p1, dtw, lon)

    X2 = df[["idw", "hand_m", "x", "y"]].values
    p2 = cv_gbm(X2, dtw, groups)
    report("M2  GBM(idw, hand, x, y)", p2, dtw, lon)

    # marginal value of HAND: GBM(idw) vs GBM(idw, hand) head to head
    p_idw_gbm = cv_gbm(df[["idw"]].values, dtw, groups)
    print("\n==== HAND marginal (apples-to-apples GBM) ====")
    print(f"  GBM(idw)       MAD={mad(p_idw_gbm, dtw):.3f}")
    print(f"  GBM(idw,hand)  MAD={mad(p1, dtw):.3f}")
    print(f"  delta from adding HAND: {mad(p_idw_gbm, dtw) - mad(p1, dtw):+.3f} m")


if __name__ == "__main__":
    main()
