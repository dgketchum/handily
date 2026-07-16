"""P3 monitoring-population OOF diagnostic scatter (Ma / Janssen / handily), NM.

The monitoring wells are handily's own training / assimilation population, so
handily's column MUST be the leak-free out-of-fold (OOF) crossfit prediction
(``gnn_dtw_m`` in ``gnn_oof_predictions.parquet``), NEVER the shipped raster
(which trained on these wells). Ma and Janssen are shown for context but are
LIKELY IN-SAMPLE here -- the NM monitoring set is ~88% NWIS, the direct USGS
source both trained on -- so P3 is a DIAGNOSTIC panel that structurally favors the
incumbents, not a fair out-of-sample test (that is P1/P2 in
``plot_shallow_scatter_intercomp.py``). Ma and Janssen are sampled fresh from the
same rasters used there so their construction is identical across panels; only
handily switches to the OOF prediction. The OOF parquet is self-contained for the
join (carries ``obs_dtw_m``, ``gnn_dtw_m``, ``x5070``/``y5070``) -- no
``canonical_id`` join to the live GWX index is needed (and the OOF snapshot's
8-hex ids would not match the index's 16-hex ids anyway).

Reuses the shallow-grid cell drawer (single-color scatter, dashed 1:1, red OLS
fit, standard n/r/slope/RMSE/bias box + off-scale fraction) so the P3 cells are
visually identical to the P1/P2 cells. The depth-banded MAD/medR/p95/floor panel
lives in the report JSON and ``notes/BENCHMARK_INTERCOMPARISON.md``.

Usage:
    uv run python utils/plot_p3_oof_intercomp.py            # obs <= 10 m
    uv run python utils/plot_p3_oof_intercomp.py --cap 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import sample_raster, shallow_skill  # noqa: E402
from plot_shallow_scatter_intercomp import draw_cell  # noqa: E402

OOF_PARQUET = (
    "/data/ssd2/handily/conus/wte_gnn/"
    "gnn_conus_monitoring_water_gate_mirror_sigma_w25_prod/gnn_oof_predictions.parquet"
)
MA_RASTER = "/nas/gwx/wtd_states/wtd_new_mexico.tif"
JANSSEN_RASTER = "/nas/gwx/janssen/V2_140.tif"
# NM lon/lat bbox (matches the P1/P2 fair-panel window).
NM_BBOX_LONLAT = (-109.1, 31.2, -102.9, 37.1)
# LABEL -> per-well column holding that product's DTW prediction (m).
PRODUCTS = (
    ("Ma", "ma_dtw_m"),
    ("Janssen V2", "janssen_dtw_m"),
    ("handily", "gnn_dtw_m"),
)


def build_panel(oof_path: str) -> pd.DataFrame:
    """NM monitoring OOF anchors with Ma/Janssen sampled fresh; common footprint only."""
    df = pd.read_parquet(oof_path)
    df = df[~df["is_water_pseudo"]].copy()
    lon, lat = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True).transform(
        df["x5070"].to_numpy(), df["y5070"].to_numpy()
    )
    df["lon"], df["lat"] = lon, lat
    left, bottom, right, top = NM_BBOX_LONLAT
    nm = df[df.lon.between(left, right) & df.lat.between(bottom, top)].copy()
    nm["ma_dtw_m"] = sample_raster(
        MA_RASTER, nm["lon"].to_numpy(), nm["lat"].to_numpy()
    )
    nm["janssen_dtw_m"] = sample_raster(
        JANSSEN_RASTER, nm["lon"].to_numpy(), nm["lat"].to_numpy()
    )
    finite = np.isfinite(nm["obs_dtw_m"].to_numpy())
    for _, col in PRODUCTS:
        finite &= np.isfinite(nm[col].to_numpy())
    return nm.loc[finite].copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out-dir", default="/data/ssd2/handily/nm/regional/nm_scatter_intercomp"
    )
    ap.add_argument("--cap", type=float, default=10.0, help="Observed-DTW ceiling (m).")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = args.cap

    panel = build_panel(OOF_PARQUET)
    sh = panel[panel["obs_dtw_m"].between(0, cap)].copy()
    n_nwis = int(sh["is_nwis"].sum())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    report = {
        "population": "NM monitoring wells (handily training set); handily = leak-free OOF",
        "cap_m": cap,
        "common_footprint_n_all_depth": int(len(panel)),
        "n_0_to_cap": int(len(sh)),
        "n_nwis_0_to_cap": n_nwis,
        "n_non_nwis_0_to_cap": int(len(sh) - n_nwis),
        "cells": {},
        "shallow_skill_all_depth": {},
        "full_depth_r": {},
    }
    for ci, (label, col) in enumerate(PRODUCTS):
        sub = pd.DataFrame({"mean_dtw": sh["obs_dtw_m"], "pred_dtw_m": sh[col]})
        st = draw_cell(
            axes[ci],
            sub,
            cap,
            show_ylabel=(ci == 0),
            ylabel="NM monitoring wells",
            show_title=True,
            title=label,
        )
        report["cells"][label] = st
        report["shallow_skill_all_depth"][label] = shallow_skill(
            panel[col].to_numpy(), panel["obs_dtw_m"].to_numpy()
        )
        report["full_depth_r"][label] = float(
            np.corrcoef(panel["obs_dtw_m"], panel[col])[0, 1]
        )

    fig.suptitle(
        f"NM P3 monitoring-population OOF diagnostic — observed DTW ≤ {cap:g} m "
        f"(handily = leak-free OOF; Ma/Janssen likely in-sample, {n_nwis}/{len(sh)} NWIS)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out_png = out_dir / f"p3_oof_intercomp_obs_le_{cap:g}m.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # Durable per-well join for reproduction / QGIS.
    keep = ["canonical_id", "source", "is_nwis", "lon", "lat", "obs_dtw_m"] + [
        c for _, c in PRODUCTS
    ]
    sh[keep].to_csv(out_dir / "p3_oof_intercomp_wells.csv", index=False)
    with open(out_dir / f"p3_oof_intercomp_report_le_{cap:g}m.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nFigure: {out_png}")


if __name__ == "__main__":
    main()
