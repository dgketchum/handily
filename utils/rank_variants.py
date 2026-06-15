"""Phase 7 of the Ruby HUC8 pilot: rank the FAC10 tuning variants.

Each variant (baseline / v2_drier / v3_wetter / v4_sag) has been scored against
the GWX wells + NHD springs by ``compare_fac_ma.py``, which wrote a long-format
``score_summary.csv`` and a ``well_residuals.fgb`` (per-well ``residual_fac`` +
``weight``) into its own validation subdir. This reads those back and ranks the
variants by the plan's weighted objective (lower = better):

    objective = weighted_huber(usable-well residuals, delta)              # data fit
              + W_SPRING  * (1 - spring_capture_60m)                      # anchor miss
              + W_SHALLOW * (1 - mean(shallow_recall_<2m, shallow_recall_<5m))
              + W_SMOOTH  * mean_abs_laplacian(REM)                       # roughness

The well term is a Huber loss (delta = 2 m) weighted by the observation ``weight``
column (monitoring 1.0 / primary 0.6 / secondary 0.25; diagnostic 0.0 drops out),
so a few deep outliers can't dominate. The spring and shallow penalties convert a
full miss to ~W meters of Huber-equivalent cost (default 5 m each). The smoothness
term is a mild regularizer against the IDW/relaxation staircase.

The weights are deliberate, documented constants — tune them, do NOT tune the FAC
prior to Ma. Ma is included only as a reference row. All raw components are printed
and written so the ranking can be redone by any single criterion.

Usage:
    uv run python utils/rank_variants.py \
        --validation-root /data/ssd2/handily/mt/regional/ruby_huc8/validation \
        --rem-root /data/ssd2/handily/mt/regional/ruby_huc8/rem \
        --variants baseline:ruby_fac10_baseline v2_drier:ruby_v2_drier \
                   v3_wetter:ruby_v3_wetter v4_sag:ruby_v4_sag \
        --out /data/ssd2/handily/mt/regional/ruby_huc8/validation/variant_ranking.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

log = logging.getLogger("rank_variants")

HUBER_DELTA_M = 2.0
W_SPRING = 5.0  # a full spring-capture miss ~ 5 m of Huber-equivalent cost
W_SHALLOW = 5.0  # a full shallow-recall miss ~ 5 m
W_SMOOTH = 1.0  # mean |laplacian| (m) of the REM, mild regularizer
REM_NAME = "fac_head_depth_rem_10m.tif"


def _get(summary: pd.DataFrame, raster: str, gt: str, g: str, stat: str) -> float:
    m = summary[
        (summary.raster == raster)
        & (summary.group_type == gt)
        & (summary.group == g)
        & (summary.statistic == stat)
    ]
    return float("nan") if m.empty else float(m["value"].iloc[0])


def weighted_huber(resid: np.ndarray, weight: np.ndarray, delta: float) -> float:
    r = np.asarray(resid, dtype="float64")
    w = np.asarray(weight, dtype="float64")
    ok = np.isfinite(r) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    r, w = r[ok], w[ok]
    a = np.abs(r)
    h = np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))
    return float(np.sum(w * h) / np.sum(w))


def rem_roughness(path: Path) -> float:
    """Mean absolute discrete Laplacian (m) over finite interior pixels."""
    try:
        with rasterio.open(path) as src:
            a = src.read(1).astype("float64")
            nod = src.nodata
        if nod is not None and np.isfinite(nod):
            a[a == nod] = np.nan
        c = a[1:-1, 1:-1]
        lap = 4.0 * c - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
        lap = lap[np.isfinite(lap)]
        return float(np.mean(np.abs(lap))) if lap.size else float("nan")
    except (FileNotFoundError, rasterio.errors.RasterioIOError) as e:
        log.warning("roughness unavailable for %s: %s", path, e)
        return float("nan")


def variant_row(label: str, val_dir: Path, rem_path: Path) -> dict:
    summ_path = val_dir / "score_summary.csv"
    resid_path = val_dir / "well_residuals.fgb"
    if not summ_path.exists():
        log.warning("%s: no score_summary.csv, skipping", label)
        return {"variant": label, "objective": float("nan"), "status": "missing"}

    summary = pd.read_csv(summ_path)
    capture60 = _get(summary, "fac", "capture", "60m", "capture_rate")
    rec2 = _get(summary, "fac", "shallow_class", "<2m", "recall")
    rec5 = _get(summary, "fac", "shallow_class", "<5m", "recall")
    mean_recall = np.nanmean([rec2, rec5])

    huber = float("nan")
    if resid_path.exists():
        wr = gpd.read_file(resid_path)
        if "residual_fac" in wr and "weight" in wr:
            huber = weighted_huber(
                wr["residual_fac"].to_numpy(), wr["weight"].to_numpy(), HUBER_DELTA_M
            )
    rough = rem_roughness(rem_path)

    spring_pen = W_SPRING * (1.0 - capture60)
    shallow_pen = W_SHALLOW * (1.0 - mean_recall)
    smooth_pen = W_SMOOTH * rough
    objective = np.nansum([huber, spring_pen, shallow_pen, smooth_pen])

    return {
        "variant": label,
        "objective": float(objective),
        "huber_m": huber,
        "spring_capture_60m": capture60,
        "spring_median_resid_m": _get(
            summary, "fac", "residual", "all", "median_residual_m"
        ),
        "shallow_recall_2m": rec2,
        "shallow_recall_5m": rec5,
        "primary_mad_m": _get(
            summary, "fac", "tier", "primary", "median_abs_residual_m"
        ),
        "primary_bias_m": _get(summary, "fac", "tier", "primary", "bias_m"),
        "valley_mad_m": _get(
            summary, "fac", "setting", "valley", "median_abs_residual_m"
        ),
        "roughness_m": rough,
        "spring_penalty": spring_pen,
        "shallow_penalty": shallow_pen,
        "smooth_penalty": smooth_pen,
        "status": "ok",
    }


def ma_reference(val_dir: Path) -> dict:
    """Ma benchmark row (reference only — never a tuning target)."""
    summ_path = val_dir / "score_summary.csv"
    resid_path = val_dir / "well_residuals.fgb"
    if not summ_path.exists():
        return {}
    summary = pd.read_csv(summ_path)
    huber = float("nan")
    if resid_path.exists():
        wr = gpd.read_file(resid_path)
        if "residual_ma" in wr and "weight" in wr:
            huber = weighted_huber(
                wr["residual_ma"].to_numpy(), wr["weight"].to_numpy(), HUBER_DELTA_M
            )
    return {
        "variant": "ma_benchmark",
        "objective": float("nan"),
        "huber_m": huber,
        "spring_capture_60m": _get(summary, "ma", "capture", "60m", "capture_rate"),
        "spring_median_resid_m": _get(
            summary, "ma", "residual", "all", "median_residual_m"
        ),
        "shallow_recall_2m": _get(summary, "ma", "shallow_class", "<2m", "recall"),
        "shallow_recall_5m": _get(summary, "ma", "shallow_class", "<5m", "recall"),
        "primary_mad_m": _get(
            summary, "ma", "tier", "primary", "median_abs_residual_m"
        ),
        "primary_bias_m": _get(summary, "ma", "tier", "primary", "bias_m"),
        "valley_mad_m": _get(
            summary, "ma", "setting", "valley", "median_abs_residual_m"
        ),
        "roughness_m": float("nan"),
        "status": "reference",
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-root", required=True)
    p.add_argument("--rem-root", required=True)
    p.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="label:rem_subdir entries, e.g. baseline:ruby_fac10_baseline",
    )
    p.add_argument("--rem-name", default=REM_NAME)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    val_root = Path(args.validation_root)
    rem_root = Path(args.rem_root)

    rows = []
    for spec in args.variants:
        label, _, rem_sub = spec.partition(":")
        rem_path = rem_root / rem_sub / args.rem_name
        rows.append(variant_row(label, val_root / label, rem_path))

    # Ma reference from the first variant that has a summary.
    for spec in args.variants:
        label = spec.partition(":")[0]
        ref = ma_reference(val_root / label)
        if ref:
            rows.append(ref)
            break

    df = pd.DataFrame(rows)
    ranked = df[df.status == "ok"].sort_values("objective")
    rest = df[df.status != "ok"]
    out = pd.concat([ranked, rest], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    cols = [
        "variant",
        "objective",
        "huber_m",
        "spring_capture_60m",
        "shallow_recall_5m",
        "primary_mad_m",
        "valley_mad_m",
        "roughness_m",
        "status",
    ]
    log.info("Wrote %s", args.out)
    present = [c for c in cols if c in out.columns]
    print(out[present].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not ranked.empty:
        print(f"\nBest variant: {ranked.iloc[0]['variant']}")
    else:
        log.warning("no variant produced a score_summary.csv — nothing to rank")


if __name__ == "__main__":
    main()
