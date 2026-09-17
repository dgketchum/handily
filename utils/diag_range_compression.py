"""Range-compression diagnostics for the four-expert gated CONUS GNN.

Answers, without training anything, where the pred-vs-obs slope < 1 in the
observed 0-10 m band comes from (``notes/RANGE_COMPRESSION_PLAN.md`` D2-D6).

The model is a convex mixture in WTE space, ``wte = sum_i w_i * wte_i`` with
experts fac, deep, mirror (``z_surf - d``) and the free head, so in DTW space
``dtw = sum_i w_i * dtw_i`` exactly (the weights sum to one). Every expert's
DTW is reconstructed from the artifacts and the residual is decomposed as
``sum_i w_i * (dtw_i - obs)`` per observed band, which sums to the bias of the
mixture. Inference-time counterfactuals then remix the SAME expert values with
a different gate (top-1, drop-mirror, head-only, sharpened ``w^tau``) to show
what the gate alone is doing to the slope.

CONUS: the out-of-fold parquet (fold-test rows carry the fold's own gate, so
the identity closes to float precision). Nevada: the 100 m rendered layers at
the statewide-eval sites; the renderer aggregates ``wte`` by fold median and the
gate by fold mean, so the identity closes only approximately there and the
closure error is reported, with the FAC expert taken from the point-path
``fac_rem_dtw_m`` (the FAC surface is not written by the renderer).

Slopes: OLS on all rows plus Theil-Sen on a fixed-seed subsample of 4,000 rows
(the pairwise median is quadratic in n). CI95 on the OLS slope from a block
bootstrap (CONUS blocks = HUC12 ``cv_unit``; NV blocks = HUC8 ``basin_hit``).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import theilslopes

BAND_EDGES = (0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, np.inf)
BAND_LABELS = (
    "0-1",
    "1-2",
    "2-3",
    "3-5",
    "5-7",
    "7-10",
    "10-15",
    "15-20",
    "20-30",
    "30-50",
    "50+",
)
SLOPE_WINDOWS = ((0.0, 5.0), (0.0, 10.0), (2.0, 10.0), (0.0, 30.0))
EXPERTS = ("fac", "deep", "mirror", "head")
W_MIN = 0.05  # weight floor for back-solving an expert from its contribution
NV_LAYERS = {
    "wte": "gnn_wte_100m.tif",
    "dtw": "gnn_dtw_100m.tif",
    "head_wte": "gnn_head_wte_100m.tif",
    "deep_wte": "gnn_deep_wte_100m.tif",
    "r_wte": "gnn_r_wte_100m.tif",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def band_of(v: np.ndarray) -> pd.Categorical:
    return pd.cut(v, BAND_EDGES, labels=BAND_LABELS, include_lowest=True, right=False)


def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    b, a = np.polyfit(x, y, 1)
    return float(b), float(a)


def theil(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    if len(x) > 4000:
        idx = np.random.default_rng(seed).choice(len(x), 4000, replace=False)
        x, y = x[idx], y[idx]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # scipy's unused CI sqrt
        b, a, _, _ = theilslopes(y, x)
    return float(b), float(a)


def slope_rows(
    obs: np.ndarray,
    pred: np.ndarray,
    blocks: np.ndarray,
    label: str,
    n_boot: int,
    seed: int,
) -> list[dict]:
    rows = []
    rng = np.random.default_rng(seed)
    for lo, hi in SLOPE_WINDOWS:
        m = (obs >= lo) & (obs < hi) & np.isfinite(pred)
        if m.sum() < 30:
            continue
        x, y, blk = obs[m], pred[m], blocks[m]
        b_ols, a_ols = ols(x, y)
        b_ts, a_ts = theil(x, y, seed)
        ub, inv = np.unique(blk, return_inverse=True)
        per = [np.flatnonzero(inv == i) for i in range(len(ub))]
        bs = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(ub), len(ub))
            idx = np.concatenate([per[i] for i in pick])
            bs.append(np.polyfit(x[idx], y[idx], 1)[0])
        lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5])
        rows.append(
            {
                "predictor": label,
                "obs_window_m": f"{lo:g}-{hi:g}",
                "n": int(m.sum()),
                "n_blocks": int(len(ub)),
                "ols_slope": round(b_ols, 3),
                "ols_intercept_m": round(a_ols, 2),
                "ols_slope_ci95_lo": round(float(lo_ci), 3),
                "ols_slope_ci95_hi": round(float(hi_ci), 3),
                "theilsen_slope": round(b_ts, 3),
                "theilsen_intercept_m": round(a_ts, 2),
            }
        )
    return rows


def band_rows(obs: np.ndarray, pred: np.ndarray, label: str, extra: dict | None = None):
    """Per observed-band panel: n, medR, bias, MAD, RMSE (m); optional extra
    per-row arrays summarised by band median."""
    res = pred - obs
    df = pd.DataFrame({"band": band_of(obs), "res": res})
    for k, v in (extra or {}).items():
        df[k] = v
    rows = []
    for b, g in df.groupby("band", observed=True):
        ok = g["res"].notna()
        r = g.loc[ok, "res"].to_numpy()
        row = {
            "predictor": label,
            "band_m": str(b),
            "n": int(ok.sum()),
            "medR_m": round(float(np.median(r)), 2),
            "bias_m": round(float(np.mean(r)), 2),
            "MAD_m": round(float(np.median(np.abs(r))), 2),
            "RMSE_m": round(float(np.sqrt(np.mean(r**2))), 2),
        }
        for k in extra or {}:
            row[f"{k}_med"] = round(float(g.loc[ok, k].median()), 3)
        rows.append(row)
    ok = np.isfinite(res)
    rows.append(
        {
            "predictor": label,
            "band_m": "all",
            "n": int(ok.sum()),
            "medR_m": round(float(np.median(res[ok])), 2),
            "bias_m": round(float(np.mean(res[ok])), 2),
            "MAD_m": round(float(np.median(np.abs(res[ok]))), 2),
            "RMSE_m": round(float(np.sqrt(np.mean(res[ok] ** 2))), 2),
        }
    )
    return rows


def decomposition_rows(obs: np.ndarray, dtw: dict, w: dict) -> list[dict]:
    """Exact per-band residual decomposition: mean of w_i*(dtw_i - obs) per
    expert (m), which sums to the mixture bias; plus each expert's own medR
    over the rows where its weight exceeds W_MIN (its value is only defined
    there for the back-solved experts)."""
    band = band_of(obs)
    rows = []
    for b in list(BAND_LABELS) + ["all"]:
        m = np.ones(len(obs), bool) if b == "all" else np.asarray(band == b)
        if not m.any():
            continue
        row = {"band_m": b, "n": int(m.sum())}
        total = 0.0
        for e in EXPERTS:
            # an undefined expert only occurs where its weight is 0: no contribution
            err = np.where(np.isfinite(dtw[e][m]), dtw[e][m] - obs[m], 0.0)
            c = float(np.mean(w[e][m] * err))
            total += c
            row[f"contrib_{e}_m"] = round(c, 2)
            row[f"w_{e}_mean"] = round(float(np.mean(w[e][m])), 3)
            sel = m & (w[e] > W_MIN) & np.isfinite(dtw[e])
            row[f"medR_{e}_m"] = (
                round(float(np.median(dtw[e][sel] - obs[sel])), 2)
                if sel.sum()
                else np.nan
            )
            row[f"n_{e}_defined"] = int(sel.sum())
        row["contrib_sum_m"] = round(total, 2)
        rows.append(row)
    return rows


def remixes(dtw: dict, w: dict) -> dict[str, np.ndarray]:
    """Inference-time gate counterfactuals on the same expert values (DTW, m)."""
    W = np.stack([w[e] for e in EXPERTS], axis=1)
    D = np.stack([dtw[e] for e in EXPERTS], axis=1)
    out = {}
    out["mix"] = np.nansum(W * D, axis=1)
    top = np.argmax(W, axis=1)
    out["top1"] = D[np.arange(len(top)), top]
    for tau in (2.0, 4.0):
        Wt = W**tau
        Wt /= Wt.sum(1, keepdims=True)
        out[f"sharp_tau{tau:g}"] = np.nansum(Wt * D, axis=1)
    Wd = W.copy()
    Wd[:, EXPERTS.index("mirror")] = 0.0
    s = Wd.sum(1, keepdims=True)
    Wd = np.where(s > 0, Wd / np.where(s > 0, s, 1.0), W)
    out["drop_mirror"] = np.nansum(Wd * D, axis=1)
    out["head_only"] = D[:, EXPERTS.index("head")]
    out["fac_only"] = D[:, EXPERTS.index("fac")]
    return out


def load_conus(
    arm_dir: Path, graph_dir: Path
) -> tuple[pd.DataFrame, dict, dict, float]:
    o = pd.read_parquet(arm_dir / "gnn_oof_predictions.parquet")
    q = pd.read_parquet(
        graph_dir / "query_nodes.parquet",
        columns=[
            "canonical_id",
            "fac_rem_wte_anom_m",
            "deep_regional_wte_anom_m",
            "cv_unit",
        ],
    )
    m = o.merge(q, on="canonical_id", how="left", validate="one_to_one")
    real = ~m["is_water_pseudo"].astype(bool) & ~m["is_shore_pseudo"].astype(bool)
    m = m[real].reset_index(drop=True)
    hp = json.loads((arm_dir / "gnn_run.json").read_text())["hyperparams"]
    has_mirror = "gate_w_mirror" in m.columns
    d = float(hp["mirror_depth_m"]) if has_mirror else np.nan
    if not has_mirror:
        log("  no mirror expert in this arm: w_mirror = 0, mirror expert undefined")
    z = m["z_surf_well_m"].to_numpy("float64")
    R = m["regional_wte_idw_oof_m"].to_numpy("float64")
    w = {
        e: m[f"gate_w_{e}"].to_numpy("float64")
        if f"gate_w_{e}" in m.columns
        else np.zeros(len(m))
        for e in EXPERTS
    }
    wte = {
        "fac": R + m["fac_rem_wte_anom_m"].to_numpy("float64"),
        "deep": R + m["deep_regional_wte_anom_m"].to_numpy("float64"),
        "mirror": z - d,
    }
    head_contrib = m["gnn_wte_hat_m"].to_numpy("float64") - sum(
        w[e] * np.where(np.isfinite(wte[e]), wte[e], 0.0)
        for e in ("fac", "deep", "mirror")
    )
    nz = w["head"] > 1e-6
    wte["head"] = np.where(nz, head_contrib / np.where(nz, w["head"], 1.0), np.nan)
    dtw = {e: z - wte[e] for e in EXPERTS}
    mix = sum(w[e] * np.where(np.isfinite(wte[e]), wte[e], 0.0) for e in EXPERTS)
    closure = mix - m["gnn_wte_hat_m"].to_numpy("float64")
    log(
        f"  CONUS closure |mix - wte_hat|: max {np.max(np.abs(closure)):.2e} m; "
        f"head weight below 1e-6 on {int((~nz).sum())}/{len(nz)} rows, "
        f"below {W_MIN} on {int((w['head'] <= W_MIN).sum())} rows"
    )
    # remixes need a value on every row; a numerically-zero head carries the mixture
    dtw["head"] = np.where(nz, dtw["head"], m["gnn_dtw_m"].to_numpy("float64"))
    return m, dtw, w, d


def load_nv(
    sites: pd.DataFrame, huc8_root: Path, arm_dirname: str, pt_preds: Path
) -> tuple[pd.DataFrame, dict, dict, float, dict]:
    s = sites.copy()
    pt = pd.read_parquet(
        pt_preds, columns=["x5070", "y5070", "fac_rem_dtw_m", "z_surf_m"]
    )
    s = s.merge(pt, on=["x5070", "y5070"], how="left", validate="one_to_one")
    fac_absent = s["fac_rem_dtw_m"].isna().to_numpy()
    n = len(s)
    x = s["x5070"].to_numpy("float64")
    y = s["y5070"].to_numpy("float64")
    samp = {
        k: np.full(n, np.nan) for k in list(NV_LAYERS) + [f"w_{e}" for e in EXPERTS]
    }
    d = None
    for basin, idx in s.groupby("basin_hit").indices.items():
        bdir = huc8_root / basin / "gnn" / arm_dirname
        run = json.loads((bdir / "infer_run.json").read_text())
        man = json.loads(
            (Path(run["model_dir"]) / "models" / "inference_manifest.json").read_text()
        )
        md = float(man["flags"]["mirror_depth_m"])
        if d is None:
            d = md
        elif md != d:
            raise SystemExit(f"{basin}: mirror depth {md} != {d}")
        with rasterio.open(bdir / "gnn_gate_w_100m.tif") as ds:
            if tuple(ds.descriptions) != tuple(f"w_{e}" for e in EXPERTS):
                raise SystemExit(f"{basin}: gate bands {ds.descriptions}")
            tr = ds.transform
            col = np.floor((x[idx] - tr.c) / tr.a).astype("int64")
            row = np.floor((y[idx] - tr.f) / tr.e).astype("int64")
            g = ds.read().astype("float64")
            nd = ds.nodata
            for i, e in enumerate(EXPERTS):
                v = g[i, row, col]
                if nd is not None:
                    v = np.where(v == nd, np.nan, v)
                samp[f"w_{e}"][idx] = v
        for k, name in NV_LAYERS.items():
            with rasterio.open(bdir / name) as ds:
                a = ds.read(1).astype("float64")
                v = a[row, col]
                if ds.nodata is not None:
                    v = np.where(v == ds.nodata, np.nan, v)
                samp[k][idx] = v
    for k, v in samp.items():
        s[k] = v
    if not np.isfinite(s["dtw"]).all():
        raise SystemExit("NV: sampled dtw has non-finite values at sites; investigate")
    z = s["wte"].to_numpy("float64") + s["dtw"].to_numpy("float64")
    w = {e: s[f"w_{e}"].to_numpy("float64") for e in EXPERTS}
    # The point path's FAC-REM sample is nodata at some sites while the 100 m
    # render still carries a FAC expert there (the renderer samples the FAC mosaic
    # on its own grid). Where the point value is absent: back-solve the FAC
    # expert from the mixture identity if the gate weights it (> W_MIN), else
    # carry the mixture DTW (an expert the gate masks out has no value to solve).
    mix_no_fac = (
        w["deep"] * s["deep_wte"].to_numpy("float64")
        + w["mirror"] * (z - d)
        + w["head"] * s["head_wte"].to_numpy("float64")
    )
    solvable = fac_absent & (w["fac"] > W_MIN)
    fac_wte_solved = (s["wte"].to_numpy("float64") - mix_no_fac) / np.where(
        w["fac"] > W_MIN, w["fac"], 1.0
    )
    fac_dtw = s["fac_rem_dtw_m"].to_numpy("float64").copy()
    fac_dtw[solvable] = (z - fac_wte_solved)[solvable]
    fac_dtw[fac_absent & ~solvable] = s["dtw"].to_numpy("float64")[
        fac_absent & ~solvable
    ]
    log(
        f"  FAC point value absent at {int(fac_absent.sum())}/{n} sites: "
        f"{int(solvable.sum())} back-solved (w_fac > {W_MIN}), "
        f"{int((fac_absent & ~solvable).sum())} carry the mixture"
    )
    wte = {
        "fac": z - fac_dtw,
        "deep": s["deep_wte"].to_numpy("float64"),
        "mirror": z - d,
        "head": s["head_wte"].to_numpy("float64"),
    }
    mix = sum(w[e] * wte[e] for e in EXPERTS)
    closure = mix - s["wte"].to_numpy("float64")
    fac_closure_free = closure[w["fac"] <= 0.001]
    info = {
        "closure_median_abs_m": round(float(np.median(np.abs(closure))), 3),
        "closure_p90_abs_m": round(float(np.percentile(np.abs(closure), 90)), 3),
        "closure_max_abs_m": round(float(np.max(np.abs(closure))), 3),
        "closure_no_fac_median_abs_m": round(
            float(np.median(np.abs(fac_closure_free))), 3
        )
        if len(fac_closure_free)
        else np.nan,
        "n_no_fac_rows": int(len(fac_closure_free)),
        "z_vs_pt_zsurf_max_abs_m": round(
            float(np.max(np.abs(z - s["z_surf_m"].to_numpy("float64")))), 3
        ),
        "mirror_depth_m": d,
    }
    log(f"  NV closure: {info}")
    dtw = {e: z - wte[e] for e in EXPERTS}
    return s, dtw, w, d, info


def run_arm(
    label: str,
    obs: np.ndarray,
    mix_pred: np.ndarray,
    dtw: dict,
    w: dict,
    blocks: np.ndarray,
    extra: dict,
    out_dir: Path,
    n_boot: int,
    seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rm = remixes(dtw, w)
    dm = np.abs(rm["mix"] - mix_pred)
    log(
        f"  remix 'mix' vs artifact dtw: median {np.nanmedian(dm):.4f} m, "
        f"p99 {np.nanpercentile(dm, 99):.3f} m, max {np.nanmax(dm):.3f} m"
    )
    rm["artifact"] = mix_pred
    preds = {"artifact": mix_pred}
    for e in EXPERTS:
        preds[f"expert_{e}"] = np.where(w[e] > W_MIN, dtw[e], np.nan)
    preds |= {k: v for k, v in rm.items() if k not in ("mix", "artifact")}
    band_tab = []
    slope_tab = []
    for k, p in preds.items():
        band_tab += band_rows(obs, p, k, extra if k == "artifact" else None)
        slope_tab += slope_rows(obs, p, blocks, k, n_boot, seed)
    band_df = pd.DataFrame(band_tab)
    slope_df = pd.DataFrame(slope_tab)
    dec_df = pd.DataFrame(decomposition_rows(obs, dtw, w))
    band_df.to_csv(out_dir / "band_panel.csv", index=False)
    slope_df.to_csv(out_dir / "slopes.csv", index=False)
    dec_df.to_csv(out_dir / "residual_decomposition.csv", index=False)
    log(f"  [{label}] slopes 0-10 m:")
    log(
        slope_df[slope_df.obs_window_m == "0-10"][
            [
                "predictor",
                "n",
                "ols_slope",
                "ols_slope_ci95_lo",
                "ols_slope_ci95_hi",
                "theilsen_slope",
                "ols_intercept_m",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--oof-arm", action="append", default=[], help="NAME=run_dir (CONUS OOF)"
    )
    ap.add_argument(
        "--graph-dir",
        default="/data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2",
    )
    ap.add_argument(
        "--nv-arm",
        action="append",
        default=[],
        help="NAME=huc8_arm_dirname=pt_preds.parquet",
    )
    ap.add_argument(
        "--nv-sites",
        default="/data/ssd2/handily/nv/regional/statewide_eval_m20_ord/sites_sampled.parquet",
    )
    ap.add_argument("--huc8-root", default="/data/ssd2/handily/huc8")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}

    for spec in args.oof_arm:
        name, run_dir = spec.split("=", 1)
        log(f"CONUS OOF arm {name}: {run_dir}")
        m, dtw, w, d = load_conus(Path(run_dir), Path(args.graph_dir))
        obs = m["obs_dtw_m"].to_numpy("float64")
        extra = {
            "sigma_m": m["gnn_sigma_m"].to_numpy("float64"),
            "nll_attention": (1.0 / m["gnn_sigma_m"].to_numpy("float64"))
            / np.mean(1.0 / m["gnn_sigma_m"].to_numpy("float64")),
            **{f"w_{e}": w[e] for e in EXPERTS},
        }
        run_arm(
            f"conus/{name}",
            obs,
            m["gnn_dtw_m"].to_numpy("float64"),
            dtw,
            w,
            m["cv_unit"].to_numpy(),
            extra,
            out / "conus" / name,
            args.n_boot,
            args.seed,
        )
        summary[f"conus/{name}"] = {"n": int(len(obs)), "mirror_depth_m": d}

    sites = None
    if args.nv_arm:
        sites = pd.read_parquet(args.nv_sites)
        sites = sites[np.isfinite(sites["obs_dtw_m"])].reset_index(drop=True)
    for spec in args.nv_arm:
        name, dirname, pt = spec.split("=", 2)
        log(f"NV arm {name}: {dirname}")
        s, dtw, w, d, info = load_nv(sites, Path(args.huc8_root), dirname, Path(pt))
        for set_name, mask in (
            ("primary", s["is_admission_eligible"].to_numpy(bool)),
            ("all_sites", np.ones(len(s), bool)),
        ):
            sub = {e: dtw[e][mask] for e in EXPERTS}
            ww = {e: w[e][mask] for e in EXPERTS}
            run_arm(
                f"nv/{name}/{set_name}",
                s.loc[mask, "obs_dtw_m"].to_numpy("float64"),
                s.loc[mask, "dtw"].to_numpy("float64"),
                sub,
                ww,
                s.loc[mask, "basin_hit"].to_numpy(),
                {f"w_{e}": ww[e] for e in EXPERTS},
                out / "nv" / name / set_name,
                args.n_boot,
                args.seed,
            )
            summary[f"nv/{name}/{set_name}"] = {"n": int(mask.sum()), **info}
        keep = ["x5070", "y5070", "obs_dtw_m", "is_admission_eligible", "basin_hit"]
        keep += [c for c in NV_LAYERS if c not in keep]
        s[keep + [f"w_{e}" for e in EXPERTS] + ["fac_rem_dtw_m"]].to_parquet(
            out / "nv" / name / "sites_experts.parquet", index=False
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
