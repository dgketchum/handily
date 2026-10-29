"""Sweep pumping-well R variants (utils/pumping_r_core.Variant) at scoring sites.

Donors load once; every variant in ``--variants`` (a JSON list of Variant field
dicts, each with a unique ``name``) is evaluated at every ``--sites`` set and
scored against the observed water table and the sites' reference R.

Site files (utils/build_r_sweep_sites_nm.py): site_id, x5070, y5070, z_m,
obs_dtw_m, block, r_ref_m (uncapped reference R), r_ref_cap_m (reference R under
its ground cap), optional ma_dtw_m (context DTW map).

Residual r = R - observed WTE = obs DTW - predicted DTW, m; positive = R above the
table = DTW too shallow. Candidates per variant: ``<name>:blend`` (the variant's R,
capped at its cap_m), ``<name>:median`` and ``<name>:q75`` (components, same cap).
References: ``ref`` (uncapped) and ``ref_cap``; ``ma`` (Ma DTW map, context only,
residual = obs - Ma) where the site file carries it.

Deltas: MAD(reference) - MAD(candidate) and RMSE(reference) - RMSE(candidate), m;
positive = candidate better; paired block bootstrap over ``block`` (HUC8), one
resample set per site set reused for every candidate, CI95 = 2.5 / 97.5
percentiles of the resampled delta. References for deltas: ``ref_cap``, and the
same statistic of the ``--baseline`` variant.

Strata: observed depth 0-2 / 2-5 / 5-10 / 10-30 / 30+ m; donor density = horizontal
distance from the site to its 32nd nearest donor (fixed k, independent of the
variant), < 1 / 1-3 / 3-10 / > 10 km; horizontal distance to the nearest donor
0-0.3 / 0.3-1 / 1-5 / 5+ km. Shallow precision / recall at < 2 / 5 / 10 m from
predicted DTW = z - R. Support calibration: Spearman rank correlation of |r| of the
blend with the weighted donor-WTE IQR, n_eff, and the variant's k-th donor
horizontal distance (dimensionless, -1..1; positive = larger support value goes
with larger error).

Outputs (``--out-dir``): ``<variant>/sites_<set>.parquet`` + ``<variant>/variant.json``,
``panel_core.csv``, ``panel_depth.csv``, ``panel_density.csv``, ``panel_dist.csv``,
``shallow_pr.csv``, ``deltas.csv``, ``support.csv``, ``site_diag.csv``,
``sweep_run.json``, ``report.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pumping_r_core import (  # noqa: E402
    DEM,
    HAND,
    DonorIndex,
    Variant,
    compute,
    exclude_donors,
    load_donors,
)

log = logging.getLogger("pumping_r")

DEPTH_EDGES = [0, 2, 5, 10, 30, np.inf]
DEPTH_LABELS = ["0-2", "2-5", "5-10", "10-30", "30+"]
DENS_EDGES_KM = [0, 1, 3, 10, np.inf]
DENS_LABELS = ["<1km", "1-3km", "3-10km", ">10km"]
DIST_EDGES_KM = [0, 0.3, 1, 5, np.inf]
DIST_LABELS = ["0-0.3km", "0.3-1km", "1-5km", "5+km"]
SHALLOW_M = (2.0, 5.0, 10.0)
COMPONENTS = ("blend", "median", "q75")


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="+", required=True, help="NAME=path.parquet")
    ap.add_argument("--variants", required=True, help="JSON list of Variant dicts")
    ap.add_argument("--baseline", required=True, help="variant name for stage deltas")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hand", default=HAND)
    ap.add_argument("--dem", default=DEM)
    ap.add_argument("--pool-classes", default="pumping,unknown")
    ap.add_argument("--exclude-points", nargs="*", default=[])
    ap.add_argument("--exclude-m", type=float, default=10.0)
    ap.add_argument("--exclude-same-source", nargs="*", default=[])
    ap.add_argument("--exclude-same-source-m", type=float, default=0.0)
    ap.add_argument(
        "--exclude-ids", nargs="*", default=[], help="parquet files with canonical_id"
    )
    ap.add_argument("--donor-buffer-km", type=float, default=200.0)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def panel(r: np.ndarray) -> dict:
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


class BlockBoot:
    """One set of paired block resamples per site set, reused for every delta."""

    def __init__(self, blocks: np.ndarray, n: int, seed: int):
        ub, inv = np.unique(blocks, return_inverse=True)
        groups = [np.flatnonzero(inv == i) for i in range(len(ub))]
        rng = np.random.RandomState(seed)
        self.n_blocks = len(ub)
        self.idx = [
            np.concatenate([groups[i] for i in rng.randint(0, len(ub), len(ub))])
            for _ in range(n)
        ]

    def delta(self, cand: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> dict:
        ok = mask & np.isfinite(cand) & np.isfinite(ref)
        ac, ar = np.abs(cand), np.abs(ref)
        sc, sr = cand**2, ref**2
        dm, dr = [], []
        for ix in self.idx:
            ix = ix[ok[ix]]
            if ix.size == 0:
                continue
            dm.append(np.median(ar[ix]) - np.median(ac[ix]))
            dr.append(np.sqrt(sr[ix].mean()) - np.sqrt(sc[ix].mean()))
        dm, dr = np.asarray(dm), np.asarray(dr)
        return {
            "n": int(ok.sum()),
            "d_mad_m": float(np.median(ar[ok]) - np.median(ac[ok]))
            if ok.any()
            else np.nan,
            "d_mad_lo": float(np.percentile(dm, 2.5)) if dm.size else np.nan,
            "d_mad_hi": float(np.percentile(dm, 97.5)) if dm.size else np.nan,
            "d_rmse_m": float(np.sqrt(sr[ok].mean()) - np.sqrt(sc[ok].mean()))
            if ok.any()
            else np.nan,
            "d_rmse_lo": float(np.percentile(dr, 2.5)) if dr.size else np.nan,
            "d_rmse_hi": float(np.percentile(dr, 97.5)) if dr.size else np.nan,
        }


def bands(values: np.ndarray, edges, labels) -> list[tuple[str, np.ndarray]]:
    return [
        (lab, (values >= lo) & (values < hi))
        for lab, lo, hi in zip(labels, edges[:-1], edges[1:])
    ]


def shallow_pr(obs: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for t in SHALLOW_M:
        po, oo = pred < t, obs < t
        tp = int((po & oo).sum())
        out[f"prec_lt{t:g}"] = tp / po.sum() if po.sum() else np.nan
        out[f"rec_lt{t:g}"] = tp / oo.sum() if oo.sum() else np.nan
        out[f"npred_lt{t:g}"] = int(po.sum())
    return out


def fmt_ci(v, lo, hi) -> str:
    return f"{v:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse()
    t0 = time.time()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    variants = [Variant.from_dict(d) for d in json.load(open(args.variants))]
    names = [v.name for v in variants]
    if len(set(names)) != len(names):
        raise SystemExit("variant names must be unique")
    if args.baseline not in names:
        raise SystemExit(f"baseline {args.baseline} not among variants {names}")

    sites = {}
    for spec in args.sites:
        nm, path = spec.split("=", 1)
        sites[nm] = pd.read_parquet(path)
        log.info("sites %s: %d (%s)", nm, len(sites[nm]), path)

    donors = load_donors(args.hand, args.dem, args.pool_classes.split(","))
    allxy = np.vstack(
        [s[["x5070", "y5070"]].to_numpy("float64") for s in sites.values()]
    )
    buf = args.donor_buffer_km * 1000.0
    lo, hi = allxy.min(0) - buf, allxy.max(0) + buf
    near = donors["x5070"].between(lo[0], hi[0]) & donors["y5070"].between(lo[1], hi[1])
    donors = donors[near].reset_index(drop=True)
    log.info(
        "donors within %.0f km of the site extent: %d",
        args.donor_buffer_km,
        len(donors),
    )
    donors, excl = exclude_donors(
        donors,
        args.exclude_points,
        args.exclude_m,
        args.exclude_same_source,
        args.exclude_same_source_m,
        args.exclude_ids,
    )

    indices: dict[float, DonorIndex] = {}

    def index_for(vw: float) -> DonorIndex:
        if vw not in indices:
            indices[vw] = DonorIndex(donors, vw)
        return indices[vw]

    fixed = {}
    for nm, s in sites.items():
        ix = index_for(variants[0].vw)
        qxy = s[["x5070", "y5070"]].to_numpy("float64")
        d1, _ = ix.h_tree.query(qxy, k=1, workers=-1)
        fixed[nm] = {
            "dens_km": ix.density_km(qxy, 32),
            "near_km": d1 / 1000.0,
            "boot": BlockBoot(s["block"].astype(str).to_numpy(), args.boot, args.seed),
        }
        log.info(
            "%s: sites with a donor within 10 m: %d; nearest donor p50 %.2f km; 32nd donor p50 %.2f km; %d blocks",
            nm,
            int((d1 < 10).sum()),
            float(np.median(d1) / 1000),
            float(np.median(fixed[nm]["dens_km"])),
            fixed[nm]["boot"].n_blocks,
        )

    resid: dict[str, dict[str, np.ndarray]] = {nm: {} for nm in sites}
    preds: dict[str, dict[str, np.ndarray]] = {nm: {} for nm in sites}
    support_rows, diag_rows = [], []
    for nm, s in sites.items():
        z = s["z_m"].to_numpy("float64")
        obs_wte = z - s["obs_dtw_m"].to_numpy("float64")
        for ref in ("ref", "ref_cap"):
            col = "r_ref_m" if ref == "ref" else "r_ref_cap_m"
            r = s[col].to_numpy("float64")
            resid[nm][ref] = r - obs_wte
            preds[nm][ref] = z - r
        if "ma_dtw_m" in s and np.isfinite(s["ma_dtw_m"]).any():
            ma = s["ma_dtw_m"].to_numpy("float64")
            resid[nm]["ma"] = s["obs_dtw_m"].to_numpy("float64") - ma
            preds[nm]["ma"] = ma

    for v in variants:
        vdir = out / v.name
        vdir.mkdir(exist_ok=True)
        (vdir / "variant.json").write_text(v.to_json())
        for nm, s in sites.items():
            tv = time.time()
            qxy = s[["x5070", "y5070"]].to_numpy("float64")
            z = s["z_m"].to_numpy("float64")
            res = compute(index_for(v.vw), qxy, z, v)
            obs_wte = z - s["obs_dtw_m"].to_numpy("float64")
            comp = {
                "blend": res["r_blend"],
                "median": np.minimum(res["r_median"], z - v.cap_m),
                "q75": np.minimum(res["r_q75"], z - v.cap_m),
            }
            for c, r in comp.items():
                resid[nm][f"{v.name}:{c}"] = r - obs_wte
                preds[nm][f"{v.name}:{c}"] = z - r
            df = s[["site_id", "x5070", "y5070", "z_m", "obs_dtw_m", "block"]].copy()
            for k, a in res.items():
                df[k] = a
            df["dens32_km"] = fixed[nm]["dens_km"]
            df["resid_blend_m"] = resid[nm][f"{v.name}:blend"]
            df.to_parquet(vdir / f"sites_{nm}.parquet", index=False)

            ab = np.abs(resid[nm][f"{v.name}:blend"])
            row = {"site_set": nm, "variant": v.name}
            for lab, a in (
                ("iqr_m", res["iqr_m"]),
                ("n_eff", res["n_eff"]),
                ("dk_h_km", res["dk_h_km"]),
                ("dens32_km", fixed[nm]["dens_km"]),
            ):
                ok = np.isfinite(a) & np.isfinite(ab)
                row[f"rho_{lab}"] = float(spearmanr(a[ok], ab[ok]).statistic)
            q = np.nanpercentile(res["iqr_m"], [25, 50, 75])
            qb = np.digitize(res["iqr_m"], q)
            for i in range(4):
                row[f"mad_iqr_q{i + 1}_m"] = float(np.nanmedian(ab[qb == i]))
            support_rows.append(row)
            diag_rows.append(
                {
                    "site_set": nm,
                    "variant": v.name,
                    "frac_cap_bound": float(np.mean(res["cap_bound"])),
                    "frac_w_gt0": float(np.mean(res["w_blend"] > 0)),
                    "frac_w_eq1": float(np.mean(res["w_blend"] >= 1)),
                    "top_share_max": float(np.nanmax(res["top_share"])),
                    "n_eff_p10": float(np.nanpercentile(res["n_eff"], 10)),
                    "iqr_p50_m": float(np.nanmedian(res["iqr_m"])),
                    "dk_h_km_p50": float(np.nanmedian(res["dk_h_km"])),
                    "dk_h_km_p90": float(np.nanpercentile(res["dk_h_km"], 90)),
                    "n_screened_mean": float(np.nanmean(res["n_screened"])),
                    "frac_nan": float(np.mean(~np.isfinite(res["r_blend"]))),
                }
            )
            log.info("variant %s on %s: %.1f s", v.name, nm, time.time() - tv)

    core, depth, dens, dist, spr, deltas = [], [], [], [], [], []
    for nm, s in sites.items():
        obs = s["obs_dtw_m"].to_numpy("float64")
        fx = fixed[nm]
        boot = fx["boot"]
        all_mask = np.ones(len(s), bool)
        dbands = bands(obs, DEPTH_EDGES, DEPTH_LABELS)
        for c, r in resid[nm].items():
            core.append({"site_set": nm, "candidate": c, **panel(r)})
            for lab, m in dbands:
                depth.append(
                    {
                        "site_set": nm,
                        "obs_depth_band_m": lab,
                        "candidate": c,
                        **panel(r[m]),
                    }
                )
            for lab, m in bands(fx["dens_km"], DENS_EDGES_KM, DENS_LABELS):
                dens.append(
                    {"site_set": nm, "density_band": lab, "candidate": c, **panel(r[m])}
                )
            for lab, m in bands(fx["near_km"], DIST_EDGES_KM, DIST_LABELS):
                dist.append(
                    {
                        "site_set": nm,
                        "nearest_donor_band": lab,
                        "candidate": c,
                        **panel(r[m]),
                    }
                )
            spr.append(
                {"site_set": nm, "candidate": c, **shallow_pr(obs, preds[nm][c])}
            )
        for v in variants:
            for comp in COMPONENTS:
                c = f"{v.name}:{comp}"
                refs = [("ref_cap", resid[nm]["ref_cap"])]
                if v.name != args.baseline:
                    refs.append(
                        (
                            f"{args.baseline}:{comp}",
                            resid[nm][f"{args.baseline}:{comp}"],
                        )
                    )
                for ref_name, rr in refs:
                    for lab, m in [("all", all_mask)] + dbands:
                        deltas.append(
                            {
                                "site_set": nm,
                                "candidate": c,
                                "reference": ref_name,
                                "obs_depth_band_m": lab,
                                **boot.delta(resid[nm][c], rr, m),
                            }
                        )
        log.info("scored %s (%.0f s elapsed)", nm, time.time() - t0)

    tabs = {
        "panel_core": pd.DataFrame(core),
        "panel_depth": pd.DataFrame(depth),
        "panel_density": pd.DataFrame(dens),
        "panel_dist": pd.DataFrame(dist),
        "shallow_pr": pd.DataFrame(spr),
        "deltas": pd.DataFrame(deltas),
        "support": pd.DataFrame(support_rows),
        "site_diag": pd.DataFrame(diag_rows),
    }
    for k, df in tabs.items():
        df.to_csv(out / f"{k}.csv", index=False)

    run = {
        "variants": [json.loads(v.to_json()) for v in variants],
        "baseline": args.baseline,
        "sites": args.sites,
        "pool_classes": args.pool_classes,
        "exclusion": {**excl, "files": args.exclude_points},
        "donors_used": int(len(donors)),
        "boot": args.boot,
        "blocks": {nm: fixed[nm]["boot"].n_blocks for nm in sites},
        "elapsed_s": round(time.time() - t0, 1),
    }
    json.dump(run, open(out / "sweep_run.json", "w"), indent=2)
    write_report(out / "report.md", tabs, run, list(sites))
    log.info("wrote %s (%.0f s)", out, time.time() - t0)


def write_report(path: Path, tabs: dict, run: dict, site_sets: list[str]) -> None:
    core, depth, dens = tabs["panel_core"], tabs["panel_depth"], tabs["panel_density"]
    spr, deltas, sup, diag = (
        tabs["shallow_pr"],
        tabs["deltas"],
        tabs["support"],
        tabs["site_diag"],
    )
    ex = run["exclusion"]
    with open(path, "w") as f:
        f.write("# Pumping-well R sweep\n\n")
        f.write(
            f"Donors: HAND wells, well_class in {run['pool_classes']}, {run['donors_used']:,} used; "
            f"{ex['dropped_radius']:,} dropped within {ex['radius_m']:.0f} m of a funnel point, "
            f"{ex['dropped_same_source']:,} more of source {ex['same_source']} within {ex['same_source_m']:.0f} m, "
            f"{ex['dropped_ids']:,} by id list {ex['id_files']}. "
            f"Baseline variant `{run['baseline']}`. Blocks (HUC8): {run['blocks']}; {run['boot']} resamples.\n\n"
            "Residual r = R - observed WTE, m (positive = DTW too shallow). Delta = reference minus "
            "candidate, m; positive = candidate better; CI95 from the paired HUC8-block bootstrap. "
            "`:blend`, `:median`, `:q75` are capped at the variant's cap_m; `ref_cap` is the crossfit R "
            "under its 2 m cap; `ma` is the Ma DTW map (context).\n\n"
        )
        for nm in site_sets:
            f.write(f"## Site set `{nm}`\n\n### Core panel (m)\n\n")
            f.write(
                core[core.site_set == nm]
                .drop(columns="site_set")
                .to_markdown(index=False, floatfmt=".2f")
                + "\n\n"
            )
            for title, tab, col, val in (
                ("MAD (m) by observed depth band", depth, "obs_depth_band_m", "mad_m"),
                (
                    "Median residual (m) by observed depth band",
                    depth,
                    "obs_depth_band_m",
                    "median_r_m",
                ),
                (
                    "RMSE (m) by observed depth band",
                    depth,
                    "obs_depth_band_m",
                    "rmse_m",
                ),
                (
                    "MAD (m) by donor density (distance to 32nd donor)",
                    dens,
                    "density_band",
                    "mad_m",
                ),
            ):
                t = tab[tab.site_set == nm]
                order = DEPTH_LABELS if col == "obs_depth_band_m" else DENS_LABELS
                piv = t.pivot(index="candidate", columns=col, values=val)[order]
                piv = piv.reindex(t.candidate.drop_duplicates())
                if col == "density_band":
                    nrow = (
                        t[t.candidate == "ref_cap"].set_index(col)["n"].reindex(order)
                    )
                    f.write(f"### {title}\n\nn per band: {dict(nrow)}\n\n")
                else:
                    f.write(f"### {title}\n\n")
                f.write(piv.to_markdown(floatfmt=".2f") + "\n\n")
            t = spr[spr.site_set == nm].drop(columns="site_set")
            f.write("### Shallow precision / recall (predicted DTW = z - R)\n\n")
            f.write(t.to_markdown(index=False, floatfmt=".2f") + "\n\n")
            d = deltas[deltas.site_set == nm].copy()
            d["d_mad"] = [
                fmt_ci(a, b, c) for a, b, c in zip(d.d_mad_m, d.d_mad_lo, d.d_mad_hi)
            ]
            d["d_rmse"] = [
                fmt_ci(a, b, c) for a, b, c in zip(d.d_rmse_m, d.d_rmse_lo, d.d_rmse_hi)
            ]
            for ref_kind, title in (
                ("ref_cap", "vs ref_cap"),
                ("baseline", f"vs baseline `{run['baseline']}`"),
            ):
                sel = (
                    d[d.reference == "ref_cap"]
                    if ref_kind == "ref_cap"
                    else d[d.reference != "ref_cap"]
                )
                if sel.empty:
                    continue
                piv = sel.pivot(
                    index="candidate", columns="obs_depth_band_m", values="d_mad"
                )[["all"] + DEPTH_LABELS]
                piv = piv.reindex(sel.candidate.drop_duplicates())
                f.write(
                    f"### MAD delta {title} (m, CI95)\n\n" + piv.to_markdown() + "\n\n"
                )
                piv = sel[sel.obs_depth_band_m == "all"].set_index("candidate")[
                    ["d_rmse"]
                ]
                f.write(
                    f"### RMSE delta {title}, all sites (m, CI95)\n\n"
                    + piv.to_markdown()
                    + "\n\n"
                )
            f.write(
                "### Support calibration (blend; Spearman of |r| with the support value)\n\n"
            )
            f.write(
                sup[sup.site_set == nm]
                .drop(columns="site_set")
                .to_markdown(index=False, floatfmt=".2f")
                + "\n\n"
            )
            f.write("### Site diagnostics\n\n")
            f.write(
                diag[diag.site_set == nm]
                .drop(columns="site_set")
                .to_markdown(index=False, floatfmt=".3f")
                + "\n\n"
            )


if __name__ == "__main__":
    main()
