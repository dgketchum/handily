"""Rewrite a GNN training bundle so its level-0 inputs match the render path.

Round-3 "parity" bundle (notes/NV_FIX_ROUND_REVIEW_2026-10-07.md section 5).
The shipped ndwrlbl bundle built three level-0 inputs differently from the way
the renderer builds them on the lattice, so the checkpoints learned one frame
and are applied in another. This script copies a bundle (symlinks for every
file it does not change), rewrites ``query_nodes.parquet`` and edits
``graph_manifest.json``:

1. **R at appended labels in the lattice frame.** Appended rows (e.g. NV NDWR)
   got a fold-inherited crossfit R; the renderer interpolates the ARCHIVED
   crossfit R of the base real wells (relief-kNN-IDW, no exclusion). Recompute
   R and the deep datum for the appended rows exactly as ``infer_conus_gnn``
   does on the lattice (``idw_exclude_at_points`` with exclude 0 over the base
   pool).
1b. **R replacement** (``--r-replace``): R := a per-row table (query_node_idx,
   r_wte_m), e.g. the pumping-well R of utils/build_pumping_r_query_nodes.py, on
   every row; the deep datum is kept. Applied before the cap.
2. **Ground cap.** R := min(R, z_surf - cap) on EVERY row (water pseudo-rows
   included; this bundle has no water-v3 pins). Mirrors ``--r-ground-cap-m``.
3. **FAC recipe.** ``fac_rem_dtw_m`` resampled from an explicit FAC-REM raster
   list (``--fac-recipe`` [+ ``--fac-fallback-recipe``] discovered on disk, or
   ``--fac-manifest``); the list is written to the out bundle as
   ``fac_rem_manifest.json`` and the trainer carries it into the arm. Rows
   outside ``--fac-basins`` are expected unchanged and reported.
4. **Drilled-depth floor.** ``drilled_depth_idw_m``/``_p90_m`` rebuilt with the
   same 100 m self-exclusion guard plus ``--dd-dist-floor-m`` (render passes the
   same floor). The 1 m construction is first re-run and must reproduce the
   bundle columns exactly, or the script stops.
5. Dependent columns recomputed: wte_residual_m, wte_resid_base_m,
   fac_rem_wte_anom_m, deep_regional_wte_anom_m.
6. ``--drop-feature`` removes columns from ``query_feature_cols`` (the column
   stays in the parquet).

Rows are never dropped: lateral/source edges index ``query_node_idx``.
Each step is independently switchable so ablation bundles share every other
byte with the full one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-bundle", required=True)
    ap.add_argument("--out-bundle", required=True)
    ap.add_argument(
        "--appended-source",
        action="append",
        default=None,
        help="source name(s) whose R/deep are recomputed in the lattice frame "
        "(default: manifest appended_wells.source_name); pass 'none' to skip",
    )
    ap.add_argument(
        "--frozen-r",
        default=None,
        help="optional parquet with x5070/y5070/r_wte_m (render point-path payload) "
        "to parity-check the recomputed lattice R against",
    )
    ap.add_argument(
        "--r-replace",
        default=None,
        help="parquet (query_node_idx, r_wte_m): replace R on every row before the cap",
    )
    ap.add_argument("--r-ground-cap-m", type=float, default=None)
    ap.add_argument(
        "--fac-recipe",
        default=None,
        help="resample fac_rem_dtw_m from the rasters of this recipe (discovered under "
        "huc8/); writes fac_rem_manifest.json into the out bundle",
    )
    ap.add_argument(
        "--fac-fallback-recipe",
        default=None,
        help="with --fac-recipe: basins lacking it are listed on this recipe instead",
    )
    ap.add_argument(
        "--fac-manifest",
        default=None,
        help="resample fac_rem_dtw_m from this existing manifest instead of discovering",
    )
    ap.add_argument(
        "--fac-basins",
        default=None,
        help="text file of HUC8s whose FAC is expected to change; changes outside are "
        "reported per basin (report only)",
    )
    ap.add_argument("--dd-dist-floor-m", type=float, default=None)
    ap.add_argument("--drop-feature", action="append", default=[])
    ap.add_argument("--dry-run", action="store_true", help="measure, write nothing")
    return ap.parse_args()


def _stats(d: np.ndarray) -> str:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return "n=0"
    a = np.abs(d)
    return (
        f"n={len(d)} p50={np.median(d):+.3f} p90|.|={np.percentile(a, 90):.3f} "
        f"max|.|={a.max():.3f} m"
    )


def main() -> None:
    args = parse()
    from build_conus_graph_inputs import _relief_coords, sample_drilled_depth
    from fac_rem_registry import (
        MANIFEST_NAME,
        count_by_recipe,
        discover,
        load_manifest,
        sample_fac_rem,
        select_paths,
        write_manifest,
    )
    from infer_conus_gnn import cap_r_ground, idw_exclude_at_points, well_pool

    if args.fac_recipe and args.fac_manifest:
        raise SystemExit("pass --fac-recipe or --fac-manifest, not both")
    resample_fac = bool(args.fac_recipe or args.fac_manifest)

    base = Path(args.base_bundle)
    out = Path(args.out_bundle)
    man = json.loads((base / "graph_manifest.json").read_text())
    qn = pd.read_parquet(base / "query_nodes.parquet")
    if not (qn["query_node_idx"].to_numpy() == np.arange(len(qn))).all():
        raise SystemExit("query_node_idx not contiguous; edges would break")
    n = len(qn)
    z = qn["z_surf_well_m"].to_numpy("float64")
    xy = qn[["x5070", "y5070"]].to_numpy("float64")
    r = qn["regional_wte_idw_oof_m"].to_numpy("float64").copy()
    deep = qn["deep_regional_wte_idw_oof_m"].to_numpy("float64").copy()
    fac = qn["fac_rem_dtw_m"].to_numpy("float64").copy()
    rewrite: dict = {"base_bundle": str(base), "n_rows": n}

    # --- 1. lattice-frame R/deep at appended rows --------------------------------
    app = args.appended_source
    if app is None:
        aw = man.get("appended_wells")
        app = [aw["source_name"]] if aw else []
    if app == ["none"]:
        app = []
    if app:
        is_app = qn["source"].isin(app).to_numpy(bool)
        pool_df = well_pool(qn)
        pool_df = pool_df[~pool_df["source"].isin(app)]
        k, p, vw = int(man["idw_k"]), float(man["idw_power"]), float(man["r_relief_vw"])
        pxy = pool_df[["x5070", "y5070"]].to_numpy("float64")
        pz = pool_df["z_surf_well_m"].to_numpy("float64")
        pool_co = _relief_coords(pxy, pz, vw)
        q_co = _relief_coords(xy[is_app], z[is_app], vw)
        r_lat = idw_exclude_at_points(
            pool_co,
            pool_df["regional_wte_idw_oof_m"].to_numpy("float64"),
            pxy,
            q_co,
            xy[is_app],
            k,
            p,
            0.0,
        )
        d_lat = idw_exclude_at_points(
            pool_co,
            pool_df["deep_regional_wte_idw_oof_m"].to_numpy("float64"),
            pxy,
            q_co,
            xy[is_app],
            k,
            p,
            0.0,
        )
        if not (np.isfinite(r_lat).all() and np.isfinite(d_lat).all()):
            raise SystemExit("lattice R/deep non-finite at appended rows")
        print(
            f"[1] appended rows {int(is_app.sum())} (sources {app}), pool {len(pool_df)}"
        )
        print(f"    lattice R - bundle R:    {_stats(r_lat - r[is_app])}")
        print(f"    lattice deep - bundle:   {_stats(d_lat - deep[is_app])}")
        rewrite["appended_r"] = {
            "sources": app,
            "n": int(is_app.sum()),
            "pool_n": int(len(pool_df)),
            "frame": "lattice: relief-kNN-IDW of archived base-well crossfit R/deep, exclude 0",
            "k": k,
            "power": p,
            "relief_vw": vw,
            "delta_vs_bundle_R_p50_m": float(np.median(r_lat - r[is_app])),
            "delta_vs_bundle_R_p90abs_m": float(
                np.percentile(np.abs(r_lat - r[is_app]), 90)
            ),
        }
        if args.frozen_r:
            fz = pd.read_parquet(args.frozen_r, columns=["x5070", "y5070", "r_wte_m"])
            key = pd.DataFrame(
                {"x5070": xy[is_app, 0], "y5070": xy[is_app, 1], "r_lat": r_lat}
            )
            m = key.merge(fz, on=["x5070", "y5070"], how="left")
            d = m["r_lat"].to_numpy() - m["r_wte_m"].to_numpy()
            print(
                f"    lattice R - frozen r_wte_m ({int(np.isfinite(d).sum())} matched): {_stats(d)}"
            )
            rewrite["appended_r"]["frozen_check"] = {
                "path": args.frozen_r,
                "n_matched": int(np.isfinite(d).sum()),
                "max_abs_m": float(np.nanmax(np.abs(d))),
            }
        r[is_app] = r_lat
        deep[is_app] = d_lat

    # --- 1b. R replacement ---------------------------------------------------------
    if args.r_replace:
        if app:
            raise SystemExit(
                "--r-replace with appended-source R: pass --appended-source none"
            )
        rr = pd.read_parquet(args.r_replace, columns=["query_node_idx", "r_wte_m"])
        rr = rr.set_index("query_node_idx")["r_wte_m"].reindex(np.arange(n))
        r_new = rr.to_numpy("float64")
        if not np.isfinite(r_new).all():
            raise SystemExit(
                f"--r-replace covers {int(np.isfinite(r_new).sum())}/{n} rows"
            )
        print(f"[1b] R replaced from {args.r_replace}: new - old {_stats(r_new - r)}")
        rewrite["r_replace"] = {
            "path": args.r_replace,
            "delta_vs_bundle_R_p50_m": float(np.median(r_new - r)),
            "delta_vs_bundle_R_p90abs_m": float(np.percentile(np.abs(r_new - r), 90)),
        }
        r = r_new

    # --- 2. ground cap -------------------------------------------------------------
    if args.r_ground_cap_m is not None:
        r_pre = r.copy()
        r, bound = cap_r_ground(r, z, float(args.r_ground_cap_m))
        by = (
            qn.assign(_b=bound, _p=qn["is_water_pseudo"].astype(bool))
            .groupby(
                np.where(
                    qn["is_water_pseudo"].astype(bool),
                    "water_pseudo",
                    qn["source"].where(qn["source"].isin(app), "monitoring"),
                )
            )["_b"]
            .mean()
        )
        print(
            f"[2] cap R <= z - {args.r_ground_cap_m} m: bound {int(bound.sum())}/{n} ({100 * bound.mean():.1f}%), median lowering {np.median((r_pre - r)[bound]) if bound.any() else 0:.2f} m"
        )
        print(
            "    bound share by group: "
            + ", ".join(f"{g} {v:.3f}" for g, v in by.items())
        )
        rewrite["r_ground_cap"] = {
            "cap_m": float(args.r_ground_cap_m),
            "n_bound": int(bound.sum()),
            "pct_bound": float(100 * bound.mean()),
            "median_lowering_m": float(np.median((r_pre - r)[bound]))
            if bound.any()
            else 0.0,
            "bound_share_by_group": {str(g): float(v) for g, v in by.items()},
            "note": "deep datum left uncapped (render caps r_wte only); water pseudo-rows capped (bundle has no water-v3 pins)",
        }

    # --- 3. FAC recipe ---------------------------------------------------------------
    if resample_fac:
        if args.fac_manifest:
            fac_doc = load_manifest(args.fac_manifest)
            fac_entries, fac_recipe = fac_doc["entries"], fac_doc["recipe"]
            fac_fallback = fac_doc.get("fallback_recipe")
        else:
            fac_recipe, fac_fallback = args.fac_recipe, args.fac_fallback_recipe
            fac_entries = discover(fac_recipe, fallback=fac_fallback)
        fac_reg = select_paths(fac_entries, strict=True)
        print(
            f"[3] FAC-REM manifest: {len(fac_reg)} rasters {count_by_recipe(fac_entries)}"
        )
        fac_new = sample_fac_rem(xy[:, 0], xy[:, 1], fac_reg)
        same = (
            np.isfinite(fac_new)
            & np.isfinite(fac)
            & np.isclose(fac_new, fac, atol=1e-6)
        ) | (~np.isfinite(fac_new) & ~np.isfinite(fac))
        if args.fac_basins:
            allowed = set(Path(args.fac_basins).read_text().split())
            inside = qn["huc8"].astype(str).isin(allowed).to_numpy(bool)
        else:
            inside = np.ones(n, bool)
        n_out_changed = int((~same & ~inside).sum())
        print(
            f"[3] FAC recipe {fac_recipe}: {int((~same).sum())} rows changed, {n_out_changed} of them OUTSIDE the expected basins"
        )
        if n_out_changed:
            # a rebuilt basin's raster can overlap a neighbouring basin's edge, and
            # dropping the legacy pilots re-sources their footprint; the renderer
            # samples the same manifest, so this is parity -- reported, not fatal
            bad = qn.loc[~same & ~inside, "huc8"].astype(str).value_counts()
            print(
                "    outside-basin changes by huc8: "
                + ", ".join(f"{h} {c}" for h, c in bad.items())
            )
        new_nan = ~np.isfinite(fac_new) & np.isfinite(fac)
        new_fin = np.isfinite(fac_new) & ~np.isfinite(fac)
        d = fac_new - fac
        print(
            f"    inside: changed {int((~same & inside).sum())}/{int(inside.sum())}, newly NaN {int(new_nan.sum())}, newly finite {int(new_fin.sum())}; delta {_stats(d[inside])}"
        )
        for grp in ("monitoring", "appended", "water_pseudo"):
            if grp == "water_pseudo":
                g = qn["is_water_pseudo"].astype(bool).to_numpy()
            elif grp == "appended":
                g = qn["source"].isin(app).to_numpy(bool)
            else:
                g = ~qn["is_water_pseudo"].astype(bool).to_numpy() & ~qn["source"].isin(
                    app
                ).to_numpy(bool)
            g &= inside
            print(
                f"      {grp:13s} n={int(g.sum())} newly NaN {int((new_nan & g).sum())} delta {_stats(d[g])}"
            )
        rewrite["fac"] = {
            "recipe": fac_recipe,
            "fallback_recipe": fac_fallback,
            "n_rasters": len(fac_reg),
            "n_by_recipe": count_by_recipe(fac_entries),
            "n_changed": int((~same).sum()),
            "n_newly_nan": int(new_nan.sum()),
            "n_newly_finite": int(new_fin.sum()),
            "n_changed_outside_basins": n_out_changed,
            "delta_p50_m": float(np.nanmedian(d[inside]))
            if np.isfinite(d[inside]).any()
            else None,
            "basins_file": args.fac_basins,
        }
        fac = fac_new

    # --- 4. drilled depth floor ----------------------------------------------------
    if args.dd_dist_floor_m is not None:
        dd = man["drilled_depth"]
        ids = qn["canonical_id"].to_numpy()
        chk = sample_drilled_depth(
            dd["points"],
            xy,
            int(dd["k"]),
            float(dd["power"]),
            query_ids=ids,
            self_exclude_m=float(dd["self_exclude_m"]),
            dist_floor_m=1.0,
        )
        for c in dd["feature_cols"]:
            dc = chk[c] - qn[c].to_numpy("float64")
            print(f"[4] dd parity (floor 1 m) {c}: {_stats(dc)}")
            if np.nanmax(np.abs(dc)) > 1e-6:
                raise SystemExit(f"drilled-depth parity FAILED on {c}")
        new = sample_drilled_depth(
            dd["points"],
            xy,
            int(dd["k"]),
            float(dd["power"]),
            query_ids=ids,
            self_exclude_m=float(dd["self_exclude_m"]),
            dist_floor_m=float(args.dd_dist_floor_m),
        )
        for c in dd["feature_cols"]:
            print(
                f"    floor {args.dd_dist_floor_m:.0f} m - bundle {c}: {_stats(new[c] - qn[c].to_numpy('float64'))}"
            )
            qn[c] = new[c]
        rewrite["drilled_depth"] = {
            "dist_floor_m": float(args.dd_dist_floor_m),
            "self_exclude_m": float(dd["self_exclude_m"]),
            "parity_check": "floor 1 m reproduced bundle columns to 1e-6 m",
        }

    # --- 5. dependent columns --------------------------------------------------------
    wte = qn["wte_obs_m"].to_numpy("float64")
    qn["regional_wte_idw_oof_m"] = r
    qn["deep_regional_wte_idw_oof_m"] = deep
    qn["fac_rem_dtw_m"] = fac
    qn["wte_residual_m"] = wte - r
    qn["wte_resid_base_m"] = z - r
    qn["fac_rem_wte_anom_m"] = (z - fac) - r
    qn["deep_regional_wte_anom_m"] = deep - r
    for c in (
        "regional_wte_idw_oof_m",
        "deep_regional_wte_idw_oof_m",
        "wte_residual_m",
        "wte_resid_base_m",
        "deep_regional_wte_anom_m",
    ):
        if not np.isfinite(qn[c].to_numpy("float64")).all():
            raise SystemExit(f"non-finite {c} after rewrite")
    print(
        f"[5] target wte_residual_m: {_stats(qn['wte_residual_m'].to_numpy())}; "
        f"share R>z-2 now {float((r > z - 2).mean()):.3f}"
    )

    # --- 6. manifest ---------------------------------------------------------------
    cols = list(man["query_feature_cols"])
    for c in args.drop_feature:
        if c not in cols:
            raise SystemExit(f"--drop-feature {c}: not in query_feature_cols")
        cols.remove(c)
    man["query_feature_cols"] = cols
    man["drilled_depth"] = dict(
        man["drilled_depth"],
        **(
            {"dist_floor_m": float(args.dd_dist_floor_m)}
            if args.dd_dist_floor_m is not None
            else {}
        ),
    )
    if args.r_ground_cap_m is not None:
        man["r_ground_cap_m"] = float(args.r_ground_cap_m)
    if resample_fac:
        man["fac_recipe"] = fac_recipe
        man["fac_fallback_recipe"] = fac_fallback
        man["fac_rem_manifest"] = MANIFEST_NAME
        man["fac_rem_n_rasters"] = len(fac_reg)
    elif not (base / MANIFEST_NAME).exists():
        raise SystemExit(
            f"{base.name} carries no {MANIFEST_NAME} and no FAC resample was requested: "
            "pass --fac-recipe/--fac-manifest so the out bundle carries an explicit "
            "FAC-REM raster list"
        )
    rewrite["dropped_features"] = list(args.drop_feature)
    rewrite["render_contract"] = {
        "r_ground_cap_m": args.r_ground_cap_m,
        "r_replace": args.r_replace,
        "dd_dist_floor_m": args.dd_dist_floor_m,
        "fac_rem_manifest": MANIFEST_NAME,
        "exclude_source_name": app,
    }
    man["parity_rewrite"] = rewrite
    print(
        f"[6] query_feature_cols {len(man['query_feature_cols'])} (dropped {args.drop_feature})"
    )

    if args.dry_run:
        print("dry run: nothing written")
        return
    out.mkdir(parents=True, exist_ok=False)
    skip = {"query_nodes.parquet", "graph_manifest.json"}
    if resample_fac:
        skip.add(MANIFEST_NAME)
    for f in sorted(base.iterdir()):
        if f.name in skip:
            continue
        os.symlink(f.resolve(), out / f.name)
    qn.to_parquet(out / "query_nodes.parquet", index=False)
    (out / "graph_manifest.json").write_text(json.dumps(man, indent=2, default=str))
    if resample_fac:
        write_manifest(
            out / MANIFEST_NAME,
            fac_entries,
            fac_recipe,
            note=f"written by build_parity_bundle for {out.name} (base {base.name})",
            fallback_recipe=fac_fallback,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
