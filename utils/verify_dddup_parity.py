"""Verify a canonical first-class graph bundle reproduces a derived (symlink-overlay)
bundle bit-for-bit on the drilled-depth / Dupuit-hang query features.

Context (E5, notes/PROMOTION_E5.md): the ladder-winning dd/dup arm was first produced
by an *overlay* bundle (utils/augment_drilled_dupuit.py: symlink the heavy base files +
regenerate query_nodes.parquet with dd/dup columns appended last). Promotion requires a
*canonical* bundle built straight from build_conus_graph_inputs.py --drilled-depth-points
/--dupuit-hang-features. Before trusting the canonical bundle we must prove the two agree
on feature VALUES (column order may differ — the builder places dd/dup before the water
block; the overlay appends them last — that is expected and harmless).

The comparison joins on canonical_id (the stable per-query key: gwx_* for real wells,
water_* for pseudo-labels), so a difference in query_node_idx ORDER does not register as a
value mismatch. Column values are compared with allclose over the finite intersection AND
an exact NaN-pattern match (a feature present in one bundle but absent in the other is a
real mismatch, not a rounding artefact).

Usage:
    uv run python utils/verify_dddup_parity.py \
        --canonical /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_v2 \
        --derived   /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water_dddup \
        --base      /data/ssd2/handily/conus/wte_gnn/graph_conus_monitoring_water

--base is optional; when given, ALL of its query_feature_cols are also checked
canonical-vs-base (this predicts whether a confirmation training arm on the canonical
bundle reproduces the derived-bundle arm within seed noise).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DD_DUP_COLS = [
    "drilled_depth_idw_m",
    "drilled_depth_p90_m",
    "dupuit_hang_dtw_m",
    "log1p_dupuit_d_m",
]


def compare_columns(df_a, df_b, cols, join_key="canonical_id", rtol=1e-9, atol=1e-6):
    """Compare `cols` between two query-node frames, aligned on `join_key`.

    Returns a report dict with, per column: allclose (finite intersection),
    nan_pattern_match (identical finite/NaN masks), max_abs_diff, and an overall
    `passed` flag. A column absent from either frame is reported as a mismatch.
    """
    a = df_a.set_index(join_key)
    b = df_b.set_index(join_key)
    common = a.index.intersection(b.index)
    per_col = {}
    passed = True
    for c in cols:
        if c not in a.columns or c not in b.columns:
            per_col[c] = {
                "present_a": c in a.columns,
                "present_b": c in b.columns,
                "allclose": False,
                "nan_pattern_match": False,
                "max_abs_diff": None,
            }
            passed = False
            continue
        x = a.loc[common, c].to_numpy(dtype=float)
        y = b.loc[common, c].to_numpy(dtype=float)
        fx, fy = np.isfinite(x), np.isfinite(y)
        nan_match = bool(np.array_equal(fx, fy))
        both = fx & fy
        if both.any():
            allclose = bool(np.allclose(x[both], y[both], rtol=rtol, atol=atol))
            max_abs = float(np.max(np.abs(x[both] - y[both])))
        else:
            allclose = True
            max_abs = 0.0
        per_col[c] = {
            "present_a": True,
            "present_b": True,
            "allclose": allclose,
            "nan_pattern_match": nan_match,
            "max_abs_diff": max_abs,
        }
        if not (allclose and nan_match):
            passed = False
    return {"n_common": int(len(common)), "columns": per_col, "passed": passed}


def _load(bundle):
    d = Path(bundle)
    man = json.load(open(d / "graph_manifest.json"))
    q = pd.read_parquet(d / "query_nodes.parquet")
    return man, q


def _print_block(title, rep):
    print(
        f"\n[{title}] n_common={rep['n_common']}  ->  {'PASS' if rep['passed'] else 'FAIL'}"
    )
    for c, r in rep["columns"].items():
        if not r["present_a"] or not r["present_b"]:
            print(f"    {c:26s} MISSING (a={r['present_a']} b={r['present_b']})")
            continue
        flag = "OK" if (r["allclose"] and r["nan_pattern_match"]) else "MISMATCH"
        print(
            f"    {c:26s} allclose={r['allclose']} nan_match={r['nan_pattern_match']} "
            f"max|Δ|={r['max_abs_diff']:.3e}  [{flag}]"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--canonical", required=True, help="first-class rebuild bundle dir")
    ap.add_argument(
        "--derived", required=True, help="symlink-overlay dd/dup bundle dir"
    )
    ap.add_argument("--base", default=None, help="optional base water bundle dir")
    ap.add_argument("--join-key", default="canonical_id")
    args = ap.parse_args(argv)

    man_c, qc = _load(args.canonical)
    man_d, qd = _load(args.derived)
    fail = False

    print("=== manifests ===")
    for name, man in (("canonical", man_c), ("derived", man_d)):
        qfc = man["query_feature_cols"]
        present = [c for c in qfc if c in DD_DUP_COLS]
        print(f"{name:10s} query_feature_cols n={len(qfc)}  dd/dup present: {present}")

    # population + order
    set_c, set_d = set(qc[args.join_key]), set(qd[args.join_key])
    same_set = set_c == set_d
    print(f"\nidentical {args.join_key} set (canonical vs derived): {same_set}")
    if not same_set:
        fail = True
        print(
            f"    only canonical: {len(set_c - set_d)}  only derived: {len(set_d - set_c)}"
        )

    rep_dd = compare_columns(qc, qd, DD_DUP_COLS, args.join_key)
    _print_block("dd/dup: canonical vs derived", rep_dd)
    fail = fail or not rep_dd["passed"]

    if args.base:
        man_b, qb = _load(args.base)
        base_cols = man_b["query_feature_cols"]
        rep_base = compare_columns(qc, qb, base_cols, args.join_key)
        n_ok = sum(
            1
            for r in rep_base["columns"].values()
            if r["allclose"] and r["nan_pattern_match"]
        )
        print(
            f"\n[base query_feature_cols: canonical vs base] {n_ok}/{len(base_cols)} cols "
            f"allclose+nan-match  ->  {'PASS' if rep_base['passed'] else 'FAIL'}"
        )
        for c, r in rep_base["columns"].items():
            if not (r["allclose"] and r["nan_pattern_match"]):
                print(f"    MISMATCH {c:26s} {r}")
        fail = fail or not rep_base["passed"]

    print("\n=== PARITY VERDICT:", "FAIL" if fail else "PASS", "===")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
