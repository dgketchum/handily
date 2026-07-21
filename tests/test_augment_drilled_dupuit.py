"""Unit tests for the dd/dup bundle-augment pure logic (build_augmented_manifest):
query_feature_cols extension in builder order, block population, non-mutation of the
base manifest, idempotency, and the drilled-depth self-exclusion leak guard.
Loads utils/augment_drilled_dupuit.py by path (repo test convention).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


aug = _load("augment_drilled_dupuit")


def _base_man():
    return {
        "query_feature_cols": ["fac_rem_wte_anom_m", "slope_deg", "gsw_occ_pct"],
        "drilled_depth": None,
        "dupuit_hang": None,
    }


def test_appends_dd_then_dup_in_order():
    base = _base_man()
    dd = {"feature_cols": ["drilled_depth_idw_m", "drilled_depth_p90_m"], "k": 32}
    dup = {"feature_cols": ["dupuit_hang_dtw_m", "log1p_dupuit_d_m"], "top_orders": 2}
    man = aug.build_augmented_manifest(base, dd, dup)
    assert man["query_feature_cols"] == [
        "fac_rem_wte_anom_m",
        "slope_deg",
        "gsw_occ_pct",
        "drilled_depth_idw_m",
        "drilled_depth_p90_m",
        "dupuit_hang_dtw_m",
        "log1p_dupuit_d_m",
    ]
    assert man["drilled_depth"] == dd
    assert man["dupuit_hang"] == dup
    assert man["dd_dup_augment"]["added_query_feature_cols"] == [
        "drilled_depth_idw_m",
        "drilled_depth_p90_m",
        "dupuit_hang_dtw_m",
        "log1p_dupuit_d_m",
    ]


def test_does_not_mutate_base():
    base = _base_man()
    before = list(base["query_feature_cols"])
    dd = {"feature_cols": ["drilled_depth_idw_m", "drilled_depth_p90_m"]}
    aug.build_augmented_manifest(base, dd, None)
    assert base["query_feature_cols"] == before
    assert base["drilled_depth"] is None


def test_idempotent_no_duplicate_cols():
    base = _base_man()
    dd = {"feature_cols": ["drilled_depth_idw_m", "drilled_depth_p90_m"]}
    once = aug.build_augmented_manifest(base, dd, None)
    twice = aug.build_augmented_manifest(once, dd, None)
    assert twice["query_feature_cols"] == once["query_feature_cols"]


def test_only_requested_block_added():
    base = _base_man()
    dup = {"feature_cols": ["dupuit_hang_dtw_m", "log1p_dupuit_d_m"]}
    man = aug.build_augmented_manifest(base, None, dup)
    assert man["drilled_depth"] is None
    assert man["dupuit_hang"] == dup
    assert "drilled_depth_idw_m" not in man["query_feature_cols"]
    assert man["query_feature_cols"][-2:] == ["dupuit_hang_dtw_m", "log1p_dupuit_d_m"]


def test_self_exclusion_drops_colocated_record(tmp_path):
    """A co-located pool record (<self_exclude_m) must NOT dominate the query's own
    drilled-depth feature -- the leak guard that keeps dd target-blind."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
    from build_conus_graph_inputs import sample_drilled_depth

    # One query at origin; a co-located deep record at 5 m + far shallow records.
    pool = pd.DataFrame(
        {
            "canonical_id": ["self", "a", "b", "c", "d"],
            "x5070": [5.0, 1000.0, 1100.0, 1200.0, 1300.0],
            "y5070": [0.0, 0.0, 0.0, 0.0, 0.0],
            "drilled_depth_m": [500.0, 10.0, 11.0, 12.0, 13.0],
        }
    )
    pp = tmp_path / "pool.parquet"
    pool.to_parquet(pp, index=False)
    qxy = np.array([[0.0, 0.0]])
    out = sample_drilled_depth(
        str(pp),
        qxy,
        k=4,
        power=2.0,
        query_ids=np.array(["query"]),
        self_exclude_m=100.0,
    )
    # The 500 m co-located record is excluded; feature reflects the ~10-13 m neighbours.
    assert out["drilled_depth_idw_m"][0] < 50.0
