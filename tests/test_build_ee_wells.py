"""Unit tests for utils/build_ee_wells.py: the ramp bake and the bundle mode."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import build_ee_wells as B  # noqa: E402


def test_ramp_hex_hits_stops_and_clamps():
    hexes = B.ramp_hex(np.array([0.0, 100.0, -5.0, 500.0]))
    assert hexes[0] == B.DTW_COLORS[0]
    assert hexes[1] == B.DTW_COLORS[-1]
    assert hexes[2] == B.DTW_COLORS[0]  # clamped below
    assert hexes[3] == B.DTW_COLORS[-1]  # clamped above


def _bundle(tmp_path, huc8s, pseudo):
    n = len(huc8s)
    z = np.full(n, 1500.0)
    wte = z - np.arange(n, dtype=float)
    pd.DataFrame(
        {
            "canonical_id": [f"w{i}" for i in range(n)],
            "source": ["nwis"] * n,
            "well_class": ["monitoring"] * (n - 1) + ["ose"],
            "confinement_class": ["unconfined"] * n,
            "huc8": huc8s,
            "x5070": np.full(n, -1_000_000.0),
            "y5070": np.full(n, 1_300_000.0),
            "mean_dtw": z - wte,
            "z_surf_well_m": z,
            "wte_obs_m": wte,
            "is_water_pseudo": pseudo,
        }
    ).to_parquet(tmp_path / "query_nodes.parquet")
    return tmp_path


def test_build_bundle_filters_footprint_and_pseudo(tmp_path):
    bundle = _bundle(
        tmp_path,
        huc8s=["13020201", "13020201", "16060001", "13020201"],
        pseudo=[False, True, False, False],
    )
    out = tmp_path / "wt.csv"
    df = B.build_bundle(bundle, ["13020201"], out, label="t")
    # pseudo-row and the out-of-footprint basin are dropped
    assert sorted(df["canonical_id"]) == ["w0", "w3"]
    assert set(df["role"]) == {"monitoring", "ose"}
    assert df["dtw_m"].tolist() == [0.0, 3.0]
    assert (df["fillcolor"].str.len() == 6).all()
    assert df["longitude"].between(-180, 180).all()
    assert out.exists()


def test_build_bundle_accepts_unpadded_basin_codes(tmp_path):
    bundle = _bundle(tmp_path, huc8s=["01020304", "01020304"], pseudo=[False, False])
    df = B.build_bundle(bundle, [1020304], tmp_path / "wt.csv", label="t")
    assert len(df) == 2
