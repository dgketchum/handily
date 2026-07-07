"""Unit tests for the Dupuit hang-surface helpers (build_dupuit_wte.py).

These two functions also feed the builder's --dupuit-hang-features query bank,
so their contracts matter beyond the standalone diagnostic:
  * build_boundaries selects the top-2 Strahler orders PER BASIN (each basin's
    own max, not a global cut) and drops reaches with non-finite coords/elev.
  * hang_interp is exact-at-boundary IDW (distance clamp, not division blowup)
    and returns the distance to the NEAREST boundary reach.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_dupuit_wte.py"
_spec = importlib.util.spec_from_file_location("build_dupuit_wte", _MOD)
bd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bd)


def _write_inputs(tmp_path):
    rn = pd.DataFrame(
        {
            "comid": [1, 2, 3, 4, 5, 6],
            "basin": ["a", "a", "a", "b", "b", "b"],
            "streamorde": [1, 2, 3, 5, 6, 6],
            "reach_elev_m": [900.0, 800.0, 700.0, 500.0, 400.0, np.nan],
        }
    )
    geom = pd.DataFrame(
        {
            "comid": [1, 2, 3, 4, 5, 6],
            "cx": [0.0, 1000.0, 2000.0, 50000.0, 60000.0, 70000.0],
            "cy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    rn_path, g_path = tmp_path / "reach_nodes.parquet", tmp_path / "geom.parquet"
    rn.to_parquet(rn_path)
    geom.to_parquet(g_path)
    return str(rn_path), str(g_path)


def test_build_boundaries_per_basin_top2_and_finite_filter(tmp_path):
    rn_path, g_path = _write_inputs(tmp_path)
    b = bd.build_boundaries(rn_path, g_path, top_orders=2)
    # basin a: max order 3 -> orders {2,3} = comids 2,3. basin b: max 6 ->
    # orders {5,6} = comids 4,5,6, but 6 has NaN elev -> dropped.
    assert sorted(b.comid.tolist()) == [2, 3, 4, 5]


def test_hang_interp_exact_at_boundary_and_nearest_distance(tmp_path):
    rn_path, g_path = _write_inputs(tmp_path)
    b = bd.build_boundaries(rn_path, g_path, top_orders=2)
    bxy = b[["cx", "cy"]].to_numpy("float64")
    bhead = b.reach_elev_m.to_numpy("float64")
    # query 1 sits ON comid 2 (clamped dist 1 m dominates IDW); query 2 is
    # midway between comids 2 and 3 with comid 4 far away.
    q = np.array([[1000.0, 0.0], [1500.0, 0.0]])
    h, d = bd.hang_interp(bxy, bhead, q, k=8, power=2.0)
    assert abs(h[0] - 800.0) < 1.0
    assert d[0] == 0.0
    # midpoint: 500 m from heads 800/700 (comid 1 is order-1, NOT a boundary);
    # comids 4/5 at ~50 km carry ~1e-4 of the weight -> ~750
    assert abs(h[1] - 750.0) < 0.5
    assert d[1] == 500.0


def test_hang_interp_k_exceeding_boundary_count(tmp_path):
    rn_path, g_path = _write_inputs(tmp_path)
    b = bd.build_boundaries(rn_path, g_path, top_orders=2)
    # single boundary point = comid 2 at (1000, 0); query 100 m north of it
    h, d = bd.hang_interp(
        b[["cx", "cy"]].to_numpy("float64")[:1],
        b.reach_elev_m.to_numpy("float64")[:1],
        np.array([[1000.0, 100.0]]),
        k=8,
        power=2.0,
    )
    assert np.isfinite(h).all() and d[0] == 100.0
