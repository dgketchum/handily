"""Unit tests for the WP2 assignment-prior builder (utils/build_two_surface_priors.py).

Covers the four load-bearing mechanics: (a) the local depth-rank prior is monotone
in the neighborhood depth rank; (b) the confined-type hard cap is enforced; (c) an
empty neighborhood falls back to the CONUS-neutral 0.5; (d) the log-odds combination
matches a hand-computed case.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bp = _load("build_two_surface_priors")


# --------------------------------------------------------------------------- #
# (a) rank prior monotone in local depth rank
# --------------------------------------------------------------------------- #
def test_rank_prior_monotone_in_local_depth_rank():
    # 20 co-located GWX wells with completion depths 0..19; a target at that spot.
    pool_comp = np.arange(20, dtype=float)
    pool_xy = np.zeros((20, 2))
    tree = cKDTree(pool_xy)

    qs, ps = [], []
    for t in [0.0, 5.0, 10.0, 19.0]:
        q, n, r = bp.neighborhood_quantile(
            tree, pool_comp, 0.0, 0.0, t, bp.RADII_M, bp.MIN_NEIGHBORS
        )
        assert n == 20 and r == bp.RADII_M[0]
        qs.append(q)
        ps.append(float(bp.p_rank_from_quantile(q)))

    # deeper target -> higher quantile -> lower phreatic prior (strictly monotone)
    assert qs[0] < qs[1] < qs[2] < qs[3]
    assert ps[0] > ps[1] > ps[2] > ps[3]
    # shallowest well is phreatic-leaning, deepest is regional-leaning
    assert ps[0] > 0.5 > ps[3]


# --------------------------------------------------------------------------- #
# (b) confined cap enforced
# --------------------------------------------------------------------------- #
def test_confined_cap_enforced():
    # very shallow local rank (would be strongly phreatic) but confined -> capped
    p_rank = bp.p_rank_from_quantile(0.0)  # -> 0.95
    p = bp.combine_logodds(
        p_rank,
        beta_conf=bp.BETA_CONF["confined"],
        beta_role=bp.BETA_ROLE["monitoring"],
        beta_head=0.0,
        is_confined=True,
        unconf_floor=False,
    )
    assert float(p) <= bp.CONFINED_CAP + 1e-12


# --------------------------------------------------------------------------- #
# (c) empty-neighborhood fallback -> neutral 0.5
# --------------------------------------------------------------------------- #
def test_empty_neighborhood_fallback_neutral():
    # pool is 50 km away: no neighbors within 2 km or 10 km
    pool_comp = np.arange(10, dtype=float)
    pool_xy = np.full((10, 2), 50_000.0)
    tree = cKDTree(pool_xy)
    q, n, r = bp.neighborhood_quantile(
        tree, pool_comp, 0.0, 0.0, 12.0, bp.RADII_M, bp.MIN_NEIGHBORS
    )
    assert math.isnan(q) and math.isnan(r) and n == 0
    # downstream mapping: NaN q -> neutral p_rank 0.5
    p_rank = 0.5 if not np.isfinite(q) else float(bp.p_rank_from_quantile(q))
    assert p_rank == 0.5

    # too-few-neighbors (< MIN_NEIGHBORS) also falls back
    pool_xy2 = np.zeros((3, 2))
    tree2 = cKDTree(pool_xy2)
    q2, n2, r2 = bp.neighborhood_quantile(
        tree2, np.arange(3, dtype=float), 0.0, 0.0, 1.0, bp.RADII_M, bp.MIN_NEIGHBORS
    )
    assert math.isnan(q2) and math.isnan(r2) and n2 == 3


# --------------------------------------------------------------------------- #
# (d) log-odds combination matches a hand-computed case
# --------------------------------------------------------------------------- #
def test_logodds_combination_hand_case():
    # p_rank=0.7 (unconfined, monitoring, no head); floor 0.5 applies but is below.
    p_rank = 0.7
    beta_conf = bp.BETA_CONF["unconfined"]  # +1.0
    beta_role = bp.BETA_ROLE["monitoring"]  # +0.5
    L = math.log(p_rank / (1 - p_rank)) + beta_conf + beta_role  # 0.847298 + 1.5
    expected = 1.0 / (1.0 + math.exp(-L))  # ~0.91279
    expected = min(max(expected, bp.P_LO), bp.P_HI)  # within [0.05,0.95]
    p = float(bp.combine_logodds(p_rank, beta_conf, beta_role, 0.0, False, True))
    assert abs(p - expected) < 1e-9

    # head geometry case: large head_above_screen pushes toward regional.
    h = 50.0
    bh = float(bp.beta_head_from_h(h))  # clip(-(50-5)/15,-2,2) = -2.0
    assert abs(bh - (-2.0)) < 1e-12
    L2 = (
        math.log(0.6 / 0.4) + 0.0 + 0.0 + bh
    )  # unconfined_marginal (0), role unknown (0)
    exp2 = min(max(1.0 / (1.0 + math.exp(-L2)), bp.P_LO), bp.P_HI)
    p2 = float(bp.combine_logodds(0.6, 0.0, 0.0, bh, False, False))
    assert abs(p2 - exp2) < 1e-9

    # confined-marginal floor NOT applied when locally deep (unconf_floor False)
    p_deep = float(bp.combine_logodds(0.2, 1.0, 0.0, 0.0, False, False))
    assert p_deep < 0.5  # allowed below the floor because it's locally deep
