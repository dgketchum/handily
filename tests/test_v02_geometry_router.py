"""Unit tests for the E2 aquifer-geometry builder + two-surface router pure logic.

Covers the unit conversion / thickness clip, bank-grid snapping, contour
interpolation, the HUC4-blocked fold guarantee (no group spans folds), the
oracle-assignment label, the oracle-fraction-recovered arithmetic and the
soft-routed-blend identity.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import v02_build_geometry_layers as gb  # noqa: E402
import v02_route_two_surface_geometry as rt  # noqa: E402


# --------------------------- builder pure logic --------------------------- #
def test_feet_to_m():
    assert gb.feet_to_m(np.array([3.28084, 0.0])) == pytest.approx([1.0, 0.0], abs=1e-6)
    assert np.isnan(gb.feet_to_m(np.array([np.nan]))[0])


def test_thickness_clip_and_nan():
    top = np.array([1000.0, 100.0, 100.0, np.nan])
    bot = np.array([0.0, 200.0, 0.0, 0.0])  # row1: negative -> 0
    thk = gb.thickness_from_top_bot(top, bot, clip_max_m=200.0)
    assert thk[0] == pytest.approx(200.0)  # (1000-0)/3.28084=305 -> clipped 200
    assert thk[1] == 0.0  # negative thickness -> 0
    assert thk[2] == pytest.approx(100.0 / gb.FT_PER_M)
    assert np.isnan(thk[3])  # NaN input propagates


def test_snap_bounds_alignment_and_outward():
    left, bottom, right, top = gb.snap_bounds(
        -2539500.3, 3255001.2, -2537000.7, 3257900.0, res=1000.0
    )
    # aligned to the 1 km grid coincident with the canonical origin
    for v in (left, right):
        assert (v - gb.ORIGIN_X) % 1000.0 == 0.0
    for v in (bottom, top):
        assert (v - gb.ORIGIN_Y) % 1000.0 == 0.0
    # snapped box contains the input box
    assert left <= -2539500.3 and right >= -2537000.7
    assert bottom <= 3255001.2 and top >= 3257900.0


def test_interp_contours_recovers_plane():
    # z = 2x + 3y on scattered points -> linear interp must recover it inside hull
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 10, size=(200, 2))
    z = 2 * pts[:, 0] + 3 * pts[:, 1]
    xs = np.linspace(2, 8, 7)
    ys = np.linspace(8, 2, 7)  # descending
    grid = gb.interp_contours(pts, z, xs, ys)
    gx, gy = np.meshgrid(xs, ys)
    expect = 2 * gx + 3 * gy
    ok = np.isfinite(grid)
    assert ok.any()
    assert np.allclose(grid[ok], expect[ok], atol=1e-6)


# ----------------------------- router pure logic --------------------------- #
def test_assign_group_folds_no_group_spans_fold():
    groups = np.array(["a", "a", "b", "c", "c", "c", "d", "e", "e", "f"])
    folds = rt.assign_group_folds(groups, k=3)
    # every group lands in exactly one fold
    for g in np.unique(groups):
        assert len(np.unique(folds[groups == g])) == 1
    # all folds populated for this input, deterministic
    assert set(np.unique(folds)) <= {0, 1, 2}
    assert np.array_equal(folds, rt.assign_group_folds(groups, k=3))


def test_assign_group_folds_balances():
    groups = np.repeat(np.arange(20), 5)  # 20 equal groups of 5
    folds = rt.assign_group_folds(groups, k=4)
    counts = np.bincount(folds, minlength=4)
    assert counts.min() == counts.max()  # perfectly balanced when groups equal


def test_oracle_phreatic_label():
    p = np.array([1.0, 5.0, 3.0])
    r = np.array([4.0, 1.0, 3.0])
    obs = np.array([0.0, 0.0, 0.0])
    # |p-obs| vs |r-obs|: [1<4 ->1], [5>1 ->0], [3<=3 ->1 (tie to phreatic)]
    assert list(rt.oracle_phreatic_label(p, r, obs)) == [1, 0, 1]


def test_oracle_fraction_recovered():
    # base 10, oracle 4 -> headroom 6; routed 7 -> recovered (10-7)/6 = 0.5
    assert rt.oracle_fraction_recovered(10.0, 7.0, 4.0) == pytest.approx(0.5)
    # no headroom -> None
    assert rt.oracle_fraction_recovered(5.0, 5.0, 5.0) is None
    # worse than baseline -> negative
    assert rt.oracle_fraction_recovered(10.0, 12.0, 4.0) == pytest.approx(-1.0 / 3.0)


def test_routed_soft_identity_and_endpoints():
    p = np.array([0.0, 1.0, 0.25])
    ph = np.array([2.0, 2.0, 4.0])
    rg = np.array([8.0, 8.0, 8.0])
    out = rt.routed_soft(p, ph, rg)
    assert out[0] == pytest.approx(8.0)  # p=0 -> regional
    assert out[1] == pytest.approx(2.0)  # p=1 -> phreatic
    assert out[2] == pytest.approx(0.25 * 4.0 + 0.75 * 8.0)
