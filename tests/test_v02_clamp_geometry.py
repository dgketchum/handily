"""Unit tests for the E2b aquifer-geometry boundary clamp (utils/v02_clamp_geometry).

Pins the constraint ORIENTATION (a depth cap, not a floor) and the clamp/margin
behaviour on synthetic cases — a sign error here would flip the whole experiment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from v02_clamp_geometry import (  # noqa: E402
    bind_mask,
    composite_bound,
    dtw_cap_from_base_alt,
    hard_clamp,
    paired_skill_ci,
    soft_clamp,
    valid_depth,
)


# --------------------------------------------------------------------------- #
# orientation
# --------------------------------------------------------------------------- #
def test_cap_is_depth_to_base_positive_when_surface_above_base():
    # land surface 1000 m, aquifer base 940 m -> 60 m of aquifer -> cap 60 m depth.
    cap = dtw_cap_from_base_alt(1000.0, 940.0)
    assert cap == pytest.approx(60.0)


def test_cap_orientation_is_a_ceiling_not_a_floor():
    # A physically valid water table (obs) must be shallower than the cap; a model
    # that predicts DEEPER than the base of the aquifer is the one that gets pulled
    # back UP to the cap. If orientation were flipped (a floor), a shallow obs would
    # be pushed deeper — the sign error we are guarding against.
    z_surf, base_alt = 1000.0, 940.0
    cap = dtw_cap_from_base_alt(z_surf, base_alt)  # 60 m
    obs = 25.0  # real water table well inside the aquifer
    pred_too_deep = 90.0  # deeper than the aquifer base -> impossible
    clamped = hard_clamp(np.array([pred_too_deep]), np.array([cap]))[0]
    assert clamped == pytest.approx(60.0)
    assert clamped < pred_too_deep  # depth reduced
    assert clamped > obs  # still above obs (bound is above the true water table)


def test_valid_depth_rejects_nonpositive_and_nonfinite():
    out = valid_depth([50.0, 0.0, -5.0, np.nan, np.inf])
    assert out[0] == pytest.approx(50.0)
    assert (
        np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3]) and np.isnan(out[4])
    )


# --------------------------------------------------------------------------- #
# hard clamp
# --------------------------------------------------------------------------- #
def test_hard_clamp_only_reduces_never_increases():
    pred = np.array([5.0, 50.0, 200.0])
    cap = np.array([100.0, 100.0, 100.0])
    out = hard_clamp(pred, cap)
    assert np.all(out <= pred)
    np.testing.assert_allclose(out, [5.0, 50.0, 100.0])


def test_hard_clamp_untouched_below_cap():
    # prediction below the cap is identical -> clamp cannot touch it ("shallow-safe"
    # wherever the aquifer is thick).
    pred = np.array([3.0, 12.0])
    cap = np.array([500.0, 500.0])
    np.testing.assert_allclose(hard_clamp(pred, cap), pred)


def test_hard_clamp_passthrough_when_cap_nan():
    # no geometry coverage -> constraint not applicable -> prediction unchanged.
    pred = np.array([80.0, 80.0])
    cap = np.array([np.nan, 40.0])
    np.testing.assert_allclose(hard_clamp(pred, cap), [80.0, 40.0])


def test_bind_mask_matches_definition():
    pred = np.array([80.0, 30.0, 80.0])
    cap = np.array([40.0, 40.0, np.nan])
    np.testing.assert_array_equal(bind_mask(pred, cap), [True, False, False])


# --------------------------------------------------------------------------- #
# soft clamp
# --------------------------------------------------------------------------- #
def test_soft_lambda_zero_equals_hard():
    pred = np.array([200.0, 5.0])
    cap = np.array([100.0, 100.0])
    np.testing.assert_allclose(soft_clamp(pred, cap, 0.0), hard_clamp(pred, cap))


def test_soft_lambda_one_is_noop():
    pred = np.array([200.0, 5.0])
    cap = np.array([100.0, 100.0])
    np.testing.assert_allclose(soft_clamp(pred, cap, 1.0), pred)


def test_soft_lambda_shrinks_exceedance():
    # pred 200, cap 100, lam 0.25 -> 100 + 0.25*100 = 125; still <= pred, >= cap.
    out = soft_clamp(np.array([200.0]), np.array([100.0]), 0.25)[0]
    assert out == pytest.approx(125.0)
    assert 100.0 <= out <= 200.0


# --------------------------------------------------------------------------- #
# margin loosening
# --------------------------------------------------------------------------- #
def test_positive_margin_reduces_binds_monotonically():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 300, 2000)
    comp = rng.uniform(20, 150, 2000)
    counts = [int(bind_mask(pred, comp + mg).sum()) for mg in (0, 25, 50, 100, 400)]
    assert counts == sorted(counts, reverse=True)  # non-increasing with margin
    assert counts[-1] == 0 or counts[-1] < counts[0]


def test_margin_clamped_closer_to_pred():
    pred = np.array([200.0])
    comp = np.array([50.0])
    c0 = hard_clamp(pred, comp + 0)[0]
    c50 = hard_clamp(pred, comp + 50)[0]
    assert c0 == pytest.approx(50.0)
    assert c50 == pytest.approx(100.0)
    assert abs(c50 - pred[0]) < abs(c0 - pred[0])  # looser margin -> closer to pred


# --------------------------------------------------------------------------- #
# composite coalesce
# --------------------------------------------------------------------------- #
def test_composite_priority_hp_then_fill_then_grav():
    b_hp = np.array([10.0, np.nan, np.nan, np.nan])
    b_fill = np.array([20.0, 20.0, np.nan, np.nan])
    b_grav = np.array([30.0, 30.0, 30.0, np.nan])
    out = composite_bound(b_hp, b_fill, b_grav)
    np.testing.assert_allclose(out[:3], [10.0, 20.0, 30.0])
    assert np.isnan(out[3])


def test_composite_drops_nonpositive_layers():
    # a zero/negative HP bound is invalid -> fall through to fill.
    out = composite_bound(np.array([-5.0]), np.array([42.0]), np.array([np.nan]))
    assert out[0] == pytest.approx(42.0)


# --------------------------------------------------------------------------- #
# skill helper
# --------------------------------------------------------------------------- #
def test_paired_skill_ci_zero_when_identical():
    err = np.array([1.0, -2.0, 3.0, -4.0])
    out = paired_skill_ci(err, err, n_boot=200)
    assert out["mad_skill"] == pytest.approx(0.0)
    assert out["rmse_skill"] == pytest.approx(0.0)


def test_paired_skill_ci_positive_when_model_better():
    base = np.array([10.0, 10.0, 10.0, 10.0])
    model = np.array([2.0, 2.0, 2.0, 2.0])
    out = paired_skill_ci(model, base, n_boot=200)
    assert out["mad_skill"] > 0 and out["rmse_skill"] > 0
    assert out["mad_skill_ci95"][0] > 0  # tight, all-positive
