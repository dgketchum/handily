"""Unit tests for utils/build_wte_surface.py solve_head().

Covers the Dirichlet Laplace/Poisson solve core: harmonic interpolation between
anchors, the no-information (anchor-less component -> NaN) rule, and the recharge
mound term raising the table above the harmonic baseline.
"""

import importlib.util
from pathlib import Path

import numpy as np

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_wte_surface.py"
_spec = importlib.util.spec_from_file_location("build_wte_surface", _MOD)
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


def test_1d_linear_interpolation_between_two_anchors():
    # 1x5 row, anchors at the ends (0 and 40), Laplace -> linear ramp.
    domain = np.ones((1, 5), dtype=bool)
    anchor_mask = np.zeros((1, 5), dtype=bool)
    anchor_mask[0, 0] = True
    anchor_mask[0, 4] = True
    anchor_val = np.zeros((1, 5))
    anchor_val[0, 0] = 0.0
    anchor_val[0, 4] = 40.0
    head = bs.solve_head(domain, anchor_mask, anchor_val, mound_c=0.0)
    np.testing.assert_allclose(head[0], [0, 10, 20, 30, 40], atol=1e-4)


def test_2d_harmonic_gradient_across_columns():
    # Anchors on the left (0) and right (10) columns -> linear in x, flat in y.
    domain = np.ones((4, 5), dtype=bool)
    anchor_mask = np.zeros((4, 5), dtype=bool)
    anchor_mask[:, 0] = True
    anchor_mask[:, 4] = True
    anchor_val = np.zeros((4, 5))
    anchor_val[:, 4] = 10.0
    head = bs.solve_head(domain, anchor_mask, anchor_val, mound_c=0.0)
    expected = np.tile(np.array([0, 2.5, 5.0, 7.5, 10.0]), (4, 1))
    np.testing.assert_allclose(head, expected, atol=1e-4)


def test_anchorless_component_is_nan():
    # Two disconnected land blocks; only the left has an anchor. The right block
    # has no boundary information and must come back NaN (never fabricated).
    domain = np.zeros((1, 5), dtype=bool)
    domain[0, 0:2] = True  # left component (has anchor)
    domain[0, 3:5] = True  # right component (no anchor)
    anchor_mask = np.zeros((1, 5), dtype=bool)
    anchor_mask[0, 0] = True
    anchor_val = np.zeros((1, 5))
    anchor_val[0, 0] = 5.0
    head = bs.solve_head(domain, anchor_mask, anchor_val, mound_c=0.0)
    assert head[0, 0] == 5.0
    assert head[0, 1] == 5.0  # harmonic with one anchor -> constant
    assert np.isnan(head[0, 3]) and np.isnan(head[0, 4])
    assert np.isnan(head[0, 2])  # gap (not in domain)


def test_mound_raises_table_above_harmonic_baseline():
    # Same end anchors at 0; pure Laplace -> all interior 0. A positive recharge
    # mound (nabla^2 h = -c) bulges the interior above the baseline.
    domain = np.ones((1, 7), dtype=bool)
    anchor_mask = np.zeros((1, 7), dtype=bool)
    anchor_mask[0, 0] = True
    anchor_mask[0, 6] = True
    anchor_val = np.zeros((1, 7))
    flat = bs.solve_head(domain, anchor_mask, anchor_val, mound_c=0.0)
    mounded = bs.solve_head(domain, anchor_mask, anchor_val, mound_c=1.0)
    np.testing.assert_allclose(flat[0], 0.0, atol=1e-4)
    assert mounded[0, 3] > mounded[0, 1] > 0.0  # peak at center, above zero
    assert mounded[0, 1] == mounded[0, 5]  # symmetric


def test_outside_domain_is_nan():
    domain = np.ones((1, 3), dtype=bool)
    domain[0, 2] = False
    anchor_mask = np.zeros((1, 3), dtype=bool)
    anchor_mask[0, 0] = True
    anchor_val = np.zeros((1, 3))
    head = bs.solve_head(domain, anchor_mask, anchor_val, mound_c=0.0)
    assert np.isnan(head[0, 2])
