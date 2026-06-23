"""Unit tests for utils/build_wte_anchors.py (pure functions).

Covers the two subtle correctness points: the accumulation nodata sentinel
(-9999999) must be masked BEFORE the magnitude test so it cannot pose as a giant
river, and a large *negative* accumulation (Hydrography90m inter-region inflow at
a 20-degree tile seam) IS a valid channel.
"""

import importlib.util
from pathlib import Path

import numpy as np

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_wte_anchors.py"
_spec = importlib.util.spec_from_file_location("build_wte_anchors", _MOD)
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)

ACC_ND = -9999999.0
ELEV_ND = 32767.0
THR = 1000.0  # cells


def _grid():
    # 1x6 row of cells:
    #  0: big river                 acc=5000   elev=100  -> channel, head 100
    #  1: below threshold           acc=200    elev=110  -> off-channel
    #  2: accumulation nodata       acc=ND     elev=120  -> NOT channel (the bug)
    #  3: seam inflow (negative)    acc=-3000  elev=130  -> channel, head 130
    #  4: elev nodata on a river    acc=8000   elev=ND   -> off-channel (no surface)
    #  5: exactly at threshold      acc=1000   elev=140  -> channel (>=), head 140
    acc = np.array([[5000.0, 200.0, ACC_ND, -3000.0, 8000.0, 1000.0]])
    elev = np.array([[100.0, 110.0, 120.0, 130.0, ELEV_ND, 140.0]])
    return acc, elev


def test_anchor_head_values_and_nodata():
    acc, elev = _grid()
    head, mask = ba.compute_anchor_head(
        acc, elev, acc_nodata=ACC_ND, elev_nodata=ELEV_ND, threshold_cells=THR
    )
    nd = ba.OUT_NODATA
    np.testing.assert_array_equal(
        head, np.array([[100.0, nd, nd, 130.0, nd, 140.0]], dtype=np.float32)
    )
    # mask: 1 channel / 0 off-channel-valid / 255 input-nodata
    np.testing.assert_array_equal(
        mask, np.array([[1, 0, ba.MASK_NODATA, 1, ba.MASK_NODATA, 1]], dtype=np.uint8)
    )
    assert head.dtype == np.float32 and mask.dtype == np.uint8


def test_nodata_sentinel_is_not_a_channel():
    # |-9999999| >> threshold; only correct nodata-first masking keeps it off.
    acc = np.array([[ACC_ND]])
    elev = np.array([[50.0]])
    head, mask = ba.compute_anchor_head(
        acc, elev, acc_nodata=ACC_ND, elev_nodata=ELEV_ND, threshold_cells=THR
    )
    assert head[0, 0] == ba.OUT_NODATA
    assert mask[0, 0] == ba.MASK_NODATA


def test_negative_seam_inflow_is_a_channel():
    acc = np.array([[-50000.0]])
    elev = np.array([[200.0]])
    head, mask = ba.compute_anchor_head(
        acc, elev, acc_nodata=ACC_ND, elev_nodata=ELEV_ND, threshold_cells=THR
    )
    assert head[0, 0] == np.float32(200.0)
    assert mask[0, 0] == 1


def test_count_ladder_is_monotone_nonincreasing():
    acc, elev = _grid()
    valid = (acc != ACC_ND) & (elev != ELEV_ND)
    counts = ba.count_channel_cells(acc, valid, [200.0, 1000.0, 4000.0, 10000.0])
    assert list(counts) == sorted(counts, reverse=True)
    # at 200 cells: cells 0,3(|-3000|),5 plus the acc=200 cell? acc=200 is exactly
    # 200 -> >=200 true; cell 4 invalid (elev nodata); cell 2 invalid (acc nodata).
    assert counts[0] == 4  # acc 5000,200,-3000,1000 valid & >=200 (8000 has elev ND)
    assert counts[1] == 3  # >=1000: 5000,-3000,1000
