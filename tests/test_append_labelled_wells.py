"""Unit checks for utils/append_labelled_wells.py helpers."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
import append_labelled_wells as A  # noqa: E402


def test_ids_deterministic_and_unique():
    x = np.array([1000.0, 1000.0, 2000.5])
    y = np.array([500.0, 501.0, 500.0])
    a = A._ids("ndwr", x, y)
    b = A._ids("ndwr", x, y)
    assert list(a) == list(b)
    assert len(set(a)) == 3
    assert all(s.startswith("ndwr_") for s in a)


def test_ids_prefix_changes_id():
    x = np.array([1.0])
    y = np.array([2.0])
    assert A._ids("ndwr", x, y)[0] != A._ids("other", x, y)[0]


def test_novel_unit_folds_never_split_a_unit():
    # base wells: fold 0 at x=0, fold 1 at x=100
    base_xy = np.array([[0.0, 0.0], [100.0, 0.0]])
    base_fold = np.array([0, 1])
    # unit "a": rows straddle the midpoint -> a per-row rule would split it;
    # its row nearest to any base well (x=10) sits by the fold-0 well -> fold 0
    units = np.array(["a", "a", "a", "b"])
    xy = np.array([[10.0, 0.0], [60.0, 0.0], [90.0, 0.0], [99.0, 0.0]])
    f = A.novel_unit_folds(units, xy, base_xy, base_fold)
    assert list(f) == [0, 0, 0, 1]
