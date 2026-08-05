"""Unit tests for the MAE embedding extractor's output naming: the --tag suffix must
keep same-arm checkpoints (v1 vs v2 pyr_wide) from clobbering each other's parquets.

Loads utils/extract_mae_embeddings.py by path (repo test convention).
"""

import importlib.util
import sys
from pathlib import Path


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


eme = _load("extract_mae_embeddings")


def test_out_basename_untagged_matches_v1_convention():
    assert (
        eme.out_basename("pyr_wide", None, False, None)
        == "mae_embeddings_pyr_wide.parquet"
    )
    assert (
        eme.out_basename("pyr_wide", None, True, None)
        == "mae_embeddings_pyr_wide_allq.parquet"
    )
    assert (
        eme.out_basename("pyr_wide", None, False, "hellgate_coords")
        == "mae_embeddings_pyr_wide_hellgate_coords.parquet"
    )


def test_out_basename_tag_disambiguates_same_arm():
    tagged = eme.out_basename("pyr_wide", "v2hard", False, None)
    assert tagged == "mae_embeddings_pyr_wide_v2hard.parquet"
    assert tagged != eme.out_basename("pyr_wide", None, False, None)
    assert (
        eme.out_basename("pyr_wide", "v2hard", True, None)
        == "mae_embeddings_pyr_wide_v2hard_allq.parquet"
    )
    assert (
        eme.out_basename("pyr_wide", "v2hard", False, "coords")
        == "mae_embeddings_pyr_wide_v2hard_coords.parquet"
    )
