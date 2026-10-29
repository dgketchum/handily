"""Footprint handling in the statewide raster eval and the ``--drop-uncovered``
mode of the CONUS OOF comparer (AZ round, 2026-10-29: a partial reference arm
silently shrank every panel, and state parity arms could not be compared)."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_UTILS = Path(__file__).resolve().parents[1] / "utils"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _UTILS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(_UTILS))
    spec.loader.exec_module(mod)
    return mod


def test_footprint_masks_partial_reference():
    ev = _load("eval_nv_statewide_raster")
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan])
    prim = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0])
    ref = np.array([1.0, 2.0, np.nan, np.nan, 5.0, 6.0])
    ma = np.ones(6)
    basin = np.array(["a", "a", "b", "b", "c", "c"])
    fps = ev.footprint_masks(obs, prim, ref, ma, basin)
    assert list(fps) == ["primary_ma", "all_arms"]
    pm, nb_pm = fps["primary_ma"]
    al, nb_al = fps["all_arms"]
    assert pm.tolist() == [True, True, True, True, False, False] and nb_pm == 2
    assert al.tolist() == [True, True, False, False, False, False] and nb_al == 1


def test_footprint_masks_full_reference_single_footprint():
    ev = _load("eval_nv_statewide_raster")
    v = np.arange(4.0)
    fps = ev.footprint_masks(v, v, v, v, np.array(["a", "a", "b", "b"]))
    assert list(fps) == ["all_arms"] and fps["all_arms"][1] == 2
    rows = ev.tag_rows([{"x": 1}, {"x": 2}], "all_arms", 2)
    assert all(
        r["footprint"] == "all_arms" and r["footprint_basins"] == 2 for r in rows
    )


def _write_run(d: Path, ids, rng, graph: Path):
    d.mkdir(parents=True)
    n = len(ids)
    obs = rng.uniform(0.5, 40.0, n)
    pd.DataFrame(
        {
            "canonical_id": ids,
            "is_water_pseudo": False,
            "obs_dtw_m": obs,
            "gnn_dtw_m": obs + rng.normal(0, 2.0, n),
            "gnn_sigma_m": rng.uniform(1, 5, n),
            "huc2": "16",
        }
    ).to_parquet(d / "gnn_oof_predictions.parquet")
    (d / "gnn_run.json").write_text(json.dumps({"graph_dir": str(graph), "folds": {}}))


def test_oof_compare_drop_uncovered(tmp_path):
    rng = np.random.default_rng(0)
    base = [f"w{i:03d}" for i in range(60)]
    ref_ids = base + [f"ref_only{i}" for i in range(5)]
    arm_ids = base + [f"arm_only{i}" for i in range(7)]
    graph = tmp_path / "graph"
    graph.mkdir()
    all_ids = sorted(set(ref_ids) | set(arm_ids))
    pd.DataFrame(
        {"canonical_id": all_ids, "cv_unit": [f"u{i % 4}" for i in range(len(all_ids))]}
    ).to_parquet(graph / "query_nodes.parquet")
    _write_run(tmp_path / "ref", ref_ids, rng, graph)
    _write_run(tmp_path / "arm", arm_ids, rng, graph)
    cmd = [
        sys.executable,
        str(_UTILS / "compare_conus_oof_arms.py"),
        "--ref",
        f"ref={tmp_path / 'ref'}",
        "--arm",
        f"arm={tmp_path / 'arm'}",
        "--n-boot",
        "20",
    ]
    r = subprocess.run(
        cmd + ["--out-dir", str(tmp_path / "o1")], capture_output=True, text=True
    )
    assert r.returncode != 0 and "does not cover" in (r.stdout + r.stderr)
    r = subprocess.run(
        cmd + ["--out-dir", str(tmp_path / "o2"), "--drop-uncovered"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    summ = json.loads((tmp_path / "o2" / "run_summary.json").read_text())
    assert summ["n_wells"] == 60 and summ["drop_uncovered"] is True
    cov = summ["coverage"]["arm"]
    assert cov == {
        "n_arm_wells": 67,
        "n_ref_wells": 65,
        "n_arm_only_dropped": 7,
        "n_ref_only_dropped": 5,
    }
    panel = pd.read_csv(tmp_path / "o2" / "panel.csv")
    assert (panel.loc[panel.stratum == "all", "n"] == 60).all()
