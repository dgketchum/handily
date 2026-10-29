import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from pumping_r_core import (  # noqa: E402
    DonorIndex,
    Variant,
    cap_shares,
    compute,
    exclude_donors,
    weighted_stats,
)


def _donors(n=200, seed=0, dtw=10.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 20_000, n)
    y = rng.uniform(0, 20_000, n)
    z = np.full(n, 1000.0)
    d = np.full(n, dtw)
    return pd.DataFrame(
        {
            "canonical_id": [f"w{i}" for i in range(n)],
            "source": "s",
            "x5070": x,
            "y5070": y,
            "z_surf_m": z,
            "mean_dtw": d,
            "wte_m": z - d,
        }
    )


def test_cap_shares_bounds_top_share():
    w = np.array([[100.0, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    c = cap_shares(w, 0.25)
    assert (c / c.sum(1, keepdims=True)).max() <= 0.25 + 1e-9
    assert np.array_equal(cap_shares(w, 0.0), w)


def test_weighted_stats_quantiles():
    vals = np.array([[1.0, 2.0, 3.0, 4.0]])
    wts = np.ones((1, 4))
    st = weighted_stats(vals, wts)
    assert st["median"][0] == 2.0
    assert st["q75"][0] == 3.0
    assert st["mean"][0] == 2.5
    st0 = weighted_stats(vals, np.zeros((1, 4)))
    assert np.isnan(st0["median"][0])


def test_uniform_field_reproduced_and_capped():
    d = _donors(dtw=10.0)
    ix = DonorIndex(d, 100.0)
    q = np.array([[10_000.0, 10_000.0]])
    for kern in ("idw", "gauss"):
        for gate in ("depth", "support", "median"):
            v = Variant(kernel=kern, gate=gate, k=16)
            r = compute(ix, q, np.array([1000.0]), v)
            assert abs(r["r_blend"][0] - 990.0) < 1e-9
    # ground cap: a query 5 m below the donor table is capped at z - cap_m
    r = compute(ix, q, np.array([985.0]), Variant(cap_m=1.0))
    assert r["cap_bound"][0] and abs(r["r_blend"][0] - 984.0) < 1e-9


def test_single_bad_donor_cannot_set_median_under_share_cap():
    d = _donors(dtw=50.0)
    q = np.array([[5_000.0, 5_000.0]])
    bad = d.iloc[[0]].copy()
    bad["x5070"], bad["y5070"], bad["mean_dtw"] = 5_000.0, 5_000.0, 2.0
    bad["wte_m"] = bad["z_surf_m"] - 2.0
    d = pd.concat([d, bad], ignore_index=True)
    ix = DonorIndex(d, 100.0)
    plain = compute(ix, q, np.array([1000.0]), Variant(max_share=0.0, gate="median"))
    capped = compute(ix, q, np.array([1000.0]), Variant(max_share=0.25, gate="median"))
    assert (
        abs(plain["r_median"][0] - 998.0) < 1e-9
    )  # the co-located donor owns the cell
    assert abs(capped["r_median"][0] - 950.0) < 1e-9
    assert capped["top_share"][0] <= 0.25 + 1e-9


def test_screen_drops_outlier():
    d = _donors(dtw=50.0)
    d.loc[:2, "mean_dtw"] = 500.0
    d["wte_m"] = d["z_surf_m"] - d["mean_dtw"]
    ix = DonorIndex(d, 100.0)
    q = d[["x5070", "y5070"]].to_numpy()[:1]
    r = compute(ix, q, np.array([1000.0]), Variant(screen=True, gate="median"))
    assert r["n_screened"][0] >= 1
    assert abs(r["r_median"][0] - 950.0) < 1e-9


def test_exclude_donors_radius_same_source_and_ids(tmp_path):
    d = _donors(n=4)
    d["x5070"] = [0.0, 50.0, 500.0, 5000.0]
    d["y5070"] = 0.0
    d["source"] = ["a", "b", "b", "a"]
    pts = tmp_path / "p.parquet"
    pd.DataFrame({"x5070": [0.0], "y5070": [0.0]}).to_parquet(pts)
    ids = tmp_path / "ids.parquet"
    pd.DataFrame({"canonical_id": ["w3"]}).to_parquet(ids)
    out, c = exclude_donors(d, [str(pts)], 10.0, ["b"], 100.0, [str(ids)])
    assert list(out["canonical_id"]) == ["w2"]
    assert c["dropped_radius"] == 1 and c["dropped_same_source"] == 1
    assert c["dropped_ids"] == 1
