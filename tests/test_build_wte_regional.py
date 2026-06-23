"""Unit tests for utils/build_wte_regional.py robust_smooth_fit().

Covers the three behaviors that define the estimator: robust rejection of
trend outliers, the one-sided stream cap (upper bound, not equality), and the
table rising into the uplands where wells justify it.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from rasterio.transform import Affine

_MOD = Path(__file__).resolve().parents[1] / "utils" / "build_wte_regional.py"
_spec = importlib.util.spec_from_file_location("build_wte_regional", _MOD)
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)

# One huge 2x2 EPSG:5070 grid that swallows all of CONUS, so binning never drops a
# synthetic well for being out-of-grid -- isolates the filter/sample/manifest logic.
_CONUS_TR = Affine(4.0e6, 0.0, -3.0e6, 0.0, -4.0e6, 4.0e6)


def _synthetic_wells(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    lon = rng.uniform(-120.0, -75.0, n)
    lat = rng.uniform(28.0, 48.0, n)
    obs = np.where(np.arange(n) < 100, 1, 3)  # first 100 single-obs, rest multi-obs
    df = pd.DataFrame(
        {
            "longitude": lon,
            "latitude": lat,
            "mean_wte_m": rng.uniform(100.0, 2000.0, n),
            "confinement_class": "unconfined",
            "source": np.where(np.arange(n) % 2 == 0, "nwis", "state_db"),
            "obs_count": obs,
            "canonical_id": [f"w{i:04d}" for i in range(n)],
        }
    )
    p = tmp_path / "wells.parquet"
    df.to_parquet(p, index=False)
    return p, df


def test_single_obs_and_sample_define_the_retirement_manifest(tmp_path):
    # The fit set (manifest) must be exactly the single-obs, sampled, in-grid wells:
    # source-agnostic (nwis kept), obs_count==1 only, and a deterministic 1/2 sample.
    p, df = _synthetic_wells(tmp_path)
    single_ids = set(df.loc[df.obs_count == 1, "canonical_id"])

    _, well_weight, well_count, fit_ids, n_used, _ = br.aggregate_wells(
        str(p),
        _CONUS_TR,
        (2, 2),
        single_obs_only=True,
        sample_frac=0.5,
        seed=0,
    )
    assert set(fit_ids) <= single_ids  # never a multi-obs well
    assert abs(len(fit_ids) - 50) <= 8  # ~1/2 of the 100 single-obs wells
    assert n_used == len(fit_ids)  # every binned well is in the manifest
    assert int(well_count.sum()) == len(fit_ids)  # support counts == fit set

    # Deterministic: same seed -> identical manifest.
    *_, fit_ids2, _, _ = br.aggregate_wells(
        str(p), _CONUS_TR, (2, 2), single_obs_only=True, sample_frac=0.5, seed=0
    )
    assert list(fit_ids) == list(fit_ids2)


def test_source_agnostic_by_default_but_excludable(tmp_path):
    # Default keeps nwis (source-agnostic dev); --exclude-sources opts out.
    p, df = _synthetic_wells(tmp_path)
    *_, fit_all, _, _ = br.aggregate_wells(str(p), _CONUS_TR, (2, 2))
    *_, fit_noidx, _, _ = br.aggregate_wells(
        str(p), _CONUS_TR, (2, 2), exclude_sources=("nwis",)
    )
    assert len(fit_all) == len(df)  # all wells kept by default
    assert len(fit_noidx) == int((df.source != "nwis").sum())


def _empty(shape):
    return (
        np.full(shape, np.nan),  # well_mean
        np.zeros(shape),  # well_n
        np.full(shape, np.nan),  # cap_value
        np.zeros(shape, dtype=bool),  # cap_mask
    )


def test_robust_fit_ignores_a_gross_outlier():
    # A linear ramp of wells along a row, with one absurd shallow/high outlier.
    # The robust fit should track the ramp, not get yanked toward the outlier.
    domain = np.ones((1, 11), dtype=bool)
    wm, wn, cv, cm = _empty((1, 11))
    ramp = np.arange(11, dtype=float)  # 0..10
    wm[0] = ramp
    wn[0] = 1.0
    wm[0, 5] = 500.0  # gross outlier at the midpoint
    s = br.robust_smooth_fit(
        domain, wm, wn, cv, cm, lam=0.5, robust_scale=2.0, n_iter=12
    )
    # midpoint should stay near the underlying ramp value (5), nowhere near 500
    assert abs(s[0, 5] - 5.0) < 1.5
    assert s[0, 5] < 50.0


def test_stream_cap_is_a_one_sided_upper_bound():
    # A well wants the surface at 100 in a cell that is also a major-stream cell
    # capped at 10. The cap must win (S <= 10), since the table cannot dome above
    # the stream.
    domain = np.ones((1, 5), dtype=bool)
    wm, wn, cv, cm = _empty((1, 5))
    wm[0, 2] = 100.0
    wn[0, 2] = 1.0
    cm[0, 2] = True
    cv[0, 2] = 10.0
    s = br.robust_smooth_fit(domain, wm, wn, cv, cm, lam=0.1, cap_weight=1e4, n_iter=10)
    assert s[0, 2] <= 10.0 + 0.5


def test_cap_does_not_pull_surface_up():
    # Where a well sits BELOW the stream cap (arid losing reach), the one-sided cap
    # must not lift the surface up to the cap.
    domain = np.ones((1, 5), dtype=bool)
    wm, wn, cv, cm = _empty((1, 5))
    wm[0, 2] = 3.0  # well well below the cap
    wn[0, 2] = 1.0
    cm[0, 2] = True
    cv[0, 2] = 50.0  # cap far above
    s = br.robust_smooth_fit(domain, wm, wn, cv, cm, lam=0.1, cap_weight=1e4, n_iter=10)
    assert abs(s[0, 2] - 3.0) < 1.0  # stays at the well, not lifted to 50


def test_table_rises_into_uplands_where_wells_justify():
    # Valley wells low at the ends, mountain wells high in the middle (no cap
    # overhead). The fitted surface must rise in the middle, not stay flat/low.
    domain = np.ones((1, 9), dtype=bool)
    wm, wn, cv, cm = _empty((1, 9))
    wm[0, 0] = 0.0
    wm[0, 8] = 0.0
    wm[0, 4] = 40.0  # mountain well
    wn[0, [0, 4, 8]] = 1.0
    s = br.robust_smooth_fit(
        domain, wm, wn, cv, cm, lam=0.2, robust_scale=50.0, n_iter=8
    )
    assert s[0, 4] > 20.0  # surface rose toward the mountain well
    assert s[0, 4] > s[0, 1]  # higher in the middle than near the valley


def test_dataless_component_is_nan():
    # Two disconnected blocks; only the left has a well. The right has no data and
    # must stay NaN (not fabricated).
    domain = np.zeros((1, 5), dtype=bool)
    domain[0, 0:2] = True
    domain[0, 3:5] = True
    wm, wn, cv, cm = _empty((1, 5))
    wm[0, 0] = 7.0
    wn[0, 0] = 1.0
    s = br.robust_smooth_fit(domain, wm, wn, cv, cm, lam=0.5, n_iter=5)
    assert np.isfinite(s[0, 0]) and np.isfinite(s[0, 1])
    assert np.isnan(s[0, 3]) and np.isnan(s[0, 4])
    assert np.isnan(s[0, 2])  # outside domain
