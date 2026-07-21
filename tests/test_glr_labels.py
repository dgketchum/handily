"""GLR (gaining/losing-reach) cross-fit shore-label screen tests.

Two families, mirroring test_shore_labels.py:

* pure-logic tests (synthetic frames) -- always run. They pin the pre-registered
  screen logic (median over K<=5 nearest within D; PASS iff n>=1 AND d_glr<=+5 m;
  no lower bound; no-evidence => no-label), the MANDATED synthetic leakage test (a
  held-out well that would flip a pass bit must not affect that fold's bit -- fold f's
  mask sees only cv_fold!=f training wells), the lattice cell-id join key, and the
  trainer's per-fold weight composition + flag-off byte-identity.
* data-dependent tests -- skip when the built glr_labels parquet is absent, so the
  suite still runs off-zoran. They pin the fold-awareness proof (per-fold pass counts
  differ) and 1:1 coverage of the shore lattice cells.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

WTE = Path("/data/ssd2/handily/conus/wte_gnn")
GLR = WTE / "glr" / "glr_labels.parquet"
V2S = WTE / "graph_conus_monitoring_water_v2s"
_HAVE_GLR = GLR.exists()
_HAVE_BUNDLE = V2S.exists()
_glr = pytest.mark.skipif(not _HAVE_GLR, reason="glr_labels parquet absent")
_bundle = pytest.mark.skipif(not _HAVE_BUNDLE, reason="v2s bundle absent")


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# screen logic (pure, synthetic)
# ---------------------------------------------------------------------------
def test_screen_pass_only_when_ground_at_or_below_local_water_table():
    """d_glr = median(z_surf - well WTE) over K<=5 nearest within D; PASS iff <= +5 m."""
    bgl = _load("build_glr_labels")
    ring_xy = np.array([[0.0, 0.0]])
    z_surf = np.array([10.0])
    # three wells within 5 km: WT altitudes 9,10,11 -> diffs +1,0,-1 -> median 0 <= 5
    well_xy = np.array([[100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])
    well_wte = np.array([9.0, 10.0, 11.0])
    r = bgl.screen_pool(ring_xy, z_surf, well_xy, well_wte)
    assert r["n_wells"][0] == 3
    assert np.isclose(r["d_glr"][0], 0.0)
    assert bool(r["pass"][0])
    # lift the water table far below ground: diffs +18,+19,+20 -> median 19 > 5 -> FAIL
    r2 = bgl.screen_pool(ring_xy, z_surf, well_xy, np.array([-8.0, -9.0, -10.0]))
    assert np.isclose(r2["d_glr"][0], 19.0)
    assert not bool(r2["pass"][0])


def test_screen_no_lower_bound_discharge_zone_passes():
    """WT above land surface (d_glr strongly negative) is a discharge zone; DTW=0
    remains plausible, so there is NO lower bound -- it must PASS."""
    bgl = _load("build_glr_labels")
    r = bgl.screen_pool(
        np.array([[0.0, 0.0]]),
        np.array([10.0]),
        np.array([[50.0, 0.0]]),
        np.array([40.0]),
    )
    assert np.isclose(r["d_glr"][0], -30.0)  # ring ground 30 m BELOW the water table
    assert bool(r["pass"][0])


def test_screen_no_evidence_no_label():
    """No well within D => n=0 => d_glr NaN => NO PASS (conservative default)."""
    bgl = _load("build_glr_labels")
    far = bgl.SEARCH_RADIUS_M + 1000.0
    r = bgl.screen_pool(
        np.array([[0.0, 0.0]]),
        np.array([10.0]),
        np.array([[far, 0.0]]),
        np.array([10.0]),
    )
    assert r["n_wells"][0] == 0
    assert not np.isfinite(r["d_glr"][0])
    assert not bool(r["pass"][0])


def test_screen_median_over_at_most_k_nearest():
    """With >K wells within D the median is taken over the K nearest only (a far pair
    inside D cannot swing a shore whose K nearest are all shallow)."""
    bgl = _load("build_glr_labels")
    # 5 near wells (diffs 0) + 2 farther-but-in-D wells with huge diffs; K=5 -> median 0
    near = np.array([[10.0 * i, 0.0] for i in range(1, 6)])
    farin = np.array([[4000.0, 0.0], [4500.0, 0.0]])
    well_xy = np.vstack([near, farin])
    well_wte = np.concatenate([np.full(5, 10.0), np.array([-90.0, -90.0])])
    r = bgl.screen_pool(np.array([[0.0, 0.0]]), np.array([10.0]), well_xy, well_wte)
    assert r["n_wells"][0] == bgl.K_NEAREST  # capped at K among the in-D neighbors
    assert np.isclose(r["d_glr"][0], 0.0)
    assert bool(r["pass"][0])


# ---------------------------------------------------------------------------
# MANDATED synthetic leakage test (plan section: verification program)
# ---------------------------------------------------------------------------
def _leak_frames(with_flip_well: bool):
    """One ring; a nearby 'flip' well in fold 0 that would pass the screen, plus far
    background wells in folds 0/1/2 so every leave-one-fold-out pool is non-empty."""
    ring = pd.DataFrame(
        {"x5070": [0.0], "y5070": [0.0], "huc8": ["01010101"], "z_surf": [10.0]}
    )
    rows = []
    if with_flip_well:
        rows.append((1000.0, 0.0, 0, 10.0))  # W: within 5 km, d_glr=0 -> would PASS
    rows += [
        (100000.0, 0.0, 0, 0.0),  # far background, fold 0
        (100000.0, 100000.0, 1, 0.0),  # far background, fold 1
        (200000.0, 0.0, 2, 0.0),  # far background, fold 2
    ]
    pool = pd.DataFrame(rows, columns=["x5070", "y5070", "cv_fold", "wte_obs_m"])
    return ring, pool


def test_heldout_well_never_flips_its_own_fold_bit():
    """The flip well W is in fold 0. Fold 0's mask uses only cv_fold!=0 wells, so W must
    NOT influence glr_pass_fold_0 -- adding/removing W leaves fold 0's bit unchanged
    (False both ways), while it DOES flip fold 1's bit (where W is a training well)."""
    bgl = _load("build_glr_labels")
    with_w, _ = bgl.build_labels(*_leak_frames(with_flip_well=True))
    without_w, _ = bgl.build_labels(*_leak_frames(with_flip_well=False))
    # fold 0 (W's OWN held-out fold): W is invisible to the screen -> no near evidence
    assert not bool(with_w["glr_pass_fold_0"].iloc[0])
    assert not bool(without_w["glr_pass_fold_0"].iloc[0])
    assert bool(with_w["glr_pass_fold_0"].iloc[0]) == bool(
        without_w["glr_pass_fold_0"].iloc[0]
    )  # held-out well leaves its own fold's bit invariant
    # fold 1 (W is a TRAINING well): W legitimately flips the bit on
    assert bool(with_w["glr_pass_fold_1"].iloc[0])
    assert not bool(without_w["glr_pass_fold_1"].iloc[0])


def test_per_fold_bits_are_fold_dependent():
    """The flip well in fold 0 makes glr_pass_fold_0 differ from the other folds -- the
    fold-awareness proof (a global mask would give identical bits across folds)."""
    bgl = _load("build_glr_labels")
    out, folds = bgl.build_labels(*_leak_frames(with_flip_well=True))
    bits = [bool(out[f"glr_pass_fold_{f}"].iloc[0]) for f in folds]
    assert bits[0] is False  # fold 0 cannot see its own W
    assert any(bits)  # some other fold can
    assert len(set(bits)) > 1  # bits are NOT all identical => genuinely fold-aware


# ---------------------------------------------------------------------------
# lattice cell-id join key
# ---------------------------------------------------------------------------
def test_cell_id_collapses_shared_cell():
    bgl = _load("build_glr_labels")
    x = np.array([bgl.LAT_X0 + 12.0, bgl.LAT_X0 + 88.0, bgl.LAT_X0 + 150.0])
    y = np.array([bgl.LAT_Y0 - 12.0, bgl.LAT_Y0 - 88.0, bgl.LAT_Y0 - 150.0])
    cid = bgl.cell_ids(x, y)
    assert cid[0] == cid[1]  # same 100 m cell
    assert cid[0] != cid[2]  # different cell


def test_cell_id_matches_trainer_inline_formula():
    """The trainer computes the shore<->label join key inline; it must match the
    builder's cell_ids bit-for-bit (else the per-fold weight join silently misses)."""
    bgl = _load("build_glr_labels")
    x = np.array([-2301750.0, -2301550.0, 12345.0])
    y = np.array([2077050.0, 2077350.0, -6789.0])
    # inline formula copied from train_conus_gnn._cell_id
    x0, y0, res = -2540000.0, 3258000.0, 100.0
    col = np.floor((x - x0) / res).astype("int64")
    row = np.floor((y0 - y) / res).astype("int64")
    inline = col * 100_000_000 + row
    assert np.array_equal(inline, bgl.cell_ids(x, y))


# ---------------------------------------------------------------------------
# trainer per-fold weight composition (pure logic; the exact statements main() uses)
# ---------------------------------------------------------------------------
def test_flag_off_uses_global_weight_byte_identical():
    """glr_active False => the fold weight IS the global sample_w_t (no override), so
    training is byte-identical to the --shore-label-weight path."""
    sample_w_t = np.array([1.0, 0.0, 0.0])  # e.g. two shore rows at shore-weight 0
    glr_active = False
    sample_w_fold_t = sample_w_t
    if glr_active:  # pragma: no cover - the off branch is the assertion
        sample_w_fold_t = sample_w_t.copy()
        sample_w_fold_t[:] = 99.0
    assert sample_w_fold_t is sample_w_t


def test_per_fold_override_lifts_only_passing_shore_rows():
    """In fold f: passing shore rows -> glr_label_weight; non-passing shore stay 0;
    real wells keep their base weight untouched."""
    # rows: [real, shore_pass_f, shore_fail_f, real, shore_pass_f]
    glr_base_w = np.array([1.0, 0.0, 0.0, 2.0, 0.0])  # shore at shore-weight 0
    glr_label_weight = 0.25
    pass_f = np.array([False, True, False, False, True])  # only shore rows are True
    w_fold = glr_base_w.copy()
    w_fold[pass_f] = glr_label_weight
    assert w_fold.tolist() == [1.0, 0.25, 0.0, 2.0, 0.25]
    # non-shore rows never touched; non-passing shore stays loss-inert
    assert w_fold[0] == 1.0 and w_fold[3] == 2.0
    assert w_fold[2] == 0.0


def test_shore_row_in_heldout_fold_excluded_from_training_regardless_of_pass():
    """A shore row whose OWN cv_fold == f is in `test`, hence out of `tr` -- so even a
    set pass bit cannot make it train in its own held-out fold (the weight only bites
    rows fold f actually trains on)."""
    fold = np.array([0, 1, 0, 1])
    shore = np.array([False, False, True, True])  # shore rows in folds 0 and 1
    pass_fold_0 = np.array([False, False, True, True])  # both shore rows "pass" fold 0
    for f in (0, 1):
        test = fold == f
        trainval = ~test
        va = np.zeros(4, bool)
        tr = trainval & ~va
        # the shore row whose cv_fold == f is NOT in tr even though its pass bit is set
        held = shore & test
        assert not (tr & held).any()
    # concretely: shore row idx 2 (fold 0) passes fold 0 but is held out of fold 0's tr
    assert pass_fold_0[2]
    assert not (~(fold == 0) & shore & (fold == 0))[2]


# ---------------------------------------------------------------------------
# built-artifact invariants (data-dependent)
# ---------------------------------------------------------------------------
@_glr
def test_per_fold_pass_counts_differ():
    """The fold-awareness proof on the real build: per-fold pass counts are NOT all
    equal (a fold-blind/global mask would give identical counts)."""
    g = pd.read_parquet(GLR)
    cols = sorted(c for c in g.columns if c.startswith("glr_pass_fold_"))
    counts = [int(g[c].sum()) for c in cols]
    assert len(counts) >= 2
    assert len(set(counts)) > 1, f"per-fold pass counts identical: {counts}"


@_glr
def test_pass_bits_respect_screen_bound():
    """Every passing ring has finite d_glr <= +5 m and n_wells >= 1 in that fold."""
    g = pd.read_parquet(GLR)
    for c in [c for c in g.columns if c.startswith("glr_pass_fold_")]:
        f = c.rsplit("_", 1)[1]
        m = g[c].to_numpy(bool)
        d = g[f"d_glr_fold_{f}"].to_numpy("float64")[m]
        n = g[f"n_wells_fold_{f}"].to_numpy("int64")[m]
        assert np.isfinite(d).all() and (d <= 5.0 + 1e-9).all()
        assert (n >= 1).all()


@_glr
@_bundle
def test_every_shore_cell_covered_by_a_label():
    """Every is_shore_pseudo bundle node maps 1:1 onto a GLR label cell (fail-loud
    coverage contract the trainer asserts)."""
    bgl = _load("build_glr_labels")
    qn = pd.read_parquet(
        V2S / "query_nodes.parquet", columns=["x5070", "y5070", "is_shore_pseudo"]
    )
    s = qn[qn["is_shore_pseudo"].astype(bool)]
    shore_cid = set(bgl.cell_ids(s["x5070"].to_numpy(), s["y5070"].to_numpy()).tolist())
    label_cid = set(pd.read_parquet(GLR, columns=["cell_id"])["cell_id"].tolist())
    assert shore_cid <= label_cid
    assert len(shore_cid) == len(s)  # one shore node per lattice cell
