"""Toy base model (softmax-gated four-expert mixture) and source assimilation.

Run as::

    uv run python -m utils.figures.toy_model    # fit, render, assimilate, print metrics

The base model is a fitted gated mixture, not a graph network. That is deliberate
(plan section 1.4): the figures are about what the outputs look like, and this
emits every layer the production renderer emits.

Everything is solved in **WTE-residual space over R**: the target at a point is
``y = observed WTE - R``, and the model predicts a small residual that is added
back to R. The four experts are

===== =========================== =================================
index expert                      value (a WTE residual over R)
===== =========================== =================================
0     FAC-REM                     ``fac_water_surface - R``
1     deep prior                  ``deep_wte - R``
2     3 m mirror                  ``(DEM - 3) - R``
3     free head                   ``h . features`` (fitted linear)
===== =========================== =================================

mixed by ``g = softmax(W . features)`` with features
``[dist_to_stream, slope, HAND, log1p(flow_acc), 1]`` (the first four
standardised on grid statistics, which use no labels). A Laplace scale
``b = exp(c . [features, |expert disagreement|])`` is fitted in a second stage
on the residuals of the mean fit, so the model reports an uncertainty that is
allowed to depend on how much the experts argue. Fitting minimises the Laplace
negative log-likelihood on the training folds with the permanent-water
pseudo-observations carried at weight 0.25; only real wells enter any metric.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.figures.fig_common import CELL, NX, NY, TOY_DIR  # noqa: E402
from utils.figures.toy_priors import (  # noqa: E402
    Z_SCALE,
    build_all,
    training_pool,
    water_table_wells,
)
from utils.figures.toy_world import HOLDOUT_REGION, N_FOLDS, load_world  # noqa: E402

TOY_STACK_PKL = TOY_DIR / "toy_stack.pkl"

N_EXPERTS = 4
EXPERT_NAMES = ("fac", "deep", "mirror", "head")
FEATURE_NAMES = ("dist_to_stream", "slope", "hand", "log1p_flow_acc", "bias")
N_FEAT = len(FEATURE_NAMES)
N_PARAMS = (N_EXPERTS - 1) * N_FEAT + N_FEAT + (N_FEAT + 1)

#: Ridge penalty on the gate logit weights. Keeps the gate a blend, not a mask.
#: It has to be small: with the smoothed gate features doing most of the work
#: against saturation, a large value flattens the gate to a uniform 0.25 and the
#: regime story disappears.
GATE_L2 = 0.5

#: Ridge penalty on the free-head *slope* coefficients (its intercept is free).
#: Unpenalised, the head is a second unconstrained regression on the same features
#: and absorbs most of the weight, leaving no regime story; pinned to an intercept
#: it is what it should be -- a fitted offset from R -- and the physical experts
#: have to explain the structure. It also fits better here.
HEAD_L2 = 10000.0

#: Constant Laplace scale (m) held during the mean-fit stage of :func:`_fit_one`.
#: It only sets the weight of the data term against the two ridges above, which
#: are calibrated to it; the reported ``b`` comes from the second stage.
MEAN_FIT_SCALE_M = float(np.exp(6.0))

#: The gate features are smoothed by this length before use (m). The gate is a
#: statement about *regime*, so it must not switch on a single rough cell.
GATE_SMOOTH_M = 300.0


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def gate_grids(world: dict) -> list[np.ndarray]:
    """The four gate-feature grids, smoothed to :data:`GATE_SMOOTH_M`.

    Order matches the first four entries of :data:`FEATURE_NAMES`. Smoothing is
    what makes the fitted gate render as continuous regime fields instead of a
    mesh of hairlines following every rough cell.
    """
    sigma = GATE_SMOOTH_M / CELL
    raw = [
        np.asarray(world["dist_to_stream"], float),
        np.asarray(world["slope"], float),
        np.asarray(world["hand"], float),
        np.log1p(np.asarray(world["flow_acc"], float)),
    ]
    return [ndimage.gaussian_filter(a, sigma, mode="nearest") for a in raw]


def feature_stats(world: dict) -> dict:
    """Standardisation statistics for the gate features, from the grid alone."""
    raw = np.stack([a.ravel() for a in gate_grids(world)])
    return {"mean": raw.mean(axis=1), "std": raw.std(axis=1)}


def _features(world, stats, rows=None, cols=None):
    """Feature matrix (n, N_FEAT). Whole grid when ``rows`` is None."""
    src = gate_grids(world)
    if rows is None:
        cols4 = np.stack([a.ravel() for a in src], axis=1)
    else:
        cols4 = np.stack([a[rows, cols] for a in src], axis=1)
    z = (cols4 - stats["mean"]) / stats["std"]
    return np.column_stack([z, np.ones(z.shape[0])])


# --------------------------------------------------------------------------
# Model algebra
# --------------------------------------------------------------------------


def _unpack(theta):
    i = 0
    w = np.zeros((N_EXPERTS, N_FEAT))
    w[1:] = theta[i : i + (N_EXPERTS - 1) * N_FEAT].reshape(N_EXPERTS - 1, N_FEAT)
    i += (N_EXPERTS - 1) * N_FEAT
    h = theta[i : i + N_FEAT]
    i += N_FEAT
    c = theta[i : i + N_FEAT + 1]
    return w, h, c


def _predict(theta, feats, experts3):
    """Return (residual prediction, Laplace scale b, gate weights, expert values).

    ``experts3`` is (n, 3): the FAC, deep and mirror residuals over R.
    """
    w, h, c = _unpack(theta)
    head = feats @ h
    e = np.column_stack([experts3, head])
    logits = feats @ w.T
    logits -= logits.max(axis=1, keepdims=True)
    g = np.exp(logits)
    g /= g.sum(axis=1, keepdims=True)
    pred = (g * e).sum(axis=1)
    disagree = e.std(axis=1)
    lb = np.column_stack([feats, disagree]) @ c
    b = np.exp(np.clip(lb, -3.0, 6.0))
    return pred, b, g, e


def _nll(theta, feats, experts3, y, weight, gate_l2, head_l2):
    """Weighted Laplace NLL plus ridge penalties on the gate and the free head.

    Without the gate penalty the gate saturates into a hard mask (weights 0 or 1
    with a one-cell transition) -- it fits marginally better and is useless to look
    at, and it makes the mixture brittle wherever a feature crosses the switch.
    Without the head penalty the unconstrained head absorbs most of the weight and
    the physical experts never get to speak.
    """
    pred, b, _, _ = _predict(theta, feats, experts3)
    w, h, _ = _unpack(theta)
    return float(
        np.sum(weight * (np.abs(y - pred) / b + np.log(2.0 * b)))
        + gate_l2 * np.sum(w[1:] ** 2)
        + head_l2 * np.sum(h[:-1] ** 2)
    )


def _fit_one(feats, experts3, y, weight, seed=0, gate_l2=None, head_l2=None):
    """Two-stage fit: the mean under a fixed reference scale, then the scale.

    Stage 1 holds ``log b`` at ``log(MEAN_FIT_SCALE_M)`` everywhere and fits the
    gate and the free head. With a constant scale the Laplace NLL is a weighted
    L1 loss divided by that constant, so :data:`GATE_L2` and :data:`HEAD_L2` are
    ridge strengths *relative to* ``MEAN_FIT_SCALE_M`` (that is how they were
    calibrated). Stage 2 holds the mean fixed and fits the scale coefficients
    ``c`` on the resulting residuals. Fitting both jointly from a cold start
    collapses: the scale shrinks wherever the mean fits and the gate degenerates
    to the free head (OOF MAD 4.75 -> 5.09 m, coverage of the +-b band 0.30).
    """
    gate_l2 = GATE_L2 if gate_l2 is None else gate_l2
    head_l2 = HEAD_L2 if head_l2 is None else head_l2
    rng = np.random.default_rng(seed)
    n_mean = N_PARAMS - (N_FEAT + 1)
    c_fixed = np.zeros(N_FEAT + 1)
    c_fixed[N_FEAT - 1] = np.log(MEAN_FIT_SCALE_M)

    def nll_mean(t):
        return _nll(
            np.concatenate([t, c_fixed]), feats, experts3, y, weight, gate_l2, head_l2
        )

    best = None
    for attempt in range(2):
        t0 = np.zeros(n_mean)
        if attempt:
            t0 += 0.1 * rng.standard_normal(n_mean)
        res = minimize(
            nll_mean, t0, method="L-BFGS-B", options={"maxiter": 800, "maxfun": 40000}
        )
        if best is None or res.fun < best.fun:
            best = res
    if not np.isfinite(best.fun):
        raise RuntimeError(
            "Laplace NLL (mean stage) did not converge to a finite value"
        )
    mean_theta = best.x

    pred, _, _, e = _predict(np.concatenate([mean_theta, c_fixed]), feats, experts3)
    resid = np.abs(y - pred)
    design = np.column_stack([feats, e.std(axis=1)])

    def nll_scale(c):
        lb = np.clip(design @ c, -3.0, 6.0)
        b = np.exp(lb)
        return float(np.sum(weight * (resid / b + lb)))

    c0 = np.zeros(N_FEAT + 1)
    c0[N_FEAT - 1] = np.log(max(np.median(resid), 0.5))
    res = minimize(nll_scale, c0, method="L-BFGS-B", options={"maxiter": 800})
    if not np.isfinite(res.fun):
        raise RuntimeError(
            "Laplace NLL (scale stage) did not converge to a finite value"
        )
    return np.concatenate([mean_theta, res.x])


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------


def fit_base_model(
    wells: pd.DataFrame,
    world: dict,
    priors: dict,
    folds=None,
    pseudo_weight: float = 0.25,
    gate_l2: float | None = None,
    head_l2: float | None = None,
):
    """Fit one gated mixture per fold and return ``(params_by_fold, oof_table)``.

    ``folds`` defaults to ``range(N_FOLDS)``. Fold ``k`` trains on every
    water-table well in folds 0..5 except ``k``, using that fold's cross-fit R and
    fold-pure deep prior, plus the permanent-water pseudo-observations at
    ``pseudo_weight``. The out-of-fold table carries, for every water-table well,
    ``pred_wte``, ``pred_dtw``, ``b`` and the four gate weights, predicted by the
    model that never saw it. Buffered-holdout wells (fold ``HOLDOUT_REGION``) are
    predicted by the median of the six fold models.
    """
    folds = list(range(N_FOLDS)) if folds is None else list(folds)
    stats = feature_stats(world)
    dem = np.asarray(world["dem"], float)
    pseudo = priors["pseudo"]

    params_by_fold: dict[int, np.ndarray] = {}
    for fold in folds:
        r = priors["r_crossfit"][fold]
        deep = priors["deep"][fold]
        pool = training_pool(wells, exclude_fold=fold)
        rows, cols = pool["row"].to_numpy(int), pool["col"].to_numpy(int)

        prow = pseudo["row"].to_numpy(int)
        pcol = pseudo["col"].to_numpy(int)
        allr = np.concatenate([rows, prow])
        allc = np.concatenate([cols, pcol])

        feats = _features(world, stats, allr, allc)
        rv = r[allr, allc]
        experts3 = np.column_stack(
            [
                priors["fac_ws"][allr, allc] - rv,
                deep[allr, allc] - rv,
                priors["mirror"][allr, allc] - rv,
            ]
        )
        y = (
            np.concatenate(
                [pool["wte_obs"].to_numpy(float), pseudo["wte_obs"].to_numpy(float)]
            )
            - rv
        )
        weight = np.concatenate([np.ones(len(pool)), pseudo["weight"].to_numpy(float)])
        params_by_fold[fold] = _fit_one(
            feats, experts3, y, weight, seed=fold, gate_l2=gate_l2, head_l2=head_l2
        )

    wt = water_table_wells(wells)
    rows, cols = wt["row"].to_numpy(int), wt["col"].to_numpy(int)
    wfold = wt["fold"].to_numpy(int)
    feats = _features(world, stats, rows, cols)

    per_fold = {}
    for fold in folds:
        r = priors["r_crossfit"][fold]
        deep = priors["deep"][fold]
        rv = r[rows, cols]
        experts3 = np.column_stack(
            [
                priors["fac_ws"][rows, cols] - rv,
                deep[rows, cols] - rv,
                priors["mirror"][rows, cols] - rv,
            ]
        )
        pred, b, g, _ = _predict(params_by_fold[fold], feats, experts3)
        per_fold[fold] = (rv + pred, b, g)

    wte = np.empty(len(wt))
    bb = np.empty(len(wt))
    gg = np.empty((len(wt), N_EXPERTS))
    for fold in folds:
        m = wfold == fold
        if m.any():
            wte[m], bb[m], gg[m] = (
                per_fold[fold][0][m],
                per_fold[fold][1][m],
                per_fold[fold][2][m],
            )
    m = wfold == HOLDOUT_REGION
    if m.any():
        wte[m] = np.median(np.stack([per_fold[f][0] for f in folds]), axis=0)[m]
        bb[m] = np.median(np.stack([per_fold[f][1] for f in folds]), axis=0)[m]
        gg[m] = np.median(np.stack([per_fold[f][2] for f in folds]), axis=0)[m]

    oof = wt[
        [
            "row",
            "col",
            "x",
            "y",
            "dem",
            "fold",
            "dtw_obs",
            "wte_obs",
            "is_source",
            "is_holdout_buffered",
        ]
    ].copy()
    oof["pred_wte"] = wte
    oof["pred_dtw"] = np.asarray(dem)[rows, cols] - wte
    oof["b"] = bb
    for j, name in enumerate(EXPERT_NAMES):
        oof[f"gate_{name}"] = gg[:, j]
    return params_by_fold, oof


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_base(params_by_fold: dict, world: dict, priors: dict) -> dict:
    """Fold-median field render. Keys match the production renderer's layer names.

    Returns ``wte``, ``dtw``, ``head_wte``, ``deep_wte``, ``r_wte``,
    ``fold_spread``, ``sigma`` (the Laplace scale b) and ``gate_w`` (shape
    ``(4, NY, NX)``, ordered fac, deep, mirror, head).

    The render uses the **inference R** (the interpolation of the archived
    cross-fit values), not any single fold's cross-fit field: that is the
    production arrangement, and it is what makes the mapped surface the same
    surface the model was trained against.
    """
    stats = feature_stats(world)
    dem = np.asarray(world["dem"], float)
    feats = _features(world, stats)
    r = np.asarray(priors["r_inference"], float)
    rv = r.ravel()

    folds = sorted(params_by_fold)
    wte_f, b_f, g_f, head_f, deep_f = [], [], [], [], []
    for fold in folds:
        deep = np.asarray(priors["deep"][fold], float)
        experts3 = np.column_stack(
            [
                priors["fac_ws"].ravel() - rv,
                deep.ravel() - rv,
                priors["mirror"].ravel() - rv,
            ]
        )
        pred, b, g, e = _predict(params_by_fold[fold], feats, experts3)
        wte_f.append(rv + pred)
        b_f.append(b)
        g_f.append(g)
        head_f.append(rv + e[:, 3])
        deep_f.append(deep.ravel())

    wte_s = np.stack(wte_f)
    return {
        "wte": np.median(wte_s, axis=0).reshape(NY, NX),
        "dtw": dem - np.median(wte_s, axis=0).reshape(NY, NX),
        "head_wte": np.median(np.stack(head_f), axis=0).reshape(NY, NX),
        "deep_wte": np.median(np.stack(deep_f), axis=0).reshape(NY, NX),
        "r_wte": r,
        "fold_spread": wte_s.std(axis=0).reshape(NY, NX),
        "sigma": np.median(np.stack(b_f), axis=0).reshape(NY, NX),
        "gate_w": np.median(np.stack(g_f), axis=0).T.reshape(N_EXPERTS, NY, NX),
    }


# --------------------------------------------------------------------------
# Source-well assimilation
# --------------------------------------------------------------------------


def _apply_correction(base_wte, world, sx, sy, sz, sres, k, tau, length, z_scale):
    """kNN softmax-attention residual correction on a grid. Returns (wte, d_nearest)."""
    tree = cKDTree(np.column_stack([sx, sy, z_scale * sz]))
    q = np.column_stack(
        [
            world["X"].ravel(),
            world["Y"].ravel(),
            z_scale * np.asarray(world["dem"], float).ravel(),
        ]
    )
    kk = int(min(k, sx.size))
    d, idx = tree.query(q, k=kk)
    if kk == 1:
        d, idx = d[:, None], idx[:, None]
    a = -d / tau
    a -= a.max(axis=1, keepdims=True)
    wts = np.exp(a)
    wts /= wts.sum(axis=1, keepdims=True)
    corr = (wts * sres[idx]).sum(axis=1)

    ptree = cKDTree(np.column_stack([sx, sy]))
    dn, _ = ptree.query(q[:, :2], k=1)
    corr *= np.exp(-dn / length)
    return (base_wte.ravel() + corr).reshape(NY, NX), dn.reshape(NY, NX)


def assimilate(
    base_fields: dict,
    wells: pd.DataFrame,
    world: dict,
    sources_mask,
    k: int = 16,
    tau: float | None = None,
    z_scale: float = Z_SCALE,
    length: float | None = None,
    fit_radius_m: float | None = None,
):
    """Pull the base surface onto visible source wells.

    ``sources_mask`` is a boolean array aligned with ``wells``: the observations
    the assimilated surface is allowed to see. For each grid cell the k nearest
    visible sources in the relief-lifted metric vote with weights
    ``softmax(-distance / tau)`` on their own WTE residual against the base
    prediction; the weighted residual is added to the base WTE and attenuated by
    ``exp(-nearest-source distance / length)`` so the far field returns to the
    base model exactly.

    ``tau`` (m, softmax temperature) and ``length`` (m, attenuation scale) are
    fitted when either is None, on the training-fold wells that are *not* sources.
    ``fit_radius_m`` restricts that fit to validation wells within a given distance
    of a source; the default (None, every validation well) is what balances the
    near-field gain against the far-field harm of over-long attenuation, and a
    finite radius makes the fit near-field-greedy.

    Returns ``{"wte", "dtw", "nearest_source_dist", "tau", "length"}``; the first
    three are ``(NY, NX)`` arrays in m.
    """
    mask = np.asarray(sources_mask, bool)
    if mask.shape[0] != len(wells):
        raise ValueError("sources_mask must align with wells")
    dem = np.asarray(world["dem"], float)
    base = np.asarray(base_fields["wte"], float)

    src = wells.loc[mask & ~wells["is_confined_flag"].to_numpy(bool)]
    if len(src) == 0:
        out = base.copy()
        big = np.full((NY, NX), np.inf)
        return {
            "wte": out,
            "dtw": dem - out,
            "nearest_source_dist": big,
            "tau": np.nan,
            "length": np.nan,
        }

    sr, sc = src["row"].to_numpy(int), src["col"].to_numpy(int)
    sx, sy = src["x"].to_numpy(float), src["y"].to_numpy(float)
    sz = src["dem"].to_numpy(float)
    sres = src["wte_obs"].to_numpy(float) - base[sr, sc]

    val = training_pool(wells)
    val = val.loc[~mask[wells.index.get_indexer(val.index)]]
    vr, vc = val["row"].to_numpy(int), val["col"].to_numpy(int)
    vobs = val["dtw_obs"].to_numpy(float)

    if (tau is None or length is None) and fit_radius_m is not None:
        vd, _ = cKDTree(np.column_stack([sx, sy])).query(
            np.column_stack([val["x"].to_numpy(float), val["y"].to_numpy(float)]), k=1
        )
        near = vd <= float(fit_radius_m)
        if near.sum() < 20:
            raise ValueError(
                f"only {int(near.sum())} validation wells within {fit_radius_m:g} m of a source"
            )
        vr, vc, vobs = vr[near], vc[near], vobs[near]

    if tau is None or length is None:
        grid = []
        for t in (500.0, 1000.0, 2000.0, 4000.0, 8000.0):
            for lscale in (1000.0, 2000.0, 3000.0, 5000.0):
                w_, _ = _apply_correction(
                    base, world, sx, sy, sz, sres, k, t, lscale, z_scale
                )
                mad = float(np.median(np.abs((dem[vr, vc] - w_[vr, vc]) - vobs)))
                grid.append((mad, t, lscale))
        # Parsimony rule: among settings within 1 % of the best validation MAD, take
        # the shortest attenuation length. The MAD surface is nearly flat in
        # ``length`` once it is long enough to cover the source network, and the
        # unpenalised argmin drifts to the top of the search grid, which smears the
        # correction across the far field for no measurable skill.
        floor = min(g[0] for g in grid) * 1.01
        tau, length = min(
            ((t, ls) for mad, t, ls in grid if mad <= floor), key=lambda p: (p[1], p[0])
        )

    wte, dn = _apply_correction(base, world, sx, sy, sz, sres, k, tau, length, z_scale)
    return {
        "wte": wte,
        "dtw": dem - wte,
        "nearest_source_dist": dn,
        "tau": tau,
        "length": length,
    }


def density_curve(
    base_fields: dict,
    wells: pd.DataFrame,
    world: dict,
    fractions=(0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9),
    seed: int = 7,
    k: int = 16,
    tau: float = 2000.0,
    length: float = 3000.0,
    n_draws: int = 40,
    near_m: float | None = None,
) -> pd.DataFrame:
    """Held-out MAD as a function of the fraction of source wells made visible.

    The candidate sources are the wells flagged ``is_source``; for each fraction
    ``n_draws`` independent seeded subsets are revealed and the scores averaged --
    a single draw is noisy enough at small fractions to invert the curve. Scoring
    is always on the *same* held-out set -- the water-table wells that are not
    candidate sources and lie within ``near_m`` of one that could be -- so the
    curve is not confounded by a changing evaluation population, and is not
    flattened by wells no source density could ever reach. ``near_m`` defaults to
    the attenuation scale ``length``, which is exactly the radius within which
    assimilation can move anything; pass a number to override it, or
    ``near_m=0`` to score every held-out well.

    ``tau`` and ``length`` (m) should be the values ``assimilate`` fitted for the
    full source set, so the curve isolates the effect of source density alone.

    Returns a DataFrame with ``fraction``, ``n_sources``, ``mad_m``, ``mad_sd_m``,
    ``rmse_m`` and ``n_wells``; ``mad_sd_m`` is the standard deviation of MAD
    across the draws (0 at fraction 0, which is deterministic).
    """
    near_m = float(length) if near_m is None else float(near_m)
    rng = np.random.default_rng(seed)
    cand = np.flatnonzero(
        wells["is_source"].to_numpy(bool) & ~wells["is_confined_flag"].to_numpy(bool)
    )

    held = water_table_wells(wells)
    held = held.loc[~held["is_source"].to_numpy(bool)]
    if near_m > 0:
        cx = wells["x"].to_numpy(float)[cand]
        cy = wells["y"].to_numpy(float)[cand]
        d, _ = cKDTree(np.column_stack([cx, cy])).query(
            np.column_stack([held["x"].to_numpy(float), held["y"].to_numpy(float)]), k=1
        )
        held = held.loc[d <= float(near_m)]
    hr, hc = held["row"].to_numpy(int), held["col"].to_numpy(int)
    hobs = held["dtw_obs"].to_numpy(float)

    rows = []
    for p in fractions:
        n_src = int(round(p * cand.size))
        mads, rmses = [], []
        draws = 1 if n_src == 0 else n_draws
        for _ in range(draws):
            if n_src == 0:
                dtw = np.asarray(base_fields["dtw"], float)
            else:
                mask = np.zeros(len(wells), bool)
                mask[rng.permutation(cand)[:n_src]] = True
                dtw = assimilate(
                    base_fields, wells, world, mask, k=k, tau=tau, length=length
                )["dtw"]
            s = score(dtw[hr, hc], hobs)
            mads.append(s["mad_m"])
            rmses.append(s["rmse_m"])
        rows.append(
            {
                "fraction": p,
                "n_sources": n_src,
                "mad_m": float(np.mean(mads)),
                "mad_sd_m": float(np.std(mads)),
                "rmse_m": float(np.mean(rmses)),
                "n_wells": int(hobs.size),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score(pred_dtw_at_wells, obs_dtw) -> dict:
    """Metric panel for a set of DTW predictions. All values in metres.

    ``residual = predicted DTW - observed DTW`` (positive = predicted too deep).
    Reports MAD (median |residual|), the median residual and the mean bias side by
    side -- when the bias greatly exceeds the median residual the error is a
    skewed tail, not an offset -- plus RMSE and n.
    """
    pred = np.asarray(pred_dtw_at_wells, float)
    obs = np.asarray(obs_dtw, float)
    if pred.shape != obs.shape:
        raise ValueError("prediction and observation arrays must match")
    if not np.isfinite(pred).all():
        raise ValueError(
            "non-finite predictions reached score(); find the cause upstream"
        )
    r = pred - obs
    return {
        "mad_m": float(np.median(np.abs(r))),
        "median_resid_m": float(np.median(r)),
        "bias_m": float(r.mean()),
        "rmse_m": float(np.sqrt((r**2).mean())),
        "n": int(r.size),
    }


def load_stack(force: bool = False) -> dict:
    """Build (or load) the whole toy stack once and cache it.

    Returns ``{"world", "wells", "priors", "params", "oof", "fields"}``. Fitting
    the six fold models takes about a minute, so every figure script should call
    this rather than :func:`fit_base_model` directly. Cached to
    ``/data/ssd2/handily/figures/toy/toy_stack.pkl``; delete that file or pass
    ``force=True`` to refit.
    """
    if not force and TOY_STACK_PKL.exists():
        with open(TOY_STACK_PKL, "rb") as fh:
            return pickle.load(fh)
    world, wells = load_world()
    priors = build_all(wells, world)
    params, oof = fit_base_model(wells, world, priors)
    fields = render_base(params, world, priors)
    stack = {
        "world": world,
        "wells": wells,
        "priors": priors,
        "params": params,
        "oof": oof,
        "fields": fields,
    }
    TOY_STACK_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(TOY_STACK_PKL, "wb") as fh:
        pickle.dump(stack, fh)
    print(f"wrote {TOY_STACK_PKL}")
    return stack


def _main() -> None:
    stack = load_stack(force=True)
    world, wells = stack["world"], stack["wells"]
    oof, fields = stack["oof"], stack["fields"]

    print("base model, out-of-fold at water-table wells:")
    print("  ", score(oof["pred_dtw"].to_numpy(), oof["dtw_obs"].to_numpy()))
    print(
        "  gate means:",
        {n: round(float(oof[f"gate_{n}"].mean()), 3) for n in EXPERT_NAMES},
    )

    d = world["dist_to_stream"]
    g = fields["gate_w"]
    near = d < 300
    far = (d > 2000) | np.asarray(world["playa_basin"], bool)
    print(
        f"  gate FAC near channels {g[0][near].mean():.3f} vs upland/playa {g[0][far].mean():.3f}"
    )
    print(
        f"  gate deep near channels {g[1][near].mean():.3f} vs upland/playa {g[1][far].mean():.3f}"
    )

    truth = world["dtw_true"]
    print(
        f"  field DTW vs truth: MAD {np.median(np.abs(fields['dtw'] - truth)):.2f} m over the whole grid"
    )

    src = wells["is_source"].to_numpy(bool) & ~wells["is_confined_flag"].to_numpy(bool)
    asm = assimilate(fields, wells, world, src)
    print(f"assimilation: tau = {asm['tau']:.0f} m, length = {asm['length']:.0f} m")

    held = water_table_wells(wells)
    held = held.loc[~held["is_source"].to_numpy(bool)]
    hr, hc = held["row"].to_numpy(int), held["col"].to_numpy(int)
    hobs = held["dtw_obs"].to_numpy(float)
    print("  held-out base:      ", score(np.asarray(fields["dtw"])[hr, hc], hobs))
    print("  held-out assimilated:", score(asm["dtw"][hr, hc], hobs))
    hd = asm["nearest_source_dist"][hr, hc]
    print("  held-out MAD (m) by distance to nearest source:")
    for lo, hi in [(0, 1000), (1000, 2000), (2000, 4000), (4000, 1e9)]:
        m = (hd >= lo) & (hd < hi)
        if m.any():
            b = score(np.asarray(fields["dtw"])[hr, hc][m], hobs[m])
            q = score(asm["dtw"][hr, hc][m], hobs[m])
            print(
                f"    {lo:>5.0f}-{hi:<8.0f} n={m.sum():>3d}  base {b['mad_m']:6.2f} -> assimilated {q['mad_m']:6.2f}"
                f"   (RMSE {b['rmse_m']:6.2f} -> {q['rmse_m']:6.2f})"
            )

    dn = asm["nearest_source_dist"]
    diff = asm["dtw"] - fields["dtw"]
    for lo, hi in [(0, 500), (500, 1500), (1500, 3000), (3000, 6000), (6000, 1e9)]:
        m = (dn >= lo) & (dn < hi)
        if m.any():
            print(
                f"  |difference| {lo:>5.0f}-{hi:<8.0f} m from a source: median {np.median(np.abs(diff[m])):.3f} m"
            )

    curve = density_curve(fields, wells, world, tau=asm["tau"], length=asm["length"])
    print(curve.to_string(index=False))


if __name__ == "__main__":
    _main()
