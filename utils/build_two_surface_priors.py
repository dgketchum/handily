"""WP2 privileged assignment-prior table — per-well P(observation sampled the
PHREATIC surface) for the handily v0.2 two-surface (phreatic vs regional-aquifer)
GNN mixture model.

The two-surface feasibility gate (utils/v02_two_surface_feasibility.py) showed the
collocated vertical dispersion is IDENTIFIABLE but that construction metadata
predicts component membership only as a LOCAL, cell-relative signal (OOF AUC 0.71
cell-relative vs ~0.55 absolute; global corr(completion_depth, DTW) ~ 0.05). This
script turns that into a per-well prior a_i = P(well i sampled the phreatic /
water-table surface), used ONLY in the training loss responsibilities of the GNN
mixture (never as a query feature).

HARD RULE — the prior is UNSUPERVISED with respect to water level. No parameter of
the prior is estimated from observed DTW/WTE. Evidence is purely local / geometric
/ categorical. (Completion depth correlates with depth-to-water by physics — that
is the intended semantics — but nothing here is *fit* to water level.)

Two variants are emitted side by side so the trainer can choose:

  * ``p_phreatic_construction`` — construction-only. Local completion-depth
    evidence + confinement evidence + well role. Contains NO water-level-derived
    quantity.
  * ``p_phreatic_geom`` — additionally blends screen/head geometry
    (``head_above_screen``). ``head_above_screen`` is a partial water-level proxy
    (head height above the screen top), so it is acceptable for a privileged
    prior but kept in a SEPARATE column.

--------------------------------------------------------------------------------
Evidence and the log-odds combination
--------------------------------------------------------------------------------
Let ``logit(p)=log(p/(1-p))`` and ``sigmoid`` its inverse.

1. LOCAL DEPTH-RANK PRIOR (the identifiable signal; neighborhood-relative).
   For each monitoring well, gather every GWX well with a plausible completion
   depth (0 < comp <= 3000 m) within 2 km in EPSG:5070 (fall back to 10 km if the
   2 km neighborhood has < 5 wells; CONUS-neutral 0.5 if the 10 km neighborhood
   still has < 5). The target's completion-depth quantile in that neighborhood is

       q = ( #(neighbor_comp < target_comp) + 0.5 * #(neighbor_comp == target_comp) ) / n

   (mid-rank empirical quantile, dimensionless in [0, 1]; the target's own GWX
   record sits in the pool at distance ~0 and contributes to the == term).
   Shallowest-in-neighborhood (q -> 0) is phreatic-leaning:

       p_rank = clip(1 - q, 0.05, 0.95)          L_rank = logit(p_rank)

   No usable neighborhood or no target completion depth => p_rank = 0.5 (L_rank=0).

2. CONFINEMENT EVIDENCE (log-odds shift + hard bound). Taken from the v0.2
   contract's own ``confinement_class`` (the authoritative screening for these
   wells — the contract population is entirely unconfined / unconfined_marginal;
   the GWX 8-hex-prefix join is ambiguous and must NOT override this
   safety-critical field). Additive shift ``beta_conf``:

       confined / likely_confined / artesian : -3.0  AND cap p_phreatic <= 0.10
       unconfined                            : +1.0  AND floor p_phreatic >= 0.50
                                                       UNLESS depth-rank says deep
                                                       (p_rank < 0.5)
       unconfined_marginal                   :  0.0  (neutral weight)
       unknown / missing                     :  0.0

3. WELL ROLE (log-odds shift). Taken from the GWX-matched record's
   ``well_class`` (the contract labels every well 'monitoring', so it carries no
   discriminating signal; GWX gives the real role). Additive shift ``beta_role``:

       monitoring          : +0.5   (screened at the water table -> phreatic)
       pumping / production : -0.5   (deeper productive screen -> regional)
       unknown / missing    :  0.0

4. HEAD GEOMETRY (``p_phreatic_geom`` only). ``head_above_screen`` h (m), from the
   GWX-matched record. A water-table well is screened across the table so h is
   small; a pressurized/deep well stands with head far above its screen so h is
   large. Additive shift

       beta_head = clip( -(h - 5.0) / 15.0, -2.0, +2.0 )   (0.0 if h missing)

Combination (both variants; ``beta_head`` = 0 for the construction variant):

       L = L_rank + beta_conf + beta_role [+ beta_head]
       p = clip(sigmoid(L), 0.05, 0.95)
       if confined-type:            p = min(p, 0.10)     # hard cap
       elif unconfined and not deep: p = max(p, 0.50)    # floor

Water-pseudo rows (open-water monitoring pseudo-labels) are phreatic by
construction and get p = 0.98 in both variants.

Sources per field (documented so the join ambiguity is auditable):
  * confinement_class  <- v0.2 contract (authoritative screening).
  * completion depth   <- GWX matched record (contract's well_depth is 100% null).
  * head_above_screen  <- GWX matched record.
  * well_class (role)  <- GWX matched record.

The join: contract ``canonical_id`` is 'gwx_' + 8 hex (a PREFIX of the GWX table's
'gwx_' + 16 hex id). Match on the 12-char prefix; where a prefix matches >1 GWX row
(a collision), pick deterministically: has-construction-data, then max obs_count,
then lexicographic canonical_id. ~11% of contract wells have no GWX match; they get
the neutral 0.5 prior and gwx_matched=False.

Usage:
    uv run python utils/build_two_surface_priors.py
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from scipy.spatial import cKDTree

log = logging.getLogger("build_two_surface_priors")

WELLS = "/data/ssd2/gwx/products/current/wells.geoparquet"
PANELS = "/data/ssd2/handily/conus/wte_gnn/v02/contract/wells_panels.parquet"
OUT_DIR = "/data/ssd2/handily/conus/wte_gnn/v02/wp2"

# --- constants (all part of the documented prior; none fit to water level) ---
COMP_MIN_M = 0.0  # exclusive: completion depth must be > 0
COMP_MAX_M = 3000.0  # inclusive: reject physically-impossible deeper "depths"
RADII_M = (2000.0, 10000.0)  # 2 km then 10 km fallback
MIN_NEIGHBORS = 5  # neighborhood floor before CONUS-neutral 0.5
PREFIX_LEN = 12  # 'gwx_' + 8 hex

P_LO, P_HI = 0.05, 0.95  # depth-rank / combined prior clip
CONFINED_CAP = 0.10  # hard upper bound for confined-type wells
UNCONF_FLOOR = 0.50  # lower bound for unconfined (shallow/mid) wells
WATER_PSEUDO_P = 0.98  # open-water pseudo-label prior (both variants)

BETA_CONF = {
    "confined": -3.0,
    "likely_confined": -3.0,
    "artesian": -3.0,
    "unconfined": 1.0,
    "unconfined_marginal": 0.0,
    "unknown": 0.0,
}
CONFINED_TYPES = ("confined", "likely_confined", "artesian")
BETA_ROLE = {
    "monitoring": 0.5,
    "pumping": -0.5,
    "production": -0.5,
    "unknown": 0.0,
}
HEAD_REF_M = 5.0
HEAD_SCALE_M = 15.0
BETA_HEAD_CLIP = 2.0

SACRIFICIAL_HUC4 = ("0707", "1019", "1605")
DEPTH_BANDS = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 30.0), (30.0, np.inf)]


# --------------------------------------------------------------------------- #
# pure math (unit-testable)
# --------------------------------------------------------------------------- #
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def p_rank_from_quantile(q: np.ndarray) -> np.ndarray:
    """Local depth-rank prior: shallowest-in-neighborhood (q->0) is phreatic.

    p_rank = clip(1 - q, 0.05, 0.95). Monotonically DECREASING in q.
    """
    return np.clip(1.0 - np.asarray(q, float), P_LO, P_HI)


def beta_head_from_h(h: np.ndarray) -> np.ndarray:
    """Head-geometry log-odds shift; 0 where head_above_screen is missing."""
    h = np.asarray(h, float)
    b = np.clip(-(h - HEAD_REF_M) / HEAD_SCALE_M, -BETA_HEAD_CLIP, BETA_HEAD_CLIP)
    return np.where(np.isfinite(h), b, 0.0)


def combine_logodds(p_rank, beta_conf, beta_role, beta_head, is_confined, unconf_floor):
    """Combine evidence in log-odds space, then apply the confined cap / unconfined
    floor. Works elementwise on scalars or arrays.

        L = logit(p_rank) + beta_conf + beta_role + beta_head
        p = clip(sigmoid(L), 0.05, 0.95)
        p = min(p, 0.10)  where is_confined
        p = max(p, 0.50)  where unconf_floor (and not is_confined)

    ``is_confined`` and ``unconf_floor`` are mutually exclusive by construction.
    """
    L = _logit(p_rank) + beta_conf + beta_role + beta_head
    p = np.clip(_sigmoid(L), P_LO, P_HI)
    p = np.where(is_confined, np.minimum(p, CONFINED_CAP), p)
    p = np.where(
        unconf_floor & ~np.asarray(is_confined, bool), np.maximum(p, UNCONF_FLOOR), p
    )
    return p


# --------------------------------------------------------------------------- #
# neighborhood depth rank
# --------------------------------------------------------------------------- #
def neighborhood_quantile(tree, pool_comp, x, y, target_comp, radii, min_neighbors):
    """Mid-rank empirical quantile of ``target_comp`` among GWX completion depths
    within the smallest radius in ``radii`` that yields >= ``min_neighbors``.

    Returns (q, n_neighbors, radius_used). q is NaN and radius NaN when no radius
    reaches the floor (CONUS-neutral fallback) or the target has no completion depth.
    """
    if not np.isfinite(target_comp):
        return np.nan, 0, np.nan
    n_last = 0
    for r in radii:
        idx = tree.query_ball_point([x, y], r)
        n_last = len(idx)
        if n_last >= min_neighbors:
            comps = pool_comp[idx]
            q = (
                np.sum(comps < target_comp) + 0.5 * np.sum(comps == target_comp)
            ) / n_last
            return float(q), int(n_last), float(r)
    return np.nan, int(n_last), np.nan


def _batch_neighborhood(mon_xy, target_comp, tree, pool_comp):
    """Vectorized two-pass ball query + per-well mid-rank quantile."""
    n = len(mon_xy)
    q = np.full(n, np.nan)
    n_nb = np.zeros(n, dtype=np.int64)
    rad = np.full(n, np.nan)

    have_comp = np.isfinite(target_comp)
    # pass 1: 2 km for every well that has a target completion depth
    idx2 = tree.query_ball_point(mon_xy, RADII_M[0], workers=-1)
    need_fallback = []
    for i in range(n):
        if not have_comp[i]:
            continue
        nb = idx2[i]
        n_nb[i] = len(nb)
        if len(nb) >= MIN_NEIGHBORS:
            comps = pool_comp[nb]
            t = target_comp[i]
            q[i] = (np.sum(comps < t) + 0.5 * np.sum(comps == t)) / len(nb)
            rad[i] = RADII_M[0]
        else:
            need_fallback.append(i)

    # pass 2: 10 km only for the wells that missed the floor at 2 km
    if need_fallback:
        fb = np.asarray(need_fallback)
        idx10 = tree.query_ball_point(mon_xy[fb], RADII_M[1], workers=-1)
        for j, i in enumerate(fb):
            nb = idx10[j]
            n_nb[i] = len(nb)
            if len(nb) >= MIN_NEIGHBORS:
                comps = pool_comp[nb]
                t = target_comp[i]
                q[i] = (np.sum(comps < t) + 0.5 * np.sum(comps == t)) / len(nb)
                rad[i] = RADII_M[1]
    return q, n_nb, rad


# --------------------------------------------------------------------------- #
# loaders / join
# --------------------------------------------------------------------------- #
def load_contract(panels_path: str) -> pd.DataFrame:
    df = pq.read_table(
        panels_path,
        columns=[
            "canonical_id",
            "is_water_pseudo",
            "x5070",
            "y5070",
            "mean_dtw",
            "huc4",
            "confinement_class",
            "well_class",
        ],
    ).to_pandas()
    df = df.rename(columns={"confinement_class": "confinement_class_contract"})
    return df


def load_gwx_pool_and_join(wells_path: str, needed_prefixes: set) -> tuple:
    """Return (pool_xy [N,2] EPSG:5070, pool_comp [N], construction join frame,
    n_dropped_implausible, n_ambiguous_prefixes)."""
    w = pq.read_table(
        wells_path,
        columns=[
            "canonical_id",
            "longitude",
            "latitude",
            "well_depth",
            "screen_bottom",
            "well_class",
            "head_above_screen",
            "obs_count",
        ],
    ).to_pandas()
    comp = (
        w["well_depth"]
        .where(np.isfinite(w["well_depth"]), w["screen_bottom"])
        .to_numpy()
    )

    finite_comp = np.isfinite(comp)
    plausible = finite_comp & (comp > COMP_MIN_M) & (comp <= COMP_MAX_M)
    n_dropped = int(finite_comp.sum() - plausible.sum())
    log.info(
        "GWX pool: %d wells with plausible completion depth (%d finite dropped as "
        "implausible outside (%g, %g] m)",
        int(plausible.sum()),
        n_dropped,
        COMP_MIN_M,
        COMP_MAX_M,
    )

    tr = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    lon = w["longitude"].to_numpy()
    lat = w["latitude"].to_numpy()
    finite_ll = np.isfinite(lon) & np.isfinite(lat)
    pool_mask = plausible & finite_ll
    px, py = tr.transform(lon[pool_mask], lat[pool_mask])
    # a handful of finite-but-bogus coords (e.g. latitude > 90) project to inf
    # under the CONUS Albers grid; drop them (they cannot be spatial neighbors).
    fin_xy = np.isfinite(px) & np.isfinite(py)
    n_bad_proj = int((~fin_xy).sum())
    if n_bad_proj:
        log.info("dropped %d GWX pool points that did not project finitely", n_bad_proj)
    pool_xy = np.column_stack([px[fin_xy], py[fin_xy]])
    pool_comp = comp[pool_mask][fin_xy]

    # construction join: only GWX rows whose 12-char prefix is needed
    w["prefix"] = w["canonical_id"].str.slice(0, PREFIX_LEN)
    w["comp_m"] = comp
    matched = w[w["prefix"].isin(needed_prefixes)].copy()
    per_prefix = matched.groupby("prefix").size()
    n_ambiguous = int((per_prefix > 1).sum())
    ambiguous_prefixes = set(per_prefix.index[per_prefix > 1])

    # deterministic dedup: has-construction, then max obs_count, then id
    matched["has_constr"] = (
        np.isfinite(matched["well_depth"]) | np.isfinite(matched["screen_bottom"])
    ).astype(int)
    matched["obs_sort"] = matched["obs_count"].fillna(-1.0)
    matched = matched.sort_values(
        ["prefix", "has_constr", "obs_sort", "canonical_id"],
        ascending=[True, False, False, True],
    )
    dedup = matched.drop_duplicates("prefix", keep="first").copy()
    dedup["gwx_ambiguous"] = dedup["prefix"].isin(ambiguous_prefixes)
    join = dedup[
        ["prefix", "comp_m", "well_class", "head_above_screen", "gwx_ambiguous"]
    ].rename(columns={"well_class": "well_class_gwx"})
    return pool_xy, pool_comp, join, n_dropped, n_ambiguous


# --------------------------------------------------------------------------- #
# report helpers
# --------------------------------------------------------------------------- #
def _band_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}m" if np.isfinite(hi) else f"{lo:g}+m"


def _p_dist(p: np.ndarray) -> dict:
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return {"n": 0}
    return {
        "n": int(len(p)),
        "mean": round(float(np.mean(p)), 4),
        "median": round(float(np.median(p)), 4),
        "p10": round(float(np.quantile(p, 0.10)), 4),
        "p90": round(float(np.quantile(p, 0.90)), 4),
        "frac_ge_0.5": round(float(np.mean(p >= 0.5)), 4),
    }


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str) -> float | None:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return None
    aa, bb = a[m], b[m]
    if method == "spearman":
        aa = pd.Series(aa).rank().to_numpy()
        bb = pd.Series(bb).rank().to_numpy()
    if np.std(aa) == 0 or np.std(bb) == 0:
        return None
    return round(float(np.corrcoef(aa, bb)[0, 1]), 4)


DEFINITIONS = {
    "unit_convention": "Depths/heads in metres; probabilities dimensionless in "
    "[0,1]; quantiles dimensionless in [0,1]. DTW = depth to the unconfined water "
    "table (m, larger = deeper). The prior is UNSUPERVISED wrt water level: no "
    "parameter is fit to DTW/WTE; the observed-DTW bands below are diagnostics "
    "computed AFTER the fact and never used in construction.",
    "p_phreatic_construction": "P(observation sampled the phreatic/water-table "
    "surface) from construction-only evidence: local completion-depth rank + "
    "confinement + well role. Higher = more likely water-table. Range [0.05,0.95] "
    "(0.98 for water pseudo; <=0.10 for confined-type).",
    "p_phreatic_geom": "As p_phreatic_construction but additionally blends the "
    "head_above_screen geometry shift (a partial water-level proxy). Kept separate "
    "so the trainer can opt out of the proxy.",
    "depth_rank_q": "Mid-rank empirical quantile (dimensionless [0,1]) of the "
    "well's completion depth among GWX wells with plausible completion depth "
    "within the neighborhood radius; 0 = shallowest locally, 1 = deepest locally. "
    "NaN when no neighborhood reached MIN_NEIGHBORS or no completion depth.",
    "p_rank_formula": "p_rank = clip(1 - depth_rank_q, 0.05, 0.95); monotonically "
    "decreasing in depth_rank_q (shallow -> phreatic).",
    "combination_formula": "L = logit(p_rank) + beta_conf + beta_role [+ beta_head "
    "for _geom]; p = clip(sigmoid(L), 0.05, 0.95); then p=min(p,0.10) if "
    "confined-type else p=max(p,0.50) if unconfined and depth_rank_q<0.5 is false "
    "(i.e. not locally deep).",
    "beta_conf": "confined/likely_confined/artesian=-3.0 (+cap 0.10); "
    "unconfined=+1.0 (+floor 0.50 unless locally deep); unconfined_marginal=0.0; "
    "unknown/missing=0.0. Source: v0.2 contract confinement_class (authoritative "
    "screening; GWX confinement NOT used to avoid ambiguous-join contamination).",
    "beta_role": "monitoring=+0.5; pumping/production=-0.5; unknown/missing=0.0. "
    "Source: GWX-matched well_class (contract well_class is uniformly 'monitoring').",
    "beta_head": "clip(-(head_above_screen - 5.0)/15.0, -2.0, +2.0); 0 if missing. "
    "Source: GWX-matched head_above_screen (m above screen top).",
    "n_neighbors": "Count of GWX wells with plausible completion depth within "
    "neighbor_radius_m (the radius actually used: 2000 m, or 10000 m fallback).",
    "neighbor_radius_m": "Radius used for depth_rank_q (2000 or 10000 m); NaN if "
    "the CONUS-neutral 0.5 fallback fired.",
    "gwx_matched": "True if the contract 8-hex prefix matched >=1 GWX record.",
    "gwx_ambiguous": "True if the prefix matched >1 GWX record (collision, "
    "deduplicated by has-construction / max obs_count / lexicographic id).",
    "confinement_class": "From the v0.2 contract (authoritative), not GWX.",
    "evidence_flags": "Compact '|'-joined tags recording which evidence fired: "
    "rank2km / rank10km / rank_neutral / no_comp / unmatched_neutral / water_pseudo; "
    "conf_<class>; role_<class or none>; head / no_head; confined_cap / unconf_floor.",
    "sacrificial_huc4": "HUC4 0707/1019/1605 rows still receive priors (trainer "
    "needs them) but are EXCLUDED from every reported diagnostic below.",
}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wells", default=WELLS)
    ap.add_argument("--panels", default=PANELS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("loading contract panels %s", args.panels)
    con = load_contract(args.panels)
    n_total = len(con)
    is_pseudo = con["is_water_pseudo"].to_numpy()
    real = con[~is_pseudo].copy()
    log.info(
        "contract: %d rows (%d real, %d water pseudo)",
        n_total,
        len(real),
        int(is_pseudo.sum()),
    )

    needed_prefixes = set(real["canonical_id"].tolist())
    log.info("loading GWX wells + building pool/join %s", args.wells)
    pool_xy, pool_comp, join, n_dropped, n_ambiguous = load_gwx_pool_and_join(
        args.wells, needed_prefixes
    )

    # attach construction join to real wells (by prefix == contract canonical_id)
    real = real.merge(join, left_on="canonical_id", right_on="prefix", how="left")
    real["gwx_matched"] = real["prefix"].notna()
    real["gwx_ambiguous"] = real["gwx_ambiguous"] == True  # noqa: E712 (NaN->False)
    n_matched = int(real["gwx_matched"].sum())
    n_unmatched = int((~real["gwx_matched"]).sum())
    log.info(
        "join: %d matched (%d ambiguous prefixes), %d unmatched (neutral 0.5)",
        n_matched,
        n_ambiguous,
        n_unmatched,
    )

    # neighborhood depth rank (only meaningful where a target completion exists)
    target_comp = real["comp_m"].to_numpy(float)
    mon_xy = real[["x5070", "y5070"]].to_numpy(float)
    log.info("building cKDTree over %d GWX pool points", len(pool_comp))
    tree = cKDTree(pool_xy)
    log.info("computing neighborhood depth-rank quantiles for %d real wells", len(real))
    q, n_nb, rad = _batch_neighborhood(mon_xy, target_comp, tree, pool_comp)
    real["depth_rank_q"] = q
    real["n_neighbors"] = n_nb
    real["neighbor_radius_m"] = rad

    # evidence vectors
    p_rank = np.where(np.isfinite(q), p_rank_from_quantile(q), 0.5)
    cclass = real["confinement_class_contract"].fillna("unknown").to_numpy()
    beta_conf = np.array([BETA_CONF.get(c, 0.0) for c in cclass])
    is_confined = np.array([c in CONFINED_TYPES for c in cclass])
    wclass = real["well_class_gwx"].fillna("unknown").to_numpy()
    beta_role = np.array([BETA_ROLE.get(w, 0.0) for w in wclass])
    head = real["head_above_screen"].to_numpy(float)
    beta_head = beta_head_from_h(head)
    # unconfined floor applies for unconfined wells that are not locally deep
    locally_deep = np.isfinite(q) & (q >= 0.5)
    unconf_floor = (cclass == "unconfined") & (~locally_deep)

    p_constr = combine_logodds(
        p_rank, beta_conf, beta_role, 0.0, is_confined, unconf_floor
    )
    p_geom = combine_logodds(
        p_rank, beta_conf, beta_role, beta_head, is_confined, unconf_floor
    )

    # unmatched wells -> strict neutral 0.5 (no GWX evidence at all)
    unmatched = ~real["gwx_matched"].to_numpy()
    p_constr = np.where(unmatched, 0.5, p_constr)
    p_geom = np.where(unmatched, 0.5, p_geom)

    # evidence flags
    flags = []
    for i in range(len(real)):
        if unmatched[i]:
            flags.append("unmatched_neutral")
            continue
        parts = []
        if not np.isfinite(target_comp[i]):
            parts.append("no_comp")
        elif np.isfinite(q[i]):
            parts.append("rank2km" if rad[i] == RADII_M[0] else "rank10km")
        else:
            parts.append("rank_neutral")
        parts.append(f"conf_{cclass[i]}")
        parts.append(f"role_{wclass[i]}")
        parts.append("head" if np.isfinite(head[i]) else "no_head")
        if is_confined[i]:
            parts.append("confined_cap")
        elif unconf_floor[i]:
            parts.append("unconf_floor")
        flags.append("|".join(parts))
    real["evidence_flags"] = flags
    real["p_phreatic_construction"] = p_constr
    real["p_phreatic_geom"] = p_geom
    real["confinement_class"] = real["confinement_class_contract"]

    # assemble real-well output
    real_out = real[
        [
            "canonical_id",
            "is_water_pseudo",
            "p_phreatic_construction",
            "p_phreatic_geom",
            "depth_rank_q",
            "n_neighbors",
            "neighbor_radius_m",
            "gwx_matched",
            "gwx_ambiguous",
            "confinement_class",
            "evidence_flags",
        ]
    ].copy()

    # water-pseudo output rows (phreatic by construction)
    pseudo = con[is_pseudo].copy()
    pseudo_out = pd.DataFrame(
        {
            "canonical_id": pseudo["canonical_id"].to_numpy(),
            "is_water_pseudo": True,
            "p_phreatic_construction": WATER_PSEUDO_P,
            "p_phreatic_geom": WATER_PSEUDO_P,
            "depth_rank_q": np.nan,
            "n_neighbors": 0,
            "neighbor_radius_m": np.nan,
            "gwx_matched": False,
            "gwx_ambiguous": False,
            "confinement_class": pseudo["confinement_class_contract"].to_numpy(),
            "evidence_flags": "water_pseudo",
        }
    )

    table = pd.concat([real_out, pseudo_out], ignore_index=True)
    assert len(table) == n_total, (len(table), n_total)
    priors_path = out / "assignment_priors.parquet"
    table.to_parquet(priors_path)
    log.info("wrote %s (%d rows)", priors_path, len(table))

    # ---------------- diagnostics (exclude sacrificial HUC4) ---------------- #
    real["huc4"] = real["huc4"].astype(object)
    diag_mask = (~real["huc4"].isin(SACRIFICIAL_HUC4)).to_numpy() & real[
        "gwx_matched"
    ].to_numpy()
    n_sac = int(real["huc4"].isin(SACRIFICIAL_HUC4).sum())

    dtw = real["mean_dtw"].to_numpy(float)
    band = np.full(len(real), "", dtype=object)
    for lo, hi in DEPTH_BANDS:
        m = np.isfinite(dtw) & (dtw >= lo) & (dtw < hi)
        band[m] = _band_label(lo, hi)

    by_band_constr, by_band_geom = {}, {}
    for lo, hi in DEPTH_BANDS:
        lab = _band_label(lo, hi)
        m = diag_mask & (band == lab)
        if m.sum() > 0:
            by_band_constr[lab] = _p_dist(p_constr[m])
            by_band_geom[lab] = _p_dist(p_geom[m])

    corr_diag = {
        "pearson_p_constr_vs_completion_depth": _safe_corr(
            p_constr[diag_mask], target_comp[diag_mask], "pearson"
        ),
        "spearman_p_constr_vs_completion_depth": _safe_corr(
            p_constr[diag_mask], target_comp[diag_mask], "spearman"
        ),
        "spearman_p_constr_vs_depth_rank_q": _safe_corr(
            p_constr[diag_mask], q[diag_mask], "spearman"
        ),
        "spearman_p_constr_vs_observed_dtw_DIAGNOSTIC_ONLY": _safe_corr(
            p_constr[diag_mask], dtw[diag_mask], "spearman"
        ),
    }

    rad_used = real["neighbor_radius_m"].to_numpy()
    coverage = {
        "n_contract_rows": n_total,
        "n_real_wells": int(len(real)),
        "n_water_pseudo": int(is_pseudo.sum()),
        "n_gwx_matched": n_matched,
        "n_gwx_unmatched_neutral": n_unmatched,
        "frac_matched": round(n_matched / len(real), 4),
        "n_ambiguous_prefix_collisions": n_ambiguous,
        "n_gwx_records_dropped_implausible_comp": n_dropped,
        "n_rank_from_2km": int(np.sum(rad_used == RADII_M[0])),
        "n_rank_from_10km": int(np.sum(rad_used == RADII_M[1])),
        "n_rank_neutral_fallback": int(
            np.sum(real["gwx_matched"].to_numpy() & ~np.isfinite(rad_used))
        ),
        "n_sacrificial_huc4_excluded_from_diagnostics": n_sac,
        "n_diagnostic_wells": int(diag_mask.sum()),
    }

    report = {
        "generated": str(date.today()),
        "inputs": {"wells": args.wells, "panels": args.panels},
        "coverage": coverage,
        "prior_distribution_overall_diagnostic": {
            "p_phreatic_construction": _p_dist(p_constr[diag_mask]),
            "p_phreatic_geom": _p_dist(p_geom[diag_mask]),
        },
        "prior_distribution_by_observed_dtw_band_DIAGNOSTIC_ONLY": {
            "p_phreatic_construction": by_band_constr,
            "p_phreatic_geom": by_band_geom,
        },
        "correlation_diagnostics": corr_diag,
        "constants": {
            "comp_bounds_m": [COMP_MIN_M, COMP_MAX_M],
            "radii_m": list(RADII_M),
            "min_neighbors": MIN_NEIGHBORS,
            "p_clip": [P_LO, P_HI],
            "confined_cap": CONFINED_CAP,
            "unconf_floor": UNCONF_FLOOR,
            "water_pseudo_p": WATER_PSEUDO_P,
            "beta_conf": BETA_CONF,
            "beta_role": BETA_ROLE,
            "head_ref_m": HEAD_REF_M,
            "head_scale_m": HEAD_SCALE_M,
            "beta_head_clip": BETA_HEAD_CLIP,
        },
        "definitions": DEFINITIONS,
    }
    report_path = out / "assignment_priors_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info("wrote %s", report_path)
    log.info(
        "DONE matched=%d/%d ambiguous=%d unmatched=%d p_constr median(diag)=%s",
        n_matched,
        len(real),
        n_ambiguous,
        n_unmatched,
        report["prior_distribution_overall_diagnostic"]["p_phreatic_construction"][
            "median"
        ],
    )


if __name__ == "__main__":
    main()
