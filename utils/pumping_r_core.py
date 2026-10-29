"""Density-adaptive pumping-well R: donor index, kernels, gates, statistics.

R is the regional water-table prior handily learns a residual over. Here it is a
deterministic function of position built from donor wells (pumping and
class-unknown wells, never monitoring wells, never the state-funnel wells), so it
needs no folds and is identical at training nodes, at panel sites, and at raster
cells. One ``Variant`` is one recipe; ``compute`` evaluates it at any point set.

Factors (notes/R_DENSITY_SWEEP_PLAN.md):

  A kernel   idw   w = 1 / max(d, 1 m)^power, then share cap
             gauss w = exp(-0.5 (d / h)^2), h = max(h_min, h_frac * d_k), then share cap
  B gate     depth   w_blend = clip((ramp_hi - d_med) / (ramp_hi - ramp_lo), 0, 1)
             support depth ramp x clip((support_hi - dk_h) / (support_hi - support_lo), 0, 1)
                     x [n_eff >= n_min]
             median  w_blend = 0 (median only)
  C select   lifted     k nearest in (x, y, vw z)
             horizontal k nearest in (x, y), weighted by lifted distance; neighbours
                        beyond radius_max get zero weight unless fewer than n_min lie inside
  F screen   drop donors whose DTW departs from the weighted local median DTW by more
             than max(screen_abs_m, screen_mad_k * weighted MAD)

Distances are metres in the lifted metric unless named ``_h`` (horizontal) or
``_km``. Residual convention everywhere: r = R - observed WTE, m; positive = R
above the table = DTW too shallow.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_stacker_features import sample_coarse  # noqa: E402

log = logging.getLogger("pumping_r")

HAND = "/data/ssd2/handily/conus/wte_gnn/conus_wells_hand.parquet"
DEM = "/data/ssd2/handily/conus/covariates/elev48i0100a.tif"
STATS = ("mean", "median", "q25", "q75")


@dataclass
class Variant:
    name: str = "a1"
    kernel: str = "idw"  # idw | gauss
    k: int = 32
    power: float = 2.0
    h_min_m: float = 300.0
    h_frac: float = 1.0
    max_share: float = 0.125
    select: str = "lifted"  # lifted | horizontal
    vw: float = 100.0
    radius_max_km: float = 10.0
    gate: str = "depth"  # depth | support | median
    ramp_lo_m: float = 5.0
    ramp_hi_m: float = 15.0
    support_lo_km: float = 2.0
    support_hi_km: float = 8.0
    n_min: int = 8
    screen: bool = False
    screen_abs_m: float = 30.0
    screen_mad_k: float = 4.0
    cap_m: float = 1.0
    stat_shallow: str = "q75"

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Variant":
        known = {f.name for f in fields(cls)}
        bad = set(d) - known
        if bad:
            raise ValueError(f"unknown Variant fields {sorted(bad)}")
        return cls(**d)

    def label(self) -> str:
        bits = [self.kernel, f"k{self.k}"]
        if self.kernel == "gauss":
            bits.append(f"h{int(self.h_min_m)}")
        bits += [f"s{self.max_share:g}", self.select, f"vw{self.vw:g}", self.gate]
        if self.screen:
            bits.append("scr")
        return "_".join(bits)


def load_donors(hand: str, dem: str, classes: list[str]) -> pd.DataFrame:
    """HAND-table wells of the given well_class values, one row per canonical well
    (min-HAND record), with the 100 m DEM surface and WTE = z - mean_dtw."""
    w = pd.read_parquet(
        hand,
        columns=[
            "canonical_id",
            "source",
            "well_class",
            "confinement_class",
            "mean_dtw",
            "x5070",
            "y5070",
            "hand_m",
        ],
    )
    n0 = len(w)
    w = w.sort_values("hand_m", na_position="last").drop_duplicates(
        "canonical_id", keep="first"
    )
    log.info("HAND table: %d rows -> %d unique wells", n0, len(w))
    w = w[w["well_class"].isin(classes) & np.isfinite(w["mean_dtw"])].reset_index(
        drop=True
    )
    log.info(
        "donor pool (%s): %d wells; class counts %s",
        classes,
        len(w),
        w["well_class"].value_counts().to_dict(),
    )
    z = sample_coarse(
        dem, w["x5070"].to_numpy("float64"), w["y5070"].to_numpy("float64")
    )
    ok = np.isfinite(z)
    log.info("DEM surface at donors: %.4f finite", float(ok.mean()))
    w = w[ok].reset_index(drop=True)
    w["z_surf_m"] = z[ok]
    w["wte_m"] = w["z_surf_m"] - w["mean_dtw"]
    log.info(
        "donor mean_dtw: p50 %.1f m, p90 %.1f m, %d negative",
        float(w["mean_dtw"].median()),
        float(w["mean_dtw"].quantile(0.9)),
        int((w["mean_dtw"] < 0).sum()),
    )
    return w


def exclude_donors(
    donors: pd.DataFrame,
    files: list[str],
    radius_m: float,
    same_source: list[str] | None = None,
    same_source_m: float = 0.0,
    id_files: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop donors within radius_m (horizontal) of any point in the given parquet
    files (x5070, y5070): bundle labels, admitted sources, admission targets --
    everything that made it through a state funnel into handily.

    ``same_source`` / ``same_source_m``: donors whose ``source`` is in the list are
    dropped out to the wider radius. The HAND table carries funnel-source records
    from an older GWX build whose coordinates can sit tens of metres from the
    current build's, so the same physical well can survive a 10 m radius.

    ``id_files``: parquet files with a ``canonical_id`` column; those donors are
    dropped outright (e.g. funnel-source records that cannot be tied to the
    current build, utils/build_r_sweep_sites_nm.py)."""
    counts = {
        "radius_m": radius_m,
        "same_source": same_source or [],
        "same_source_m": same_source_m,
        "id_files": id_files or [],
        "dropped_radius": 0,
        "dropped_same_source": 0,
        "dropped_ids": 0,
    }
    if id_files:
        ids = set()
        for p in id_files:
            ids |= set(pd.read_parquet(p, columns=["canonical_id"])["canonical_id"])
        hit = donors["canonical_id"].isin(ids).to_numpy()
        counts["dropped_ids"] = int(hit.sum())
        log.info(
            "donors dropped by id list (%d ids): %d", len(ids), counts["dropped_ids"]
        )
        donors = donors[~hit].reset_index(drop=True)
    if not files:
        return donors, counts
    pts = []
    for p in files:
        e = pd.read_parquet(p, columns=["x5070", "y5070"]).dropna()
        pts.append(e.to_numpy("float64"))
        log.info("exclusion points %s: %d", p, len(e))
    ex = np.vstack(pts)
    d, _ = cKDTree(ex).query(donors[["x5070", "y5070"]].to_numpy("float64"), k=1)
    drop = d < radius_m
    counts["dropped_radius"] = int(drop.sum())
    if same_source and same_source_m > radius_m:
        wide = (
            donors["source"].isin(same_source).to_numpy() & (d < same_source_m) & ~drop
        )
        counts["dropped_same_source"] = int(wide.sum())
        drop |= wide
    log.info(
        "donors dropped: %d within %.0f m of an exclusion point, %d more of source %s "
        "within %.0f m; donors left %d",
        counts["dropped_radius"],
        radius_m,
        counts["dropped_same_source"],
        same_source,
        same_source_m,
        len(donors) - int(drop.sum()),
    )
    return donors[~drop].reset_index(drop=True), counts


def cap_shares(wts: np.ndarray, max_share: float) -> np.ndarray:
    """Row-wise cap on any one donor's share of the total weight. IDW weights with
    a 1 m floor hand a cell to its nearest donor (share -> 1 within the radius where
    the second donor's weight catches up), so a single bad record prints a bullseye
    and flips the weighted median (needs share >= 0.5) or q75 (>= 0.25). Capping the
    share at s guarantees n_eff >= 1/s donors in every statistic.

    Exact solution of w' = min(w, c) with c = s * sum(w'): with the m largest weights
    capped, c = s * (sum of the rest) / (1 - s m); the smallest m whose (m+1)-th
    largest weight is <= c is the answer. Rows with fewer than 1/s positive weights
    cannot meet the cap and fall back to equal weights on their positive entries.
    Zero rows stay zero."""
    if max_share <= 0 or max_share >= 1:
        return wts
    w = wts.astype("float64", copy=True)
    n, k = w.shape
    srt = -np.sort(-w, axis=1)
    suffix = np.cumsum(srt[:, ::-1], axis=1)[:, ::-1]  # sum of srt[:, m:]
    m = np.arange(k)
    denom = 1.0 - max_share * m
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(denom > 0, max_share * suffix / denom, np.inf)
    ok = (srt <= c + 1e-12 * np.maximum(c, 1.0)) & (denom > 0)
    has = ok.any(1)
    first = ok.argmax(1)
    cap = c[np.arange(n), first]
    out = np.where(has[:, None], np.minimum(w, cap[:, None]), (w > 0).astype("float64"))
    out[w.sum(1) <= 0] = 0.0
    return out


def weighted_stats(vals: np.ndarray, wts: np.ndarray) -> dict[str, np.ndarray]:
    """Row-wise weighted mean / median / q25 / q75 over (n, k) arrays; zero weights
    drop a neighbour. Quantiles are 'lower' type on the weighted CDF."""
    wsum = wts.sum(1)
    safe = np.where(wsum > 0, wsum, np.nan)
    mean = (wts * vals).sum(1) / safe
    order = np.argsort(vals, axis=1)
    v = np.take_along_axis(vals, order, 1)
    w = np.take_along_axis(wts, order, 1)
    cdf = np.cumsum(w, 1) / safe[:, None]
    out = {"mean": mean}
    for name, q in (("median", 0.5), ("q25", 0.25), ("q75", 0.75)):
        idx = (cdf >= q).argmax(1)
        out[name] = np.take_along_axis(v, idx[:, None], 1)[:, 0]
    for name in out:
        out[name][wsum <= 0] = np.nan
    return out


def weighted_median_mad(
    vals: np.ndarray, wts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    med = weighted_stats(vals, wts)["median"]
    mad = weighted_stats(np.abs(vals - med[:, None]), wts)["median"]
    return med, mad


class DonorIndex:
    """Donor arrays plus the horizontal and relief-lifted trees for one vw."""

    def __init__(self, donors: pd.DataFrame, vw: float):
        self.xy = donors[["x5070", "y5070"]].to_numpy("float64")
        self.z = donors["z_surf_m"].to_numpy("float64")
        self.wte = donors["wte_m"].to_numpy("float64")
        self.dtw = donors["mean_dtw"].to_numpy("float64")
        self.vw = float(vw)
        self.h_tree = cKDTree(self.xy)
        self.l_tree = cKDTree(np.column_stack([self.xy, self.vw * self.z]))

    def lift(self, qxy: np.ndarray, qz: np.ndarray) -> np.ndarray:
        return np.column_stack([qxy, self.vw * qz])

    def density_km(self, qxy: np.ndarray, k: int = 32) -> np.ndarray:
        """Horizontal distance to the k-th nearest donor, km: the fixed density axis
        used for stratification, independent of the variant's own k."""
        d, _ = self.h_tree.query(qxy, k=k, workers=-1)
        return d[:, -1] / 1000.0

    def neighbours(self, qxy: np.ndarray, qz: np.ndarray, v: Variant):
        """Candidate set per query: lifted distance, donor index, horizontal distance,
        and a keep mask (selection C)."""
        if v.select == "lifted":
            dist, idx = self.l_tree.query(self.lift(qxy, qz), k=v.k, workers=-1)
            geo = np.hypot(
                self.xy[idx, 0] - qxy[:, 0][:, None],
                self.xy[idx, 1] - qxy[:, 1][:, None],
            )
            keep = np.ones_like(dist, dtype=bool)
        elif v.select == "horizontal":
            geo, idx = self.h_tree.query(qxy, k=v.k, workers=-1)
            dz = self.vw * (self.z[idx] - qz[:, None])
            dist = np.hypot(geo, dz)
            keep = geo <= v.radius_max_km * 1000.0
            # never fewer than n_min neighbours: the n_min nearest always stay
            keep[:, : v.n_min] = True
        else:
            raise ValueError(f"select {v.select}")
        return dist, idx, geo, keep


def kernel_weights(
    dist: np.ndarray, keep: np.ndarray, v: Variant
) -> tuple[np.ndarray, np.ndarray]:
    """Kernel A on the candidate set; returns weights and the bandwidth used (m;
    NaN for idw)."""
    if v.kernel == "idw":
        w = 1.0 / np.maximum(dist, 1.0) ** v.power
        h = np.full(dist.shape[0], np.nan)
    elif v.kernel == "gauss":
        dk = np.where(keep, dist, 0.0).max(1)
        h = np.maximum(v.h_min_m, v.h_frac * dk)
        w = np.exp(-0.5 * (dist / h[:, None]) ** 2)
    else:
        raise ValueError(f"kernel {v.kernel}")
    w = np.where(keep, w, 0.0)
    return cap_shares(w, v.max_share), h


def screen_outliers(
    dtw: np.ndarray, w: np.ndarray, v: Variant
) -> tuple[np.ndarray, np.ndarray]:
    """Factor F: zero the weight of donors far from the weighted local median DTW."""
    med, mad = weighted_median_mad(dtw, w)
    tol = np.maximum(v.screen_abs_m, v.screen_mad_k * mad)
    drop = np.abs(dtw - med[:, None]) > tol[:, None]
    drop &= w > 0
    w2 = np.where(drop, 0.0, w)
    return cap_shares(w2, v.max_share), drop.sum(1)


def compute(
    index: DonorIndex, qxy: np.ndarray, qz: np.ndarray, v: Variant, chunk: int = 500_000
) -> dict[str, np.ndarray]:
    """Evaluate one variant at (x, y, z) points. Returns per-point arrays:
    r_blend (capped), r_median, r_q75, r_q25, r_mean (uncapped WTE, m asl),
    w_blend, n_eff, h_m, iqr_m, dk_h_km, nearest_h_km, top_share, cap_bound,
    n_screened."""
    if abs(index.vw - v.vw) > 1e-9:
        raise ValueError(f"index built with vw {index.vw}, variant wants {v.vw}")
    n = len(qxy)
    out = {
        k: np.full(n, np.nan)
        for k in (
            "r_median",
            "r_q75",
            "r_q25",
            "r_mean",
            "n_eff",
            "h_m",
            "iqr_m",
            "dk_h_km",
            "nearest_h_km",
            "top_share",
            "n_screened",
        )
    }
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        dist, idx, geo, keep = index.neighbours(qxy[s:e], qz[s:e], v)
        w, h = kernel_weights(dist, keep, v)
        nscr = np.zeros(e - s)
        if v.screen:
            w, nscr = screen_outliers(index.dtw[idx], w, v)
        st = weighted_stats(index.wte[idx], w)
        wsum = w.sum(1)
        out["r_median"][s:e] = st["median"]
        out["r_q75"][s:e] = st["q75"]
        out["r_q25"][s:e] = st["q25"]
        out["r_mean"][s:e] = st["mean"]
        out["n_eff"][s:e] = np.where(
            wsum > 0, wsum**2 / np.maximum((w**2).sum(1), 1e-300), 0
        )
        out["h_m"][s:e] = h
        out["iqr_m"][s:e] = st["q75"] - st["q25"]
        out["dk_h_km"][s:e] = np.where(w > 0, geo, 0.0).max(1) / 1000.0
        out["nearest_h_km"][s:e] = geo.min(1) / 1000.0
        out["top_share"][s:e] = np.where(
            wsum > 0, w.max(1) / np.maximum(wsum, 1e-300), np.nan
        )
        out["n_screened"][s:e] = nscr
    r_sh = out["r_q75"] if v.stat_shallow == "q75" else out[f"r_{v.stat_shallow}"]
    if v.gate == "median":
        wb = np.zeros(n)
    else:
        d_med = qz - out["r_median"]
        wb = np.clip((v.ramp_hi_m - d_med) / (v.ramp_hi_m - v.ramp_lo_m), 0.0, 1.0)
        if v.gate == "support":
            sup = np.clip(
                (v.support_hi_km - out["dk_h_km"])
                / (v.support_hi_km - v.support_lo_km),
                0.0,
                1.0,
            )
            sup = sup * (out["n_eff"] >= v.n_min)
            wb = wb * sup
        elif v.gate != "depth":
            raise ValueError(f"gate {v.gate}")
    r = wb * r_sh + (1.0 - wb) * out["r_median"]
    out["cap_bound"] = r > qz - v.cap_m
    out["r_blend"] = np.minimum(r, qz - v.cap_m)
    out["w_blend"] = wb
    return out
