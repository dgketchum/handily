"""Optional regional-aquifer graph substrate for the CONUS WTE/DTW GNN.

A SEPARATE build stage from ``build_conus_graph_inputs.py`` so the proven query/
reach bundle stays byte-stable and the aquifer graph can be rebuilt independently.
It augments an existing ``graph_dir`` bundle with three target-blind files plus an
``aquifer`` block in ``graph_manifest.json``:

  aquifer_nodes.parquet       coarse (default 1 km) grid cells over the well footprint,
                              carrying static GLHYMPS/USGS/Pelletier geology only.
  aquifer_edges.parquet       8-neighbour grid edges, gated by principal-aquifer /
                              rock-type compatibility, with a harmonic-K conductance.
  aquifer_to_query_edges.pq    each query well attached to its k nearest aquifer cells.

Phase 1 is TARGET-BLIND: no observed head/DTW, no Ma/Janssen, no deep-OOF label is
stored on any aquifer node/edge. The intent is a smooth, long-range hydrogeologic
context the GNN can route to deep wells -- the trainer/model add a GATED residual
correction branch over the FAC-residual stream head (notes/regional_aquifer_graph.md),
so this substrate is an exact no-op until the learned route is switched on.

    uv run python utils/build_regional_aquifer_graph.py \\
        --graph-dir /data/ssd2/handily/conus/wte_gnn/graph_fac_clipped_facresid \\
        --covariate-root /nas/handily/covariates \\
        --grid-res-m 1000 --extent-buffer-km 25 --knn-aquifer-query 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_regional_aquifer_graph")

LN10 = np.log(10.0)

# Target-blind aquifer-node model features (numeric only). Categorical codes
# (principal_aquifer_code / rocktype / karst) are kept on the node table for
# adjacency + diagnostics but are NEVER fed to the model as ordinal continuous
# values -- only their 0/1 "present" flags enter here.
AQUIFER_FEATURE_COLS = [
    "principal_aquifer_present",
    "aquifer_rocktype_present",
    "logk_glhymps",
    "porosity_glhymps",
    "depth_to_bedrock_m",
    "sediment_thickness_m",
    "basinfill_thickness_m",
]
AQUIFER_EDGE_FEATURE_COLS = [
    "log1p_aq_dist_m",
    "same_principal_aquifer",
    "same_rocktype",
    "same_component",
    "logk_hmean",
    "conductance",
]
AQUIFER_QUERY_EDGE_FEATURE_COLS = [
    "log1p_aq_query_dist_m",
    "rank",
    "is_controlling",
    "same_principal_aquifer",
    "same_component",
]

# A target/benchmark column has no business on a Phase-1 aquifer node/edge. The
# guard is substring-based so a renamed leak (mean_dtw, obs_wte_m, janssen, ma,
# regional_deep, ...) is still caught. Legit geology features carry none of these.
FORBIDDEN_SUBSTRINGS = ("dtw", "wte", "janssen", "regional_deep", "deep_regional")


def assert_target_blind(cols: list[str]) -> None:
    """Raise if any model-feature column looks like a label/benchmark (leakage guard)."""
    bad = []
    for c in cols:
        cl = c.lower()
        if (
            cl == "ma"
            or cl.startswith("ma_")
            or any(s in cl for s in FORBIDDEN_SUBSTRINGS)
        ):
            bad.append(c)
    if bad:
        raise SystemExit(
            f"aquifer feature cols contain forbidden target/benchmark columns: {bad}"
        )


# --- raster source registry ------------------------------------------------
def _raster_specs(geo: Path) -> dict[str, dict]:
    """Phase-1 static geology sources (all EPSG:5070, documented in COVARIATE_ACQUISITION.md).

    ``scale`` converts stored units to model units (logk/porosity are stored x100;
    depth-to-bedrock is cm -> m). ``kind`` is ``code`` (categorical id; nodata 0 ->
    absent) or ``cont`` (continuous; nodata -> NaN). ``glhymps_logk`` flags the one
    raster whose stored value must be on the x100 scale (an unconverted-input guard).
    """
    return {
        "principal_aquifer_code": dict(
            path=geo / "aquifer_code.tif", kind="code", required=True
        ),
        "aquifer_rocktype_code": dict(
            path=geo / "aquifer_rocktype.tif", kind="code", required=True
        ),
        "karst_type_code": dict(
            path=geo / "karst_type.tif", kind="code", required=False
        ),
        "logk_glhymps": dict(
            path=geo / "permeability_logk_x100.tif",
            kind="cont",
            scale=0.01,
            required=True,
            glhymps_logk=True,
        ),
        "porosity_glhymps": dict(
            path=geo / "porosity_x100.tif", kind="cont", scale=0.01, required=True
        ),
        "depth_to_bedrock_m": dict(
            path=geo / "depth_to_bedrock_abs_cm.tif",
            kind="cont",
            scale=0.01,
            required=False,
        ),
        "sediment_thickness_m": dict(
            path=geo / "sediment_thickness_avg_m.tif", kind="cont", required=False
        ),
        "basinfill_thickness_m": dict(
            path=geo / "sediment_thickness_basinfill_m.tif",
            kind="cont",
            required=False,
        ),
    }


def _sample_raster(path: Path, x5070: np.ndarray, y5070: np.ndarray) -> np.ndarray:
    """Nearest-cell sample of an EPSG:5070 raster; nodata + out-of-grid -> NaN."""
    import rasterio

    out = np.full(x5070.shape, np.nan, dtype="float64")
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 5070:
            raise SystemExit(f"raster not EPSG:5070: {path} (crs={src.crs})")
        t = src.transform
        col = np.floor((x5070 - t.c) / t.a).astype(np.int64)
        row = np.floor((y5070 - t.f) / t.e).astype(np.int64)
        ok = (row >= 0) & (row < src.height) & (col >= 0) & (col < src.width)
        if ok.any():
            vals = np.array(
                [v[0] for v in src.sample(list(zip(x5070[ok], y5070[ok])))],
                dtype="float64",
            )
            nd = src.nodata
            if nd is not None:
                vals[vals == nd] = np.nan
            out[ok] = vals
    return out


def check_and_sample(
    specs: dict[str, dict], x5070: np.ndarray, y5070: np.ndarray
) -> dict[str, np.ndarray]:
    """Source-check each raster over the extent, then sample it at the grid centres.

    Logs CRS/res/nodata/finite-fraction/percentiles (post-conversion); aborts on a
    missing required raster, all-nodata over the extent, or a GLHYMPS logk raster
    that is not on the expected x100 scale (a double-conversion / wrong-file guard).
    """
    import rasterio

    samples: dict[str, np.ndarray] = {}
    for name, spec in specs.items():
        path = Path(spec["path"])
        if not path.exists():
            if spec.get("required"):
                raise SystemExit(f"required geology raster missing: {path}")
            log.warning("optional geology raster missing, skipping: %s", path)
            continue
        with rasterio.open(path) as src:
            log.info(
                "%-24s %s crs=%s res=%.0f nodata=%s",
                name,
                path.name,
                src.crs.to_epsg() if src.crs else None,
                src.res[0],
                src.nodata,
            )
        raw = _sample_raster(path, x5070, y5070)
        finite = np.isfinite(raw)
        if not finite.any():
            # A required layer with no signal is a real failure; an optional layer
            # (karst, sediment) may legitimately be all-nodata over a small/clipped
            # footprint -> warn and treat as absent (skipped, like a missing file).
            if spec.get("required"):
                raise SystemExit(f"{name}: all-nodata over the well footprint ({path})")
            log.warning(
                "optional geology raster all-nodata over footprint, skipping: %s", path
            )
            continue
        if spec.get("glhymps_logk"):
            # stored value is logk x100 (~ -1000 .. -1600); a finite median > -50
            # means the file is already logk (unconverted) -> the x0.01 scale is wrong.
            med_raw = float(np.median(raw[finite]))
            if med_raw > -50.0:
                raise SystemExit(
                    f"{name}: median raw value {med_raw:.1f} is not on the x100 logk "
                    f"scale -- looks unconverted ({path})"
                )
        conv = raw * spec.get("scale", 1.0) if spec["kind"] == "cont" else raw
        c = conv[np.isfinite(conv)]
        log.info(
            "  %-22s finite=%.1f%% min=%.3g p50=%.3g p99=%.3g max=%.3g",
            name,
            100.0 * finite.mean(),
            float(np.min(c)),
            float(np.percentile(c, 50)),
            float(np.percentile(c, 99)),
            float(np.max(c)),
        )
        samples[name] = conv
    return samples


# --- grid construction -------------------------------------------------------
def make_grid_centers(
    bounds: tuple[float, float, float, float], res: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Snapped coarse-grid cell centres + (grid_i row, grid_j col) indices.

    Row 0 is the top (max y); grid index space is dense and contiguous so the
    8-neighbour edge build is a pure index-offset lookup.
    """
    minx, miny, maxx, maxy = bounds
    minx = np.floor(minx / res) * res
    miny = np.floor(miny / res) * res
    maxx = np.ceil(maxx / res) * res
    maxy = np.ceil(maxy / res) * res
    nx = int(round((maxx - minx) / res))
    ny = int(round((maxy - miny) / res))
    jj, ii = np.meshgrid(np.arange(nx), np.arange(ny))
    ii = ii.ravel()
    jj = jj.ravel()
    xc = minx + (jj + 0.5) * res
    yc = maxy - (ii + 0.5) * res
    return xc, yc, ii.astype("int32"), jj.astype("int32")


def assemble_nodes(
    samples: dict[str, np.ndarray],
    xc: np.ndarray,
    yc: np.ndarray,
    gi: np.ndarray,
    gj: np.ndarray,
) -> pd.DataFrame:
    """Node table with dense ``aquifer_node_idx``; cells all-nodata in geology dropped.

    A cell is kept if it carries ANY valid geology signal (a present aquifer/rock
    code or any finite continuous field); cells with nothing are noise on the
    coarse grid and would only add dead edges.
    """

    def code(name: str) -> np.ndarray:
        v = samples.get(name)
        if v is None:
            return np.zeros(len(xc), dtype="int32")
        return np.where(np.isfinite(v), v, 0).astype("int32")

    def cont(name: str) -> np.ndarray:
        v = samples.get(name)
        return (
            v.astype("float32")
            if v is not None
            else np.full(len(xc), np.nan, "float32")
        )

    aq_code = code("principal_aquifer_code")
    rt_code = code("aquifer_rocktype_code")
    karst = code("karst_type_code")
    logk = cont("logk_glhymps")
    poros = cont("porosity_glhymps")
    dtb = cont("depth_to_bedrock_m")
    sed = cont("sediment_thickness_m")
    bf = cont("basinfill_thickness_m")

    aq_present = (aq_code > 0).astype("float32")
    rt_present = (rt_code > 0).astype("float32")
    any_geology = (
        (aq_code > 0)
        | (rt_code > 0)
        | np.isfinite(logk)
        | np.isfinite(poros)
        | np.isfinite(dtb)
        | np.isfinite(sed)
        | np.isfinite(bf)
    )
    df = pd.DataFrame(
        {
            "x5070": xc,
            "y5070": yc,
            "grid_i": gi,
            "grid_j": gj,
            "principal_aquifer_code": aq_code,
            "principal_aquifer_present": aq_present,
            "aquifer_rocktype_code": rt_code,
            "aquifer_rocktype_present": rt_present,
            "karst_type_code": karst,
            "logk_glhymps": logk,
            "porosity_glhymps": poros,
            "depth_to_bedrock_m": dtb,
            "sediment_thickness_m": sed,
            "basinfill_thickness_m": bf,
        }
    )[any_geology].reset_index(drop=True)
    df.insert(0, "aquifer_node_idx", np.arange(len(df), dtype="int64"))
    return df


# --- K conductance -----------------------------------------------------------
def harmonic_k_linear(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """Harmonic mean of linear K: 2 / (1/k1 + 1/k2)."""
    return 2.0 / (1.0 / k1 + 1.0 / k2)


def log_harmonic_k(logk1: np.ndarray, logk2: np.ndarray) -> np.ndarray:
    """log10 of the harmonic mean of K from two log10-K endpoints, stable in log space.

    H = 2 / (10^-logk1 + 10^-logk2); computed via logaddexp on natural-log terms so
    the deep-negative logk values (~-12 .. -16) never under/overflow. Never converts
    to linear K and never multiplies by saturated thickness (a static, target-blind
    conductivity only).
    """
    lnk1 = np.asarray(logk1, "float64") * LN10
    lnk2 = np.asarray(logk2, "float64") * LN10
    ln_inv_sum = np.logaddexp(-lnk1, -lnk2)  # ln(1/K1 + 1/K2)
    return (np.log(2.0) - ln_inv_sum) / LN10


# --- edge construction -------------------------------------------------------
# canonical "forward" 8-neighbour offsets (each undirected pair generated once,
# then both directions emitted). 4-neighbour uses the first two.
_OFFSETS_8 = [(0, 1), (1, 0), (1, 1), (1, -1)]
_OFFSETS_4 = [(0, 1), (1, 0)]


def build_grid_edges(
    nodes: pd.DataFrame,
    neighborhood: int = 8,
    conductance_p: float = 0.5,
    barrier: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """8- (or 4-) neighbour aquifer edges, adjacency-gated, with K conductance + components.

    Adjacency gate (kept iff): same principal aquifer where both endpoints have a
    valid code, OR same rock type when one endpoint's principal-aquifer code is
    missing. Barrier (optional, Phase 2): a per-node 0/1 mask -- an edge touching a
    barrier cell is dropped, and a diagonal edge whose two shoulder cells are both
    barriers is dropped (no leak through a barrier corner). Connected components are
    labelled on the kept undirected edges and written back to nodes.

    Returns (edges_df with both directions, component_label_per_node).
    """
    offsets = _OFFSETS_8 if neighborhood == 8 else _OFFSETS_4
    n = len(nodes)
    gi = nodes["grid_i"].to_numpy("int64")
    gj = nodes["grid_j"].to_numpy("int64")
    x = nodes["x5070"].to_numpy("float64")
    y = nodes["y5070"].to_numpy("float64")
    aq = nodes["principal_aquifer_code"].to_numpy("int64")
    rt = nodes["aquifer_rocktype_code"].to_numpy("int64")
    logk = nodes["logk_glhymps"].to_numpy("float64")
    cell = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(gi, gj))}
    is_barrier = (
        barrier.astype(bool) if barrier is not None else np.zeros(n, dtype=bool)
    )

    src, dst = [], []
    for k in range(n):
        i, j = gi[k], gj[k]
        for di, dj in offsets:
            nb = cell.get((int(i + di), int(j + dj)))
            if nb is None:
                continue
            if is_barrier[k] or is_barrier[nb]:
                continue
            if di != 0 and dj != 0:  # diagonal: block if both shoulder cells barrier
                s1 = cell.get((int(i + di), int(j)))
                s2 = cell.get((int(i), int(j + dj)))
                if (
                    s1 is not None
                    and s2 is not None
                    and is_barrier[s1]
                    and is_barrier[s2]
                ):
                    continue
            src.append(k)
            dst.append(nb)
    src = np.asarray(src, "int64")
    dst = np.asarray(dst, "int64")

    both_aq = (aq[src] > 0) & (aq[dst] > 0)
    same_aq = both_aq & (aq[src] == aq[dst])
    same_rt = (rt[src] > 0) & (rt[dst] > 0) & (rt[src] == rt[dst])
    keep = same_aq | (~both_aq & same_rt)
    src, dst = src[keep], dst[keep]
    same_principal = same_aq[keep].astype("float32")
    same_rocktype = same_rt[keep].astype("float32")

    dist = np.hypot(x[src] - x[dst], y[src] - y[dst])
    logk_hmean = log_harmonic_k(logk[src], logk[dst])
    conductance = logk_hmean - conductance_p * np.log1p(dist)

    # components on the kept undirected graph (csgraph treats coo as directed; we
    # symmetrize so a one-direction edge still joins the two cells).
    if len(src):
        a = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
        a = a + a.T
        _, comp = connected_components(a, directed=False)
    else:
        comp = np.arange(n, dtype="int64")
    comp = comp.astype("int64")

    # both directions, matching the channel-edge convention.
    s2 = np.concatenate([src, dst])
    d2 = np.concatenate([dst, src])
    dist2 = np.concatenate([dist, dist])
    sp2 = np.concatenate([same_principal, same_principal])
    srt2 = np.concatenate([same_rocktype, same_rocktype])
    hmean2 = np.concatenate([logk_hmean, logk_hmean])
    cond2 = np.concatenate([conductance, conductance])
    edges = pd.DataFrame(
        {
            "src_aquifer_idx": s2,
            "dst_aquifer_idx": d2,
            "aq_dist_m": dist2.astype("float32"),
            "log1p_aq_dist_m": np.log1p(dist2).astype("float32"),
            "same_principal_aquifer": sp2,
            "same_rocktype": srt2,
            # within-graph edges are same-component by construction (Phase 1 grid
            # edges only); the column is the contract for future cross-component KNN.
            "same_component": np.ones(len(s2), dtype="float32"),
            "logk_hmean": hmean2.astype("float32"),
            "conductance": cond2.astype("float32"),
            "edge_kind": np.zeros(len(s2), dtype="int16"),  # 0 = grid_neighbor
        }
    )
    return edges, comp


# --- aquifer -> query attachment --------------------------------------------
def build_aquifer_query_edges(
    aq_xy: np.ndarray,
    aq_code: np.ndarray,
    aq_comp: np.ndarray,
    q_node_idx: np.ndarray,
    q_xy: np.ndarray,
    q_code: np.ndarray,
    k: int = 4,
    max_dist_m: float = 5000.0,
) -> pd.DataFrame:
    """Attach each query to its k nearest aquifer cells (prefer same principal aquifer).

    Same-principal candidates are found by a **radius** query within ``max_dist_m`` (so a
    same-aquifer cell that ranks far in the global kNN is still considered), taken in
    distance order; the nearest remaining **global** cells then backfill to k. The
    rank-0 selection is the query's *controlling primary* — the nearest same-principal
    cell within range, else the nearest global cell. ``is_controlling`` marks that
    primary and ``same_component`` is membership in the primary's component, so both
    fields share one anchor. Every query gets up to k edges as long as >=1 aquifer node
    exists (100% coverage).
    """
    if len(aq_xy) == 0:
        raise SystemExit("no aquifer nodes to attach queries to")
    tree = cKDTree(aq_xy)
    # Generous global-NN pool for backfill + the always-attach guarantee.
    kk = min(len(aq_xy), max(k * 4, 16))
    g_dists, g_idxs = tree.query(q_xy, k=kk)
    if kk == 1:
        g_dists = g_dists[:, None]
        g_idxs = g_idxs[:, None]
    # All cells within max_dist_m, regardless of global-kNN rank (the P2a fix).
    within = tree.query_ball_point(q_xy, r=max_dist_m)

    rows = []
    for qi in range(len(q_xy)):
        qc = q_code[qi]
        gi = g_idxs[qi]
        order = np.argsort(g_dists[qi], kind="stable")
        gi = gi[order]
        # same-principal candidates in range, distance-sorted
        cand = np.asarray(within[qi], dtype="int64")
        if qc > 0 and len(cand):
            cand = cand[(aq_code[cand] == qc) & (aq_code[cand] > 0)]
            if len(cand):
                cd = np.linalg.norm(aq_xy[cand] - q_xy[qi], axis=1)
                cand = cand[np.argsort(cd, kind="stable")]
        else:
            cand = np.empty(0, dtype="int64")
        # backfill with nearest global cells not already chosen
        chosen: list[int] = [int(c) for c in cand[:k]]
        seen = set(chosen)
        for c in gi:
            if len(chosen) >= k:
                break
            ci = int(c)
            if ci not in seen:
                chosen.append(ci)
                seen.add(ci)
        sel = np.asarray(chosen[:k], dtype="int64")
        # single anchor for both fields: the rank-0 (controlling) primary.
        home_comp = aq_comp[sel[0]]
        for rank, pos in enumerate(sel):
            dist = float(np.linalg.norm(aq_xy[pos] - q_xy[qi]))
            same_p = qc > 0 and aq_code[pos] == qc and aq_code[pos] > 0
            rows.append(
                (
                    int(pos),
                    int(q_node_idx[qi]),
                    dist,
                    int(rank),
                    1.0 if rank == 0 else 0.0,
                    1.0 if same_p else 0.0,
                    1.0 if aq_comp[pos] == home_comp else 0.0,
                )
            )
    out = pd.DataFrame(
        rows,
        columns=[
            "aquifer_node_idx",
            "query_node_idx",
            "aq_query_dist_m",
            "rank",
            "is_controlling",
            "same_principal_aquifer",
            "same_component",
        ],
    )
    out["log1p_aq_query_dist_m"] = np.log1p(out["aq_query_dist_m"]).astype("float32")
    out["aq_query_dist_m"] = out["aq_query_dist_m"].astype("float32")
    out["rank"] = out["rank"].astype("int16")
    for c in ("is_controlling", "same_principal_aquifer", "same_component"):
        out[c] = out[c].astype("float32")
    return out


# --- manifest ----------------------------------------------------------------
def update_manifest(gdir: Path, block: dict | None) -> None:
    """Add/replace the optional ``aquifer`` block in graph_manifest.json (only)."""
    man_path = gdir / "graph_manifest.json"
    man = json.loads(man_path.read_text())
    man["aquifer"] = block
    man_path.write_text(json.dumps(man, indent=2))


def _remove_stale(gdir: Path) -> None:
    for f in (
        "aquifer_nodes.parquet",
        "aquifer_edges.parquet",
        "aquifer_to_query_edges.parquet",
    ):
        p = gdir / f
        if p.exists():
            p.unlink()
            log.info("removed stale aquifer file: %s", p.name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir", required=True, help="existing bundle to augment")
    p.add_argument("--covariate-root", default="/nas/handily/covariates")
    p.add_argument("--grid-res-m", type=float, default=1000.0)
    p.add_argument(
        "--extent-source",
        choices=["query-buffer"],
        default="query-buffer",
        help="query-buffer: cells within --extent-buffer-km of any retained well",
    )
    p.add_argument("--extent-buffer-km", type=float, default=25.0)
    p.add_argument("--knn-aquifer-query", type=int, default=4)
    p.add_argument("--max-aquifer-query-dist-m", type=float, default=5000.0)
    p.add_argument("--edge-neighborhood", type=int, choices=[4, 8], default=8)
    p.add_argument("--conductance-p", type=float, default=0.5)
    p.add_argument(
        "--allow-missing-aquifer",
        action="store_true",
        help="write a missingness flag instead of failing if a query gets no edge",
    )
    p.add_argument(
        "--remove",
        action="store_true",
        help="delete stale aquifer files + set manifest aquifer=null, then exit",
    )
    args = p.parse_args()

    gdir = Path(args.graph_dir)
    if not (gdir / "graph_manifest.json").exists():
        raise SystemExit(f"no graph_manifest.json in {gdir}")
    if args.remove:
        _remove_stale(gdir)
        update_manifest(gdir, None)
        log.info("aquifer files removed; manifest aquifer=null")
        return

    assert_target_blind(AQUIFER_FEATURE_COLS)
    geo = Path(args.covariate_root) / "geology"
    res = args.grid_res_m
    buffer_m = args.extent_buffer_km * 1000.0

    qn = pd.read_parquet(gdir / "query_nodes.parquet")
    qx = qn["x5070"].to_numpy("float64")
    qy = qn["y5070"].to_numpy("float64")
    if not (np.isfinite(qx).all() and np.isfinite(qy).all()):
        raise SystemExit("query nodes carry non-finite x5070/y5070")
    q_node_idx = qn["query_node_idx"].to_numpy("int64")
    log.info("queries=%d; grid res=%.0fm buffer=%.0fkm", len(qn), res, buffer_m / 1000)

    # --- grid over the well footprint, masked to within buffer of a well ----------
    bounds = (
        qx.min() - buffer_m,
        qy.min() - buffer_m,
        qx.max() + buffer_m,
        qy.max() + buffer_m,
    )
    xc, yc, gi, gj = make_grid_centers(bounds, res)
    well_tree = cKDTree(np.column_stack([qx, qy]))
    nn_dist, _ = well_tree.query(np.column_stack([xc, yc]), k=1)
    within = nn_dist <= buffer_m
    xc, yc, gi, gj = xc[within], yc[within], gi[within], gj[within]
    log.info(
        "grid cells within buffer: %d (of %d in bbox)", int(within.sum()), within.size
    )

    # --- sample geology, drop all-nodata cells, build nodes -----------------------
    specs = _raster_specs(geo)
    samples = check_and_sample(specs, xc, yc)
    nodes = assemble_nodes(samples, xc, yc, gi, gj)
    log.info("aquifer nodes after drop-all-nodata: %d", len(nodes))

    # --- edges + components -------------------------------------------------------
    edges, comp = build_grid_edges(
        nodes, neighborhood=args.edge_neighborhood, conductance_p=args.conductance_p
    )
    nodes["aquifer_component_id"] = comp[nodes["aquifer_node_idx"].to_numpy()]
    n_comp = int(len(np.unique(comp[nodes["aquifer_node_idx"].to_numpy()])))
    _, csize = np.unique(nodes["aquifer_component_id"], return_counts=True)
    log.info(
        "edges=%d (both dirs); components=%d largest=%.1f%%",
        len(edges),
        n_comp,
        100.0 * csize.max() / len(nodes),
    )

    # --- aquifer -> query attachment ---------------------------------------------
    aq_code_q = _sample_raster(geo / "aquifer_code.tif", qx, qy)
    aq_code_q = np.where(np.isfinite(aq_code_q), aq_code_q, 0).astype("int64")
    aq_xy = nodes[["x5070", "y5070"]].to_numpy("float64")
    qedges = build_aquifer_query_edges(
        aq_xy,
        nodes["principal_aquifer_code"].to_numpy("int64"),
        nodes["aquifer_component_id"].to_numpy("int64"),
        q_node_idx,
        np.column_stack([qx, qy]),
        aq_code_q,
        k=args.knn_aquifer_query,
        max_dist_m=args.max_aquifer_query_dist_m,
    )
    covered = qedges["query_node_idx"].nunique()
    if covered != len(qn) and not args.allow_missing_aquifer:
        raise SystemExit(
            f"only {covered}/{len(qn)} queries got an aquifer edge "
            f"(pass --allow-missing-aquifer to write a missingness flag instead)"
        )
    ctrl = qedges[qedges["is_controlling"] == 1.0]
    log.info(
        "query attachment: %d/%d covered (%.1f%%); same-principal edge frac=%.2f; "
        "controlling dist median=%.0fm p95=%.0fm; beyond max-dist=%d",
        covered,
        len(qn),
        100.0 * covered / len(qn),
        float(qedges["same_principal_aquifer"].mean()),
        float(ctrl["aq_query_dist_m"].median()),
        float(ctrl["aq_query_dist_m"].quantile(0.95)),
        int((ctrl["aq_query_dist_m"] > args.max_aquifer_query_dist_m).sum()),
    )

    # --- write files + manifest ---------------------------------------------------
    assert (nodes["aquifer_node_idx"].to_numpy() == np.arange(len(nodes))).all()
    for cols, df, what in (
        (AQUIFER_FEATURE_COLS, nodes, "node"),
        (AQUIFER_EDGE_FEATURE_COLS, edges, "edge"),
        (AQUIFER_QUERY_EDGE_FEATURE_COLS, qedges, "query-edge"),
    ):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise SystemExit(f"declared aquifer {what} feature cols absent: {missing}")
    nodes.to_parquet(gdir / "aquifer_nodes.parquet")
    edges.to_parquet(gdir / "aquifer_edges.parquet")
    qedges.to_parquet(gdir / "aquifer_to_query_edges.parquet")

    block = {
        "enabled": True,
        "builder": "build_regional_aquifer_graph.py",
        "node_file": "aquifer_nodes.parquet",
        "edge_file": "aquifer_edges.parquet",
        "query_edge_file": "aquifer_to_query_edges.parquet",
        "aquifer_feature_cols": AQUIFER_FEATURE_COLS,
        "aquifer_edge_feature_cols": AQUIFER_EDGE_FEATURE_COLS,
        "aquifer_query_edge_feature_cols": AQUIFER_QUERY_EDGE_FEATURE_COLS,
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "query_edge_count": int(len(qedges)),
        "component_count": n_comp,
        "largest_component_frac": float(csize.max() / len(nodes)),
        "grid_resolution_m": res,
        "extent_source": args.extent_source,
        "extent_buffer_km": args.extent_buffer_km,
        "edge_neighborhood": args.edge_neighborhood,
        "knn_aquifer_query": args.knn_aquifer_query,
        "max_aquifer_query_dist_m": args.max_aquifer_query_dist_m,
        "conductance_p": args.conductance_p,
        "query_coverage_frac": float(covered / len(qn)),
        "same_principal_query_edge_frac": float(
            qedges["same_principal_aquifer"].mean()
        ),
        "data_sources": {k: str(v["path"]) for k, v in specs.items()},
        "leakage_notes": [
            "Aquifer node/edge features are target-blind only in this build.",
            "No observed WTE/DTW, Ma, Janssen, or deep-OOF labels are stored on "
            "aquifer nodes; only static GLHYMPS/USGS/Pelletier geology + 0/1 flags.",
            "Aquifer tensors are constant across CV folds (no label to hold out).",
        ],
    }
    update_manifest(gdir, block)
    log.info(
        "wrote aquifer_nodes/edges/to_query_edges + manifest aquifer block -> %s", gdir
    )


if __name__ == "__main__":
    main()
