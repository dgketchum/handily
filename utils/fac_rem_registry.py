"""Unified raster source for the FAC-REM shallow water-table prior.

The GNN's FAC feature and the wall-to-wall inference grid must source FAC-REM
from the SAME place, with the SAME precedence, or train/infer skew silently
corrupts the prediction. This module is that single source: the per-basin 10 m
EPSG:5070 ``fac_head_depth_rem_10m.tif`` rasters and one function,
``sample_fac_rem(x5070, y5070) -> fac_rem_dtw_m``, that samples them in an
explicit best->worst order (first finite hit wins).

This replaces the ``build_stacker_features.join_fac`` shard concatenation for the
GNN path. ``join_fac`` deduped overlapping shards by lexical filename sort
(``drop_duplicates(keep="first")``), so a digit-named shard silently won the
overlap over a letter-named one -- a precedence bug. Here the precedence is the
explicit ``FAC_REM_REGISTRY`` order, identical at train and inference time.

FAC-REM is a terrain-following HAND DEPTH below ground (metres, >= 0), NOT an
elevation. The rasters carry no nodata tag; out-of-channel / out-of-grid cells
are non-finite, so ``np.isfinite`` is the in-band-gap filter (the same rule
``build_conus_fac_rem._sample_wells`` uses).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio

log = logging.getLogger("fac_rem_registry")

_SCALABLE_ROOT = "/data/ssd2/handily/scalable_fac_rem"


def _fac_path(basin: str) -> str:
    return f"{_SCALABLE_ROOT}/{basin}/rem/{basin}_scalable/fac_head_depth_rem_10m.tif"


# Ordered best -> worst. First finite sample wins where basins overlap (they do
# not today, but the order is the contract). Grows as basins are built; keep the
# 5 native-10 m EPSG:5070 scalable products that back the wall-to-wall product.
FAC_REM_REGISTRY: list[str] = [
    _fac_path(b)
    for b in (
        "nm_rio_grande_abq",
        "nv_upper_humboldt",
        "mt_big_hole",
        "mt_beaverhead",
        "mt_ruby",
    )
]


def existing_registry(registry: list[str] | None = None) -> list[str]:
    """The registry entries that are actually on disk (warn + skip the rest)."""
    reg = registry if registry is not None else FAC_REM_REGISTRY
    out = []
    for p in reg:
        if Path(p).exists():
            out.append(p)
        else:
            log.warning("FAC-REM registry entry missing, skipping: %s", p)
    return out


def _sample_one(path: str, x5070: np.ndarray, y5070: np.ndarray) -> np.ndarray:
    """Nearest-cell sample of a single 10 m EPSG:5070 FAC-REM depth raster.

    Reads only the windowed rows the points touch is not worth it at this point
    density; the raster is opened, the point rows/cols are computed against its
    transform, and only in-bounds points are read cell-by-cell so a multi-GB
    raster is never fully materialised. Out-of-grid + non-finite -> NaN.
    """
    out = np.full(x5070.shape, np.nan, dtype="float64")
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 5070:
            raise SystemExit(f"FAC-REM raster not EPSG:5070: {path} (crs={src.crs})")
        t = src.transform
        h, w = src.height, src.width
        col = np.floor((x5070 - t.c) / t.a).astype(np.int64)
        row = np.floor((y5070 - t.f) / t.e).astype(np.int64)
        ok = (row >= 0) & (row < h) & (col >= 0) & (col < w)
        if not ok.any():
            return out
        # sample() reprojection-free read at native coords; pass (x, y) pairs.
        pts = list(zip(x5070[ok], y5070[ok]))
        vals = np.array([v[0] for v in src.sample(pts)], dtype="float64")
        nd = src.nodata
        if nd is not None:
            vals[vals == nd] = np.nan
        vals[~np.isfinite(vals)] = np.nan
        out[ok] = vals
    return out


def sample_fac_rem(
    x5070: np.ndarray, y5070: np.ndarray, registry: list[str] | None = None
) -> np.ndarray:
    """FAC-REM depth (m) at EPSG:5070 coords; registry precedence, first finite wins.

    Identical sourcing for training well nodes and the inference grid -- no shard
    concatenation, no lexical-sort precedence. Points outside every registered
    basin stay NaN (the trainer median-imputes + flags via the NaN-indicator).
    """
    x5070 = np.asarray(x5070, dtype="float64")
    y5070 = np.asarray(y5070, dtype="float64")
    out = np.full(x5070.shape, np.nan, dtype="float64")
    for path in existing_registry(registry):
        need = ~np.isfinite(out)
        if not need.any():
            break
        vals = _sample_one(path, x5070[need], y5070[need])
        idx = np.where(need)[0]
        out[idx] = vals
        log.info(
            "FAC-REM %s: filled %d/%d remaining",
            Path(path).parents[1].name,
            int(np.isfinite(vals).sum()),
            int(need.sum()),
        )
    return out
