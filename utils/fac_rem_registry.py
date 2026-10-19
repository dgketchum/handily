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
import os
from pathlib import Path

import numpy as np
import rasterio

log = logging.getLogger("fac_rem_registry")

_SCALABLE_ROOT = "/data/ssd2/handily/huc8"


# FAC-REM recipe pointer. Each basin resolves to
# rem/{basin}_{HANDILY_FAC_RECIPE}/ when that product exists and falls back to the
# baseline rem/{basin}_scalable/ otherwise, so a variant recipe rebuilt on a
# subset of basins (e.g. scalable_v2 on the NV closed basins) overlays the
# baseline without touching it. Read once at import (the registry is built at
# import); set the env var before importing.
FAC_RECIPE_BASELINE = "scalable"
FAC_RECIPE = os.environ.get("HANDILY_FAC_RECIPE", "").strip() or FAC_RECIPE_BASELINE


def _fac_recipe_path(basin: str, recipe: str) -> str:
    return f"{_SCALABLE_ROOT}/{basin}/rem/{basin}_{recipe}/fac_head_depth_rem_10m.tif"


def _fac_path(basin: str) -> str:
    """Preferred-recipe raster when it exists, else the baseline ``_scalable`` one."""
    if FAC_RECIPE != FAC_RECIPE_BASELINE:
        pref = _fac_recipe_path(basin, FAC_RECIPE)
        if Path(pref).exists():
            return pref
    return _fac_recipe_path(basin, FAC_RECIPE_BASELINE)


# The 5 native-10 m EPSG:5070 pilots that back the wall-to-wall product. Listed
# first so they take precedence (first finite sample wins) where a pilot overlaps
# its HUC8-code equivalent.
_NAMED_PILOTS = (
    "nm_rio_grande_abq",
    "nv_upper_humboldt",
    "mt_big_hole",
    "mt_beaverhead",
    "mt_ruby",
)


def _discover(path_fn) -> list[str]:
    """Auto-discover every basin under huc8/ whose raster exists, ordered best->worst.

    Named pilots first (precedence contract), then the remaining basins (HUC8 codes)
    sorted. Globbing the root means new HUC8 builds are picked up automatically -- the
    registry no longer needs hand-editing per basin (see notes/HUC8_DATA.md).
    """
    root = Path(_SCALABLE_ROOT)
    out: list[str] = []
    seen: set[str] = set()
    for b in _NAMED_PILOTS:
        p = path_fn(b)
        if Path(p).exists():
            out.append(p)
        seen.add(b)
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name in seen:
                continue
            p = path_fn(d.name)
            if Path(p).exists():
                out.append(p)
    return out


def _recipe_of(path: str) -> tuple[str, str]:
    """(basin, recipe) from ``.../{basin}/rem/{basin}_{recipe}/<raster>``."""
    p = Path(path)
    basin = p.parents[2].name
    return basin, p.parent.name[len(basin) + 1 :]


# Registry pin. The registry is auto-discovered from whatever is on disk, so two
# machines with different basin sets staged render DIFFERENT maps at raster
# overlaps (first finite sample wins). HANDILY_FAC_PIN names a manifest written
# by utils/pin_fac_registry.py (``basin<TAB>recipe`` per line, '#' comments):
# the registry is restricted to exactly those basins, in discovery order, and
# import fails if any pinned basin is missing or resolves to another recipe.
# Same manifest on both machines => same registry => same map.
FAC_PIN = os.environ.get("HANDILY_FAC_PIN", "").strip() or None


def read_pin(path: str) -> dict[str, str]:
    """``{basin: recipe}`` from a pin manifest (order = file order)."""
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        basin, recipe = line.split("\t")[:2]
        out[basin] = recipe
    return out


def _apply_pin(registry: list[str], pin_path: str) -> list[str]:
    pin = read_pin(pin_path)
    found = {_recipe_of(p)[0]: p for p in registry}
    missing = sorted(b for b in pin if b not in found)
    wrong = sorted(
        f"{b}: pinned {r}, resolved {_recipe_of(found[b])[1]}"
        for b, r in pin.items()
        if b in found and _recipe_of(found[b])[1] != r
    )
    if missing or wrong:
        raise SystemExit(
            f"HANDILY_FAC_PIN={pin_path}: registry does not satisfy the pin; "
            f"missing {len(missing)} {missing[:10]}; recipe mismatch {len(wrong)} {wrong[:10]}"
        )
    kept = [p for p in registry if _recipe_of(p)[0] in pin]
    log.info(
        "FAC-REM registry pinned to %s: %d of %d discovered basins kept",
        pin_path,
        len(kept),
        len(registry),
    )
    return kept


# Ordered best -> worst. First finite sample wins where basins overlap.
FAC_REM_REGISTRY: list[str] = _discover(_fac_path)
if FAC_PIN:
    FAC_REM_REGISTRY = _apply_pin(FAC_REM_REGISTRY, FAC_PIN)

# basin -> recipe each registry entry resolved to (for run records / renderers).
FAC_REM_RECIPES: dict[str, str] = dict(_recipe_of(p) for p in FAC_REM_REGISTRY)


def fac_rem_recipe(basin: str) -> str | None:
    """Recipe the registry resolved ``basin`` to, or None if it has no FAC-REM."""
    return FAC_REM_RECIPES.get(basin)


def _log_recipe_resolution() -> None:
    n_pref = sum(r == FAC_RECIPE for r in FAC_REM_RECIPES.values())
    if FAC_RECIPE != FAC_RECIPE_BASELINE:
        for basin, recipe in FAC_REM_RECIPES.items():
            log.info("FAC-REM recipe %s -> %s", basin, recipe)
    log.info(
        "FAC-REM recipe pointer HANDILY_FAC_RECIPE=%s: %d basins on it, "
        "%d fell back to %s (%d total)",
        FAC_RECIPE,
        n_pref,
        len(FAC_REM_RECIPES) - n_pref,
        FAC_RECIPE_BASELINE,
        len(FAC_REM_RECIPES),
    )


_log_recipe_resolution()


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


def _str_top2_wte_path(basin: str) -> str:
    return f"{_SCALABLE_ROOT}/{basin}/str_top2_idw_wte_100m.tif"


# The well-free streams-Strahler regional WTE prior R (top-2 Strahler orders;
# build_str7_idw_raster.py). Same basins/precedence contract as FAC_REM_REGISTRY,
# but a 100 m EPSG:5070 water-table ELEVATION (not a depth). It consumes zero well
# labels, so as the GNN's regional base it is leakage-free and needs no cross-fit.
STR_TOP2_WTE_REGISTRY: list[str] = _discover(_str_top2_wte_path)


def sample_str_top2_wte(
    x5070: np.ndarray, y5070: np.ndarray, registry: list[str] | None = None
) -> np.ndarray:
    """Streams-Strahler regional WTE elevation (m) at EPSG:5070 coords.

    Registry precedence, first finite hit wins (the basins do not overlap today).
    Points off the trunk-supported valley (no top-2 Strahler anchor within the
    prior's dmax) stay NaN -- the caller decides explicitly (the GNN builder drops
    those wells with an audited log, never silently).
    """
    reg = STR_TOP2_WTE_REGISTRY if registry is None else registry
    x5070 = np.asarray(x5070, dtype="float64")
    y5070 = np.asarray(y5070, dtype="float64")
    out = np.full(x5070.shape, np.nan, dtype="float64")
    for path in existing_registry(reg):
        need = ~np.isfinite(out)
        if not need.any():
            break
        vals = _sample_one(path, x5070[need], y5070[need])
        idx = np.where(need)[0]
        out[idx] = vals
        log.info(
            "str_top2 WTE %s: filled %d/%d remaining",
            Path(path).parents[0].name,
            int(np.isfinite(vals).sum()),
            int(need.sum()),
        )
    return out
