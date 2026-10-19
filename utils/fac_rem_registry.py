"""FAC-REM raster manifest: the explicit list of rasters an arm's FAC feature came from.

The GNN's FAC feature and the wall-to-wall inference grid must source FAC-REM
from the SAME rasters, with the SAME precedence, or train/infer skew silently
corrupts the prediction. The list is therefore an ARTEFACT, not ambient disk
state: a bundle builder writes ``fac_rem_manifest.json`` (ordered entries with
basin, recipe, path, byte size and EPSG:5070 bounds), the trainer copies it into
the arm directory, and every consumer (inference, 10 m render, point prediction)
reads it from there and refuses to run if a raster it needs is missing or has
changed size. Two machines holding the same manifest and the same files render
the same map; there is no discovery, fallback, pilot precedence or environment
variable involved at consumption time.

``discover(recipe)`` builds a fresh manifest from disk for a bundle build: every
HUC8 under ``huc8/`` that has ``rem/{basin}_{recipe}/fac_head_depth_rem_10m.tif``,
sorted by code. Basins without that recipe are NOT in it unless an explicit
``fallback`` recipe is named (each entry records which recipe it is on, and the
manifest counts them). The five historical named pilots are excluded unless
``legacy_pilots`` is requested to reproduce a pre-2026-10 arm.

``sample_fac_rem(x, y, registry)`` samples the ordered raster paths, first
finite hit wins. FAC-REM is a terrain-following HAND DEPTH below ground (metres,
>= 0), NOT an elevation. The rasters carry no nodata tag; out-of-channel /
out-of-grid cells are non-finite, so ``np.isfinite`` is the in-band-gap filter.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import rasterio

log = logging.getLogger("fac_rem_registry")

_SCALABLE_ROOT = "/data/ssd2/handily/huc8"
MANIFEST_NAME = "fac_rem_manifest.json"
RASTER_NAME = "fac_head_depth_rem_10m.tif"

# Pre-HUC8 hand-built 10 m pilots. Historical arms sampled them FIRST (ahead of the
# HUC8 raster beneath). New manifests leave them out; ``legacy_pilots=True`` puts
# them back in front for reproducing those arms only.
_NAMED_PILOTS = (
    "nm_rio_grande_abq",
    "nv_upper_humboldt",
    "mt_big_hole",
    "mt_beaverhead",
    "mt_ruby",
)


def raster_path(basin: str, recipe: str, root: str = _SCALABLE_ROOT) -> str:
    return f"{root}/{basin}/rem/{basin}_{recipe}/{RASTER_NAME}"


def _entry(basin: str, recipe: str, path: str) -> dict:
    with rasterio.open(path) as src:
        if src.crs is None or src.crs.to_epsg() != 5070:
            raise SystemExit(f"FAC-REM raster not EPSG:5070: {path} (crs={src.crs})")
        b = src.bounds
    return {
        "basin": basin,
        "recipe": recipe,
        "path": path,
        "bytes": Path(path).stat().st_size,
        "bounds": [b.left, b.bottom, b.right, b.top],
    }


def discover(
    recipe: str,
    root: str = _SCALABLE_ROOT,
    *,
    fallback: str | None = None,
    legacy_pilots: bool = False,
) -> list[dict]:
    """Ordered manifest entries for every basin holding ``recipe`` under ``root``.

    Sorted HUC8 codes. A basin without ``recipe`` is listed on ``fallback`` when
    that is named and present, otherwise left out. With ``legacy_pilots`` the
    named pilots are listed first (historical precedence).
    """
    rootp = Path(root)

    def pick(basin: str) -> dict | None:
        for r in (recipe, fallback):
            if r is None:
                continue
            p = raster_path(basin, r, root)
            if Path(p).exists():
                return _entry(basin, r, p)
        return None

    out: list[dict] = []
    if legacy_pilots:
        out += [e for e in map(pick, _NAMED_PILOTS) if e is not None]
    if rootp.is_dir():
        for d in sorted(rootp.iterdir()):
            if not d.is_dir() or d.name in _NAMED_PILOTS:
                continue
            e = pick(d.name)
            if e is not None:
                out.append(e)
    log.info(
        "FAC-REM discover recipe=%s fallback=%s root=%s: %s",
        recipe,
        fallback,
        root,
        count_by_recipe(out),
    )
    return out


def count_by_recipe(entries: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in entries:
        out[e["recipe"]] = out.get(e["recipe"], 0) + 1
    return out


def write_manifest(
    path: str | Path,
    entries: list[dict],
    recipe: str,
    note: str = "",
    fallback_recipe: str | None = None,
) -> Path:
    doc = {
        "recipe": recipe,
        "fallback_recipe": fallback_recipe,
        "n_by_recipe": count_by_recipe(entries),
        "written": dt.datetime.now().isoformat(timespec="seconds"),
        "n": len(entries),
        "bytes_total": int(sum(e["bytes"] for e in entries)),
        "precedence": "list order; first finite sample wins at overlaps",
        "note": note,
        "entries": entries,
    }
    Path(path).write_text(json.dumps(doc, indent=1))
    return Path(path)


def load_manifest(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(
            f"FAC-REM manifest missing: {p}. Write one with utils/write_fac_manifest.py "
            "(bundle builders write it; the trainer copies it into the arm)."
        )
    doc = json.loads(p.read_text())
    if "entries" not in doc or "recipe" not in doc:
        raise SystemExit(f"not a FAC-REM manifest: {p}")
    return doc


def manifest_for_bundle(bundle_dir: str | Path) -> Path:
    """The manifest a bundle was built with (``<bundle>/fac_rem_manifest.json``)."""
    return Path(bundle_dir) / MANIFEST_NAME


def manifest_for_model(model_dir: str | Path, override: str | None = None) -> Path:
    """The manifest an arm carries; ``override`` for arms trained before it existed."""
    if override:
        return Path(override)
    p = Path(model_dir) / MANIFEST_NAME
    if not p.is_file():
        raise SystemExit(
            f"{Path(model_dir).name} carries no {MANIFEST_NAME}. Arms trained before "
            "2026-10-18 need --fac-manifest <path> (write one with "
            "utils/write_fac_manifest.py --recipe scalable --legacy-pilots to "
            "reproduce their sourcing)."
        )
    return p


def select_paths(
    entries: list[dict],
    bounds: tuple[float, float, float, float] | None = None,
    *,
    strict: bool = True,
) -> list[str]:
    """Ordered raster paths from ``entries`` that can touch ``bounds`` (all if None).

    ``strict``: every selected raster must exist with the recorded byte size, else
    SystemExit naming the offenders (stage the file, or the arm's FAC changed).
    """
    sel = []
    for e in entries:
        if bounds is not None:
            left, bottom, right, top = e["bounds"]
            x0, y0, x1, y1 = bounds
            if left >= x1 or right <= x0 or bottom >= y1 or top <= y0:
                continue
        sel.append(e)
    if strict:
        bad = []
        for e in sel:
            p = Path(e["path"])
            if not p.is_file():
                bad.append(f"{e['basin']}: missing {p}")
            elif p.stat().st_size != e["bytes"]:
                bad.append(
                    f"{e['basin']}: {p.stat().st_size} bytes on disk vs {e['bytes']} in manifest"
                )
        if bad:
            raise SystemExit(
                f"FAC-REM manifest not satisfied by this machine ({len(bad)} of {len(sel)} "
                f"needed rasters): " + "; ".join(bad[:10])
            )
    return [e["path"] for e in sel]


def _sample_one(path: str, x5070: np.ndarray, y5070: np.ndarray) -> np.ndarray:
    """Nearest-cell sample of a single 10 m EPSG:5070 FAC-REM depth raster.

    The raster is opened, the point rows/cols are computed against its transform,
    and only in-bounds points are read cell-by-cell so a multi-GB raster is never
    fully materialised. Out-of-grid + non-finite -> NaN.
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
        pts = list(zip(x5070[ok], y5070[ok]))
        vals = np.array([v[0] for v in src.sample(pts)], dtype="float64")
        nd = src.nodata
        if nd is not None:
            vals[vals == nd] = np.nan
        vals[~np.isfinite(vals)] = np.nan
        out[ok] = vals
    return out


def sample_fac_rem(
    x5070: np.ndarray, y5070: np.ndarray, registry: list[str]
) -> np.ndarray:
    """FAC-REM depth (m) at EPSG:5070 coords from the ordered ``registry`` paths.

    First finite hit wins. ``registry`` comes from ``select_paths`` on a manifest --
    the same list for training wells and the inference grid. Points outside every
    raster stay NaN (the trainer median-imputes + flags via the NaN-indicator).
    """
    x5070 = np.asarray(x5070, dtype="float64")
    y5070 = np.asarray(y5070, dtype="float64")
    out = np.full(x5070.shape, np.nan, dtype="float64")
    for path in registry:
        need = ~np.isfinite(out)
        if not need.any():
            break
        vals = _sample_one(path, x5070[need], y5070[need])
        idx = np.where(need)[0]
        out[idx] = vals
        log.info(
            "FAC-REM %s: filled %d/%d remaining",
            Path(path).parents[0].name,
            int(np.isfinite(vals).sum()),
            int(need.sum()),
        )
    return out


# ---------------------------------------------------------------------------
# str_top2 regional WTE prior (well-free; separate product, unchanged contract)
# ---------------------------------------------------------------------------
def _str_top2_wte_path(basin: str) -> str:
    return f"{_SCALABLE_ROOT}/{basin}/str_top2_idw_wte_100m.tif"


def _discover_paths(path_fn) -> list[str]:
    """Named pilots first, then sorted HUC8 dirs, keeping paths that exist."""
    root = Path(_SCALABLE_ROOT)
    out: list[str] = []
    for b in _NAMED_PILOTS:
        p = path_fn(b)
        if Path(p).exists():
            out.append(p)
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if not d.is_dir() or d.name in _NAMED_PILOTS:
                continue
            p = path_fn(d.name)
            if Path(p).exists():
                out.append(p)
    return out


# The well-free streams-Strahler regional WTE prior R (top-2 Strahler orders;
# build_str7_idw_raster.py). A 100 m EPSG:5070 water-table ELEVATION (not a
# depth). It consumes zero well labels, so as the GNN's regional base it is
# leakage-free and needs no cross-fit.
STR_TOP2_WTE_REGISTRY: list[str] = _discover_paths(_str_top2_wte_path)


def sample_str_top2_wte(
    x5070: np.ndarray, y5070: np.ndarray, registry: list[str] | None = None
) -> np.ndarray:
    """Streams-Strahler regional WTE elevation (m) at EPSG:5070 coords.

    Registry precedence, first finite hit wins (the basins do not overlap today).
    Points off the trunk-supported valley stay NaN -- the caller decides
    explicitly (the GNN builder drops those wells with an audited log).
    """
    reg = STR_TOP2_WTE_REGISTRY if registry is None else registry
    x5070 = np.asarray(x5070, dtype="float64")
    y5070 = np.asarray(y5070, dtype="float64")
    out = np.full(x5070.shape, np.nan, dtype="float64")
    for path in reg:
        if not Path(path).exists():
            log.warning("str_top2 registry entry missing, skipping: %s", path)
            continue
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
