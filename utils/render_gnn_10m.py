"""Phase 3 of notes/GNN_INFERENCE_10M_PLAN.md: exact 10 m render of the GNN DTW.

CPU-only, re-runnable per basin without touching the GNN. The coarse 100 m
lattice (infer_conus_gnn.py) carries the smooth pieces of the prior-gate
mixture -- gate weights, deep-IDW expert, free-head expert, sigma -- and the
recomposition identity (wte = sum_i w_i * expert_wte_i, asserted <1e-3 m at
every coarse cell before those layers were written) makes the 10 m render
exact rather than heuristic:

  fac_wte_10    = DEM_10 - fac_dtw_10     (fine: native 10 m FAC-REM raster)
  mirror_wte_10 = DEM_10 - d              (fine: 10 m DEM minus a constant)
  num           = w_deep*deep_wte + w_mirror*mirror_wte_10 + w_head*head_wte
  WTE_10        = w_fac*fac_wte_10 + num          where 10 m FAC is finite
                = num / (1 - w_fac)               where it is not (masked-
                                                   softmax renormalization)
  guard: 1 - w_fac < 0.05 -> head_wte alone (coarse/fine footprint
  disagreement; counted and logged, never hidden)

Smooth layers upsample bilinearly; DEM warps bilinearly onto the master grid
(per-basin 10 m origins are arbitrary -- this is the alignment fix); FAC warps
NEAREST so the sparse channel-strip footprint is not eroded by NaN edges. FAC
sourcing composites the fac_rem_registry in precedence order (first finite
wins), matching sample_fac_rem at the wells exactly.

Outputs per basin -> gnn_wte_10m.tif, gnn_dtw_10m.tif (headline, unclamped),
gnn_sigma_10m.tif. A coarse-consistency check (block-mean WTE_10 vs
gnn_wte_100m, MAD split by FAC presence) runs every basin and is logged +
persisted to render_run.json.

Usage:
    uv run python utils/render_gnn_10m.py --model-dir <...> --basins 13020102
    uv run python utils/render_gnn_10m.py --model-dir <...> --state NM
    uv run python utils/render_gnn_10m.py --model-dir <...> --state NM --mosaic
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fac_rem_registry import existing_registry  # noqa: E402
from infer_conus_gnn import (  # noqa: E402
    HUC8_ROOT,
    RES,
    X0,
    Y0,
    nm_basins,
    write_tif,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("render_gnn_10m")

RES10 = 10.0
HEAD_FALLBACK_W = 0.05  # 1 - w_fac below this -> head_wte fallback (plan step 4)


def master_window(coarse_path: Path) -> tuple:
    """The basin's 100 m window x10 finer; nesting is exact by construction."""
    with rasterio.open(coarse_path) as src:
        b = src.bounds
        if (b.left - X0) % RES or (b.top - Y0) % RES:
            raise SystemExit(f"coarse grid not on the canonical origin: {coarse_path}")
        return (
            from_origin(b.left, b.top, RES10, RES10),
            src.width * 10,
            src.height * 10,
        )


def warp_band(
    path: Path, band: int, transform, width: int, height: int, resampling
) -> np.ndarray:
    """One band -> the master window, NaN nodata in and out."""
    dst = np.full((height, width), np.nan, dtype="float32")
    with rasterio.open(path) as src:
        reproject(
            rasterio.band(src, band),
            dst,
            dst_transform=transform,
            dst_crs="EPSG:5070",
            dst_nodata=np.nan,
            src_nodata=src.nodata,
            resampling=resampling,
        )
    return dst


def composite_fac(transform, width: int, height: int) -> np.ndarray:
    """FAC-REM depth on the master window: registry precedence, first finite wins.

    Same sourcing contract as ``sample_fac_rem`` at the training wells (named
    pilots first, then sorted HUC8s), so the fine expert matches what the coarse
    gate conditioned on. NEAREST resampling: the strip-fill footprint is sparse
    and bilinear against NaN would erode channel edges.
    """
    left = transform.c
    top = transform.f
    right = left + width * RES10
    bottom = top - height * RES10
    out = np.full((height, width), np.nan, dtype="float32")
    for path in existing_registry(None):
        with rasterio.open(path) as src:
            b = src.bounds
        if b.left >= right or b.right <= left or b.bottom >= top or b.top <= bottom:
            continue
        need = ~np.isfinite(out)
        if not need.any():
            break
        vals = warp_band(Path(path), 1, transform, width, height, Resampling.nearest)
        fill = need & np.isfinite(vals)
        out[fill] = vals[fill]
        log.info("FAC %s: filled %d cells", Path(path).parents[1].name, int(fill.sum()))
    return out


def recompose(
    dem10: np.ndarray,
    fac10: np.ndarray,
    w: dict[str, np.ndarray],
    deep_wte: np.ndarray,
    head_wte: np.ndarray,
    d: float,
    mirror_on: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Gate decomposition at 10 m (plan step 4). Returns (wte10, fallback_mask).

    Where the 10 m FAC is finite the mixture is the exact convex combination the
    coarse forward asserted; where it is absent the surviving weights renormalize
    by (1 - w_fac) -- the masked-softmax semantics. Cells whose gate leaned
    ~all-FAC (1 - w_fac < HEAD_FALLBACK_W) but lost FAC at 10 m fall back to the
    free-head expert and are returned in the mask for counting.
    """
    fac_wte = dem10 - fac10
    num = w["deep"] * deep_wte + w["head"] * head_wte
    if mirror_on:
        num = num + w["mirror"] * (dem10 - d)
    fac_ok = np.isfinite(fac_wte)
    with np.errstate(divide="ignore", invalid="ignore"):
        wte10 = np.where(fac_ok, w["fac"] * fac_wte + num, num / (1.0 - w["fac"]))
    fallback = ~fac_ok & (1.0 - w["fac"] < HEAD_FALLBACK_W) & np.isfinite(head_wte)
    wte10 = np.where(fallback, head_wte, wte10)
    return wte10, fallback


def render_basin(
    basin: str, man: dict, model_name: str, write_dem: bool = False
) -> None:
    gdirb = HUC8_ROOT / basin / "gnn" / model_name
    coarse_wte = gdirb / "gnn_wte_100m.tif"
    if not coarse_wte.exists():
        raise SystemExit(f"{basin}: no coarse inference output at {gdirb}")
    transform, width, height = master_window(coarse_wte)
    experts = list(man["gate_experts"])
    mirror_on = "mirror" in experts
    d = float(man["flags"]["mirror_depth_m"])

    dem10 = warp_band(
        HUC8_ROOT / basin / "dem_10m.tif",
        1,
        transform,
        width,
        height,
        Resampling.bilinear,
    )
    fac10 = composite_fac(transform, width, height)
    up = lambda p, band=1: warp_band(  # noqa: E731
        p, band, transform, width, height, Resampling.bilinear
    )
    w = {
        e: up(gdirb / "gnn_gate_w_100m.tif", band=i + 1) for i, e in enumerate(experts)
    }
    deep_wte = up(gdirb / "gnn_deep_wte_100m.tif")
    head_wte = up(gdirb / "gnn_head_wte_100m.tif")

    wte10, fallback = recompose(dem10, fac10, w, deep_wte, head_wte, d, mirror_on)
    n_fallback = int(fallback.sum())

    valid = np.isfinite(wte10) & np.isfinite(dem10)
    wte10 = np.where(valid, wte10, np.nan)
    dtw10 = dem10 - wte10
    neg = float((dtw10[valid] < 0).mean()) if valid.any() else 0.0
    log.info(
        "%s: %dx%d @10m, %d valid cells, %d head-fallback, DTW median %.2f m, "
        "%.1f%% negative",
        basin,
        width,
        height,
        int(valid.sum()),
        n_fallback,
        float(np.nanmedian(dtw10)),
        100 * neg,
    )

    # coarse-consistency: block-mean the 10 m WTE back to 100 m and MAD against
    # the lattice WTE, split by FAC presence. This is NOT expected to be ~0 in
    # relief: the lattice's z_surf is the 100 m CONUS DEM while the fine experts
    # deliberately recompute from 3DEP dem_10m (the plan's substrate upgrade),
    # and (w_fac + w_mirror) couple that DEM difference straight into WTE. The
    # substrate MAD is reported alongside so a warp/renorm bug (consistency far
    # ABOVE the substrate-explained scale) stays detectable.
    hb, wb = height // 10, width // 10
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN edge blocks
        wte_agg = np.nanmean(wte10.reshape(hb, 10, wb, 10), axis=(1, 3))
        dem_agg = np.nanmean(dem10.reshape(hb, 10, wb, 10), axis=(1, 3))
    fac_frac = np.isfinite(fac10).reshape(hb, 10, wb, 10).mean(axis=(1, 3))
    with rasterio.open(coarse_wte) as src:
        coarse = src.read(1).astype("float64")
        coarse[coarse == src.nodata] = np.nan
    with rasterio.open(gdirb / "gnn_dtw_100m.tif") as src:
        coarse_dtw = src.read(1).astype("float64")
        coarse_dtw[coarse_dtw == src.nodata] = np.nan
    zsurf100 = coarse + coarse_dtw  # the lattice's z_surf, reconstructed exactly
    both = np.isfinite(wte_agg) & np.isfinite(coarse)
    resid = np.abs(wte_agg - coarse)
    mad_all = float(np.median(resid[both])) if both.any() else np.nan
    m_fac = both & (fac_frac > 0)
    m_nofac = both & (fac_frac == 0)
    mad_fac = float(np.median(resid[m_fac])) if m_fac.any() else np.nan
    mad_nofac = float(np.median(resid[m_nofac])) if m_nofac.any() else np.nan
    dem_sub = np.abs(dem_agg - zsurf100)
    mad_dem = float(np.median(dem_sub[both])) if both.any() else np.nan
    log.info(
        "%s: coarse-consistency MAD %.3f m (FAC cells %.3f, non-FAC %.3f); "
        "DEM substrate MAD %.3f m (10 m 3DEP vs 100 m lattice DEM)",
        basin,
        mad_all,
        mad_fac,
        mad_nofac,
        mad_dem,
    )

    with rasterio.open(coarse_wte) as src:
        tags = {
            k: v for k, v in src.tags().items() if not k.startswith("AREA_OR_POINT")
        }
    layers = {"gnn_wte_10m.tif": wte10, "gnn_dtw_10m.tif": dtw10}
    if (gdirb / "gnn_sigma_100m.tif").exists():
        layers["gnn_sigma_10m.tif"] = np.where(
            valid, up(gdirb / "gnn_sigma_100m.tif"), np.nan
        )
    # The aligned 3DEP DEM the recompose already warped onto the master grid --
    # persisted (pixel-locked to gnn_dtw_10m) so a downstream gdaldem hillshade
    # needs no per-basin regridding. Kept only when requested (viewing basemap).
    if write_dem:
        layers["gnn_dem_10m.tif"] = dem10
    for name, arr in layers.items():
        write_tif(gdirb / name, arr, transform)
        with rasterio.open(gdirb / name, "r+") as dst:
            dst.update_tags(**tags)
    (gdirb / "render_run.json").write_text(
        json.dumps(
            {
                "basin": basin,
                "window_10m": [width, height],
                "n_valid": int(valid.sum()),
                "n_head_fallback": n_fallback,
                "pct_negative_dtw": neg,
                "coarse_consistency_mad_m": mad_all,
                "coarse_consistency_mad_fac_m": mad_fac,
                "coarse_consistency_mad_nofac_m": mad_nofac,
                "dem_substrate_mad_m": mad_dem,
                **tags,
            },
            indent=2,
        )
    )
    log.info("%s: wrote %d 10 m layers -> %s", basin, len(layers), gdirb)


def build_mosaic(basins: list[str], model_name: str, state: str, out_root: Path):
    """Per-state VRTs over the per-basin 10 m COGs, registry-precedence overlap.

    gdalbuildvrt paints later-listed sources on top, so the list is ordered
    LOWEST precedence first -- the highest-precedence basin (named pilots, per
    ``existing_registry``) wins overlaps, matching sample_fac_rem's
    first-finite-wins.
    """
    precedence = [Path(p).parents[1].name for p in existing_registry(None)]
    rank = {b: i for i, b in enumerate(precedence)}
    ordered = sorted(basins, key=lambda b: rank.get(b, len(rank)), reverse=True)
    out_dir = out_root / "mosaics" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in ("gnn_dtw_10m", "gnn_wte_10m", "gnn_sigma_10m"):
        tifs = [
            str(HUC8_ROOT / b / "gnn" / model_name / f"{layer}.tif")
            for b in ordered
            if (HUC8_ROOT / b / "gnn" / model_name / f"{layer}.tif").exists()
        ]
        if not tifs:
            log.info("mosaic %s: no inputs, skipped", layer)
            continue
        vrt = out_dir / f"{layer}_{state.lower()}.vrt"
        subprocess.run(
            ["gdalbuildvrt", "-overwrite", str(vrt), *tifs],
            check=True,
            capture_output=True,
        )
        log.info("mosaic: %s (%d basins)", vrt, len(tifs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--basins", nargs="+")
    ap.add_argument("--state")
    ap.add_argument("--mosaic", action="store_true", help="build state VRTs only")
    ap.add_argument(
        "--overwrite", action="store_true", help="re-render basins with existing 10 m"
    )
    ap.add_argument(
        "--write-dem",
        action="store_true",
        help="also write gnn_dem_10m.tif (aligned 3DEP DEM) for a downstream hillshade",
    )
    args = ap.parse_args()
    model_dir = Path(args.model_dir)
    model_name = model_dir.name
    man = json.loads((model_dir / "models" / "inference_manifest.json").read_text())
    if not args.basins and not args.state:
        raise SystemExit("pass --basins or --state")
    basins = args.basins or nm_basins(args.state)

    if args.mosaic:
        build_mosaic(basins, model_name, args.state or "custom", HUC8_ROOT)
        return
    done = skipped = 0
    for b in basins:
        out = HUC8_ROOT / b / "gnn" / model_name / "gnn_dtw_10m.tif"
        if not args.overwrite and out.exists():
            skipped += 1
            continue
        render_basin(b, man, model_name, write_dem=args.write_dem)
        done += 1
    log.info("render complete: %d basins, %d skipped (existing)", done, skipped)


if __name__ == "__main__":
    main()
