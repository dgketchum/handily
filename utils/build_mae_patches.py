"""Patch sampler / dataset builder for the neighborhood-MAE (notes/MAE_NEIGHBORHOOD_EMBEDDING.md).

Extracts multi-scale 64x64 windows of a co-registered, target-blind 100 m EPSG:5070
covariate stack, centered on lattice cells snapped to the canonical CONUS inference
lattice (origin X0=-2,540,000, Y0=3,258,000). The lattice origin IS the raster grid
origin for every channel (verified), so extraction is a pure padded windowed slice --
no per-pixel reprojection, no resampling seams.

Coarser scales are handled by average-pooling each FULL raster ONCE (100 m -> 500 m /
1 km / 4 km) and then gathering cheap 64x64 windows from the coarse grid -- ~100x faster
than gathering a fine window per patch and pooling it. The S=len(SCALES) scales share the
same center (fine col/row mapped to the coarse grid by integer division), giving
6.4 / 32 / 64 / 256 km footprints. A single-resolution arm uses one scale; a pyramid arm
stacks several as channel groups.

Outputs (under --out-dir):
  patches.f16      memmap [N, S, C, 64, 64] float16   (normalized, nodata->0; S=len(SCALES))
  mask.u8          memmap [N, S, 64, 64] uint8        (1=valid DEM pixel)
  centers.parquet  col,row,x5070,y5070,center_elev_m,block  per patch
  norm_stats.json  per (channel, scale) median/IQR + nan fractions
  meta.json        channel list, scales, grid, shapes, argv

The extraction primitives (snap_to_lattice, extract_stack, pool_full, gather_windows,
robust_stats, apply_norm) are imported by extract_mae_embeddings.py so wells and
lattice patches are built by identical code.

    uv run python utils/build_mae_patches.py --n-patches 45000 \
        --out-dir /data/ssd2/handily/conus/mae/patches/ladder
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_mae_patches")

# --- Canonical inference lattice == the covariate raster grid (verified 2026-07-16) ---
LATTICE_ORIGIN = (-2_540_000.0, 3_258_000.0)  # (X0 left, Y0 top), EPSG:5070
RES_M = 100.0
REF_W, REF_H = 49810, 31390  # cols, rows of every channel raster (both divisible by 10)

WIN = 64  # output tokens per side
HALF = WIN // 2
NORM_SUBSAMPLE = 30_000  # patches used for per-(channel,scale) robust norm stats
SCALES = (1, 5, 10, 40)  # 100/500/1k/4k m pool factors -> 6.4/32/64/256 km footprints
# The 4 km level (256 km footprint, +-128 km radius) is the basin-scale rung: NM rift /
# Great Basin closed basins span 50-150 km, so a window must reach >=~150 km to see the
# whole basin -- the 1 km rung (64 km) cannot. 2 km (128 km) is marginal for the widest
# basins; 5 km (320 km) over-coarsens. 4 km spans the full basin-width range with margin.

_COV = "/data/ssd2/handily/conus/covariates"
_TER = "/nas/handily/covariates/terrain"
_GEO = "/nas/handily/covariates/geology"
_DEM = "/data/ssd1/streamflow-ml-data/conus-dem/data/elev48i0100a.tif"

# (name, path, transform): transform in {None (raw), "dem_rel", float scale}
CHANNEL_SPECS: list[tuple[str, str, object]] = [
    ("dem_rel", _DEM, "dem_rel"),
    ("slope_deg", f"{_COV}/slope_deg.tif", None),
    ("tri_100m", f"{_COV}/tri_100m.tif", None),
    ("dist_to_stream_m", f"{_COV}/dist_to_stream_m.tif", None),
    ("twi_2km", f"{_TER}/twi_2km.tif", None),
    ("tpi_2km", f"{_TER}/tpi_2km.tif", None),
    ("haf_2km", f"{_TER}/haf_2km.tif", None),
    ("plan_curvature", f"{_TER}/plan_curvature.tif", None),
    ("profile_curvature", f"{_TER}/profile_curvature.tif", None),
    ("perm_logk_m2", f"{_GEO}/permeability_logk_x100.tif", 0.01),
    ("sediment_thickness_m", f"{_GEO}/sediment_thickness_basinfill_m.tif", None),
    ("depth_to_bedrock_m", f"{_GEO}/depth_to_bedrock_censored_cm.tif", 0.01),
]
CHANNEL_NAMES = [c[0] for c in CHANNEL_SPECS]

# Optional local mirror of the (network /nas) covariate rasters: if a same-basename copy
# exists here, read it instead. A one-time copy makes repeated build/extract passes
# local-SSD-fast without changing the canonical paths above (reproducible: absent cache
# -> exact same rasters).
_COV_CACHE = Path("/data/ssd2/handily/conus/mae/covariate_cache")

# Channel whose valid footprint defines the training domain. The DEM alone cannot:
# elev48i0100a carries real values (0) over ocean and across the Canada/Mexico border,
# so "DEM present" admits centers/pixels where every CONUS-only covariate is nodata
# (measured 2026-07-21: ~49% of the lattice rectangle). When set, centers are sampled
# only where this channel is valid and mask.u8 = DEM finite AND footprint finite.
FOOTPRINT_CHANNEL: str | None = None


def resolve_raster(path: str) -> str:
    """Return the local-cache copy of a raster if present, else the canonical path."""
    cached = _COV_CACHE / Path(path).name
    return str(cached) if cached.exists() else path


def apply_manifest(manifest: dict) -> None:
    """Rebind the channel roster / raster locations from a manifest dict.

    Lets the same builder run against the v2 roster or a cluster mirror without code
    edits. Schema: {"dem": path, "cache_dir": optional path, "channels": [{"name",
    "path", "transform": absent | "dem_rel" | number}, ...]}. Every channel raster must
    already be on the canonical 100 m lattice (load_channel_full still asserts this).
    Exactly one channel must carry transform "dem_rel" -- it supplies the center-elevation
    datum and the validity mask. Optional "footprint_channel": name of a channel whose
    valid area defines the training domain (see FOOTPRINT_CHANNEL); it must appear
    after the dem_rel channel so the mask AND lands on an initialized mask."""
    global CHANNEL_SPECS, CHANNEL_NAMES, _DEM, _COV_CACHE, FOOTPRINT_CHANNEL
    specs: list[tuple[str, str, object]] = []
    for ch in manifest["channels"]:
        tr = ch.get("transform")
        if tr is not None and tr != "dem_rel":
            tr = float(tr)
        specs.append((ch["name"], ch["path"], tr))
    names = [s[0] for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit("manifest: duplicate channel names")
    if sum(1 for s in specs if s[2] == "dem_rel") != 1:
        raise SystemExit("manifest: exactly one channel must have transform 'dem_rel'")
    fp = manifest.get("footprint_channel")
    if fp is not None:
        if fp not in names:
            raise SystemExit(f"manifest: footprint_channel {fp!r} not in channels")
        dem_idx = next(i for i, s in enumerate(specs) if s[2] == "dem_rel")
        if names.index(fp) <= dem_idx:
            raise SystemExit(
                "manifest: footprint_channel must come after the dem_rel channel"
            )
    CHANNEL_SPECS = specs
    CHANNEL_NAMES = names
    _DEM = manifest["dem"]
    FOOTPRINT_CHANNEL = fp
    if manifest.get("cache_dir"):
        _COV_CACHE = Path(manifest["cache_dir"])


# --------------------------------------------------------------------------- geometry
def snap_to_lattice(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x5070, y5070) -> (col, row) integer lattice indices (floor), matching sample_coarse."""
    x0, y0 = LATTICE_ORIGIN
    col = np.floor((np.asarray(x, "float64") - x0) / RES_M).astype("int64")
    row = np.floor((y0 - np.asarray(y, "float64")) / RES_M).astype("int64")
    return col, row


def lattice_center_xy(
    col: np.ndarray, row: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cell (col,row) -> its center (x5070, y5070)."""
    x0, y0 = LATTICE_ORIGIN
    x = x0 + (np.asarray(col, "float64") + 0.5) * RES_M
    y = y0 - (np.asarray(row, "float64") + 0.5) * RES_M
    return x, y


def pool_full(a: np.ndarray, f: int) -> np.ndarray:
    """Average-pool a full 2-D raster by factor f, ignoring NaN (masked sum / valid count).

    Shape (H, W) -> (H//f, W//f); a fully-NaN block stays NaN. Done ONCE per raster so
    the per-patch window gather is a cheap 64x64 slice on the coarse grid."""
    if f == 1:
        return a
    h, w = a.shape
    a = a[: h // f * f, : w // f * f].reshape(h // f, f, w // f, f)
    m = np.isfinite(a)
    s = np.where(m, a, 0.0).sum(axis=(1, 3))
    c = m.sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out


def gather_windows(
    full: np.ndarray, col: np.ndarray, row: np.ndarray, half: int = HALF
) -> np.ndarray:
    """Padded (2*half)^2 windows around (col,row) from a full HxW array (NaN pad OOB).

    Vectorized advanced-index with clamp+mask so boundary cells pad with NaN instead of
    wrapping. Returns (B, 2*half, 2*half) float32."""
    h, w = full.shape
    off = np.arange(2 * half) - half
    rr = row[:, None] + off[None, :]  # (B, win)
    cc = col[:, None] + off[None, :]
    vr, vc = (rr >= 0) & (rr < h), (cc >= 0) & (cc < w)
    rr_cl, cc_cl = np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)
    win = full[rr_cl[:, :, None], cc_cl[:, None, :]].astype("float32")
    win[~(vr[:, :, None] & vc[:, None, :])] = np.nan
    return win


def robust_stats(vals: np.ndarray) -> tuple[float, float]:
    """(median, IQR) of finite values; IQR floored at a small positive to avoid /0."""
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return 0.0, 1.0
    med = float(np.median(v))
    iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
    return med, max(iqr, 1e-6)


def apply_norm(a: np.ndarray, med: float, iqr: float, clip: float = 5.0) -> np.ndarray:
    """(a-med)/iqr clipped to +-clip; NaN -> 0 (imputed after normalization)."""
    out = (a - med) / iqr
    out = np.clip(out, -clip, clip)
    return np.nan_to_num(out, nan=0.0)


# --------------------------------------------------------------------------- raster IO
def load_channel_full(
    path: str, transform: object
) -> tuple[np.ndarray, np.ndarray | None]:
    """Full-CONUS channel as float32 with nodata->NaN and decode applied.

    Returns (decoded, raw_for_datum): raw_for_datum is the elevation array only for the
    dem_rel channel (the fine center datum), else None. Asserts the raster is on the
    reference grid so the pure-slice contract cannot silently break."""
    path = resolve_raster(path)
    with rasterio.open(path) as src:
        if (src.width, src.height) != (REF_W, REF_H):
            raise SystemExit(
                f"{path}: grid {src.width}x{src.height} != {REF_W}x{REF_H}"
            )
        t = src.transform
        if not (
            abs(t.a - RES_M) < 1e-6
            and abs(t.c - LATTICE_ORIGIN[0]) < 1e-3
            and abs(t.f - LATTICE_ORIGIN[1]) < 1e-3
        ):
            raise SystemExit(f"{path}: transform {t} not on canonical lattice")
        arr = src.read(1)
        nd = src.nodata
    a = arr.astype("float32")
    if nd is not None:
        a[arr == nd] = np.nan
    raw = None
    if transform == "dem_rel":
        raw = a  # elevation datum source (a IS elevation, undecoded)
    elif isinstance(transform, (int, float)):
        a = a * float(transform)
    return a, raw


# --------------------------------------------------------------- shared multiscale read
def extract_stack(
    col: np.ndarray,
    row: np.ndarray,
    out: np.ndarray,
    mask_out: np.ndarray | None = None,
    norm: dict | None = None,
    block: int = 6000,
) -> None:
    """Fill out[N,S,C,64,64] (and mask_out[N,S,64,64]) for lattice cells (col,row).

    One raster read per channel; each full raster is pooled ONCE per coarse scale, then
    64x64 windows are gathered from the coarse grid (center = fine col//f, row//f). If
    ``norm`` (per_channel_scale dict) is given, values are normalized in place; else raw
    values are written (build's first pass, for stats). ``mask_out`` is set from the
    dem_rel channel's finiteness (the DEM footprint drives the valid mask)."""
    n, nc = len(col), len(CHANNEL_SPECS)
    for ci, (name, path, tr) in enumerate(CHANNEL_SPECS):
        log.info("channel %d/%d %s", ci + 1, nc, name)
        full, raw = load_channel_full(path, tr)
        scales_full = [full if f == 1 else pool_full(full, f) for f in SCALES]
        st = norm["per_channel_scale"][name] if norm is not None else None
        for b0 in range(0, n, block):
            b1 = min(b0 + block, n)
            cb, rb = col[b0:b1], row[b0:b1]
            for si, f in enumerate(SCALES):
                win = gather_windows(scales_full[si], cb // f, rb // f)  # [B,64,64]
                if tr == "dem_rel":
                    datum = raw[rb, cb].astype("float32")  # fine center elevation
                    win = win - datum[:, None, None]
                    if mask_out is not None:
                        mask_out[b0:b1, si] = np.isfinite(win).astype("uint8")
                elif name == FOOTPRINT_CHANNEL and mask_out is not None:
                    # domain mask = DEM finite AND footprint finite (the DEM alone is
                    # valid over ocean/cross-border; see FOOTPRINT_CHANNEL)
                    mask_out[b0:b1, si] &= np.isfinite(win).astype("uint8")
                if norm is not None:
                    win = apply_norm(win, st[si]["median"], st[si]["iqr"])
                else:
                    # raw pass may land in a float16 memmap: clip inside f16 range
                    # (|x|<=6e4; NaN passes through). Values this far out saturate the
                    # +-5 IQR normalization clip anyway, so stats are unaffected.
                    win = np.clip(win, -6.0e4, 6.0e4)
                out[b0:b1, si, ci] = win
        del full, raw, scales_full


# --------------------------------------------------------------------------- sampling
def sample_centers(n: int, seed: int, block_px: int = 2000) -> pd.DataFrame:
    """Random valid lattice cells, block-stratified over a coarse (block_px) grid.

    Validity = DEM present at the cell (off-CONUS DEM=32767 rejected). Oversamples,
    keeps valid, then balances per block round-robin to the target n."""
    rng = np.random.default_rng(seed)
    with rasterio.open(resolve_raster(_DEM)) as src:
        dem = src.read(1)
        nd = src.nodata
    valid_full = dem != nd
    if FOOTPRINT_CHANNEL is not None:
        spec = next(s for s in CHANNEL_SPECS if s[0] == FOOTPRINT_CHANNEL)
        fp_full, _ = load_channel_full(spec[1], spec[2])
        before = valid_full.mean()
        valid_full &= np.isfinite(fp_full)
        log.info(
            "footprint %s: valid frac %.3f -> %.3f",
            FOOTPRINT_CHANNEL,
            before,
            valid_full.mean(),
        )
        del fp_full
    pool = min(n * 6, 4_000_000)
    col = rng.integers(0, REF_W, pool)
    row = rng.integers(0, REF_H, pool)
    ok = valid_full[row, col]
    col, row = col[ok], row[ok]
    if len(col) < n:
        raise SystemExit(f"only {len(col)} valid candidates for n={n}; raise pool")
    block = (row // block_px).astype("int64") * 100000 + (col // block_px)
    order = rng.permutation(
        len(col)
    )  # round-robin across blocks for geographic balance
    col, row, block = col[order], row[order], block[order]
    df = pd.DataFrame({"col": col, "row": row, "block": block})
    df["rank_in_block"] = df.groupby("block").cumcount()
    df = df.sort_values(["rank_in_block", "block"]).head(n).reset_index(drop=True)
    x, y = lattice_center_xy(df["col"].to_numpy(), df["row"].to_numpy())
    df["x5070"], df["y5070"] = x, y
    log.info("sampled %d centers over %d coarse blocks", len(df), df["block"].nunique())
    return df


# --------------------------------------------------------------------------- build
def build(out_dir: Path, n_patches: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    centers = sample_centers(n_patches, seed)
    col = centers["col"].to_numpy("int64")
    row = centers["row"].to_numpy("int64")
    n, nc, ns = len(centers), len(CHANNEL_SPECS), len(SCALES)

    patches = np.memmap(
        out_dir / "patches.f16", "float16", "w+", shape=(n, ns, nc, WIN, WIN)
    )
    mask = np.memmap(out_dir / "mask.u8", "uint8", "w+", shape=(n, ns, WIN, WIN))

    # pass 1: RAW values (f16, clipped to f16 range) straight into the final memmap.
    # No full float32 staging array: at v2 scale (53 ch x 100k patches) that array
    # would be ~350 GB RAM, and it can never work for the ~1M-patch cluster builds.
    extract_stack(col, row, patches, mask_out=mask, norm=None)

    # dem_rel center datum (already center-relative -> read the raw elevation separately)
    with rasterio.open(resolve_raster(_DEM)) as src:
        dem = src.read(1)
    center_elev = dem[row, col].astype("float32")
    center_elev[center_elev == 32767] = np.nan
    del dem

    # pass 2: per (channel, scale) robust stats from a patch subsample, then blockwise
    # in-place normalization of the memmap. Median/IQR from <=30k patches x 4096 px is
    # statistically indistinguishable from the full-set stats the old in-RAM path used.
    sub = np.sort(
        np.random.default_rng(seed + 1).choice(
            n, size=min(n, NORM_SUBSAMPLE), replace=False
        )
    )
    stats = {"channels": CHANNEL_NAMES, "scales": list(SCALES), "per_channel_scale": {}}
    for ci, name in enumerate(CHANNEL_NAMES):
        stats["per_channel_scale"][name] = []
        for si in range(ns):
            raw_sub = np.asarray(patches[sub, si, ci], "float32")
            vals = raw_sub[np.asarray(mask[sub, si]) == 1]
            med, iqr = robust_stats(vals)
            nanfrac = float(np.mean(~np.isfinite(vals))) if vals.size else 1.0
            stats["per_channel_scale"][name].append(
                {"scale": SCALES[si], "median": med, "iqr": iqr, "nan_frac": nanfrac}
            )
            for b0 in range(0, n, 6000):
                b1 = min(b0 + 6000, n)
                x = np.asarray(patches[b0:b1, si, ci], "float32")
                patches[b0:b1, si, ci] = apply_norm(x, med, iqr).astype("float16")
            if nanfrac > 0.10:
                log.warning(
                    "%s s=%d nan_frac=%.3f (>10%%) -- investigate",
                    name,
                    SCALES[si],
                    nanfrac,
                )
    patches.flush()
    mask.flush()

    centers["center_elev_m"] = center_elev
    centers.to_parquet(out_dir / "centers.parquet")
    (out_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2))
    meta = {
        "n_patches": n,
        "channels": CHANNEL_NAMES,
        "n_channels": nc,
        "scales": list(SCALES),
        "win": WIN,
        "dem": _DEM,
        "footprint_channel": FOOTPRINT_CHANNEL,
        "lattice_origin_5070": list(LATTICE_ORIGIN),
        "res_m": RES_M,
        "patches_shape": [n, ns, nc, WIN, WIN],
        "dtype": "float16",
        "seed": seed,
        "norm_subsample": int(min(n_patches, NORM_SUBSAMPLE)),
        "argv": sys.argv,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("wrote %d patches -> %s", n, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-patches", type=int, default=45000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--manifest",
        default=None,
        help="channel-manifest JSON rebinding the roster/paths (see apply_manifest); "
        "default = the built-in 12-channel zoran roster",
    )
    args = ap.parse_args()
    if args.manifest:
        apply_manifest(json.loads(Path(args.manifest).read_text()))
    build(Path(args.out_dir), args.n_patches, args.seed)


if __name__ == "__main__":
    main()
