"""Pull the AUTHORITATIVE full-domain ParFlow CONUS2 lnK via hf_hydrodata.

Companion to ``build_conus2_lnk_recon.py``, which reconstructs lnK from the open
``3d-grid.v3.tif`` indicator field. That open file is a PARTIAL domain (41.2% of
the CONUS2 box; eastern seaboard, Atlantic/Gulf coastal plain, California and
border basins absent). The full domain (54.3% of the box = every active CONUS2
cell) lives behind HydroFrame's credentialed API. Catalog entries used, all
``dataset=conus2_domain grid=conus2``:

===================  =======  ==========================================
variable             units    file
===================  =======  ==========================================
``ln_k``             m/h      ``lnK_CONUS2.pfb`` (published 2-D field)
``permeability_x``   m/h      ``spinup.wy2003.out.perm_x.pfb`` (10 layers)
``pf_indicator``     -        ``CONUS2.0.Final1km.Subsurface.pfb`` (10 layers)
``latitude``         deg      ``Latitude_CONUS2.pfb`` (orientation truth)
``mask``             -        ``CONUS2.0.Final1km.Mask.tif`` (active domain)
===================  =======  ==========================================

Credentials
-----------
Needs a HydroFrame account (https://hydrogen.princeton.edu/signup) AND a PIN
created at https://hydrogen.princeton.edu/pin -- an account alone is NOT enough;
with no PIN the API answers "The email '<addr>' is not registered with a pin."
The PIN is read from ``--pin-file`` and never logged.

Grid / orientation -- verified, not assumed
------------------------------------------
CONUS2 is 4442x3256 at 1 km on an LCC sphere (R=6370000, lat_0=40.0000076294444,
lon_0=-97, SP 30/60) with no EPSG code; georeferencing is taken from the
already-downloaded indicator GeoTIFF (identical grid). Three conventions are
checked against pulled data rather than trusted:

1. **Row order.** ``latitude`` row 0 mean 24.03 deg vs row -1 mean 53.26 deg =>
   hf arrays are SOUTH-up, so the y axis is flipped for a north-up raster.
2. **Layer order.** ``pf_indicator`` carries only geology codes (19, 20, 21-28)
   at z=0..5 and only soil codes (1-13) at z=6..9 => z index 0 is the DEEPEST
   layer and the split is 6 geology / 4 soil, matching ``LAYER_DZ_M``
   (bottom->top) and ``N_GEOLOGY_LAYERS``.
3. **Ksat table.** ``permeability_x`` equals ``KSAT_MH[pf_indicator]`` cell by
   cell to 1.1e-8 m/h (float32 storage precision) over all 78.5M active cells,
   independently confirming the runscript-parsed table used by the open-route
   reconstruction.

Any of these failing raises rather than writing a raster.

What ``ln_k`` actually is
-------------------------
Not the arithmetic ten-layer mean. Against fields derived from
``permeability_x``, the published ``ln_k`` matches a **thickness-weighted
geometric mean** best (r=0.957, median |diff| 0.158 nats, identical minimum
-5.3008) and the arithmetic ten-layer mean poorly (r=0.623, median |diff| 1.161
nats). It is written out as its own variant (``*_published``) rather than
conflated with any derived one.

Usage (hf_hydrodata is not a project dependency -- one-time acquisition only):
    uv venv /tmp/hfhd && uv pip install --python /tmp/hfhd/bin/python hf_hydrodata rasterio numpy
    PYTHONPATH=/home/dgketchum/code/handily /tmp/hfhd/bin/python \
        utils/pull_conus2_lnk_hf.py --pin-file /home/dgketchum/hydrogen_pin.txt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from utils.build_conus2_lnk_recon import (
    INDICATOR,
    KSAT_MH,
    LAYER_DZ_M,
    N_GEOLOGY_LAYERS,
    OUT_DIR,
    SRC_DIR,
    warp_to_5070,
)

log = logging.getLogger("pull_conus2_lnk_hf")

EMAIL = "dgketchum@gmail.com"
PREFIX = "lnk_conus2_full"
N_LAYERS = len(LAYER_DZ_M)
# float32 is how the cached arrays are stored; the Ksat check must not be tighter
# than that precision or it reports rounding as disagreement.
KSAT_TOL_MH = 1e-7

PULLS = {
    "ln_k": ("hf_ln_k.npy", {}),
    "permeability_x": ("hf_perm_x.npy", {}),
    "pf_indicator": ("hf_pf_indicator.npy", {}),
    "latitude": ("hf_latitude.npy", {}),
    "mask": ("hf_mask.npy", {"file_type": "tiff"}),
}


def fetch_cached(pin_file: Path) -> dict[str, np.ndarray]:
    """Pull every needed CONUS2 field once, caching to _source/*.npy."""
    missing = {v: spec for v, spec in PULLS.items() if not (SRC_DIR / spec[0]).exists()}
    if missing:
        import hf_hydrodata as hf

        pin = pin_file.read_text().strip()
        if not pin:
            raise SystemExit(f"{pin_file} is empty")
        hf.register_api_pin(EMAIL, pin)  # never logged
        log.info("registered hf_hydrodata for %s (PIN read from %s)", EMAIL, pin_file)
        for var, (fn, extra) in missing.items():
            opts = {"dataset": "conus2_domain", "variable": var, "grid": "conus2"}
            opts.update(extra)
            a = np.asarray(hf.get_gridded_data(opts), dtype="float64")
            np.save(SRC_DIR / fn, a.astype("float32"))
            log.info("pulled %s %s -> %s", var, a.shape, fn)

    out = {v: np.load(SRC_DIR / spec[0]) for v, spec in PULLS.items()}
    for v, a in out.items():
        log.info("%s: shape %s nan%% %.2f", v, a.shape, 100.0 * np.isnan(a).mean())
    return out


def verify(arrs: dict[str, np.ndarray], shape: tuple[int, int]) -> None:
    """Check row order, layer order and the Ksat table against pulled data."""
    lat = arrs["latitude"]
    north0, north1 = float(np.nanmean(lat[0])), float(np.nanmean(lat[-1]))
    if not north0 < north1:
        raise SystemExit(
            f"latitude row 0 mean {north0:.3f} >= row -1 mean {north1:.3f}: hf arrays "
            "are NOT south-up as expected. Re-check the grid convention before "
            "writing any raster."
        )
    log.info(
        "row order OK: row0 lat %.3f < row-1 lat %.3f => south-up, flipping",
        north0,
        north1,
    )

    pf = arrs["pf_indicator"]
    if pf.shape != (N_LAYERS, *shape):
        raise SystemExit(f"pf_indicator shape {pf.shape} != ({N_LAYERS}, *{shape})")
    soil_codes = set(range(1, 14))
    geo_codes = {19, 20, *range(21, 29)}
    for z in range(N_LAYERS):
        codes = {int(c) for c in np.unique(pf[z]) if c > 0}
        want = geo_codes if z < N_GEOLOGY_LAYERS else soil_codes | {26}
        if not codes <= want:
            raise SystemExit(
                f"pf_indicator z={z} carries codes {sorted(codes - want)} outside the "
                f"expected {'geology' if z < N_GEOLOGY_LAYERS else 'soil'} set. The "
                "layer split or z ordering is not what LAYER_DZ_M assumes."
            )
    log.info(
        "layer order OK: z=0..%d geology-only, z=%d..%d soil => z0 deepest, %d/%d split",
        N_GEOLOGY_LAYERS - 1,
        N_GEOLOGY_LAYERS,
        N_LAYERS - 1,
        N_GEOLOGY_LAYERS,
        N_LAYERS - N_GEOLOGY_LAYERS,
    )

    lut = np.full(int(max(KSAT_MH)) + 1, np.nan)
    for c, k in KSAT_MH.items():
        lut[c] = k
    active = pf > 0
    k_ind = np.where(active, lut[np.clip(pf.astype(np.int64), 0, len(lut) - 1)], np.nan)
    if np.isnan(k_ind[active]).any():
        bad = sorted({int(c) for c in np.unique(pf[active & np.isnan(k_ind)])})
        raise SystemExit(f"indicator codes with no Ksat entry: {bad}. Do not fill.")
    dmax = float(np.nanmax(np.abs(arrs["permeability_x"][active] - k_ind[active])))
    if dmax > KSAT_TOL_MH:
        raise SystemExit(
            f"permeability_x disagrees with KSAT_MH[pf_indicator] by up to {dmax:.3g} "
            f"m/h (> {KSAT_TOL_MH:g}). The Ksat table or the layer mapping is wrong."
        )
    log.info(
        "Ksat table OK: permeability_x == KSAT_MH[pf_indicator] to %.2g m/h over %d active cells",
        dmax,
        int(active.sum()),
    )

    mask_frac = float((arrs["mask"] == 1).mean())
    top_frac = float((pf[N_LAYERS - 1] > 0).mean())
    if abs(mask_frac - top_frac) > 1e-6:
        raise SystemExit(
            f"active-domain disagreement: mask {mask_frac:.6f} vs top-layer indicator "
            f"{top_frac:.6f}. Investigate before proceeding."
        )
    log.info(
        "active domain OK: mask == top-layer indicator, %.4f of the box", mask_frac
    )


def build(args) -> None:
    with rasterio.open(INDICATOR) as src:
        prof = dict(src.profile)
        shape = (src.height, src.width)
    prof.update(count=1, dtype="float32", nodata=-9999.0, compress="lzw")

    arrs = fetch_cached(Path(args.pin_file))
    verify(arrs, shape)

    # south-up -> north-up on every field, then mask inactive cells to NaN
    flip = {
        k: (v[..., ::-1, :] if v.ndim == 3 else v[::-1, :]) for k, v in arrs.items()
    }
    pf = flip["pf_indicator"]
    active = pf > 0
    km = np.where(active, flip["permeability_x"].astype("float64"), np.nan)
    lnk_pub = np.where(flip["mask"] == 1, flip["ln_k"].astype("float64"), np.nan)

    dz = np.asarray(LAYER_DZ_M)[:, None, None]
    k_geo = km[:N_GEOLOGY_LAYERS]
    k_soil = km[N_GEOLOGY_LAYERS:]

    variants = {
        f"{PREFIX}_published": lnk_pub,
        f"{PREFIX}_10layer_mean": np.log(np.nanmean(km, axis=0)),
        f"{PREFIX}_geology": np.log(np.nanmean(k_geo, axis=0)),
        f"{PREFIX}_soil_mean": np.log(np.nanmean(k_soil, axis=0)),
        f"{PREFIX}_thickwt": np.log(np.nansum(km * dz, axis=0) / dz.sum()),
    }

    written = {}
    for name, arr in variants.items():
        native = OUT_DIR / f"{name}_conus2grid.tif"
        with rasterio.open(native, "w", **prof) as dst:
            dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype("float32"), 1)
        out5070 = OUT_DIR / f"{name}_5070.tif"
        warp_to_5070(native, out5070)
        with rasterio.open(out5070) as s:
            a = s.read(1, masked=True)
            written[name] = {
                "path": str(out5070),
                "native_path": str(native),
                "shape": [int(s.height), int(s.width)],
                "valid_frac": float(1.0 - a.mask.mean()),
                "ln_k_min": float(a.min()),
                "ln_k_max": float(a.max()),
                "ln_k_mean": float(a.mean()),
            }
        log.info("wrote %s %s", out5070, written[name])

    # (a) agreement with the open-route reconstruction on its own footprint
    agree = {}
    for suffix in ("10layer_mean", "geology", "soil_mean", "thickwt"):
        rp = OUT_DIR / f"lnk_conus2_recon_{suffix}_5070.tif"
        fp = OUT_DIR / f"{PREFIX}_{suffix}_5070.tif"
        with rasterio.open(rp) as s:
            r = s.read(1, masked=True).filled(np.nan)
        with rasterio.open(fp) as s:
            f = s.read(1, masked=True).filled(np.nan)
        m = np.isfinite(r) & np.isfinite(f)
        d = f[m] - r[m]
        agree[suffix] = {
            "n_shared_cells": int(m.sum()),
            "recon_valid_frac": float(np.isfinite(r).mean()),
            "full_valid_frac": float(np.isfinite(f).mean()),
            "recon_cells_covered_by_full": float((m.sum() / np.isfinite(r).sum())),
            "pearson_r": float(np.corrcoef(f[m], r[m])[0, 1]),
            "frac_within_0p01_ln": float((np.abs(d) < 0.01).mean()),
            "mean_diff_ln": float(d.mean()),
            "median_abs_diff_ln": float(np.median(np.abs(d))),
            "p95_abs_diff_ln": float(np.percentile(np.abs(d), 95)),
        }
        log.info("agreement %s: %s", suffix, agree[suffix])

    # (b) what the published ln_k corresponds to among the derived fields
    ident = {}
    pub = variants[f"{PREFIX}_published"]
    derived = {
        "arith_mean_10L": np.log(np.nanmean(km, axis=0)),
        "geom_mean_10L": np.nanmean(np.log(km), axis=0),
        "thickwt_arith": np.log(np.nansum(km * dz, axis=0) / dz.sum()),
        "thickwt_geom": np.nansum(np.log(km) * dz, axis=0) / dz.sum(),
        "geology_arith": np.log(np.nanmean(k_geo, axis=0)),
        "bottom_layer": np.log(km[0]),
    }
    for nm, c in derived.items():
        m = np.isfinite(pub) & np.isfinite(c)
        d = c[m] - pub[m]
        ident[nm] = {
            "pearson_r": float(np.corrcoef(pub[m], c[m])[0, 1]),
            "median_abs_diff_ln": float(np.median(np.abs(d))),
            "max_abs_diff_ln": float(np.max(np.abs(d))),
        }
    log.info("published ln_k identity: %s", json.dumps(ident, indent=1))

    prov = OUT_DIR / "build_provenance.json"
    rec = json.loads(prov.read_text()) if prov.exists() else {}
    rec["full_domain_pull"] = {
        "route": "hf_hydrodata get_gridded_data (credentialed)",
        "dataset": "conus2_domain",
        "variables": sorted(PULLS),
        "units": "ln(Ksat [m/h]); permeability_x and ln_k both m/h",
        "email": EMAIL,
        "pin_source": args.pin_file,
        "layer_dz_m_bottom_to_top": LAYER_DZ_M,
        "n_geology_layers": N_GEOLOGY_LAYERS,
        "orientation": "hf arrays south-up (verified via latitude); flipped to north-up",
        "ksat_table_check_max_diff_mh": KSAT_TOL_MH,
        "products": written,
        "agreement_with_reconstruction": agree,
        "published_lnk_identity": ident,
    }
    prov.write_text(json.dumps(rec, indent=2))
    log.info("updated %s", prov)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pin-file", default="/home/dgketchum/hydrogen_pin.txt")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    build(args)


if __name__ == "__main__":
    main()
