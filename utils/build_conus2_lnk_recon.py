"""Reconstruct the ParFlow CONUS2 lnK field from OPEN inputs.

WHY A RECONSTRUCTION. The authoritative field is HydroFrame
``dataset=conus2_domain, variable=ln_k, grid=conus2, file_type=pfb``
(``lnK_CONUS2.pfb``, units m/h) -- "natural log of hydraulic conductivity, where k
is the average over the ten subsurface layers of ParFlow". Its catalog is anonymous
but the DATA endpoint is hard-gated: ``GET
https://hydrogen.princeton.edu/api/gridded-data?dataset=conus2_domain&variable=ln_k``
returns HTTP 400 ``{"status":"fail","message":"Invalid Authorization header."}``.
An email+PIN registered at hydrogen.princeton.edu is required and none exists here.

Both ingredients of that field ARE openly downloadable, so lnK is reconstructable:

1. CONUS2 subsurface INDICATOR grid (no auth) --
   https://raw.githubusercontent.com/hydroframe/Subsetting/master/tests/test_inputs/
   CONUS2_Inputs/3d-grid.v3.tif
   4442 x 3256, 1000 m, 5 bands Float32, nodata -99, LCC sphere R=6370000
   (lat_0=40.0000076294444, lon_0=-97, lat_1=30, lat_2=60), origin
   (-2208000.3088, 1587000.3452). Band 1 = geology+bedrock unit (19-27);
   bands 2-5 = the four soil layers (1-26).

2. Indicator -> saturated hydraulic conductivity table (no auth), read out of the
   published ParFlow runscripts
   https://raw.githubusercontent.com/hydroframe/subsettools/main/src/subsettools/
   template_runscripts/conus2_transient_solid.yaml (and the ospool_parflow spinup
   twin). ``Geom.<unit>.Perm.Value`` is Ksat in m/h; ``GeomInput.<unit>.Value`` is
   the indicator code. Cross-check that anchors the table: the CONUS2 preprint
   states bedrock indicator 19 has K = 0.005 m/h, which is exactly b1 below.

KNOWN DIVERGENCES FROM THE AUTHORITATIVE FIELD -- state these wherever a result
built on this raster is reported:

- **Vintage.** 3d-grid.v3.tif was committed 2020-09-21, before Tijerina-Kreuzer
  et al. (2024) and Yang et al. (2023). Its geology band tops out at 27 (g7) while
  the production runscript defines g8 = 28. It is a development-vintage indicator,
  NOT provably the Selected National Configuration that Ma et al. read.
- **No e-folding / anisotropy.** Production CONUS2 applies multi-level depth/slope
  K decay and vertical anisotropy on top of the per-unit Ksat. Neither is
  recoverable from indicator + table, so reconstructed lnK ~= but != ln_k.
- **Amber lineage** (inherited, not introduced here): the CONUS2 hydrostratigraphic
  CONFIGURATION was selected partly by comparison against WTD/streamflow
  observations overlapping the Fan compilation -- upstream target conditioning,
  graded low-moderate in notes/MA_JANSSEN_LEAKAGE_REVIEW.md. Not cell-by-cell
  answer leakage; the field itself is well-free.

LAYER GEOMETRY (confirmed from the runscript's dzScale x DZ=200 m). 10 layers,
392 m total; ParFlow index 0 is the DEEPEST. Thicknesses bottom -> top:
200, 100, 50, 25, 10, 5, 1, 0.6, 0.3, 0.1 m. The bottom six carry the band-1
geology/bedrock unit; the top four are the four soil bands.

OUTPUTS (EPSG:5070, 1000 m, nearest-neighbour warp -- these are class-derived
values, so interpolating them would invent units that do not exist):

- ``lnk_conus2_recon_10layer_mean_5070.tif`` -- ln(mean_{10 layers} Ksat[m/h]).
  The variant that matches the authoritative field's stated definition. The simple
  (unweighted) mean is invariant to soil-band ordering, which the source GeoTIFF
  does not document.
- ``lnk_conus2_recon_geology_5070.tif`` -- ln(Ksat) of the band-1 geology/bedrock
  unit alone: the deep aquifer parameter, the one most plausibly tied to a
  deep-regime water table.
- ``lnk_conus2_recon_soil_mean_5070.tif`` -- ln(mean Ksat) over the four soil bands.
- ``lnk_conus2_recon_thickwt_5070.tif`` -- ln(sum K_i dz_i / sum dz_i), the
  transmissivity-weighted mean; dominated by the 200 m basal layer.

Usage:
    uv run python utils/build_conus2_lnk_recon.py
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import rasterio

log = logging.getLogger("build_conus2_lnk_recon")

OUT_DIR = Path("/nas/handily/covariates/subsurface_k")
SRC_DIR = OUT_DIR / "_source"
INDICATOR = SRC_DIR / "3d-grid.v3.tif"
RUNSCRIPT = SRC_DIR / "conus2_transient_solid.yaml"

INDICATOR_URL = (
    "https://raw.githubusercontent.com/hydroframe/Subsetting/master/tests/"
    "test_inputs/CONUS2_Inputs/3d-grid.v3.tif"
)
RUNSCRIPT_URL = (
    "https://raw.githubusercontent.com/hydroframe/subsettools/main/src/subsettools/"
    "template_runscripts/conus2_transient_solid.yaml"
)

# indicator code -> Ksat (m/h), parsed out of the runscript above and hard-coded here
# so the build is reproducible if the upstream template moves. build() re-parses the
# runscript and FAILS LOUD on any disagreement.
KSAT_MH = {
    1: 0.269022595,  # s1
    2: 0.043630356,  # s2
    3: 0.015841225,  # s3
    4: 0.007582087,  # s4
    5: 0.018188160,  # s5
    6: 0.005009435,  # s6
    7: 0.005492736,  # s7
    8: 0.004675077,  # s8
    9: 0.003386794,  # s9
    10: 0.004783973,  # s10
    11: 0.003979136,  # s11
    12: 0.006162952,  # s12
    13: 0.005009435,  # s13
    19: 0.005,  # b1 bedrock
    20: 0.010,  # b2 bedrock
    21: 0.02,  # g1
    22: 0.03,  # g2
    23: 0.04,  # g3
    24: 0.05,  # g4
    25: 0.06,  # g5
    26: 0.08,  # g6
    27: 0.10,  # g7
    28: 0.20,  # g8
}

# bottom -> top; ParFlow index 0 = deepest
LAYER_DZ_M = [200.0, 100.0, 50.0, 25.0, 10.0, 5.0, 1.0, 0.6, 0.3, 0.1]
N_GEOLOGY_LAYERS = 6  # the bottom six take the band-1 unit
N_SOIL_LAYERS = 4  # the top four are bands 2-5


def parse_runscript_ksat(path: Path) -> dict[int, float]:
    """Independently re-derive indicator -> Ksat from the runscript YAML.

    The file nests ``<unit>: Perm: Value:`` and ``GeomInput: <unit>: Value:``; a real
    YAML parse is overkill and the ParFlow templates are not strictly schema-stable,
    so this walks indentation, which is what the structure actually encodes.
    """
    import re

    lines = path.read_text().split("\n")

    def _owner(i: int) -> list[str]:
        """Enclosing keys of line i, innermost first."""
        ind = len(lines[i]) - len(lines[i].lstrip())
        out = []
        for j in range(i - 1, -1, -1):
            if not lines[j].strip():
                continue
            indj = len(lines[j]) - len(lines[j].lstrip())
            if indj < ind:
                out.append(lines[j].strip().rstrip(":"))
                ind = indj
            if len(out) >= 3:
                break
        return out

    perm: dict[str, float] = {}
    code: dict[str, int] = {}
    for i, ln in enumerate(lines):
        m = re.match(r"^\s*Value:\s*([-0-9.eE]+)\s*$", ln)
        if not m:
            continue
        ctx = _owner(i)
        if len(ctx) >= 2 and ctx[0] == "Perm" and re.fullmatch(r"[sgb]\d+", ctx[1]):
            perm[ctx[1]] = float(m.group(1))
        elif ctx and re.fullmatch(r"[sgb]\d+", ctx[0]):
            v = m.group(1)
            if re.fullmatch(r"\d+", v):
                code[ctx[0]] = int(v)
    out = {code[u]: perm[u] for u in perm if u in code}
    if not out:
        raise SystemExit(f"parsed no indicator->Ksat pairs from {path}")
    return out


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        log.info("have %s (%d bytes)", dest, dest.stat().st_size)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetching %s -> %s", url, dest)
    subprocess.run(["curl", "-sSL", "-o", str(dest), url], check=True)


def warp_to_5070(native: Path, out5070: Path) -> None:
    """Warp a native-CONUS2-grid raster to EPSG:5070 at 1 km, nearest-neighbour.

    Nearest only: these are class-derived conductivities, never interpolate across
    hydrostratigraphic unit boundaries.
    """
    subprocess.run(
        [
            "gdalwarp",
            "-overwrite",
            "-t_srs",
            "EPSG:5070",
            "-tr",
            "1000",
            "1000",
            "-r",
            "near",
            "-srcnodata",
            "-9999",
            "-dstnodata",
            "-9999",
            "-co",
            "COMPRESS=LZW",
            "-co",
            "TILED=YES",
            str(native),
            str(out5070),
        ],
        check=True,
        capture_output=True,
    )


def build(args) -> None:
    fetch(INDICATOR_URL, INDICATOR)
    fetch(RUNSCRIPT_URL, RUNSCRIPT)

    parsed = parse_runscript_ksat(RUNSCRIPT)
    shared = sorted(set(parsed) & set(KSAT_MH))
    bad = [c for c in shared if abs(parsed[c] - KSAT_MH[c]) > 1e-9]
    if bad:
        raise SystemExit(
            "runscript Ksat disagrees with the hard-coded table for indicator codes "
            f"{bad}: runscript {[parsed[c] for c in bad]} vs table "
            f"{[KSAT_MH[c] for c in bad]}. The upstream template changed -- "
            "investigate before rebuilding (do NOT silently take either side)."
        )
    log.info(
        "Ksat table verified against %s (%d shared codes)", RUNSCRIPT.name, len(shared)
    )

    with rasterio.open(INDICATOR) as src:
        prof = src.profile
        nd = src.nodata
        bands = [src.read(b + 1).astype("float64") for b in range(src.count)]
        crs_wkt = src.crs.to_wkt() if src.crs else None
    if len(bands) != 1 + N_SOIL_LAYERS:
        raise SystemExit(
            f"expected {1 + N_SOIL_LAYERS} indicator bands (1 geology + "
            f"{N_SOIL_LAYERS} soil), got {len(bands)}"
        )
    log.info("indicator %s bands=%d nodata=%s", bands[0].shape, len(bands), nd)

    # indicator -> Ksat via a dense LUT; any code not in the table is a real unknown
    # unit, not something to fill -- fail loud rather than invent a conductivity.
    lut = np.full(int(max(KSAT_MH)) + 1, np.nan)
    for c, k in KSAT_MH.items():
        lut[c] = k

    def to_k(a: np.ndarray, label: str) -> np.ndarray:
        valid = np.isfinite(a) & (a != nd)
        codes = np.where(valid, a, 0).astype(np.int64)
        if (codes[valid] < 0).any() or (codes[valid] > len(lut) - 1).any():
            raise SystemExit(f"{label}: indicator code outside the LUT range")
        k = np.where(valid, lut[codes], np.nan)
        unknown = valid & ~np.isfinite(k)
        if unknown.any():
            miss = sorted(np.unique(a[unknown]).tolist())
            raise SystemExit(
                f"{label}: {unknown.sum()} cells carry indicator codes with NO Ksat "
                f"entry: {miss}. Investigate the source vintage; do not fill."
            )
        return k

    k_geo = to_k(bands[0], "band1_geology")
    k_soil = np.stack([to_k(b, f"band{i + 2}_soil") for i, b in enumerate(bands[1:])])
    log.info(
        "k_geo m/h: min %.4g max %.4g nan %.2f%% | k_soil min %.4g max %.4g",
        np.nanmin(k_geo),
        np.nanmax(k_geo),
        100.0 * np.isnan(k_geo).mean(),
        np.nanmin(k_soil),
        np.nanmax(k_soil),
    )

    # 10-layer stack: bottom six = geology unit, top four = the soil bands.
    k10 = np.concatenate(
        [np.repeat(k_geo[None], N_GEOLOGY_LAYERS, axis=0), k_soil], axis=0
    )
    dz = np.asarray(LAYER_DZ_M)[:, None, None]
    if k10.shape[0] != len(LAYER_DZ_M):
        raise SystemExit("layer stack does not match LAYER_DZ_M")

    variants = {
        "lnk_conus2_recon_10layer_mean": np.log(k10.mean(axis=0)),
        "lnk_conus2_recon_geology": np.log(k_geo),
        "lnk_conus2_recon_soil_mean": np.log(k_soil.mean(axis=0)),
        "lnk_conus2_recon_thickwt": np.log((k10 * dz).sum(0) / dz.sum()),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    native_prof = dict(prof)
    native_prof.update(count=1, dtype="float32", nodata=-9999.0, compress="lzw")
    written = {}
    for name, arr in variants.items():
        native = OUT_DIR / f"{name}_conus2grid.tif"
        with rasterio.open(native, "w", **native_prof) as dst:
            dst.write(np.where(np.isfinite(arr), arr, -9999.0).astype("float32"), 1)
        out5070 = OUT_DIR / f"{name}_5070.tif"
        warp_to_5070(native, out5070)
        with rasterio.open(out5070) as s:
            a = s.read(1, masked=True)
            written[name] = {
                "path": str(out5070),
                "native_path": str(native),
                "shape": [int(s.height), int(s.width)],
                "ln_k_min": float(a.min()),
                "ln_k_max": float(a.max()),
                "ln_k_mean": float(a.mean()),
                "valid_frac": float(1.0 - a.mask.mean()) if a.mask.ndim else 1.0,
            }
        log.info("%s -> %s %s", name, out5070, json.dumps(written[name]))

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "# CONUS2 lnK — RECONSTRUCTION (not the authoritative field)\n\n"
        "Built by `utils/build_conus2_lnk_recon.py` on open inputs because the\n"
        "authoritative HydroFrame field is credential-gated. Read that script's\n"
        "module docstring for the full provenance, the verified indicator->Ksat\n"
        "table, and the known divergences. Short version:\n\n"
        "## What this is\n\n"
        "`ln(K)` with **K in m/h**, reconstructed as\n"
        "`indicator grid (3d-grid.v3.tif) x published per-unit Ksat (ParFlow\n"
        "runscript Geom.<unit>.Perm.Value)`, over the CONUS2 10-layer column\n"
        "(bottom six layers = band-1 geology/bedrock unit; top four = soil bands\n"
        "2-5; thicknesses bottom->top 200/100/50/25/10/5/1/0.6/0.3/0.1 m).\n\n"
        "Delivered on **EPSG:5070, 1000 m, nearest-neighbour** (`*_5070.tif`), plus\n"
        "the native CONUS2 LCC grid (`*_conus2grid.tif`). Nodata -9999.\n\n"
        "| variant | meaning |\n|---|---|\n"
        "| `lnk_conus2_recon_10layer_mean` | ln(mean Ksat over all 10 layers) — "
        "matches the authoritative field's stated definition |\n"
        "| `lnk_conus2_recon_geology` | ln(Ksat) of the geology/bedrock unit alone "
        "— the deep aquifer parameter |\n"
        "| `lnk_conus2_recon_soil_mean` | ln(mean Ksat) over the four soil layers |\n"
        "| `lnk_conus2_recon_thickwt` | ln(thickness-weighted mean Ksat) — "
        "transmissivity-like, dominated by the 200 m basal layer |\n\n"
        "## What this is NOT\n\n"
        "- NOT `lnK_CONUS2.pfb`. The indicator source is **development vintage\n"
        "  (committed 2020-09-21)** — its geology band tops out at 27 (g7) while the\n"
        "  production runscript defines g8=28 — so it is not provably the Selected\n"
        "  National Configuration that Ma et al. (2026) read.\n"
        "- Production CONUS2 also applies multi-level depth/slope K **e-folding** and\n"
        "  vertical **anisotropy**, neither of which is recoverable from\n"
        "  indicator + table. Reconstructed lnK ~= but != `ln_k`.\n"
        "- **Units are m/h hydraulic conductivity**, not GLHYMPS's log10 intrinsic\n"
        "  k[m^2]. The two are different variables: GLHYMPS supplies only the polygon\n"
        "  GEOMETRY, which CONUS2 reclassifies into 8 CONUS1-calibrated classes\n"
        "  spanning 0.02-0.2 m/h — barely one order of magnitude, against GLHYMPS\n"
        "  logk's ~10 orders.\n\n"
        "## Lineage flag (amber)\n\n"
        "The CONUS2 hydrostratigraphic *configuration* was selected partly by\n"
        "comparison against WTD/streamflow observations overlapping the Fan\n"
        "compilation — upstream target conditioning, graded low-moderate in\n"
        "`notes/MA_JANSSEN_LEAKAGE_REVIEW.md`. Not cell-by-cell answer leakage; the\n"
        "field is well-free. Any result built on it must carry this flag.\n\n"
        "## To replace with the authoritative field\n\n"
        "Register at https://hydrogen.princeton.edu/signup, make a PIN at /pin, then\n"
        "```python\n"
        "import hf_hydrodata as hf\n"
        'hf.register_api_pin("<email>", "<pin>")\n'
        'hf.get_gridded_data({"dataset": "conus2_domain", "variable": "ln_k",\n'
        '                     "grid": "conus2", "file_type": "pfb",\n'
        '                     "period": "static"})\n'
        "```\n"
        "(~116 MB float64 PFB). Nothing else in the pipeline changes.\n"
    )
    (OUT_DIR / "build_provenance.json").write_text(
        json.dumps(
            {
                "indicator_url": INDICATOR_URL,
                "runscript_url": RUNSCRIPT_URL,
                "indicator_bytes": INDICATOR.stat().st_size,
                "indicator_crs_wkt": crs_wkt,
                "ksat_table_m_per_h": KSAT_MH,
                "ksat_verified_against_runscript_codes": shared,
                "layer_dz_m_bottom_to_top": LAYER_DZ_M,
                "n_geology_layers": N_GEOLOGY_LAYERS,
                "n_soil_layers": N_SOIL_LAYERS,
                "outputs": written,
                "authoritative_field_blocked": {
                    "endpoint": "https://hydrogen.princeton.edu/api/gridded-data",
                    "response": '400 {"status":"fail","message":"Invalid '
                    'Authorization header."}',
                    "requirement": "email+PIN registered at hydrogen.princeton.edu",
                },
            },
            indent=2,
        )
    )
    log.info("wrote %s and %s", readme, OUT_DIR / "build_provenance.json")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
