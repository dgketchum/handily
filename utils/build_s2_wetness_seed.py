"""Build a combined S2 wetness seed raster for the FAC head solve.

Single-date / single-season NDVI cannot separate irrigated or sub-irrigated
valley floors from forested or shrub uplands in montane basins: in summer the
conifer/aspen uplands are as green as the wet meadow (measured on MT AOI 0009:
summer-NDVI valley 0.77 vs upland 0.75, rank-AUC 0.57 — useless). Two seasonal
S2 signals do separate them:

  * spring NDVI  — the wet valley greens up early (snowmelt + groundwater +
    early irrigation) while dry uplands are still brown (AUC ~0.71);
  * fall NDWI    — the valley retains water late in the season (AUC ~0.72).

This builds a wetness index = mean of the two, each robustly min-max scaled to
[0, 1] over its valid range, written as a single-band raster suitable for the
head solve's ``ndvi_path`` (the seed sigmoid is index-agnostic). The combined
index separates valley from upland with rank-AUC ~0.89 and, fed through the
solve, detaches ~65% of upland reaches while keeping ~86% of the valley wet,
versus 0% upland detachment from summer NDVI.

A binary NDWI water-support mask is intentionally NOT produced: in semi-arid
montane valleys open water is rare (AOI 0009 fall NDWI is negative basin-wide),
so a >0.5 water mask is empty. The water signal is used continuously, in the
seed, instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rioxarray


def _minmax(arr: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    finite = np.isfinite(arr)
    lo, hi = np.nanpercentile(arr[finite], [lo_pct, hi_pct])
    if hi <= lo:
        raise ValueError(f"degenerate scaling range: p{lo_pct}={lo}, p{hi_pct}={hi}")
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def build_wetness_seed(
    spring_path: str,
    fall_path: str,
    out_path: str,
    *,
    ndvi_band: int = 1,
    ndwi_band: int = 2,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> str:
    """Write mean(minmax(spring NDVI), minmax(fall NDWI)) as a single band."""
    spring = rioxarray.open_rasterio(spring_path)
    fall = rioxarray.open_rasterio(fall_path)
    ndvi_sp = spring[ndvi_band - 1].astype("float32")
    ndwi_fa = fall[ndwi_band - 1].astype("float32")

    if ndvi_sp.shape != ndwi_fa.shape:
        ndwi_fa = ndwi_fa.rio.reproject_match(ndvi_sp)

    wet = (
        _minmax(ndvi_sp.values, lo_pct, hi_pct)
        + _minmax(ndwi_fa.values, lo_pct, hi_pct)
    ) / 2.0

    out = ndvi_sp.copy(data=wet.astype("float32"))
    out = out.rio.set_spatial_dims(x_dim="x", y_dim="y")
    out.attrs = {"long_name": "s2_wetness_seed", "units": "1"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.rio.to_raster(out_path, compress="deflate")
    print(
        f"wrote {out_path}  shape {out.shape}  "
        f"valid mean {np.nanmean(wet):.3f} [{np.nanmin(wet):.2f},{np.nanmax(wet):.2f}]"
    )
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--spring", required=True, help="S2 spring median raster (NDVI band)"
    )
    ap.add_argument("--fall", required=True, help="S2 fall median raster (NDWI band)")
    ap.add_argument("--out", required=True, help="output single-band seed raster")
    ap.add_argument("--ndvi-band", type=int, default=1)
    ap.add_argument("--ndwi-band", type=int, default=2)
    args = ap.parse_args()
    build_wetness_seed(
        args.spring,
        args.fall,
        args.out,
        ndvi_band=args.ndvi_band,
        ndwi_band=args.ndwi_band,
    )


if __name__ == "__main__":
    main()
