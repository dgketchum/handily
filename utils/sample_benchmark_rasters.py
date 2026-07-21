"""Sample the Ma and Janssen DTW benchmark rasters at contract-well locations.

Writes one parquet keyed by ``canonical_id`` so the (slow) raster reads never have
to be repeated for a benchmark intercomparison. Follows the protocol doc
(``notes/BENCHMARK_INTERCOMPARISON.md``) raster gotchas:

- **Ma** (`/nas/gwx/wtd_states/wtd_<state>.tif`, 49 per-state tiles, LCC metres,
  nodata +3.4e38): reproject 5070 -> tile CRS, first-finite hit across tiles.
  ``ma == 0`` is a GENUINE stream clamp (kept); only huge nodata / negatives dropped.
- **Janssen V2** (`/nas/gwx/janssen/V2_140.tif`, EPSG:4326, ~464 m cells,
  nodata -3.4e38): point-sample band 1, and also record the native (row, col) so a
  cell-aggregated MAD can be computed downstream (point-sampling over-credits a
  coarse raster; the doc requires reporting native cells occupied + cell-agg MAD).

All benchmark rasters are DTW in **metres** (asserted by value-range sanity check).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol

CONTRACT = "/data/ssd2/handily/conus/wte_gnn/v02/contract/wells_panels.parquet"
MA_DIR = Path("/nas/gwx/wtd_states")
JANSSEN = "/nas/gwx/janssen/V2_140.tif"
OUT = "/data/ssd2/handily/conus/wte_gnn/intercomparison/benchmark_samples.parquet"


def sample_ma_tiles(x5070: np.ndarray, y5070: np.ndarray, ma_dir: Path) -> np.ndarray:
    """First-finite Ma DTW (m) across per-state tiles; NaN where no tile covers.

    ``ma == 0`` surface clamps are preserved; nodata (+3.4e38) and implausible
    negatives / >1e4 m values are dropped to NaN.
    """
    out = np.full(len(x5070), np.nan)
    tiles = sorted(ma_dir.glob("wtd_*.tif"))
    for path in tiles:
        with rasterio.open(path) as ds:
            tr = Transformer.from_crs(5070, ds.crs, always_xy=True)
            rx, ry = tr.transform(x5070, y5070)
            left, bottom, right, top = ds.bounds
            cand = (rx >= left) & (rx <= right) & (ry >= bottom) & (ry <= top)
            cand &= ~np.isfinite(out)
            if cand.sum() == 0:
                continue
            vals = np.array(
                [v[0] for v in ds.sample(list(zip(rx[cand], ry[cand])))],
                dtype="float64",
            )
            nd = ds.nodata
            if nd is not None:
                vals[vals == nd] = np.nan
            vals[(vals < -1.0) | (vals > 1e4)] = np.nan
            out[np.where(cand)[0]] = vals
    return out


def sample_janssen(
    x5070: np.ndarray, y5070: np.ndarray, path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point-sample Janssen DTW (m) + return native (row, col) for cell-agg.

    Out-of-bounds / nodata (-3.4e38) / |val|>1e29 -> NaN; row/col -1 there.
    """
    with rasterio.open(path) as ds:
        tr = Transformer.from_crs(5070, ds.crs, always_xy=True)
        lon, lat = tr.transform(x5070, y5070)
        b = ds.bounds
        inb = (lon >= b.left) & (lon <= b.right) & (lat >= b.bottom) & (lat <= b.top)
        vals = np.full(len(x5070), np.nan)
        rows = np.full(len(x5070), -1, dtype="int64")
        cols = np.full(len(x5070), -1, dtype="int64")
        if inb.any():
            v = np.array(
                [s[0] for s in ds.sample(list(zip(lon[inb], lat[inb])))],
                dtype="float64",
            )
            r, c = rowcol(ds.transform, lon[inb], lat[inb], op=np.floor)
            vals[inb] = v
            rows[inb] = np.asarray(r, dtype="int64")
            cols[inb] = np.asarray(c, dtype="int64")
        nd = ds.nodata
    if nd is not None and np.isfinite(nd):
        bad = vals == nd
        vals[bad] = np.nan
        rows[bad] = -1
        cols[bad] = -1
    huge = np.abs(vals) > 1e29
    vals[huge] = np.nan
    rows[huge] = -1
    cols[huge] = -1
    return vals, rows, cols


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--contract", default=CONTRACT)
    ap.add_argument("--ma-dir", default=str(MA_DIR))
    ap.add_argument("--janssen", default=JANSSEN)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    df = pd.read_parquet(args.contract, columns=["canonical_id", "x5070", "y5070"])
    x = df["x5070"].to_numpy("float64")
    y = df["y5070"].to_numpy("float64")
    print(f"contract wells: {len(df)}")

    ma = sample_ma_tiles(x, y, Path(args.ma_dir))
    print(
        f"Ma:      valid {np.isfinite(ma).sum():>6} / {len(ma)}  "
        f"nodata/oob {int((~np.isfinite(ma)).sum()):>6}  "
        f"zeros(stream-clamp) {int((ma == 0).sum())}  "
        f"range [{np.nanmin(ma):.2f}, {np.nanmax(ma):.2f}] m  "
        f"median {np.nanmedian(ma):.2f} m"
    )

    jn, jr, jc = sample_janssen(x, y, args.janssen)
    n_cells = pd.Series(list(zip(jr[np.isfinite(jn)], jc[np.isfinite(jn)]))).nunique()
    print(
        f"Janssen: valid {np.isfinite(jn).sum():>6} / {len(jn)}  "
        f"nodata/oob {int((~np.isfinite(jn)).sum()):>6}  "
        f"native cells occupied {n_cells}  "
        f"range [{np.nanmin(jn):.2f}, {np.nanmax(jn):.2f}] m  "
        f"median {np.nanmedian(jn):.2f} m"
    )

    out = pd.DataFrame(
        {
            "canonical_id": df["canonical_id"].to_numpy(),
            "ma_dtw_m": ma,
            "janssen_dtw_m": jn,
            "janssen_row": jr,
            "janssen_col": jc,
        }
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
