"""Verify the multi-year NAIP water support raster along channel-head reaches.

Samples the binary support raster along reach polylines (densified to the grid
resolution) and reports the fraction of sample points where support == 1, plus
the basin-wide water-support fraction. Confirms the Rio Grande mainstem is
detected as a continuous water feature.
"""

import argparse

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


def _sample_reach(src, geom, step_m: float) -> tuple[float, int]:
    """Fraction of densified along-reach points where support == 1."""
    length = geom.length
    n = max(2, int(np.ceil(length / step_m)) + 1)
    dists = np.linspace(0, length, n)
    pts = [geom.interpolate(d) for d in dists]
    coords = [(p.x, p.y) for p in pts]
    vals = np.array([v[0] for v in src.sample(coords)], dtype=np.float32)
    hit = (vals > 0.5).mean()
    return float(hit), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support", required=True)
    ap.add_argument("--channel-heads", required=True)
    ap.add_argument("--basin", required=True)
    ap.add_argument("--reach", action="append", type=int, default=[6642, 8496, 9324])
    ap.add_argument("--step-m", type=float, default=5.0)
    args = ap.parse_args()

    ch = gpd.read_file(args.channel_heads)
    with rasterio.open(args.support) as src:
        print(
            f"support raster: crs={src.crs} res={src.res} shape={src.shape} "
            f"dtype={src.dtypes[0]} nodata={src.nodata}"
        )
        ch = ch.to_crs(src.crs)
        arr = src.read(1)
        total_water = int((arr > 0).sum())
        print(
            f"total water cells: {total_water} ({100 * total_water / arr.size:.4f}% of grid)"
        )

        # Largest connected water blob extent (river continuity check).
        try:
            from scipy import ndimage

            lab, ncomp = ndimage.label(arr > 0)
            if ncomp:
                sizes = ndimage.sum(arr > 0, lab, range(1, ncomp + 1))
                big = lab == (int(np.argmax(sizes)) + 1)
                ys, xs = np.where(big)
                h = (ys.max() - ys.min() + 1) * abs(src.res[1])
                w = (xs.max() - xs.min() + 1) * abs(src.res[0])
                print(
                    f"connected blobs: {ncomp}; largest={int(sizes.max())} cells, "
                    f"extent {h / 1000:.1f}km(NS) x {w / 1000:.1f}km(EW)"
                )
        except ImportError:
            pass

        for sid in args.reach:
            sub = ch[ch.stream_id == sid]
            if sub.empty:
                print(f"reach {sid}: NOT FOUND")
                continue
            geom = sub.geometry.iloc[0]
            hit, n = _sample_reach(src, geom, args.step_m)
            print(
                f"reach {sid}: mean water-support fraction = {hit:.3f} ({n} pts, len {geom.length:.0f}m)"
            )

        # Basin-wide fraction on the unbuffered basin polygon.
        basin_gdf = gpd.read_file(args.basin).to_crs(src.crs)
        basin = rasterize(
            basin_gdf.geometry,
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            default_value=1,
            dtype="uint8",
        ).astype(bool)
        bw = float((arr[basin] > 0).mean())
        print(f"basin-wide water-support fraction: {bw:.6f} ({100 * bw:.4f}%)")
        print(
            f"basin water cells: {int((arr[basin] > 0).sum())} of {int(basin.sum())} basin cells"
        )


if __name__ == "__main__":
    main()
