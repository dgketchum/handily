"""Restrict a NAIP open-water support mask to the FAC channel network.

The raw multi-year NAIP NDWI support mask (built by
``build_basin_naip_water_multiyear.py``) detects open water with a three-gate
test that cannot separate the river/canal surface from flood-irrigated or wet
bare agricultural fields. In irrigated basins (e.g. the Middle Rio Grande
Conservancy District) the same parcels flood every year, so K-of-N persistence
does NOT remove them — the off-channel cells survive and would wrongly pin large
swaths of cropland to the surface in the REM prior.

The fix is spatial: keep only support cells within ``buffer_m`` of the
flow-accumulation channel network at Strahler order >= ``strahler_min``. Open
water that co-locates with a mapped channel is the river/perennial canal we want
to pin on; open water out in the fields is applied irrigation and is dropped.
This is not circular — the FAC network already maps the river planform; the NAIP
water confirms which channels actually carry surface water.

Output keeps 0 as a valid "no support" value (nodata unset) so the REM support
reader treats off-channel cells as dry rather than masked.
"""

import argparse

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.ops import unary_union


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--support", required=True, help="raw binary support raster (0/1)")
    ap.add_argument("--streams", required=True, help="FAC stream network (vector)")
    ap.add_argument("--out", required=True, help="output channel-restricted raster")
    ap.add_argument(
        "--strahler-min",
        type=int,
        default=5,
        help="keep support within buffer of streams at this Strahler order or higher",
    )
    ap.add_argument(
        "--buffer-m",
        type=float,
        default=200.0,
        help="channel buffer half-width (m) defining the keep zone",
    )
    args = ap.parse_args()

    with rasterio.open(args.support) as src:
        profile = src.profile.copy()
        transform = src.transform
        shape = src.shape
        crs = src.crs
        support = src.read(1)

    streams = gpd.read_file(args.streams).to_crs(crs)
    big = streams[streams.strahler >= args.strahler_min]
    if big.empty:
        raise ValueError(
            f"no streams at strahler >= {args.strahler_min} in {args.streams}"
        )
    channel = unary_union(big.geometry.values).buffer(args.buffer_m)
    channel_mask = rasterize(
        [(channel, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8"
    )

    clean = (support.astype(bool) & channel_mask.astype(bool)).astype("uint8")

    n_in = int((support == 1).sum())
    n_out = int(clean.sum())
    dropped = n_in - n_out
    pct = n_out / n_in * 100 if n_in else float("nan")
    print(
        f"strahler>={args.strahler_min} buffer {args.buffer_m:.0f}m "
        f"({len(big)} reaches): kept {n_out:,}/{n_in:,} support cells "
        f"({pct:.1f}%); dropped {dropped:,} off-channel"
    )

    # 0 = valid "no support"; clear nodata so the REM reader does not mask zeros.
    profile.update(count=1, dtype="uint8", nodata=None, compress="deflate")
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(clean, 1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
