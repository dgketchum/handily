"""Export per-state IrrMapper irrigation-frequency rasters from Earth Engine.

For each state, takes the IrrMapperComp per-year classifications (class 0 =
irrigated), computes the fraction of years each 30 m pixel is classified
irrigated over a year range, and exports it as uint8 percent (0-100) in
EPSG:5070 to GCS. Irrigation frequency is the right well/footprint flag for
the GW-subsidy work: single-year maps miss rotation, and MIrAD 250 m is blind
to flood-irrigated hay (e.g. the Big Hole; see notes/ONE_METER_PILOT_PLAN.md).

Usage:
  uv run python utils/export_irrmapper_freq.py --states MT NV NM --years 2015-2024

Pull when tasks finish:
  gsutil -m rsync -r gs://wudr/irrmapper/ /nas/irrmapper/tif_exports/conus_freq/
"""

import argparse
import logging

import ee

from handily.ee.common import initialize_ee

LOGGER = logging.getLogger(__name__)

IRRMAPPER_COMP = "projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp"
TIGER_STATES = "TIGER/2018/States"


def _parse_years(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def export_state_freq(
    state: str,
    years: list[int],
    bucket: str,
    prefix: str,
    scale_m: int,
    crs: str,
) -> ee.batch.Task:
    coll = ee.ImageCollection(IRRMAPPER_COMP)
    ids = [f"{state}_{y}" for y in years]
    imgs = coll.filter(ee.Filter.inList("system:index", ids))
    n = imgs.size().getInfo()
    if n == 0:
        raise SystemExit(f"no IrrMapperComp images for {state} in {years}")
    if n < len(years):
        have = imgs.aggregate_array("system:index").getInfo()
        LOGGER.warning("%s: only %d/%d years present (%s)", state, n, len(years), have)

    geom = (
        ee.FeatureCollection(TIGER_STATES)
        .filter(ee.Filter.eq("STUSPS", state))
        .geometry()
    )
    freq = (
        imgs.map(lambda im: im.select("classification").eq(0).toFloat())
        .mean()
        .multiply(100)
        .round()
        .toUint8()
        .rename("irr_freq_pct")
        .clip(geom)
    )
    y0, y1 = min(years), max(years)
    name = f"irrmapper_freq_{state}_{y0}_{y1}_{scale_m}m_5070"
    task = ee.batch.Export.image.toCloudStorage(
        image=freq,
        description=name[:100],
        bucket=bucket,
        fileNamePrefix=f"{prefix.rstrip('/')}/{name}",
        region=geom,
        scale=scale_m,
        crs=crs,
        maxPixels=int(1e13),
        fileFormat="GeoTIFF",
    )
    task.start()
    LOGGER.info(
        "started export: %s (%d years) -> gs://%s/%s/%s", state, n, bucket, prefix, name
    )
    return task


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", nargs="+", required=True, help="e.g. MT NV NM")
    p.add_argument("--years", default="2015-2024", help="'2015-2024' or '2015,2017'")
    p.add_argument("--bucket", default="wudr")
    p.add_argument("--prefix", default="irrmapper")
    p.add_argument("--scale", type=int, default=30)
    p.add_argument("--crs", default="EPSG:5070")
    p.add_argument("--ee-project", default="ee-dgketchum")
    args = p.parse_args()

    initialize_ee(args.ee_project)
    years = _parse_years(args.years)
    for st in args.states:
        export_state_freq(st, years, args.bucket, args.prefix, args.scale, args.crs)


if __name__ == "__main__":
    main()
