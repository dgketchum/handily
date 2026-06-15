"""Seasonal Sentinel-2 NDVI/NDWI cloud-free mosaics via Earth Engine.

Builds per-season, per-year cloud-masked median composites of spectral
indices (NDVI, NDWI by default) from ``COPERNICUS/S2_SR_HARMONIZED`` and
exports them to Cloud Storage at 10 m. Intended to replace the diluted
single-date 20 m NAIP-derived NDVI grid used by the FAC10 head solve with a
phenology-informed, multi-date layer (see notes/FAC10_PROGRESS.md).

Cloud masking uses Cloud Score+ (``GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED``,
band ``cs_cdf``), Google's current recommended S2 masking — robust across
terrain shadow without per-scene tuning. The seasonal median is robust to
residual haze/shadow that survive masking.

Design is deliberately flexible: region, years, seasons, indices, cloud
threshold, scale, CRS, and export destination are all parameters. Defaults
target the Missouri Headwaters basin (montane Montana): green-up / peak /
senescence seasons, 2019-2024 (full S2A+S2B dual-satellite period).

Example
-------
    uv run python -m handily.s2_seasonal \
        --region /data/ssd2/handily/mt/regional/missouri_headwaters/basin_boundary.fgb \
        --bucket wudr \
        --prefix handily/mt/regional/missouri_headwaters/evidence/s2 \
        --years 2019-2024 \
        --climatology
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

import ee
import geopandas as gpd
from shapely.ops import unary_union

from handily.ee.common import initialize_ee

LOGGER = logging.getLogger("handily.s2_seasonal")

S2_SR_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_BAND = "cs_cdf"
DEFAULT_CLOUD_THRESHOLD = 0.60
DEFAULT_SCALE_M = 10
DEFAULT_CRS = "EPSG:5070"
DEFAULT_YEARS = list(range(2019, 2025))

# Spectral index definitions: name -> (band_a, band_b) for (a - b) / (a + b).
# S2 SR bands: B3 green, B4 red, B8 NIR (all 10 m); B11 SWIR1 (20 m).
INDEX_BANDS: dict[str, tuple[str, str]] = {
    "ndvi": ("B8", "B4"),  # vegetation greenness
    "ndwi": ("B3", "B8"),  # McFeeters open-water index
    "mndwi": ("B3", "B11"),  # modified NDWI (SWIR), better water discrimination
}

# Seasons for the Northern Rockies high country, by (start_md, end_md) where
# md is (month, day), inclusive of start, exclusive of end+1day. Tuned to
# montane phenology: spring green-up after snowmelt, summer peak, fall
# senescence. Override via --seasons or the `seasons` argument.
DEFAULT_SEASONS: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "spring": ((4, 1), (6, 15)),
    "summer": ((6, 16), (8, 31)),
    "fall": ((9, 1), (10, 31)),
}


@dataclass
class S2SeasonalConfig:
    region: gpd.GeoDataFrame
    bucket: str
    prefix: str
    years: list[int] = field(default_factory=lambda: list(DEFAULT_YEARS))
    seasons: dict[str, tuple[tuple[int, int], tuple[int, int]]] = field(
        default_factory=lambda: dict(DEFAULT_SEASONS)
    )
    indices: list[str] = field(default_factory=lambda: ["ndvi", "ndwi"])
    cloud_threshold: float = DEFAULT_CLOUD_THRESHOLD
    scale_m: int = DEFAULT_SCALE_M
    crs: str = DEFAULT_CRS
    climatology: bool = False
    climatology_only: bool = False
    buffer_km: float = 0.0
    dest: str = "bucket"
    drive_folder: str | None = None
    ee_project: str | None = "ee-dgketchum"


def region_to_ee_geometry(
    region: gpd.GeoDataFrame, buffer_km: float = 0.0
) -> ee.Geometry:
    """Dissolve a vector region to a single EE geometry in EPSG:4326.

    ``buffer_km`` expands the region in a projected CRS (EPSG:5070) before
    reprojection — use it to cover a DEM's buffer ring so no reach samples
    nodata at the edge.
    """
    gdf = region
    if gdf.crs is None:
        raise ValueError("region has no CRS")
    if buffer_km > 0.0:
        proj = gdf.to_crs(epsg=5070)
        geom = unary_union(list(proj.geometry)).buffer(buffer_km * 1000.0)
        geom = gpd.GeoSeries([geom], crs="EPSG:5070").to_crs(epsg=4326).iloc[0]
        return ee.Geometry(geom.__geo_interface__)
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    geom = unary_union(list(gdf.geometry))
    return ee.Geometry(geom.__geo_interface__)


def _masked_s2_collection(
    geometry: ee.Geometry,
    start: ee.Date,
    end: ee.Date,
    cloud_threshold: float,
) -> ee.ImageCollection:
    """S2 SR scenes in range, cloud-masked via Cloud Score+."""
    cs = ee.ImageCollection(CLOUD_SCORE_COLLECTION)
    coll = (
        ee.ImageCollection(S2_SR_COLLECTION)
        .filterBounds(geometry)
        .filterDate(start, end)
        .linkCollection(cs, [CLOUD_SCORE_BAND])
    )

    def _mask(img: ee.Image) -> ee.Image:
        return img.updateMask(img.select(CLOUD_SCORE_BAND).gte(cloud_threshold))

    return coll.map(_mask)


def _add_indices(img: ee.Image, indices: list[str]) -> ee.Image:
    """Append the requested normalized-difference index bands to an image."""
    out = ee.Image.constant(0).select([])
    for name in indices:
        a, b = INDEX_BANDS[name]
        out = out.addBands(img.normalizedDifference([a, b]).rename(name))
    return out.copyProperties(img, ["system:time_start"])


def season_composite(
    geometry: ee.Geometry,
    year: int,
    season_bounds: tuple[tuple[int, int], tuple[int, int]],
    indices: list[str],
    cloud_threshold: float,
) -> ee.Image:
    """Cloud-free seasonal median of the requested indices, plus n_obs.

    Returns an image with one band per index (median over cloud-masked
    scenes) and an ``n_obs`` band (valid scene count for the first index),
    clipped to ``geometry``.
    """
    (sm, sd), (em, ed) = season_bounds
    start = ee.Date.fromYMD(year, sm, sd)
    end = ee.Date.fromYMD(year, em, ed).advance(1, "day")
    masked = _masked_s2_collection(geometry, start, end, cloud_threshold)
    idx_coll = masked.map(lambda img: _add_indices(img, indices))

    median = idx_coll.select(indices).median()
    n_obs = idx_coll.select(indices[0]).count().rename("n_obs")
    return median.addBands(n_obs).toFloat().clip(geometry)


def _export_image(
    image: ee.Image,
    cfg: S2SeasonalConfig,
    geometry: ee.Geometry,
    name: str,
) -> ee.batch.Task:
    file_prefix = f"{cfg.prefix.rstrip('/')}/{name}"
    if cfg.dest == "bucket":
        task = ee.batch.Export.image.toCloudStorage(
            image=image,
            description=name[:100],
            bucket=cfg.bucket,
            fileNamePrefix=file_prefix,
            region=geometry,
            scale=cfg.scale_m,
            crs=cfg.crs,
            maxPixels=int(1e13),
            fileFormat="GeoTIFF",
        )
    elif cfg.dest == "drive":
        if not cfg.drive_folder:
            raise ValueError("drive export requires drive_folder")
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=name[:100],
            folder=cfg.drive_folder,
            fileNamePrefix=name,
            region=geometry,
            scale=cfg.scale_m,
            crs=cfg.crs,
            maxPixels=int(1e13),
            fileFormat="GeoTIFF",
        )
    else:
        raise ValueError(f"Unsupported export destination: {cfg.dest}")
    task.start()
    LOGGER.info("started export: %s -> gs://%s/%s", name, cfg.bucket, file_prefix)
    return task


def export_s2_seasonal(cfg: S2SeasonalConfig) -> list[ee.batch.Task]:
    """Submit one EE export task per (season, year), and per-season
    climatology composites if ``cfg.climatology`` is set.

    Per-(season, year) image bands: [<indices...>, n_obs].
    Climatology image bands: [<index>_median ...] reduced across years.
    """
    for name in cfg.indices:
        if name not in INDEX_BANDS:
            raise ValueError(f"unknown index '{name}'; known: {sorted(INDEX_BANDS)}")
    initialize_ee(cfg.ee_project)
    geometry = region_to_ee_geometry(cfg.region, buffer_km=cfg.buffer_km)

    tasks: list[ee.batch.Task] = []
    # per-season climatology accumulators
    season_year_imgs: dict[str, list[ee.Image]] = {s: [] for s in cfg.seasons}

    # climatology_only suppresses the per-(season, year) exports (just the
    # across-year median composites are wanted) but still accumulates the
    # per-year images so the climatology reduction below has its inputs.
    for season, bounds in cfg.seasons.items():
        for year in cfg.years:
            img = season_composite(
                geometry, year, bounds, cfg.indices, cfg.cloud_threshold
            )
            season_year_imgs[season].append(img.select(cfg.indices))
            if cfg.climatology_only:
                continue
            name = f"s2_{season}_{year}_{cfg.scale_m}m"
            tasks.append(_export_image(img, cfg, geometry, name))

    if cfg.climatology or cfg.climatology_only:
        y0, y1 = min(cfg.years), max(cfg.years)
        for season, imgs in season_year_imgs.items():
            coll = ee.ImageCollection(imgs)
            bands = [f"{i}_median" for i in cfg.indices]
            clim = (
                coll.median()
                .rename(cfg.indices)
                .select(cfg.indices, bands)
                .toFloat()
                .clip(geometry)
            )
            name = f"s2_{season}_median_{y0}_{y1}_{cfg.scale_m}m"
            tasks.append(_export_image(clim, cfg, geometry, name))

    LOGGER.info("submitted %d export tasks", len(tasks))
    return tasks


def _parse_years(spec: str) -> list[int]:
    """'2019-2024' or '2019,2021,2023' -> list of ints."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(y) for y in spec.split(",") if y.strip()]


def _parse_seasons(
    spec: str | None,
) -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """'spring:4-1:6-15,summer:6-16:8-31' -> seasons dict; None -> defaults."""
    if not spec:
        return dict(DEFAULT_SEASONS)
    out: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for part in spec.split(","):
        name, start_md, end_md = part.split(":")
        sm, sd = (int(v) for v in start_md.split("-"))
        em, ed = (int(v) for v in end_md.split("-"))
        out[name] = ((sm, sd), (em, ed))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", required=True, help="vector file (fgb/shp) of the AOI")
    p.add_argument("--bucket", default="wudr")
    p.add_argument(
        "--prefix", required=True, help="GCS key prefix (no gs://, no bucket)"
    )
    p.add_argument("--years", default="2019-2024", help="'2019-2024' or '2019,2021'")
    p.add_argument(
        "--seasons",
        default=None,
        help="name:m-d:m-d,... (default montane spring/summer/fall)",
    )
    p.add_argument(
        "--indices", default="ndvi,ndwi", help="comma list from ndvi,ndwi,mndwi"
    )
    p.add_argument("--cloud-threshold", type=float, default=DEFAULT_CLOUD_THRESHOLD)
    p.add_argument("--scale", type=int, default=DEFAULT_SCALE_M)
    p.add_argument("--crs", default=DEFAULT_CRS)
    p.add_argument(
        "--climatology",
        action="store_true",
        help="also export per-season median across years",
    )
    p.add_argument(
        "--climatology-only",
        action="store_true",
        help="export ONLY the per-season across-year medians (skip per-year)",
    )
    p.add_argument(
        "--buffer-km",
        type=float,
        default=0.0,
        help="expand region by N km (cover DEM buffer ring)",
    )
    p.add_argument("--dest", choices=["bucket", "drive"], default="bucket")
    p.add_argument("--drive-folder", default=None)
    p.add_argument("--ee-project", default="ee-dgketchum")
    args = p.parse_args()

    region = gpd.read_file(args.region)
    cfg = S2SeasonalConfig(
        region=region,
        bucket=args.bucket,
        prefix=args.prefix,
        years=_parse_years(args.years),
        seasons=_parse_seasons(args.seasons),
        indices=[s.strip() for s in args.indices.split(",") if s.strip()],
        cloud_threshold=args.cloud_threshold,
        scale_m=args.scale,
        crs=args.crs,
        climatology=args.climatology,
        climatology_only=args.climatology_only,
        buffer_km=args.buffer_km,
        dest=args.dest,
        drive_folder=args.drive_folder,
        ee_project=args.ee_project,
    )
    tasks = export_s2_seasonal(cfg)
    print(f"submitted {len(tasks)} EE export tasks")
    for t in tasks:
        print(f"  {t.id}  {t.config.get('description', '')}")


if __name__ == "__main__":
    main()
