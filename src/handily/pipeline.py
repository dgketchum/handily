from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any

import geopandas as gpd

from . import compute, dem, io
from .config import HandilyConfig

LOGGER = logging.getLogger("handily.pipeline")


@dataclass
class REMWorkflow:
    config: HandilyConfig
    aoi: Any

    def __post_init__(self) -> None:
        if isinstance(self.aoi, gpd.GeoDataFrame):
            self.aoi_gdf = self.aoi
        else:
            self.aoi_gdf = gpd.GeoDataFrame([{}], geometry=[self.aoi], crs="EPSG:4326")

        self.bounds_wsen = tuple(self.aoi_gdf.to_crs("EPSG:4326").total_bounds.tolist())
        LOGGER.info("REMWorkflow instantiated (bounds_wsen=%s)", self.bounds_wsen)

        self.flowlines = None
        self.ndwi = None

        self.dem = None
        self.streams = None
        self.rem = None
        self.fields = None
        self.stats = None

    def fetch_vectors(
        self,
        cache_flowlines: bool = False,
        overwrite_flowlines_cache: bool = False,
        flowlines_cache_name: str = "flowlines_bounds.fgb",
    ) -> None:
        io.ensure_dir(self.config.out_dir)
        flowlines_cache_path = os.path.join(self.config.out_dir, flowlines_cache_name)
        if cache_flowlines and (not overwrite_flowlines_cache) and os.path.exists(flowlines_cache_path):
            io.LOGGER.info("Loading cached flowlines: %s", flowlines_cache_path)
            self.flowlines = gpd.read_file(flowlines_cache_path)
        else:
            self.flowlines = io.get_flowlines_within_aoi(
                self.aoi_gdf, local_flowlines_dir=self.config.flowlines_local_dir
            )
            if cache_flowlines:
                io.LOGGER.info("Caching flowlines to: %s", flowlines_cache_path)
                self.flowlines.to_file(flowlines_cache_path, driver="FlatGeobuf")

        ndwi_paths = io.ndwi_files_for_bounds(self.config.ndwi_dir, self.bounds_wsen)
        if not ndwi_paths:
            raise ValueError(
                "No NDWI rasters found intersecting bounds; place NDWI GeoTIFFs covering the AOI in ndwi_dir."
            )
        self.ndwi = io.open_ndwi_mosaic_from_paths(ndwi_paths, self.bounds_wsen)

    def fetch_dem(
        self,
        target_crs_epsg: int = 5070,
        overwrite: bool = False,
        stac_collection_id: str = "usgs-3dep-1m-opr",
        cache_name: str = "dem_bounds_1m.tif",
    ) -> None:
        io.ensure_dir(self.config.out_dir)
        cache_path = os.path.join(self.config.out_dir, cache_name)
        self.dem = dem.get_dem_for_aoi_via_stac(
            aoi_gdf=self.aoi_gdf,
            stac_dir=os.path.expanduser(self.config.stac_dir),
            target_crs_epsg=int(target_crs_epsg),
            cache_path=cache_path,
            overwrite=overwrite,
            stac_download_cache_dir=os.path.join(self.config.out_dir, "stac_cache"),
            stac_collection_id=stac_collection_id,
        )

    def compute_rem(self, ndwi_threshold: float = 0.15, stats: tuple[str, ...] = ("mean",)) -> None:
        if self.dem is None:
            raise RuntimeError("DEM not available; call fetch_dem() first.")
        if self.flowlines is None or self.ndwi is None:
            raise RuntimeError("Vectors not available; call fetch_vectors() first.")

        dem_crs = self.dem.rio.crs
        flowlines_dem = self.flowlines.to_crs(dem_crs)
        self.streams = compute.build_streams_mask_from_nhd_ndwi(
            flowlines_dem, self.dem, ndwi_da=self.ndwi, ndwi_threshold=float(ndwi_threshold)
        )
        self.rem = compute.compute_rem_quick(self.dem, self.streams)
        self.fields = io.load_and_clip_fields(self.config.fields_path, self.aoi_gdf, dem_crs)
        self.stats = compute.compute_field_rem_stats(self.fields, self.rem, stats=stats)

    def results(self, ndwi_threshold: float | None = None) -> dict[str, Any]:
        return {
            "aoi": self.aoi_gdf,
            "flowlines": self.flowlines,
            "ndwi": self.ndwi,
            "streams": self.streams,
            "rem": self.rem,
            "dem": self.dem,
            "fields": self.fields,
            "fields_stats": self.stats,
            "summary": {
                "total_fields": None if self.stats is None else len(self.stats),
                "ndwi_threshold": None if ndwi_threshold is None else float(ndwi_threshold),
            },
        }

    def run(
        self,
        ndwi_threshold: float = 0.15,
        stats: tuple[str, ...] = ("mean",),
        cache_flowlines: bool = False,
        overwrite_flowlines_cache: bool = False,
    ) -> dict[str, Any]:
        self.fetch_vectors(cache_flowlines=cache_flowlines, overwrite_flowlines_cache=overwrite_flowlines_cache)
        self.fetch_dem()
        self.compute_rem(ndwi_threshold=ndwi_threshold, stats=stats)
        return self.results(ndwi_threshold=ndwi_threshold)
