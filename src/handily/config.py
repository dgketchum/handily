from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import os


@dataclass
class HandilyConfig:
    out_dir: str
    flowlines_local_dir: str
    ndwi_dir: str
    stac_dir: str
    fields_path: str
    feature_id: str = "FID"
    bounds: list[float] | None = None

    et_bucket: str | None = None
    ee_project: str | None = None
    openet_start_yr: int | None = None
    openet_end_yr: int | None = None
    openet_csv_path: str | None = None

    met_start: str | None = None
    met_end: str | None = None
    gridmet_parquet_dir: str | None = None
    gridmet_centroids_path: str | None = None
    gridmet_centroid_parquet_dir: str | None = None
    gridmet_id_col: str = "GFID"

    et_join_parquet_dir: str | None = None

    partition_joined_parquet_dir: str | None = None
    partition_out_parquet_dir: str | None = None
    partition_strata_col: str = "strata"
    partition_pattern_col: str = "pattern"

    # IrrMapper / stratification
    ee_fields_asset: str | None = None
    irrmapper_csv: str | None = None
    rem_threshold: float = 2.0
    rem_smooth_sigma: float = (
        50.0  # Gaussian sigma (meters) for EDT water surface smoothing
    )

    # REM workflow
    run_rem: bool = True
    overwrite_outputs: bool = False
    ndwi_threshold: float = 0.15
    flowlines_buffer_m: float = (
        10.0  # Buffer NHD flowlines before rasterization (meters)
    )
    rem_aoi_buffer_m: float = (
        500.0  # Buffer AOI geometry before DEM fetch/REM compute (meters)
    )
    delete_stac_cache: bool = False  # Delete raw STAC tile downloads after mosaicking
    rem_excluded_fcodes: list[int] | None = (
        None  # FCODEs excluded from REM stream mask; None = use nhd default
    )
    rem_propagate_mask: bool = (
        False  # Use network-propagated stream mask (BFS from NDWI seeds)
    )
    rem_propagate_hops: int | None = None  # Max BFS hops from seeds; None = unlimited

    # Points sampling
    points_out_dir: str | None = None
    points_seed: int = 42
    points_candidate_spacing_m: float = 60.0
    points_n_base: int = 50
    points_n_low_rem: int = 100
    points_n_riparian: int = 100
    points_n_field_edge: int = 50
    points_n_high_rem_control: int = 10
    points_low_rem_threshold_m: float = 2.0
    points_high_rem_threshold_m: float = 10.0
    points_riparian_buffer_m: float = 250.0
    points_field_edge_buffer_m: float = 100.0
    points_min_spacing_m: float | None = None
    points_ee_dest: str = "bucket"
    points_ee_bucket: str | None = None
    points_ee_drive_folder: str = "handily"
    points_year_start: int | None = None
    points_year_end: int | None = None
    points_ndvi_start_month: int = 4
    points_ndvi_end_month: int = 10

    # QGIS integration
    qgis_project: str | None = None  # Path to QGIS project file (.qgz)
    qgis_layer_group: str = "handily"  # Layer group name in QGIS project
    qgis_view_root: str | None = None  # Path prefix for viewing on different machine

    # Bucket / local mirror structure
    # Bucket path: gs://{bucket}/{bucket_prefix}/{project_name}/{subdir}/
    # Local path:  {local_data_root}/{bucket_prefix}/{project_name}/{subdir}/
    project_name: str = "default"
    bucket_prefix: str = "handily"
    local_data_root: str | None = None  # e.g., /nas/handily

    def __post_init__(self) -> None:
        self.out_dir = os.path.expanduser(self.out_dir)
        self.flowlines_local_dir = os.path.expanduser(self.flowlines_local_dir)
        self.ndwi_dir = os.path.expanduser(self.ndwi_dir)
        self.stac_dir = os.path.expanduser(self.stac_dir)
        self.fields_path = os.path.expanduser(self.fields_path)
        if self.openet_csv_path is not None:
            self.openet_csv_path = os.path.expanduser(self.openet_csv_path)
        if self.gridmet_parquet_dir is not None:
            self.gridmet_parquet_dir = os.path.expanduser(self.gridmet_parquet_dir)
        if self.gridmet_centroids_path is not None:
            self.gridmet_centroids_path = os.path.expanduser(
                self.gridmet_centroids_path
            )
        if self.gridmet_centroid_parquet_dir is not None:
            self.gridmet_centroid_parquet_dir = os.path.expanduser(
                self.gridmet_centroid_parquet_dir
            )
        if self.et_join_parquet_dir is not None:
            self.et_join_parquet_dir = os.path.expanduser(self.et_join_parquet_dir)
        if self.partition_joined_parquet_dir is not None:
            self.partition_joined_parquet_dir = os.path.expanduser(
                self.partition_joined_parquet_dir
            )
        if self.partition_out_parquet_dir is not None:
            self.partition_out_parquet_dir = os.path.expanduser(
                self.partition_out_parquet_dir
            )
        if self.irrmapper_csv is not None:
            self.irrmapper_csv = os.path.expanduser(self.irrmapper_csv)
        if self.local_data_root is not None:
            self.local_data_root = os.path.expanduser(self.local_data_root)
        if self.qgis_project is not None:
            self.qgis_project = os.path.expanduser(self.qgis_project)
        if self.points_out_dir is not None:
            self.points_out_dir = os.path.expanduser(self.points_out_dir)

    def get_bucket_path(self, subdir: str, filename: str | None = None) -> str:
        """Get bucket path for EE export (without gs:// prefix).

        Example: handily/beaverhead/irrmapper/beaverhead_irr_freq
        """
        base = f"{self.bucket_prefix}/{self.project_name}/{subdir}"
        if filename:
            return f"{base}/{filename}"
        return base

    def get_local_path(self, subdir: str, filename: str | None = None) -> str:
        """Get local path under the state project directory.

        Example: /data/ssd2/handily/mt/irrmapper/mt_irr_freq.csv
        """
        if self.local_data_root is None:
            raise ValueError("local_data_root not set in config")
        base = os.path.join(self.local_data_root, self.project_name, subdir)
        if filename:
            return os.path.join(base, filename)
        return base

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HandilyConfig":
        missing = [k for k in cls.required_keys() if k not in data]
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")
        cfg = cls(
            out_dir=str(data["out_dir"]),
            flowlines_local_dir=str(data["flowlines_local_dir"]),
            ndwi_dir=str(data["ndwi_dir"]),
            stac_dir=str(data["stac_dir"]),
            fields_path=str(data["fields_path"]),
            feature_id=str(data.get("feature_id", "FID")),
            bounds=data.get("bounds"),
            et_bucket=data.get("et_bucket"),
            ee_project=data.get("ee_project"),
            openet_start_yr=data.get("openet_start_yr"),
            openet_end_yr=data.get("openet_end_yr"),
            openet_csv_path=data.get("openet_csv_path"),
            met_start=data.get("met_start"),
            met_end=data.get("met_end"),
            gridmet_parquet_dir=data.get("gridmet_parquet_dir"),
            gridmet_centroids_path=data.get("gridmet_centroids_path"),
            gridmet_centroid_parquet_dir=data.get("gridmet_centroid_parquet_dir"),
            gridmet_id_col=str(data.get("gridmet_id_col", "GFID")),
            et_join_parquet_dir=data.get("et_join_parquet_dir"),
            partition_joined_parquet_dir=data.get("partition_joined_parquet_dir"),
            partition_out_parquet_dir=data.get("partition_out_parquet_dir"),
            partition_strata_col=str(data.get("partition_strata_col", "strata")),
            partition_pattern_col=str(data.get("partition_pattern_col", "pattern")),
            ee_fields_asset=data.get("ee_fields_asset"),
            irrmapper_csv=data.get("irrmapper_csv"),
            rem_smooth_sigma=float(data.get("rem_smooth_sigma", 50.0)),
            rem_aoi_buffer_m=float(data.get("rem_aoi_buffer_m", 500.0)),
            rem_excluded_fcodes=data.get("rem_excluded_fcodes"),
            rem_propagate_mask=bool(data.get("rem_propagate_mask", False)),
            rem_propagate_hops=int(data.get("rem_propagate_hops"))
            if data.get("rem_propagate_hops") is not None
            else None,
            run_rem=bool(data.get("run_rem", True)),
            overwrite_outputs=bool(data.get("overwrite_outputs", False)),
            rem_threshold=float(data.get("rem_threshold", 2.0)),
            ndwi_threshold=float(data.get("ndwi_threshold", 0.15)),
            flowlines_buffer_m=float(data.get("flowlines_buffer_m", 10.0)),
            delete_stac_cache=bool(data.get("delete_stac_cache", False)),
            points_out_dir=data.get("points_out_dir"),
            points_seed=int(data.get("points_seed", 42)),
            points_candidate_spacing_m=float(
                data.get("points_candidate_spacing_m", 60.0)
            ),
            points_n_base=int(data.get("points_n_base", 50)),
            points_n_low_rem=int(data.get("points_n_low_rem", 100)),
            points_n_riparian=int(data.get("points_n_riparian", 100)),
            points_n_field_edge=int(data.get("points_n_field_edge", 50)),
            points_n_high_rem_control=int(data.get("points_n_high_rem_control", 10)),
            points_low_rem_threshold_m=float(
                data.get("points_low_rem_threshold_m", 2.0)
            ),
            points_high_rem_threshold_m=float(
                data.get("points_high_rem_threshold_m", 10.0)
            ),
            points_riparian_buffer_m=float(data.get("points_riparian_buffer_m", 250.0)),
            points_field_edge_buffer_m=float(
                data.get("points_field_edge_buffer_m", 100.0)
            ),
            points_min_spacing_m=data.get("points_min_spacing_m"),
            points_ee_dest=str(data.get("points_ee_dest", "bucket")),
            points_ee_bucket=data.get("points_ee_bucket"),
            points_ee_drive_folder=str(data.get("points_ee_drive_folder", "handily")),
            points_year_start=data.get("points_year_start"),
            points_year_end=data.get("points_year_end"),
            points_ndvi_start_month=int(data.get("points_ndvi_start_month", 4)),
            points_ndvi_end_month=int(data.get("points_ndvi_end_month", 10)),
            project_name=str(data.get("project_name", "default")),
            bucket_prefix=str(data.get("bucket_prefix", "handily")),
            local_data_root=data.get("local_data_root"),
            qgis_project=data.get("qgis_project"),
            qgis_layer_group=str(data.get("qgis_layer_group", "handily")),
            qgis_view_root=data.get("qgis_view_root"),
        )
        return cfg

    @classmethod
    def from_toml(cls, toml_path: str) -> "HandilyConfig":
        import tomllib

        with open(os.path.expanduser(toml_path), "rb") as f:
            data = tomllib.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"TOML config must parse to a mapping, got {type(data).__name__}"
            )
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "HandilyConfig":
        import warnings

        warnings.warn(
            "from_yaml is deprecated, use from_toml instead",
            DeprecationWarning,
            stacklevel=2,
        )
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to load config from YAML (pip install pyyaml)."
            ) from exc
        with open(os.path.expanduser(yaml_path), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML config must parse to a mapping, got {type(data).__name__}"
            )
        return cls.from_dict(data)

    @staticmethod
    def required_keys() -> tuple[str, ...]:
        return (
            "out_dir",
            "flowlines_local_dir",
            "ndwi_dir",
            "stac_dir",
            "fields_path",
        )
