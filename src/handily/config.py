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
    ptjpl_start_yr: int | None = None
    ptjpl_end_yr: int | None = None
    ptjpl_check_dir: str | None = None

    met_start: str | None = None
    met_end: str | None = None
    gridmet_parquet_dir: str | None = None
    gridmet_centroids_path: str | None = None
    gridmet_centroid_parquet_dir: str | None = None
    gridmet_id_col: str = "GFID"

    ptjpl_csv_dir: str | None = None
    ptjpl_csv_template: str | None = None
    et_join_parquet_dir: str | None = None

    partition_joined_parquet_dir: str | None = None
    partition_out_parquet_dir: str | None = None
    partition_strata_col: str = "strata"
    partition_pattern_col: str = "pattern"

    def __post_init__(self) -> None:
        self.out_dir = os.path.expanduser(self.out_dir)
        self.flowlines_local_dir = os.path.expanduser(self.flowlines_local_dir)
        self.ndwi_dir = os.path.expanduser(self.ndwi_dir)
        self.stac_dir = os.path.expanduser(self.stac_dir)
        self.fields_path = os.path.expanduser(self.fields_path)
        if self.ptjpl_check_dir is not None:
            self.ptjpl_check_dir = os.path.expanduser(self.ptjpl_check_dir)
        if self.gridmet_parquet_dir is not None:
            self.gridmet_parquet_dir = os.path.expanduser(self.gridmet_parquet_dir)
        if self.gridmet_centroids_path is not None:
            self.gridmet_centroids_path = os.path.expanduser(self.gridmet_centroids_path)
        if self.gridmet_centroid_parquet_dir is not None:
            self.gridmet_centroid_parquet_dir = os.path.expanduser(self.gridmet_centroid_parquet_dir)
        if self.ptjpl_csv_dir is not None:
            self.ptjpl_csv_dir = os.path.expanduser(self.ptjpl_csv_dir)
        if self.et_join_parquet_dir is not None:
            self.et_join_parquet_dir = os.path.expanduser(self.et_join_parquet_dir)
        if self.partition_joined_parquet_dir is not None:
            self.partition_joined_parquet_dir = os.path.expanduser(self.partition_joined_parquet_dir)
        if self.partition_out_parquet_dir is not None:
            self.partition_out_parquet_dir = os.path.expanduser(self.partition_out_parquet_dir)

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
            ptjpl_start_yr=data.get("ptjpl_start_yr"),
            ptjpl_end_yr=data.get("ptjpl_end_yr"),
            ptjpl_check_dir=data.get("ptjpl_check_dir"),
            met_start=data.get("met_start"),
            met_end=data.get("met_end"),
            gridmet_parquet_dir=data.get("gridmet_parquet_dir"),
            gridmet_centroids_path=data.get("gridmet_centroids_path"),
            gridmet_centroid_parquet_dir=data.get("gridmet_centroid_parquet_dir"),
            gridmet_id_col=str(data.get("gridmet_id_col", "GFID")),
            ptjpl_csv_dir=data.get("ptjpl_csv_dir"),
            ptjpl_csv_template=data.get("ptjpl_csv_template"),
            et_join_parquet_dir=data.get("et_join_parquet_dir"),
            partition_joined_parquet_dir=data.get("partition_joined_parquet_dir"),
            partition_out_parquet_dir=data.get("partition_out_parquet_dir"),
            partition_strata_col=str(data.get("partition_strata_col", "strata")),
            partition_pattern_col=str(data.get("partition_pattern_col", "pattern")),
        )
        return cfg

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "HandilyConfig":
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyYAML is required to load config from YAML (pip install pyyaml).") from exc
        with open(os.path.expanduser(yaml_path), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML config must parse to a mapping, got {type(data).__name__}")
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
