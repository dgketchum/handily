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

    def __post_init__(self) -> None:
        self.out_dir = os.path.expanduser(self.out_dir)
        self.flowlines_local_dir = os.path.expanduser(self.flowlines_local_dir)
        self.ndwi_dir = os.path.expanduser(self.ndwi_dir)
        self.stac_dir = os.path.expanduser(self.stac_dir)
        self.fields_path = os.path.expanduser(self.fields_path)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HandilyConfig":
        missing = [k for k in cls.required_keys() if k not in data]
        if missing:
            raise KeyError(f"Missing required config keys: {missing}")
        return cls(
            out_dir=str(data["out_dir"]),
            flowlines_local_dir=str(data["flowlines_local_dir"]),
            ndwi_dir=str(data["ndwi_dir"]),
            stac_dir=str(data["stac_dir"]),
            fields_path=str(data["fields_path"]),
        )

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
