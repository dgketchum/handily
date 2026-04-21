"""FAC REM experiment configuration.

Loads head-solve tuning parameters, strip geometry settings, raster
resolution, and input paths from a TOML file.  Separate from the
project-level HandilyConfig which handles AOI management, STAC, NDWI,
points sampling, etc.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class FacRemConfig:
    # --- Paths (required) ---
    dem_path: str
    streams_path: str
    out_dir: str
    naip_path: str | None = None
    support_path: str | None = None

    # --- Strip geometry ---
    coarse_res_m: float = 20.0
    smooth_sigma_m: float = 500.0
    station_spacing_m: float = 200.0
    tangent_step_m: float = 10.0
    min_hit_dist_m: float = 5.0
    min_strahler: int = 0
    max_crossing_strip_m: float = 0.0
    halo_n: int = 0
    workers: int = 1

    # --- Raster resolution ---
    burn_res_m: float = 20.0
    idw_radius_m: float = 200.0
    idw_power: float = 1.0
    post_smooth_m: float = 0.0

    # --- Head-solve: seed strength ---
    ndvi_mid: float = 0.20
    ndvi_scale: float = 0.06
    ndvi_quantile: float = 0.9
    support_override: float = 1.0

    # --- Head-solve: upstream propagation ---
    distance_scale_m: float = 4000.0
    elevation_scale_m: float = 25.0
    strahler_distance_scale: float = 0.5

    # --- Head-solve: sag targets ---
    gamma: float = 2.0
    rmax_min_m: float = 2.0
    rmax_max_m: float = 60.0
    alpha_d: float = 0.75
    alpha_z: float = 1.0

    # --- Head-solve: residual solver ---
    d_min_off_support_m: float = 0.5
    support_fraction_threshold: float = 0.25
    target_weight_base: float = 2.0
    zero_weight_base: float = 2.0
    smoothness_weight: float = 1.0
    neighbor_length_floor_m: float = 200.0
    max_hydraulic_slope: float = 0.05
    max_iter: int = 500
    tol: float = 0.01

    def __post_init__(self) -> None:
        for name in (
            "dem_path",
            "streams_path",
            "out_dir",
            "naip_path",
            "support_path",
        ):
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, os.path.expanduser(val))

    @classmethod
    def from_toml(cls, toml_path: str | Path) -> FacRemConfig:
        with open(os.path.expanduser(str(toml_path)), "rb") as f:
            raw = tomllib.load(f)
        data: dict[str, Any] = {}
        # Flatten one level of TOML sections into the dataclass namespace.
        for key, val in raw.items():
            if isinstance(val, dict):
                data.update(val)
            else:
                data[key] = val
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def head_solve_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``build_channel_heads()``."""
        return {
            "ndvi_mid": self.ndvi_mid,
            "ndvi_scale": self.ndvi_scale,
            "ndvi_quantile": self.ndvi_quantile,
            "support_override": self.support_override,
            "distance_scale_m": self.distance_scale_m,
            "elevation_scale_m": self.elevation_scale_m,
            "strahler_distance_scale": self.strahler_distance_scale,
            "gamma": self.gamma,
            "rmax_min_m": self.rmax_min_m,
            "rmax_max_m": self.rmax_max_m,
            "alpha_d": self.alpha_d,
            "alpha_z": self.alpha_z,
            "d_min_off_support_m": self.d_min_off_support_m,
            "support_fraction_threshold": self.support_fraction_threshold,
            "target_weight_base": self.target_weight_base,
            "zero_weight_base": self.zero_weight_base,
            "smoothness_weight": self.smoothness_weight,
            "neighbor_length_floor_m": self.neighbor_length_floor_m,
            "max_hydraulic_slope": self.max_hydraulic_slope,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }
