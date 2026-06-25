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


CONFIG_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "paths": (
        "dem_path",
        "streams_path",
        "out_dir",
        "naip_path",
        "ndvi_path",
        "support_path",
        "fac_path",
    ),
    "strips": (
        "coarse_res_m",
        "smooth_sigma_m",
        "station_spacing_m",
        "tangent_step_m",
        "min_hit_dist_m",
        "min_strahler",
        "max_crossing_strip_m",
        "naked_fill_m",
        "halo_n",
        "workers",
        "write_strip_debug",
    ),
    "raster": (
        "burn_res_m",
        "idw_radius_m",
        "idw_power",
        "post_smooth_m",
        "base_fac_snap_cells",
        "base_smooth_stations",
    ),
    "seed": (
        "ndvi_mid",
        "ndvi_scale",
        "ndvi_quantile",
        "seed_corridor_m",
        "support_override",
        "seed_from_nhd_class",
    ),
    "propagation": (
        "distance_scale_m",
        "elevation_scale_m",
        "down_distance_scale_m",
        "strahler_distance_scale",
    ),
    "sag": (
        "gamma",
        "rmax_min_m",
        "rmax_max_m",
        "alpha_d",
        "alpha_z",
        "area_sag_lo_km2",
        "area_sag_hi_km2",
    ),
    "solver": (
        "d_min_off_support_m",
        "support_fraction_threshold",
        "strahler_pin_min",
        "area_pin_km2",
        "below_bed_offset_m",
        "target_weight_base",
        "zero_weight_base",
        "smoothness_weight",
        "neighbor_length_floor_m",
        "max_hydraulic_slope",
        "max_iter",
        "tol",
    ),
}


def _merge_nested(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], val)
        else:
            merged[key] = val
    return merged


def _resolve_profile_path(config_path: Path, profile: Any) -> Path:
    profile_path = Path(os.path.expanduser(str(profile)))
    if not profile_path.is_absolute():
        profile_path = config_path.parent / profile_path
    return profile_path.resolve()


def _load_toml_with_profile(
    toml_path: Path,
    *,
    seen: set[Path] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    toml_path = toml_path.resolve()
    seen = set() if seen is None else seen
    if toml_path in seen:
        raise ValueError(f"Recursive FAC REM profile reference: {toml_path}")
    seen.add(toml_path)

    with open(toml_path, "rb") as f:
        raw = tomllib.load(f)

    profile = raw.get("profile")
    if profile is None:
        return raw, None

    profile_path = _resolve_profile_path(toml_path, profile)
    profile_raw, _nested_profile = _load_toml_with_profile(
        profile_path,
        seen=seen,
    )
    overrides = {k: v for k, v in raw.items() if k != "profile"}
    return _merge_nested(profile_raw, overrides), profile_path


@dataclass
class FacRemConfig:
    # --- Paths (required) ---
    dem_path: str
    streams_path: str
    out_dir: str
    naip_path: str | None = None
    ndvi_path: str | None = None
    support_path: str | None = None
    fac_path: str | None = None

    # --- Strip geometry ---
    coarse_res_m: float = 20.0
    smooth_sigma_m: float = 500.0
    station_spacing_m: float = 200.0
    tangent_step_m: float = 10.0
    min_hit_dist_m: float = 5.0
    min_strahler: int = 0
    max_crossing_strip_m: float = 0.0
    # When > 0: if a station ray finds no stream, or only finds a stream beyond
    # max_crossing_strip_m, emit a bounded flat ray of this length (endpoint =
    # local channel elevation) instead of a long edge/AOI ray.  0 disables the
    # bounded fallback.
    naked_fill_m: float = 0.0
    halo_n: int = 0
    workers: int = 1
    # When false, skip writing the diagnostic ``fac_normals_cross_sections.fgb``
    # strip layer. The strips are an intermediate of the burn, not a product, and
    # materializing tens of millions of LineStrings to disk is a large, pointless
    # cost at CONUS scale. The CONUS builder disables it; local runs keep it on
    # for QGIS inspection.
    write_strip_debug: bool = True

    # --- Raster resolution ---
    burn_res_m: float = 20.0
    idw_radius_m: float = 200.0
    idw_power: float = 1.0
    post_smooth_m: float = 0.0
    base_fac_snap_cells: int = 0
    base_smooth_stations: int = 0

    # --- Head-solve: seed strength ---
    ndvi_mid: float = 0.20
    ndvi_scale: float = 0.06
    ndvi_quantile: float = 0.9
    # When true, the soft wet seed comes from each reach's NHD perenniality
    # class (streams ``nhd_class`` column) instead of an NDVI raster — the
    # static CONUS path with no per-AOI imagery. rem_fac builds the per-reach
    # [0,1] override and relaxes the NDVI/NAIP head-solve gate.
    seed_from_nhd_class: bool = False
    # Lateral half-width (m) of the swath sampled for the NDVI seed quantile.
    # 0.0 = legacy centerline-only sampling. When > 0, the seed evidence is
    # sampled across perpendicular transects spanning ±seed_corridor_m so a
    # reach whose centerline runs down a bare valley-floor arroyo still picks up
    # the adjacent irrigated/vegetated floor instead of solving spuriously deep.
    # Self-limiting: desert uplands have no adjacent green to raise the quantile.
    seed_corridor_m: float = 0.0
    support_override: float = 1.0

    # --- Head-solve: wet propagation ---
    distance_scale_m: float = 4000.0
    elevation_scale_m: float = 25.0
    down_distance_scale_m: float = 20000.0
    strahler_distance_scale: float = 0.5

    # --- Head-solve: sag targets ---
    gamma: float = 2.0
    rmax_min_m: float = 2.0
    rmax_max_m: float = 60.0
    alpha_d: float = 0.75
    alpha_z: float = 1.0
    area_sag_lo_km2: float = 50.0
    area_sag_hi_km2: float = 500.0

    # --- Head-solve: residual solver ---
    d_min_off_support_m: float = 0.5
    support_fraction_threshold: float = 0.25
    strahler_pin_min: int | None = None
    area_pin_km2: float | None = None
    # Below-bed offset (m) for order/area-pinned perennial mainstems: in
    # diverted/losing semi-arid valleys the table sits ~1-2 m below the bed, so
    # the pinned water surface drops by this much instead of sitting at the bed
    # (head_depth = 0). 0.0 keeps the legacy bed pin. See notes/lit synthesis 4c.
    below_bed_offset_m: float = 0.0
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
            "ndvi_path",
            "support_path",
            "fac_path",
        ):
            val = getattr(self, name)
            if val is not None:
                setattr(self, name, os.path.expanduser(val))

    @classmethod
    def from_toml(cls, toml_path: str | Path) -> FacRemConfig:
        path = Path(os.path.expanduser(str(toml_path)))
        raw, profile_path = _load_toml_with_profile(path)
        data: dict[str, Any] = {}
        # Flatten one level of TOML sections into the dataclass namespace.
        for key, val in raw.items():
            if isinstance(val, dict):
                data.update(val)
            else:
                data[key] = val
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        cfg = cls(**filtered)
        cfg.source_path = str(path.resolve())
        cfg.profile_path = str(profile_path) if profile_path is not None else None
        return cfg

    def to_section_dict(
        self, *, include_paths: bool = True
    ) -> dict[str, dict[str, Any]]:
        sections: dict[str, dict[str, Any]] = {}
        for section, names in CONFIG_SECTION_FIELDS.items():
            if section == "paths" and not include_paths:
                continue
            sections[section] = {name: getattr(self, name) for name in names}
        return sections

    def head_solve_kwargs(self) -> dict[str, Any]:
        """Return kwargs suitable for ``build_channel_heads()``."""
        return {
            "ndvi_mid": self.ndvi_mid,
            "ndvi_scale": self.ndvi_scale,
            "ndvi_quantile": self.ndvi_quantile,
            "seed_corridor_m": self.seed_corridor_m,
            "support_override": self.support_override,
            "distance_scale_m": self.distance_scale_m,
            "elevation_scale_m": self.elevation_scale_m,
            "down_distance_scale_m": self.down_distance_scale_m,
            "strahler_distance_scale": self.strahler_distance_scale,
            "gamma": self.gamma,
            "rmax_min_m": self.rmax_min_m,
            "rmax_max_m": self.rmax_max_m,
            "alpha_d": self.alpha_d,
            "alpha_z": self.alpha_z,
            "area_sag_lo_km2": self.area_sag_lo_km2,
            "area_sag_hi_km2": self.area_sag_hi_km2,
            "d_min_off_support_m": self.d_min_off_support_m,
            "support_fraction_threshold": self.support_fraction_threshold,
            "strahler_pin_min": self.strahler_pin_min,
            "area_pin_km2": self.area_pin_km2,
            "below_bed_offset_m": self.below_bed_offset_m,
            "target_weight_base": self.target_weight_base,
            "zero_weight_base": self.zero_weight_base,
            "smoothness_weight": self.smoothness_weight,
            "neighbor_length_floor_m": self.neighbor_length_floor_m,
            "max_hydraulic_slope": self.max_hydraulic_slope,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }
