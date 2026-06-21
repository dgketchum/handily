import json
from pathlib import Path

from handily.rem_fac import _write_run_metadata, parse_args
from handily.rem_fac_config import FacRemConfig


def test_fac_rem_config_loads_relative_profile_and_aoi_overrides(tmp_path):
    profile = tmp_path / "profile.toml"
    profile.write_text(
        """
[strips]
station_spacing_m = 50.0
halo_n = 12

[raster]
burn_res_m = 5.0
idw_radius_m = 1000.0
post_smooth_m = 100.0

[sag]
rmax_max_m = 60.0
""".lstrip(),
        encoding="utf-8",
    )
    config = tmp_path / "aoi.toml"
    config.write_text(
        """
profile = "profile.toml"

[paths]
dem_path = "/tmp/dem.tif"
streams_path = "/tmp/streams.fgb"
out_dir = "/tmp/out"
naip_path = "/tmp/naip.tif"

[raster]
burn_res_m = 10.0
""".lstrip(),
        encoding="utf-8",
    )

    cfg = FacRemConfig.from_toml(config)

    assert cfg.station_spacing_m == 50.0
    assert cfg.halo_n == 12
    assert cfg.idw_radius_m == 1000.0
    assert cfg.post_smooth_m == 100.0
    assert cfg.rmax_max_m == 60.0
    assert cfg.burn_res_m == 10.0
    assert cfg.profile_path == str(profile.resolve())


def test_run_metadata_writes_effective_profile_config(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    args = parse_args(
        [
            "--config",
            str(repo / "configs/rem/mt_0009.toml"),
            "--out-dir",
            str(out_dir),
        ]
    )
    _write_run_metadata(args, {"streams": {"count": 7448}})

    metadata = json.loads((out_dir / "fac_rem_run.json").read_text(encoding="utf-8"))

    assert metadata["profile_path"].endswith("configs/rem/profiles/mt_0009_best.toml")
    assert metadata["effective_config"]["paths"]["out_dir"] == str(out_dir)
    assert metadata["effective_config"]["strips"]["station_spacing_m"] == 50.0
    assert metadata["effective_config"]["strips"]["naked_fill_m"] == 0.0
    assert metadata["effective_config"]["strips"]["halo_n"] == 12
    assert metadata["effective_config"]["raster"]["burn_res_m"] == 5.0
    assert metadata["effective_config"]["raster"]["idw_radius_m"] == 1000.0
    assert metadata["effective_config"]["raster"]["post_smooth_m"] == 100.0
    assert metadata["effective_config"]["sag"]["rmax_max_m"] == 60.0
    assert metadata["diagnostics"]["streams"]["count"] == 7448
