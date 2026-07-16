"""Generate FAC REM TOML configs for MT AOIs.

Writes one TOML per AOI to configs/rem/mt_{aoi_id:04d}.toml.
Uses the reusable MT AOI 0009 best-run profile as the baseline.

Usage:
    uv run python utils/write_mt_toml_configs.py
"""

import os

AOIS = [1, 7, 8, 9, 10, 12, 15, 33, 34, 35, 36, 39]
DATA_ROOT = "/data/ssd2/handily/mt"
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "rem")

FIPS = {
    1: "001",
    7: "001",
    8: "001",
    9: "001",
    10: "001",
    12: "001",
    15: "001",
    33: "001",
    34: "001",
    35: "001",
    36: "023",
    39: "001",
}

NAIP_YEAR = {
    9: "2021",
}

TEMPLATE = """\
# FAC REM - MT AOI {aoi_id:04d}
# Reusable parameters come from the MT AOI 0009 best-run profile.

profile = "profiles/mt_0009_best.toml"

[paths]
dem_path = "{data_root}/aoi_{aoi_id:04d}/dem_bounds_1m.tif"
streams_path = "{data_root}/aoi_{aoi_id:04d}/streams_fac.fgb"
out_dir = "{data_root}/aoi_{aoi_id:04d}/experimental_full"
naip_path = "{naip_path}"
"""


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    for aoi_id in AOIS:
        fips = FIPS[aoi_id]
        year = NAIP_YEAR.get(aoi_id, "2023")
        naip_path = (
            f"{DATA_ROOT}/aoi_{aoi_id:04d}/naip/ortho_1-1_hc_s_mt{fips}_{year}_1.tif"
        )
        content = TEMPLATE.format(
            aoi_id=aoi_id,
            data_root=DATA_ROOT,
            naip_path=naip_path,
        )
        out_path = os.path.join(CONFIG_DIR, f"mt_{aoi_id:04d}.toml")
        with open(out_path, "w") as f:
            f.write(content)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
