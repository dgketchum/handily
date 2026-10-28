"""Fetch NMBGMR OFR-543 (southern Sacramento Mountains hydrogeology) report + appendices unchanged.

Usage: uv run python -I utils/mountain_wells/sacramento_mtns_nm_fetch.py <raw_dir> <fetch_log_json>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SOURCE_ID = "sacramento_mtns_nm"
LANDING = "https://geoinfo.nmt.edu/publications/openfile/details.cfml?Volume=543"
BASE = "https://geoinfo.nmt.edu/publications/openfile/downloads/500-599/543/"
APP = BASE + "Appendices/"
TARGETS = [
    ("landing.html", [LANDING]),
    ("OFR-543_HR.pdf", [BASE + "OFR-543_HR.pdf"]),
    ("OFR-543_LR.pdf", [BASE + "OFR-543_LR.pdf"]),
    ("OFR-543_complete.zip", [BASE + "OFR-543_complete.zip"]),
    (
        "OFR-543_Appendix-01_Site_inventory.pdf",
        [APP + "OFR-543_Appendix-01_Site_inventory.pdf"],
    ),
    (
        "OFR-543_Appendix-02_NMOSE_well_records.pdf",
        [APP + "OFR-543_Appendix-02_NMOSE_well_records.pdf"],
    ),
    (
        "OFR-543_Appendix-03_Water_level_measurements.pdf",
        [APP + "OFR-543_Appendix-03_Water_level_measurements.pdf"],
    ),
    (
        "OFR-543_Appendix-04_Data_logger_data.xls",
        [APP + "OFR-543_Appendix-04_Data_logger_data.xls"],
    ),
    (
        "OFR-543_Appendix-05_Streamflow_measurements.pdf",
        [APP + "OFR-543_Appendix-05_Streamflow_measurements.pdf"],
    ),
    (
        "OFR-543_Appendix-06_Water_table_data.pdf",
        [APP + "OFR-543_Appendix-06_Water_table_data.pdf"],
    ),
    (
        "OFR-543_Appendix-07_Water_chemistry_data.pdf",
        [APP + "OFR-543_Appendix-07_Water_chemistry_data.pdf"],
    ),
    (
        "OFR-543_Appendix-08_Mixing_of_water_sources.pdf",
        [APP + "OFR-543_Appendix-08_Mixing_of_water_sources.pdf"],
    ),
    (
        "OFR-543_Appendix-09_Quantification_of_Recharge.pdf",
        [APP + "OFR-543_Appendix-09_Quantification_of_Recharge.pdf"],
    ),
    (
        "OFR-543_Appendix-10_Estimation_of_hydrologic_properties.pdf",
        [APP + "OFR-543_Appendix-10_Estimation_of_hydrologic_properties.pdf"],
    ),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(raw_dir, log_path):
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    log = {"files": [], "failures": []}
    for name, urls in TARGETS:
        out = raw / name
        ok = False
        for url in urls:
            r = subprocess.run(
                [
                    "curl",
                    "-sSL",
                    "-A",
                    "Mozilla/5.0",
                    "-o",
                    str(out),
                    "-w",
                    "%{http_code}",
                    url,
                ],
                capture_output=True,
                text=True,
            )
            code = r.stdout.strip()
            if r.returncode == 0 and code == "200" and out.stat().st_size > 0:
                log["files"].append(
                    {
                        "path": f"raw/{SOURCE_ID}/{name}",
                        "url": url,
                        "sha256": sha256(out),
                        "bytes": out.stat().st_size,
                    }
                )
                ok = True
                break
            log["failures"].append(
                {"name": name, "url": url, "http": code, "err": r.stderr.strip()}
            )
        if not ok and out.exists():
            out.unlink()
    Path(log_path).write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
