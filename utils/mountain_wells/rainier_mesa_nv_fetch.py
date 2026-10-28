"""Fetch USGS DS 190 (Rainier Mesa, NTS water levels 1957-2005) unchanged, plus post-2005 extensions.

Usage:
  uv run python -I utils/mountain_wells/rainier_mesa_nv_fetch.py <raw_dir> <fetch_log_json>
  uv run python -I utils/mountain_wells/rainier_mesa_nv_fetch.py <raw_dir> <fetch_log_json> --sites <sites.txt>

Stage 1 (no --sites): report PDF, appendix bundle (MDB + XLS + appendix PDF) and HTML report pages.
Stage 2 (--sites, one 15-digit USGS site number per line):
  - legacy NWIS gwlevels RDB (decommissioned; attempted and logged),
  - USGS Water Data OGC API field-measurements + monitoring-locations CSV -> <raw_dir>/nwis/,
  - USGS-NNSS Regional Groundwater Explorer JSON API (per-site interpreted levels with
    general/detailed condition attributes, plus whole-database site/construction/open tables)
    -> <raw_dir>/nnss_explorer/.
The fetch log is keyed by path and merged across stages.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

SRC = "rainier_mesa_nv"
BASE = "https://pubs.usgs.gov/ds/2006/190/"
PAGES = [
    "revisionhistory.html",
    "section.html",
    "section1.html",
    "section2.html",
    "section3.html",
    "section4.html",
    "section5.html",
    "section6.html",
    "figure.html",
    "figure1.html",
    "figure2.html",
    "figure3.html",
    "table1.html",
    "table2.html",
    "table3.html",
    "table4.html",
]
TARGETS = [
    ("landing.html", [BASE]),
    ("ds2006190.pdf", [BASE + "pdf/ds2006190.pdf"]),
    ("appendixes.zip", [BASE + "pdf/appendixes.zip"]),
] + [(p, [BASE + p]) for p in PAGES]

NWIS_LEGACY = "https://waterservices.usgs.gov/nwis/gwlevels/?format=rdb&siteStatus=all&sites={sites}"
OGC = "https://api.waterdata.usgs.gov/ogcapi/v0/collections"
FM = OGC + "/field-measurements/items?monitoring_location_id={ids}&f=csv&limit=50000"
ML = OGC + "/monitoring-locations/items?id={ids}&f=csv&limit=500"
NNSS = "https://nevada.usgs.gov/doe-nnss-gw-api-v2"
NNSS_TABLES = [
    "sites",
    "lookups",
    "site-info",
    "construction-info",
    "open-info",
    "hole-info",
    "casing-info",
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(out, urls, rel, log, follow=True):
    for url in urls:
        cmd = [
            "curl",
            "-sS",
            "-A",
            "Mozilla/5.0",
            "-o",
            str(out),
            "-w",
            "%{http_code}",
            url,
        ]
        if follow:
            cmd.insert(2, "-L")
        r = subprocess.run(cmd, capture_output=True, text=True)
        code = r.stdout.strip()
        if (
            r.returncode == 0
            and code == "200"
            and out.exists()
            and out.stat().st_size > 0
        ):
            log["files"][rel] = {
                "path": rel,
                "url": url,
                "sha256": sha256(out),
                "bytes": out.stat().st_size,
            }
            return True
        log["failures"].append(
            {"name": rel, "url": url, "http": code, "err": r.stderr.strip()}
        )
    if out.exists():
        out.unlink()
    return False


def main(raw_dir, log_path, sites_file=None):
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    lp = Path(log_path)
    log = json.loads(lp.read_text()) if lp.exists() else {"files": {}, "failures": []}
    if sites_file is None:
        for name, urls in TARGETS:
            get(raw / name, urls, f"raw/{SRC}/{name}", log)
    else:
        sites = [s.strip() for s in Path(sites_file).read_text().split() if s.strip()]
        nd = raw / "nwis"
        nd.mkdir(exist_ok=True)
        ed = raw / "nnss_explorer"
        ed.mkdir(exist_ok=True)
        # legacy service: expected to fail (301 to decommission notice); do not follow the redirect
        get(
            nd / "gwlevels_legacy.rdb",
            [NWIS_LEGACY.format(sites=",".join(sites))],
            f"raw/{SRC}/nwis/gwlevels_legacy.rdb",
            log,
            follow=False,
        )
        for i in range(0, len(sites), 20):
            ids = ",".join("USGS-" + s for s in sites[i : i + 20])
            k = i // 20
            get(
                nd / f"field_measurements_{k:02d}.csv",
                [FM.format(ids=ids)],
                f"raw/{SRC}/nwis/field_measurements_{k:02d}.csv",
                log,
            )
            get(
                nd / f"monitoring_locations_{k:02d}.csv",
                [ML.format(ids=ids)],
                f"raw/{SRC}/nwis/monitoring_locations_{k:02d}.csv",
                log,
            )
        for t in NNSS_TABLES:
            get(
                ed / f"{t}.json",
                [f"{NNSS}/{t}"],
                f"raw/{SRC}/nnss_explorer/{t}.json",
                log,
            )
        sd = ed / "site_data"
        sd.mkdir(exist_ok=True)
        for s in sites:
            get(
                sd / f"{s}.json",
                [f"{NNSS}/get-site-data/{s}-USGS"],
                f"raw/{SRC}/nnss_explorer/site_data/{s}.json",
                log,
            )
    lp.write_text(json.dumps(log, indent=2))
    print(
        json.dumps(
            {"n_files": len(log["files"]), "failures": log["failures"]}, indent=2
        )
    )


if __name__ == "__main__":
    a = sys.argv[1:]
    sf = None
    if "--sites" in a:
        j = a.index("--sites")
        sf = a[j + 1]
        a = a[:j] + a[j + 2 :]
    main(a[0], a[1], sf)
