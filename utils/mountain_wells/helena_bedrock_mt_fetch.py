"""Fetch USGS WRI 00-4212 (Helena-area bedrock, MT) report, plates, data archives, plus NWIS
groundwater levels / site metadata for the report's wells, all unchanged.

Usage:
  uv run python -I utils/mountain_wells/helena_bedrock_mt_fetch.py <raw_dir> <fetch_log_json> [<site_list_txt>]

Without <site_list_txt> the publication files and an NWIS groundwater-site catalog for the Helena
bounding box (used to match the report's location numbers to NWIS station names) are fetched. With it (one 15-digit USGS site
number per line, produced by helena_bedrock_mt_parse.py --site-list), NWIS gwlevels and expanded site
metadata are fetched in batches into <raw_dir>/nwis/. Failures are logged, never retried silently.
"""

import hashlib
import json
import argparse
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://pubs.usgs.gov/wri/wri-00-4212/"
PUB_TARGETS = [
    ("landing_publication.html", "https://pubs.usgs.gov/publication/wri20004212"),
    ("index.html", BASE),
    ("WRI00-4212.pdf", BASE + "downloads/WRI00-4212.pdf"),
    ("WRI00-4212-PLATE1.pdf", BASE + "downloads/WRI00-4212-PLATE1.pdf"),
    ("WRI00-4212-PLATE2.pdf", BASE + "downloads/WRI00-4212-PLATE2.pdf"),
    ("WRI00-4212-PLATE3.pdf", BASE + "downloads/WRI00-4212-PLATE3.pdf"),
    ("wri-00-4212.met", BASE + "downloads/wri-00-4212.met"),
    ("wri-00-4212.zip", BASE + "downloads/wri-00-4212.zip"),
    ("wri-00-4212.tar.gz", BASE + "downloads/wri-00-4212.tar.gz"),
    (
        "nwis/site_catalog_bbox.rdb",
        "https://waterservices.usgs.gov/nwis/site/?format=rdb&bBox=-112.45,46.40,-111.70,46.90"
        "&siteType=GW&siteOutput=expanded&siteStatus=all",
    ),
]
OGC = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/field-measurements/items"
GWIC = "https://mbmggwic.mtech.edu"
BATCH = 40


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(url, out, raw, log):
    out.parent.mkdir(parents=True, exist_ok=True)
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
    if r.returncode == 0 and code == "200" and out.exists() and out.stat().st_size > 0:
        log["files"].append(
            {
                "path": f"raw/helena_bedrock_mt/{out.relative_to(raw)}",
                "url": url,
                "sha256": sha256(out),
                "bytes": out.stat().st_size,
            }
        )
        return True
    log["failures"].append(
        {
            "path": str(out.relative_to(raw)),
            "url": url,
            "http": code,
            "err": r.stderr.strip(),
        }
    )
    if out.exists() and code != "200":
        out.unlink()
    return False


def api_key():
    k = os.environ.get("USGS_API_KEY")
    if not k:
        p = Path.home() / "nwis_key.txt"
        k = p.read_text().strip() if p.exists() else None
    return k


def get_ogc(url, out, raw, log, key):
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"X-Api-Key": key} if key else {})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            out.write_bytes(body)
            log["files"].append(
                {
                    "path": f"raw/helena_bedrock_mt/{out.relative_to(raw)}",
                    "url": url,
                    "sha256": sha256(out),
                    "bytes": out.stat().st_size,
                }
            )
            return json.loads(body)
        except Exception as e:  # noqa: BLE001 - logged and retried with backoff
            err = repr(e)
            time.sleep(5 * (attempt + 1))
    log["failures"].append({"path": str(out.relative_to(raw)), "url": url, "err": err})
    return None


def fetch_nwis(raw, log, site_file, tag=""):
    key = api_key()
    sites = [s.strip() for s in Path(site_file).read_text().split() if s.strip()]
    for i in range(0, len(sites), BATCH):
        chunk = ",".join("USGS-" + s for s in sites[i : i + BATCH])
        url = (
            OGC
            + "?"
            + urllib.parse.urlencode(
                {"monitoring_location_id": chunk, "limit": 10000, "f": "json"}
            )
        )
        page = 0
        while url:
            d = get_ogc(
                url,
                raw
                / "nwis"
                / f"field_measurements_{tag}{i // BATCH:02d}_{page:02d}.json",
                raw,
                log,
                key,
            )
            if d is None:
                break
            nxt = [lk["href"] for lk in d.get("links", []) if lk.get("rel") == "next"]
            url = nxt[0] if nxt else None
            page += 1
            time.sleep(1)


def fetch_gwic(raw, log, id_file):
    done = {f["path"] for f in log["files"]}
    for g in [s.strip() for s in Path(id_file).read_text().split() if s.strip()]:
        for name, url in (
            (
                f"gwic/site_report_{g}.html",
                f"{GWIC}/sqlserver/v11/reports/SiteSummary.asp?gwicid={g}&agency=mbmg",
            ),
            (
                f"gwic/hydrograph_{g}.json",
                f"{GWIC}/reports/PrecipHydrographScript.py?gwicid={g}",
            ),
        ):
            if f"raw/helena_bedrock_mt/{name}" in done:
                continue
            get(url, raw / name, raw, log)
            time.sleep(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("log_path")
    ap.add_argument("--nwis-sites")
    ap.add_argument(
        "--nwis-tag",
        default="",
        help="file-name prefix for an added site batch, e.g. add_",
    )
    ap.add_argument("--gwic-ids")
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    log_p = Path(a.log_path)
    log = (
        json.loads(log_p.read_text())
        if log_p.exists()
        else {"files": [], "failures": []}
    )
    done = {f["path"] for f in log["files"]}
    if a.nwis_sites:
        pre = f"/nwis/field_measurements_{a.nwis_tag}"
        log["files"] = [
            f
            for f in log["files"]
            if pre not in f["path"] or (not a.nwis_tag and "_add" in f["path"])
        ]
        fetch_nwis(raw, log, a.nwis_sites, a.nwis_tag)
    elif a.gwic_ids:
        fetch_gwic(raw, log, a.gwic_ids)
    else:
        for name, url in PUB_TARGETS:
            if f"raw/helena_bedrock_mt/{name}" not in done:
                get(url, raw / name, raw, log)
    log_p.write_text(json.dumps(log, indent=2))
    print(
        json.dumps(
            {"n_files": len(log["files"]), "failures": log["failures"][-10:]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
