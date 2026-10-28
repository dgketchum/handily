"""Fetch CJCZO Marshall Gulch piezometer resource from HydroShare (public, no auth), unchanged.

Resource a04392de2b3448d48836ad47571a1f48:
  "CJCZO -- Piezometer, Groundwater Depth -- Marshall Gulch -- (2007-2019)"

Saves sysmeta.json, scimeta.xml, files.json, landing.html and every listed content file to
<raw_dir>/ (content paths kept relative to data/contents/), and writes a fetch log with
sha256 / bytes / url / HydroShare md5 check to <log_path>.

Usage: uv run python -I utils/mountain_wells/marshall_gulch_az_fetch.py <raw_dir> <log_path>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

RID = "a04392de2b3448d48836ad47571a1f48"
API = "https://www.hydroshare.org/hsapi/resource/"
# Linked / companion content (evidence for site setting and barometer). Failures are logged, not fatal.
EXTERNAL = [
    (
        "external/GroundwaterTableMeasurementsMG.pdf",
        "http://www.czo.arizona.edu/czo_data/MethodsSites/catalinas/GroundwaterTableMeasurementsMG.pdf",
    ),
    (
        "external/czo_archive_marshall_gulch_field_area.html",
        "https://czo-archive.criticalzone.org/catalina-jemez/infrastructure/field-area/bigelow-tower-marshall-gulch-high-elevation/",
    ),
    (
        "external/hs_7d882820166f4695ae614fe436db5172_MG_air_pressure_sysmeta.json",
        f"{API}7d882820166f4695ae614fe436db5172/sysmeta/",
    ),
]


def digest(p, algo):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl(url, out, tries=4):
    code, err = "", ""
    for _ in range(tries):
        r = subprocess.run(
            [
                "curl",
                "-sSL",
                "--connect-timeout",
                "30",
                "-m",
                "600",
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
        code, err = r.stdout.strip(), r.stderr.strip()
        if (
            r.returncode == 0
            and code == "200"
            and out.exists()
            and out.stat().st_size > 0
        ):
            return True, code, err
    return False, code, err


def list_files():
    url = f"{API}{RID}/files/"
    out = []
    while url:
        r = subprocess.run(
            ["curl", "-sSL", "-m", "120", url], capture_output=True, text=True
        )
        d = json.loads(r.stdout)
        out.extend(d["results"])
        url = d.get("next")
    return out


def main(raw_dir, log_path):
    raw_dir, log_path = Path(raw_dir), Path(log_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    listing = list_files()
    jobs = [
        ("sysmeta.json", f"{API}{RID}/sysmeta/"),
        ("scimeta.xml", f"{API}{RID}/scimeta/"),
        ("files.json", f"{API}{RID}/files/"),
        ("landing.html", f"https://www.hydroshare.org/resource/{RID}/"),
    ]
    checks = {}
    for f in listing:
        rel = f["url"].split("/data/contents/", 1)[1]
        jobs.append((rel, f"{API}{RID}/files/{rel}/"))
        checks[rel] = (f.get("checksum"), f.get("size"))
    jobs.extend(EXTERNAL)
    log = {"resource_id": RID, "n_listed": len(listing), "files": [], "failures": []}
    for name, url in jobs:
        out = raw_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        ok, code, err = curl(url, out)
        if not ok:
            log["failures"].append({"name": name, "url": url, "http": code, "err": err})
            if out.exists():
                out.unlink()
            continue
        rec = {
            "path": f"raw/marshall_gulch_az/{name}",
            "url": url,
            "sha256": digest(out, "sha256"),
            "bytes": out.stat().st_size,
        }
        if name in checks:
            md5, size = checks[name]
            rec["hydroshare_md5"] = md5
            rec["md5_match"] = digest(out, "md5") == md5
            rec["size_match"] = out.stat().st_size == size
        log["files"].append(rec)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=2))
    bad = [f["path"] for f in log["files"] if f.get("md5_match") is False]
    print(
        len(log["files"]),
        "files,",
        len(log["failures"]),
        "failures,",
        len(bad),
        "md5 mismatches",
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
