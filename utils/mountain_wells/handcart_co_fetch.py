"""Fetch USGS OFR 2006-1189 (Handcart Gulch, CO) report + data worksheets unchanged.

Usage: uv run python -I utils/mountain_wells/handcart_co_fetch.py <raw_dir> <fetch_log_json>
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

BASE = "https://pubs.usgs.gov/of/2006/1189/"
TARGETS = [
    ("landing.html", [BASE]),
    (
        "of06-1189_508.pdf",
        [
            BASE + "downloads/pdf/of06-1189_508.pdf",
            "https://pubsdata.usgs.gov/pubs/of/2006/1189/downloads/pdf/of06-1189_508.pdf",
        ],
    ),
    (
        "HandcartWellData.xls",
        [
            BASE + "downloads/xls/HandcartWellData.xls",
            "https://pubsdata.usgs.gov/pubs/of/2006/1189/downloads/xls/HandcartWellData.xls",
        ],
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
                        "path": f"raw/handcart_co/{name}",
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
