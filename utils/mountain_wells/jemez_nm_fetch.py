"""Fetch CJCZO Jemez groundwater resources from HydroShare (public, no auth), unchanged.

Sources:
  jemez_deep_nm : da61f854a1754e26a7c65de608bf5890 (2013 Burned ZOB deep slope wells, 2016-2021)
  jemez_piez_nm : 8d43148dba4c45fe8f65530d29aab7ae (2011 Burned site shallow wells)
                  36419d60a7484844acab038b1fb84828 (2013 Burned site / MCZOB shallow wells)

For each resource saves sysmeta.json, scimeta.xml, files.json, landing.html and every listed file to
<raw_root>/<source_id>/<resource_id>/, plus a fetch log with sha256/bytes/url.

Also saves White et al. (2019, HESS 23:4661) article + supplement (well construction Table S1) to
<raw_root>/jemez_deep_nm/white_etal_2019_hess/, and, with --dem-dir, two 10 m USGS 3DEP DEM clips
(UTM 13N) around the two ZOBs for the landscape check (derived data, kept out of raw/).

Usage: uv run python -I utils/mountain_wells/jemez_nm_fetch.py <raw_root> <fetch_log_dir> [--dem-dir DIR]
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

API = "https://www.hydroshare.org/hsapi/resource/"
RESOURCES = {
    "jemez_deep_nm": ["da61f854a1754e26a7c65de608bf5890"],
    "jemez_piez_nm": [
        "8d43148dba4c45fe8f65530d29aab7ae",
        "36419d60a7484844acab038b1fb84828",
    ],
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def curl(url, out, tries=3):
    code, err = "", ""
    for _ in range(tries):
        r = subprocess.run(
            [
                "curl",
                "-sSL",
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


def list_files(rid):
    url = f"{API}{rid}/files/"
    out = []
    while url:
        r = subprocess.run(
            ["curl", "-sSL", "-m", "120", url], capture_output=True, text=True
        )
        d = json.loads(r.stdout)
        out.extend(d["results"])
        url = d.get("next")
    return out


HESS = "https://hess.copernicus.org/articles/23/4661/2019/"
LIT = ["hess-23-4661-2019.pdf", "hess-23-4661-2019-supplement.pdf"]
DEM_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
    "?bbox={x0},{y0},{x1},{y1}&bboxSR=26913&imageSR=26913&size={nx},{ny}&format=tiff"
    "&pixelType=F32&interpolation=RSP_BilinearInterpolation&f=image"
)
DEM_BOXES = {
    "jemez_mczob_3dep10m_utm13.tif": (359800, 3970400, 363200, 3973600),
    "jemez_burn2011zob_3dep10m_utm13.tif": (367400, 3966700, 370600, 3969900),
}


def fetch_dem(dem_dir):
    dem_dir = Path(dem_dir)
    dem_dir.mkdir(parents=True, exist_ok=True)
    for name, (x0, y0, x1, y1) in DEM_BOXES.items():
        url = DEM_URL.format(
            x0=x0, y0=y0, x1=x1, y1=y1, nx=(x1 - x0) // 10, ny=(y1 - y0) // 10
        )
        ok, code, err = curl(url, dem_dir / name)
        print(name, "ok" if ok else f"FAILED {code} {err}")


def main(raw_root, log_dir):
    raw_root, log_dir = Path(raw_root), Path(log_dir)
    for sid, rids in RESOURCES.items():
        log = {"source_id": sid, "files": [], "failures": []}
        for rid in rids:
            rdir = raw_root / sid / rid
            rdir.mkdir(parents=True, exist_ok=True)
            meta = [
                ("sysmeta.json", f"{API}{rid}/sysmeta/"),
                ("scimeta.xml", f"{API}{rid}/scimeta/"),
                ("files.json", f"{API}{rid}/files/"),
                ("landing.html", f"https://www.hydroshare.org/resource/{rid}/"),
            ]
            listing = list_files(rid)
            for f in listing:
                rel = f["url"].split("/data/contents/", 1)[1]
                meta.append((rel, f"{API}{rid}/files/{rel}/"))
            checks = {
                f["url"].split("/data/contents/", 1)[1]: f.get("checksum")
                for f in listing
            }
            for name, url in meta:
                out = rdir / name
                out.parent.mkdir(parents=True, exist_ok=True)
                ok, code, err = curl(url, out)
                if not ok:
                    log["failures"].append(
                        {"name": name, "url": url, "http": code, "err": err}
                    )
                    if out.exists():
                        out.unlink()
                    continue
                rec = {
                    "path": f"raw/{sid}/{rid}/{name}",
                    "url": url,
                    "sha256": sha256(out),
                    "bytes": out.stat().st_size,
                }
                if name in checks:
                    rec["hydroshare_md5"] = checks[name]
                    rec["md5_match"] = md5(out) == checks[name]
                log["files"].append(rec)
        if sid == "jemez_deep_nm":
            ldir = raw_root / sid / "white_etal_2019_hess"
            ldir.mkdir(parents=True, exist_ok=True)
            for name in LIT:
                out = ldir / name
                ok, code, err = curl(HESS + name, out)
                if not ok:
                    log["failures"].append(
                        {"name": name, "url": HESS + name, "http": code, "err": err}
                    )
                    continue
                log["files"].append(
                    {
                        "path": f"raw/{sid}/white_etal_2019_hess/{name}",
                        "url": HESS + name,
                        "sha256": sha256(out),
                        "bytes": out.stat().st_size,
                    }
                )
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{sid}_fetch_log.json").write_text(json.dumps(log, indent=2))
        print(sid, len(log["files"]), "files,", len(log["failures"]), "failures")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    if "--dem-dir" in sys.argv:
        fetch_dem(sys.argv[sys.argv.index("--dem-dir") + 1])
