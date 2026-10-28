"""Parse NMBGMR OFR-543 (southern Sacramento Mountains) wells, water levels and logger data.

Reads only the unchanged downloads in <raw_dir> (PDF appendices via `pdftotext -layout`, the
Appendix 4 .xls via xlrd) and writes the shared mountain-well schema to <out_dir>.

Usage:
  uv run python -I utils/mountain_wells/sacramento_mtns_nm_parse.py <raw_dir> <out_dir> <fetch_log_json>
"""

import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
import xlrd
from rasterio.windows import Window

SOURCE_ID = "sacramento_mtns_nm"
FT = 0.3048
APP1 = "OFR-543_Appendix-01_Site_inventory.pdf"
APP3 = "OFR-543_Appendix-03_Water_level_measurements.pdf"
APP4 = "OFR-543_Appendix-04_Data_logger_data.xls"
APP6 = "OFR-543_Appendix-06_Water_table_data.pdf"
HUC8_ROOT = Path("/data/ssd2/handily/huc8")
MAYHILL_E = 455262.0  # UTM13 NAD83 easting of Mayhill, NM (32.8862 N, 105.4783 W); report: Pecos Slope lies east of Mayhill
COV = Path("/nas/handily/covariates/terrain")
GEOMORPHON = {
    1: "flat",
    2: "peak",
    3: "ridge",
    4: "shoulder",
    5: "spur",
    6: "slope",
    7: "hollow",
    8: "footslope",
    9: "valley",
    10: "pit",
}

FORMATIONS = {
    "110AVMB": "Quaternary fill (valley alluvium)",
    "260SNAN": "Permian San Andres Formation",
    "260SNAN_lower": "Lower San Andres Formation",
    "261SNGL": "San Andres - Glorieta Sandstone (Rio Bonito Member)",
    "300YESO": "Permian Yeso Formation",
    "300YESO_lower": "Lower Yeso Formation",
    "300YESO_upper": "Upper Yeso Formation",
    "310ABO": "Permian Abo Formation",
    "315LABR": "Permian Laborcita (Bursum)",
    "320HLDR": "Pennsylvanian Holder Formation",
    "322BEMN": "Pennsylvanian Beeman Formation",
    # codes used on the USGS-sourced rows but absent from the App. 3 legend (USGS NWIS aquifer codes)
    "313ARTS": "Artesia Group (USGS aquifer code; not in App. 3 legend)",
    "310YESO": "Yeso Formation (USGS aquifer code; not in App. 3 legend)",
    "313SADR": "San Andres Limestone (USGS aquifer code; not in App. 3 legend)",
}

STATUS_FLAGS = {
    "A": "other:atmospheric_pressure_affected",
    "D": "dry_censored",
    "O": "suspect|other:obstruction",
    "P": "pumping",
    "R": "recovering",
    "S": "other:nearby_well_pumping",
    "T": "other:nearby_well_pumped_recently",
    "V": "other:foreign_substance_on_water",
    "X": "other:affected_by_surface_water_stage",
    "Z": "suspect",
}
METHOD = {
    "A": "airline",
    "B": "analog_recorder",
    "C": "calibrated_airline",
    "E": "estimated",
    "F": "transducer",
    "G": "pressure_gage",
    "H": "calibrated_pressure_gage",
    "L": "geophysical_log",
    "O": "sonic_meter",
    "R": "reported_method_unknown",
    "S": "steel_tape",
    "T": "electric_tape",
    "V": "calibrated_electric_tape",
    "Z": "other",
}
SOURCE = {
    "A": "other_agency",
    "D": "driller_log",
    "G": "private_geologist_consultant_university",
    "L": "geophysical_logs",
    "M": "memory_owner_operator_driller",
    "O": "owner",
    "R": "other_person",
    "S": "NMBGMR_measured",
    "Z": "other",
}

NOTE_RULES = [
    (
        re.compile(
            r"approx|spotty|bad number|not valid|indeterminate|not repeatable|highly variable|"
            r"fluctuating|wrong side|will not repeat|not perfect|silty|difficult|gurgling|cascading|"
            r"no repeat|not repeated|only one reading",
            re.I,
        ),
        "suspect",
    ),
    (
        re.compile(
            r"pumping during|pump running|pump was running|was pumping|pump started|well pumping",
            re.I,
        ),
        "pumping",
    ),
    (
        re.compile(
            r"shut off on arrival|shut off just before|recovering|pump off for 15|pumped a few hours|"
            r"pumped 30 min|pumped to fill|just after sampling|pump turned on just after",
            re.I,
        ),
        "recovering",
    ),
    (re.compile(r"date not certain", re.I), "other:date_uncertain"),
    (
        re.compile(r"obstruction|tape caught|tape trouble|hung up|snags", re.I),
        "other:obstruction",
    ),
    (re.compile(r"flowing", re.I), "other:possibly_flowing_note"),
    (re.compile(r"\+-\s*\d", re.I), "other:stated_uncertainty"),
    (
        re.compile(r"Correction done for new MP", re.I),
        "other:mp_correction_applied_by_source",
    ),
    (re.compile(r"New wellhead and MP", re.I), "instrument_change"),
    (
        re.compile(r"nearby field was heavily irrigated", re.I),
        "other:nearby_irrigation",
    ),
    (re.compile(r"\d+\s*psi", re.I), "other:airline_psi_reading"),
]


def pdf_text(pdf, tmp):
    out = Path(tmp) / (Path(pdf).stem + ".txt")
    subprocess.run(["pdftotext", "-layout", str(pdf), str(out)], check=True)
    return out.read_text(errors="replace").splitlines()


def mdy(s):
    m, d, y = (int(x) for x in s.split("/"))
    if y < 100:
        y = 2000 + y if y <= 30 else 1900 + y
    return date(y, m, d)


# ----------------------------------------------------------------------------------------------
# Appendix 1: site inventory
# ----------------------------------------------------------------------------------------------
def parse_app1(lines):
    rows = {}
    for i, line in enumerate(lines):
        m = re.match(r"^\s*(SM-\d{4})\s+([A-Z]{1,2})\b(.*)$", line)
        if not m:
            continue
        sid, stype, rest = m.group(1), m.group(2), m.group(3).split()
        rel = rest.pop() if rest and rest[-1] in ("C", "L", "M", "U") else ""
        dinv = rest.pop() if rest and re.match(r"^\d+/\d+/\d+$", rest[-1]) else ""
        north = float(rest.pop()) if rest and re.match(r"^\d{7}$", rest[-1]) else np.nan
        east = float(rest.pop()) if rest and re.match(r"^\d{6}$", rest[-1]) else np.nan
        alt = float(rest.pop()) if rest and re.match(r"^\d{4,5}$", rest[-1]) else np.nan
        dtype = rest.pop() if rest and re.match(r"^[\d,]+$", rest[-1]) else ""
        if rest:
            raise ValueError(f"App1 unparsed tokens {rest} in line {i}: {line!r}")
        if sid in rows:
            raise ValueError(f"duplicate App1 id {sid}")
        rows[sid] = dict(
            site_type=stype,
            data_type=dtype,
            alt_ft=alt,
            easting=east,
            northing=north,
            date_inventoried=dinv,
            reliability=rel,
            app1_line=i + 1,
        )
    return rows


# ----------------------------------------------------------------------------------------------
# Appendix 3A (well table) and 3B (measurements)
# ----------------------------------------------------------------------------------------------
def split_app3(lines):
    i3b = next(i for i, ln in enumerate(lines) if "Appendix 3B" in ln)
    return lines[:i3b], lines[i3b:], i3b


def parse_app3a(lines):
    wells = {}
    pending = []
    last = None
    started = False
    for i, line in enumerate(lines):
        if "Appendix 3A" in line:
            started = True
            continue
        if not started:
            continue
        m = re.match(r"^\s*(SM-\d{4})\s+(\S+)\s+(\d{4})\s+(\d{6})\s+(\d{7})(.*)$", line)
        if m:
            toks = m.group(6).split()
            depth = mp = np.nan
            form = []
            for t in toks:
                if re.match(r"^-?\d+\.\d+$", t):
                    mp = float(t)
                elif re.match(r"^\d+$", t):
                    depth = float(t)
                else:
                    form.append(t)
            rec = dict(
                ose_id=m.group(2),
                alt_ft=float(m.group(3)),
                easting=float(m.group(4)),
                northing=float(m.group(5)),
                depth_ft=depth,
                mp_ft=mp,
                formation_parts=pending + form,
                app3_line=i + 1,
                wrapped=bool(pending),
            )
            wells[m.group(1)] = rec
            last = m.group(1)
            pending = []
            continue
        s = line.strip()
        if re.match(r"^(\d{3}[A-Z]+\S*|ower|pper|\S*/)$", s):
            if (
                last is not None
                and wells[last]["wrapped"]
                and not wells[last].get("tail")
            ):
                wells[last]["formation_parts"].append(s)
                wells[last]["tail"] = True
            else:
                pending.append(s)
    for w in wells.values():
        f = "".join(w.pop("formation_parts"))
        f = f.replace("_l" + "ower", "_lower").replace("_u" + "pper", "_upper")
        w["formation"] = f
        w.pop("wrapped", None)
        w.pop("tail", None)
    return wells


def parse_app3b(lines, offset):
    recs = []
    pending = []
    last = None
    row_re = re.compile(
        r"^\s*(SM-\d{4})\s+(\d{1,2}/\d{1,2}/\d{2})\s+(-?\d+\.?\d*)\s*(.*)$"
    )
    for j, line in enumerate(lines):
        lineno = offset + j + 1
        if (
            "Appendix 3B" in line
            or re.match(r"^\s*(Date\s+Level|Point ID\s+Measured)", line)
            or not line.strip()
        ):
            continue
        m = row_re.match(line)
        if m:
            toks = m.group(4).split()
            codes = []
            while toks and len(codes) < 3 and re.match(r"^[A-Z]$", toks[0]):
                codes.append(toks.pop(0))
            if len(codes) == 3:
                status, method, src = codes
            elif len(codes) == 2:
                status, (method, src) = "", codes
            else:
                raise ValueError(f"App3B codes not parsed at line {lineno}: {line!r}")
            rec = dict(
                site=m.group(1),
                date=mdy(m.group(2)),
                value=float(m.group(3)),
                status=status,
                method=method,
                source=src,
                notes=" ".join(pending + toks).strip(),
                lineno=lineno,
                had_prefix=bool(pending),
                had_suffix=False,
            )
            recs.append(rec)
            last = rec
            pending = []
            continue
        s = line.strip()
        if last is not None and last["had_prefix"] and not last["had_suffix"]:
            last["notes"] = (last["notes"] + " " + s).strip()
            last["had_suffix"] = True
        else:
            pending.append(s)
    if pending:
        raise ValueError(f"orphan App3B note text at end: {pending}")
    return recs


# ----------------------------------------------------------------------------------------------
# Appendix 6: water-level-elevation control points
# ----------------------------------------------------------------------------------------------
def parse_app6(lines):
    rows = []
    pat = re.compile(
        r"^\s*((?:SM-\d{4})|(?:[A-Z]{1,2} \d{5}))\s+([\d,]+)\s+([\d,]+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s*"
        r"(\d{1,2}/\d{1,2}/\d{4})?\s+([ws])\s*$"
    )
    for i, line in enumerate(lines):
        m = pat.match(line)
        if not m:
            if re.match(r"^\s*(SM-\d{4}|[A-Z]{1,2} \d{5})\s", line):
                raise ValueError(f"App6 row not parsed line {i + 1}: {line!r}")
            continue
        rows.append(
            dict(
                pid=m.group(1),
                easting=float(m.group(2).replace(",", "")),
                northing=float(m.group(3).replace(",", "")),
                surf_ft=float(m.group(4)),
                dtw_ft=float(m.group(5)),
                wse_ft=float(m.group(6)),
                date=datetime.strptime(m.group(7), "%m/%d/%Y").date()
                if m.group(7)
                else None,
                site_type=m.group(8),
                lineno=i + 1,
            )
        )
    return rows


# ----------------------------------------------------------------------------------------------
# Appendix 4: logger workbook
# ----------------------------------------------------------------------------------------------
def parse_app4(xls):
    book = xlrd.open_workbook(xls)
    sh = book.sheet_by_index(0)
    series = []
    for c in range(0, sh.ncols, 2):
        head = sh.cell_value(2, c)
        site = head.split("_")[0]
        part = str(sh.cell_value(0, c)).strip()
        wh = sh.cell_value(1, c + 1)
        rows = []
        for r in range(3, sh.nrows):
            t, v = sh.cell(r, c), sh.cell(r, c + 1)
            if t.ctype == xlrd.XL_CELL_EMPTY or (
                t.ctype == xlrd.XL_CELL_TEXT and not t.value.strip()
            ):
                if v.ctype not in (xlrd.XL_CELL_EMPTY,):
                    raise ValueError(f"value without time {site} row {r}")
                continue
            if t.ctype == xlrd.XL_CELL_DATE:
                ts = xlrd.xldate_as_datetime(t.value, book.datemode)
                ts = ts.replace(microsecond=0)
                if ts.second >= 30:
                    ts = ts + timedelta(seconds=60 - ts.second)
                else:
                    ts = ts - timedelta(seconds=ts.second)
            else:
                x = t.value.strip()
                ts = None
                for f in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
                    try:
                        ts = datetime.strptime(x, f)
                        break
                    except ValueError:
                        pass
                if ts is None:
                    raise ValueError(f"bad time {x!r}")
            val = v.value if v.ctype == xlrd.XL_CELL_NUMBER else np.nan
            rows.append((ts, val, r + 1))
        series.append(dict(site=site, col=c, part=part, wellhead_ft=wh, rows=rows))
    return series


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def flag_join(*parts):
    out = []
    for p in parts:
        for f in str(p).split("|"):
            if f and f not in out:
                out.append(f)
    return "|".join(out)


def note_flags(notes):
    return flag_join(*[f for rx, f in NOTE_RULES if rx.search(notes or "")])


def aquifer_setting(formation):
    if not formation:
        return "unknown"
    if "110AVMB" in formation:
        return "alluvium"
    return "bedrock"


def geology_text(formation):
    if not formation:
        return ""
    parts = formation.split("/")
    return "; ".join(f"{p} = {FORMATIONS.get(p, 'code not in legend')}" for p in parts)


# ----------------------------------------------------------------------------------------------
# terrain context (DEM) for landscape class
# ----------------------------------------------------------------------------------------------
def dem_tiles():
    tiles = []
    for d in sorted(HUC8_ROOT.glob("130[56]*")):
        f = d / "dem_10m.tif"
        if f.exists():
            tiles.append(f)
    return tiles


def sample_cov(xs, ys):
    out = {}
    for name in ("elev48i0100a", "tpi_500m", "tpi_2km", "geomorphons"):
        with rasterio.open(COV / f"{name}.tif") as r:
            vals = np.array(
                [v[0] for v in r.sample(list(zip(xs, ys)))], dtype="float64"
            )
            if r.nodata is not None:
                vals[vals == r.nodata] = np.nan
        out[name] = vals
    return out


def dem10_metrics(xs, ys):
    tiles = dem_tiles()
    srcs = [rasterio.open(t) for t in tiles]
    res = []
    rad = 1000.0
    for x, y in zip(xs, ys):
        best = None
        for t, r in zip(tiles, srcs):
            b = r.bounds
            if not (b.left < x < b.right and b.bottom < y < b.top):
                continue
            row, col = r.index(x, y)
            n = int(rad / r.res[0])
            win = Window(col - n, row - n, 2 * n + 1, 2 * n + 1)
            a = r.read(1, window=win, boundless=True, fill_value=r.nodata).astype(
                "float64"
            )
            a[a == r.nodata] = np.nan
            if np.isnan(a[n, n]):
                continue
            frac = np.isfinite(a).mean()
            if best is None or frac > best[0]:
                best = (frac, t.parent.name, a, n, r.res[0])
        if best is None:
            res.append({})
            continue
        frac, huc, a, n, px = best
        yy, xx = np.mgrid[-n : n + 1, -n : n + 1] * px
        dist = np.hypot(xx, yy)
        z0 = a[n, n]
        m = {
            "dem10_huc8": huc,
            "dem10_valid_frac_1km": round(float(frac), 3),
            "dem10_z_m": float(z0),
        }
        for rr in (250, 1000):
            w = a[dist <= rr]
            w = w[np.isfinite(w)]
            m[f"tpi{rr}_m"] = float(z0 - w.mean())
            m[f"relief{rr}_m"] = float(w.max() - w.min())
            m[f"relpos{rr}"] = (
                float((z0 - w.min()) / (w.max() - w.min()))
                if w.max() > w.min()
                else np.nan
            )
        c = a[n - 3 : n + 4, n - 3 : n + 4]
        gy, gx = np.gradient(c, px)
        m["slope30_deg"] = float(np.degrees(np.arctan(np.nanmean(np.hypot(gx, gy)))))
        res.append(m)
    for r in srcs:
        r.close()
    return res


def dem_class(m, geo, pecos_slope=False):
    """Strict DEM-only class. Returns (class, reason)."""
    if not m:
        return "uncertain", "no 10 m DEM coverage"
    rp, tpi250, tpi1k, rel1k = (
        m["relpos1000"],
        m["tpi250_m"],
        m["tpi1000_m"],
        m["relief1000_m"],
    )
    gtxt = GEOMORPHON.get(int(geo), "nodata") if np.isfinite(geo) else "nodata"
    desc = (
        f"10 m DEM ({m['dem10_huc8']}): relpos1km={rp:.2f}, TPI250={tpi250:+.0f} m, TPI1km={tpi1k:+.0f} m, "
        f"relief1km={rel1k:.0f} m, slope={m['slope30_deg']:.0f} deg; 100 m geomorphon={gtxt}"
    )
    if rp <= 0.25 and tpi250 < -3:
        return "mountain_valley_bottom", desc
    if m["relpos250"] <= 0.15 and tpi250 < -8:
        return (
            "mountain_valley_bottom",
            desc + " (floor of a local hollow/headwater basin at 250 m scale)",
        )
    if rp <= 0.35 and tpi1k < -20 and tpi250 < 0:
        return "mountain_valley_bottom", desc
    if rp >= 0.80 and tpi250 > 3 and tpi1k > 15:
        if pecos_slope:
            return (
                "mesa_plateau_top",
                desc + " (interfluve on the low-relief Pecos Slope, east of Mayhill)",
            )
        if rel1k >= 60:
            return "ridgetop_divide", desc
        return "mesa_plateau_top", desc
    if rel1k < 40:
        if rp >= 0.6 and tpi1k > 0:
            return "mesa_plateau_top", desc
        return "uncertain", desc
    return "mountain_slope", desc


# report statements tied to individual sites (quoted from OFR-543 text)
SITE_EVIDENCE = {
    "SM-0023": (
        "crest_regional",
        "report text: 'SM-0023 is located at the crest of the mountain range where we would expect young "
        "groundwater; however, this well is along a fault'",
    ),
    "SM-0049": (
        "crest_basin",
        "report text: 'Well SM-0049, located near the crest of the southern Sacramento Mountains'; App. 9: "
        "'The surface water basin surrounding the well is likely the only source of groundwater feeding the "
        "well' (basin area 12,220,084 ft2)",
    ),
    "SM-0026": (
        "mountain_valley_bottom",
        "report text: 'Sample SM-0026 was collected from a well adjacent to a perennial flowing stream that "
        "is spring-fed'; App. 3A formation 110AVMB/300YESO (valley alluvium)",
    ),
    "SM-1079": (
        "mountain_valley_bottom",
        "report text: 'SM-1079 is located in the valley bottom'",
    ),
    "SM-0076": (
        "mountain_valley_bottom",
        "report text: 'Approximately 3 miles upstream of well SM-0076, spring SM-1079 discharges' "
        "(Rio Penasco valley)",
    ),
}
STR_WELLS = {
    "SM-0026",
    "SM-0049",
    "SM-0022",
    "SM-0043",
    "SM-0044",
    "SM-0064",
    "SM-0066",
}
LTR_WELLS = {
    "SM-0015",
    "SM-0055",
    "SM-0016",
    "SM-0068",
    "SM-0008",
    "SM-0012",
    "SM-0048",
    "SM-0069",
}


def landscape(site, formation, m, geo, is_spring, pecos_slope):
    dcls, desc = dem_class(m, geo, pecos_slope)
    if site in SITE_EVIDENCE:
        scls, quote = SITE_EVIDENCE[site]
        if scls == "crest_basin":
            # near the crest, but inside a small surface-water basin: the local DEM decides ridge vs hollow
            conf = "medium" if dcls == "mountain_valley_bottom" else "low"
            return dcls, f"{quote}. {desc}", conf
        if scls == "crest_regional":
            # 'crest of the mountain range' is a regional statement; local position comes from the DEM
            return (
                dcls,
                f"{quote}. Regional crest zone per report, but local 10 m DEM position is NOT a ridgetop: {desc}",
                "medium",
            )
        conf = "high" if dcls == scls else "medium"
        return scls, f"{quote}. {desc}", conf
    if formation and "110AVMB" in formation:
        # report text: 'The remaining wells are completed in shallow valley-bottom alluvium or spring deposits'
        conf = "high" if dcls == "mountain_valley_bottom" else "medium"
        return (
            "mountain_valley_bottom",
            "completed in Quaternary valley fill (App. 3A 110AVMB); report text: 'The remaining wells are "
            f"completed in shallow valley-bottom alluvium or spring deposits'. {desc}",
            conf,
        )
    if is_spring and dcls == "ridgetop_divide":
        return (
            "mountain_slope",
            f"spring discharge point; DEM ridge signal overridden (springs discharge on slopes/valleys). {desc}",
            "low",
        )
    return dcls, f"DEM only. {desc}", "low"


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main(raw_dir, out_dir, fetch_log):
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rel = lambda f: f"raw/{SOURCE_ID}/{f}"  # noqa: E731

    with tempfile.TemporaryDirectory() as tmp:
        l1 = pdf_text(raw / APP1, tmp)
        l3 = pdf_text(raw / APP3, tmp)
        l6 = pdf_text(raw / APP6, tmp)
    app1 = parse_app1(l1)
    l3a, l3b, off3b = split_app3(l3)
    app3a = parse_app3a(l3a)
    app3b = parse_app3b(l3b, off3b)
    app6 = parse_app6(l6)
    app4 = parse_app4(raw / APP4)
    diag = {}

    # ---------------- inventory skeleton ----------------
    sites = {}
    for sid, r in app1.items():
        if r["site_type"] not in ("GW", "SP"):
            continue
        sites[sid] = dict(
            pid=sid,
            kind="well" if r["site_type"] == "GW" else "spring",
            easting=r["easting"],
            northing=r["northing"],
            alt_ft=r["alt_ft"],
            elev_src="App. 1 altitude (ft ASL; report: derived in ArcGIS from a 10-m DEM)",
            coord_src="table (App. 1, UTM NAD83 zone 13; hand-held Garmin 76/76S GPS)",
            reliability=r["reliability"],
            notes=[
                f"App1 site type {r['site_type']}, data type {r['data_type'] or 'blank'}, "
                f"inventoried {r['date_inventoried'] or 'n/a'}, reliability {r['reliability'] or 'blank'}"
            ],
            source_file=rel(APP1),
        )
    for sid, w in app3a.items():
        if sid not in sites:
            sites[sid] = dict(
                pid=sid,
                kind="well",
                easting=w["easting"],
                northing=w["northing"],
                alt_ft=w["alt_ft"],
                elev_src="App. 3A altitude (ft ASL; 10-m DEM per report)",
                coord_src="table (App. 3A, UTM NAD83 zone 13)",
                reliability="",
                notes=["not listed in App. 1; location from App. 3A"],
                source_file=rel(APP3),
            )
        else:
            s = sites[sid]
            if (
                abs(s["easting"] - w["easting"]) > 1
                or abs(s["northing"] - w["northing"]) > 1
            ):
                s["notes"].append(
                    f"App3A UTM {w['easting']:.0f},{w['northing']:.0f} differs from App1"
                )
            if np.isfinite(s["alt_ft"]) and abs(s["alt_ft"] - w["alt_ft"]) > 0.5:
                s["notes"].append(
                    f"App3A altitude {w['alt_ft']:.0f} ft differs from App1 {s['alt_ft']:.0f} ft"
                )
    app6_by = {}
    for r in app6:
        pid = r["pid"]
        key = pid if pid.startswith("SM-") else pid.replace(" ", "-")
        app6_by[key] = r
        if key not in sites:
            ose = not pid.startswith("SM-")
            sites[key] = dict(
                pid=key,
                kind="well" if r["site_type"] == "w" else "spring",
                easting=r["easting"],
                northing=r["northing"],
                alt_ft=r["surf_ft"],
                elev_src="App. 6 surface elevation (ft)",
                coord_src="table (App. 6, UTM NAD83 zone 13)",
                reliability="",
                notes=[
                    (
                        "NM OSE well record not part of the study network (App. 6: 'wells not part of this study'); "
                        f"OSE file '{pid}'; static level from OSE well record (Fig. 19: year 1995 or newer)"
                    )
                    if ose
                    else "listed only in App. 6 (and/or App. 4); not in App. 1 or App. 3"
                ],
                source_file=rel(APP6),
                ose=ose,
            )
        else:
            s = sites[key]
            s.setdefault("notes", []).append(
                f"App6 surface elevation {r['surf_ft']:.0f} ft"
            )
    for ser in app4:
        if ser["site"] not in sites:
            raise ValueError(f"logger site {ser['site']} has no location")

    # ---------------- observations: App. 3B manual ----------------
    obs = []
    for r in app3b:
        sid = r["site"]
        w = app3a.get(sid)
        if w is None:
            raise ValueError(f"App3B site {sid} not in App3A")
        mp = w["mp_ft"]
        fl = flag_join(STATUS_FLAGS.get(r["status"], ""), note_flags(r["notes"]))
        if r["method"] == "E":
            fl = flag_join(fl, "estimated")
        if r["method"] == "O":
            fl = flag_join(fl, "other:sonic_meter_approx")
        if r["method"] in ("A", "C"):
            fl = flag_join(fl, "other:airline")
        if r["method"] == "R":
            fl = flag_join(fl, "other:reported_method_unknown")
        if r["source"] != "S":
            fl = flag_join(
                fl, f"other:data_source_{SOURCE.get(r['source'], r['source'])}"
            )
        censored = "gt" if r["status"] == "D" else ""
        if np.isfinite(mp):
            dtw = (r["value"] - mp) * FT
            conv = f"(orig_ft - mp_height_ft {mp:.2f}) * 0.3048; MP height from App. 3A"
            mpo = mp * FT
        else:
            dtw = r["value"] * FT
            conv = "orig_ft * 0.3048; MP height blank in App. 3A, assumed 0 (App. 3A network median MP 1.0 ft, range -6.35 to 7.00 ft)"
            mpo = np.nan
            fl = flag_join(fl, "estimated|other:mp_height_unknown_assumed_0")
        if censored:
            dtw = np.nan
        obs.append(
            dict(
                source_id=SOURCE_ID,
                pid=sid,
                timestamp=r["date"].isoformat(),
                orig_value=r["value"],
                orig_units="ft",
                orig_reference="below_mp",
                mp_offset_m=mpo,
                dtw_bgs_m=dtw,
                conversion=conv,
                flags=fl,
                censored=censored,
                source_file=rel(APP3),
                source_row=f"line {r['lineno']}",
                obs_kind="manual",
                status_code=r["status"],
                method_code=r["method"],
                data_source_code=r["source"],
                obs_notes=r["notes"],
            )
        )

    # ---------------- App. 6: duplicate check + extra wells ----------------
    man_idx = defaultdict(dict)
    for r in app3b:
        man_idx[r["site"]][r["date"]] = r["value"]
    dup_exact = dup_date_only = app6_wse_ok = app6_wse_n = 0
    app6_added = []
    for key, r in app6_by.items():
        if r["site_type"] != "w":
            continue
        if key in app3a:
            # WSE consistency check vs 'Elevation + MP height - DTW'
            mp = app3a[key]["mp_ft"]
            if np.isfinite(mp):
                app6_wse_n += 1
                if abs(r["surf_ft"] + mp - r["dtw_ft"] - r["wse_ft"]) <= 1.0:
                    app6_wse_ok += 1
            if r["date"] in man_idx.get(key, {}):
                if abs(man_idx[key][r["date"]] - r["dtw_ft"]) < 0.006:
                    dup_exact += 1
                else:
                    dup_date_only += 1
                continue
            if r["date"] is None:
                continue
            # an App. 6 value for a 3A well on a date App. 3B does not carry
            diag.setdefault("app6_not_in_3b", []).append(
                f"{key} {r['date']} {r['dtw_ft']}"
            )
            continue
        ose = sites[key].get("ose", False)
        fl = "estimated"
        if ose:
            ref = "bgs"
            conv = "orig_ft * 0.3048; OSE well-record static water level, assumed below land surface"
            fl = flag_join(fl, "other:ose_well_record_static_level|other:date_unknown")
        else:
            ref = "below_mp"
            conv = (
                "orig_ft * 0.3048; App. 6 'Depth to Water' reproduces App. 3B below-MP values for study wells; "
                "MP height unknown for this well, assumed 0"
            )
            fl = flag_join(
                fl, "other:mp_height_unknown_assumed_0|other:app6_snapshot_value"
            )
        if r["date"] is None and not ose:
            fl = flag_join(fl, "other:date_unknown")
        obs.append(
            dict(
                source_id=SOURCE_ID,
                pid=key,
                timestamp=r["date"].isoformat() if r["date"] else "",
                orig_value=r["dtw_ft"],
                orig_units="ft",
                orig_reference=ref,
                mp_offset_m=np.nan,
                dtw_bgs_m=r["dtw_ft"] * FT,
                conversion=conv,
                flags=fl,
                censored="",
                source_file=rel(APP6),
                source_row=f"line {r['lineno']}",
                obs_kind="app6_control_point",
                status_code="",
                method_code="",
                data_source_code="",
                obs_notes=f"App. 6 WSE {r['wse_ft']:.0f} ft, surface {r['surf_ft']:.0f} ft",
            )
        )
        app6_added.append(key)
    diag["app6_dup_exact_with_3b"] = dup_exact
    diag["app6_same_date_value_differs"] = dup_date_only
    diag["app6_wse_check_ok_of_n"] = [app6_wse_ok, app6_wse_n]
    diag["app6_wells_added"] = app6_added

    # ---------------- App. 4 logger ----------------
    logger_summary = []
    for ser in app4:
        sid = ser["site"]
        ts = np.array([t for t, v, rr in ser["rows"]], dtype="datetime64[s]")
        vals = np.array([v for t, v, rr in ser["rows"]], dtype="float64")
        rows_xls = [rr for t, v, rr in ser["rows"]]
        days = ts.astype("datetime64[D]")
        start_d, end_d = days.min(), days.max()
        mp = app3a[sid]["mp_ft"] if sid in app3a else np.nan
        zero = vals == 0.0
        daymed = {}
        for d in np.unique(days):
            v = vals[(days == d) & ~zero & np.isfinite(vals)]
            if len(v):
                daymed[d] = float(np.median(v))
        # manual ties
        if sid in man_idx:
            visits = sorted(
                (np.datetime64(d, "D"), v)
                for d, v in man_idx[sid].items()
                if start_d <= np.datetime64(d, "D") <= end_d
            )
            tie_src = "App. 3B manual (below MP)"
        elif sid in app6_by and app6_by[sid]["date"] is not None:
            r6 = app6_by[sid]
            visits = [(np.datetime64(r6["date"], "D"), r6["dtw_ft"])]
            tie_src = "App. 6 single value (below MP assumed)"
        else:
            visits = []
            tie_src = "none"
        vdays = [d for d, v in visits]
        bounds = []  # (visit_day, off_before, off_after)
        for d, v in visits:
            ob = daymed.get(d - np.timedelta64(1, "D"))
            oa = daymed.get(d + np.timedelta64(1, "D"))
            bounds.append(
                (d, None if ob is None else ob - v, None if oa is None else oa - v)
            )
        seg_info = []
        offs = np.full(len(vals), np.nan)
        segflag = np.array([""] * len(vals), dtype=object)
        # segments: [start, v1), (v1, v2), ... (vn, end]
        edges = [None] + bounds + [None]
        for k in range(len(edges) - 1):
            a, b = edges[k], edges[k + 1]
            lo = a[0] if a else start_d - np.timedelta64(1, "D")
            hi = b[0] if b else end_d + np.timedelta64(1, "D")
            sel = (days > lo) & (days < hi)
            if not sel.any():
                continue
            oa = a[2] if a else None
            ob = b[1] if b else None
            if oa is None and ob is None:
                seg_info.append(
                    dict(
                        start=str(lo),
                        end=str(hi),
                        off_start=None,
                        off_end=None,
                        n=int(sel.sum()),
                    )
                )
                continue
            if oa is not None and ob is not None:
                t0 = (
                    (lo + np.timedelta64(1, "D"))
                    .astype("datetime64[s]")
                    .astype("int64")
                )
                t1 = (
                    (hi - np.timedelta64(1, "D"))
                    .astype("datetime64[s]")
                    .astype("int64")
                )
                tt = ts[sel].astype("int64")
                frac = np.clip((tt - t0) / max(t1 - t0, 1), 0, 1)
                offs[sel] = oa + (ob - oa) * frac
                if abs(ob - oa) > 2.0:
                    segflag[sel] = "not_defensible"
                elif abs(ob - oa) > 1.0:
                    segflag[sel] = "suspect|other:logger_offset_drift_gt_1ft"
            else:
                offs[sel] = oa if oa is not None else ob
                segflag[sel] = "other:logger_single_tie"
            seg_info.append(
                dict(
                    start=str(lo),
                    end=str(hi),
                    off_start=oa,
                    off_end=ob,
                    n=int(sel.sum()),
                )
            )
        visit_mask = (
            np.isin(days, np.array(vdays, dtype="datetime64[D]"))
            if vdays
            else np.zeros(len(vals), bool)
        )
        n_conv = 0
        for i in range(len(vals)):
            v = vals[i]
            fl = "other:logger_hourly_or_15min"
            cens = ""
            if not np.isfinite(v):
                obs.append(None)
                continue
            if sid == "SM-0035":
                dtw = np.nan
                conv = (
                    "not_defensible: no manual tie in App. 3, MP unknown; App. 6 value (49.44 ft, 2007-05-14) differs "
                    "~7 ft from logger on that day; series has zero-valued records and an unexplained ~30 ft drop "
                    "after Aug 2006"
                )
                fl = flag_join(fl, "suspect")
                if zero[i]:
                    fl = flag_join(fl, "missing")
                mpo = np.nan
            elif zero[i]:
                dtw, conv, mpo = (
                    np.nan,
                    "not_defensible: zero-valued logger record",
                    np.nan,
                )
                fl = flag_join(fl, "missing")
            elif visit_mask[i]:
                dtw = np.nan
                conv = "not_defensible: logger service day (download/reinstall disturbance; offset ambiguous)"
                fl = flag_join(fl, "other:logger_service_day")
                mpo = np.nan
            elif not np.isfinite(offs[i]):
                dtw = np.nan
                conv = "not_defensible: logger segment without a manual tie"
                mpo = np.nan
            elif segflag[i] == "not_defensible":
                dtw = np.nan
                conv = "not_defensible: manual ties at the two ends of this logger segment differ by more than 2 ft"
                fl = flag_join(fl, "suspect")
                mpo = np.nan
            else:
                mp_use = mp if np.isfinite(mp) else 0.0
                dtw = (v - offs[i] - mp_use) * FT
                conv = (
                    f"(orig_ft - tie_offset_ft {offs[i]:+.2f} - mp_height_ft {mp_use:.2f}) * 0.3048; tie = logger "
                    f"daily median (day before/after visit) minus {tie_src}, linear between bounding visits"
                )
                mpo = mp_use * FT if np.isfinite(mp) else np.nan
                fl = flag_join(fl, "other:logger_tied_to_manual", segflag[i])
                if not np.isfinite(mp):
                    fl = flag_join(fl, "estimated|other:mp_height_unknown_assumed_0")
                n_conv += 1
            obs.append(
                dict(
                    source_id=SOURCE_ID,
                    pid=sid,
                    timestamp=pd.Timestamp(ts[i]).strftime("%Y-%m-%dT%H:%M:%S"),
                    orig_value=v,
                    orig_units="ft",
                    orig_reference="unknown",
                    mp_offset_m=mpo,
                    dtw_bgs_m=dtw,
                    conversion=conv,
                    flags=fl,
                    censored=cens,
                    source_file=rel(APP4),
                    source_row=f"sheet 'data' row {rows_xls[i]} col {ser['col'] + 1}"
                    + (f" ({ser['part']})" if ser["part"] else ""),
                    obs_kind="logger",
                    status_code="",
                    method_code="F",
                    data_source_code="S",
                    obs_notes=f"App. 4 column header '{sid}_DTW'; wellhead elevation {ser['wellhead_ft']} ft",
                )
            )
        steps = np.diff(ts.astype("int64")) / 60
        logger_summary.append(
            dict(
                site=sid,
                part=ser["part"],
                wellhead_ft=ser["wellhead_ft"],
                start=str(ts.min()),
                end=str(ts.max()),
                n_values=int(np.isfinite(vals).sum()),
                n_blank=int((~np.isfinite(vals)).sum()),
                n_zero=int(zero.sum()),
                n_converted=n_conv,
                step_min_counts={
                    str(k): int(c)
                    for k, c in zip(*np.unique(steps, return_counts=True))
                    if c > 2
                },
                tie_source=tie_src,
                segments=seg_info,
                mp_ft=None if not np.isfinite(mp) else mp,
            )
        )
    obs = [o for o in obs if o is not None]
    diag["logger"] = logger_summary

    # ---------------- inventory frame ----------------
    keys = sorted(sites)
    E = np.array([sites[k]["easting"] for k in keys])
    N = np.array([sites[k]["northing"] for k in keys])
    to_wgs = pyproj.Transformer.from_crs(26913, 4326, always_xy=True)
    to_aea = pyproj.Transformer.from_crs(26913, 5070, always_xy=True)
    lon, lat = to_wgs.transform(E, N)
    xa, ya = to_aea.transform(E, N)
    cov = sample_cov(xa, ya)
    d10 = dem10_metrics(xa, ya)

    odf = pd.DataFrame(obs)
    odf["well_id"] = SOURCE_ID + ":" + odf["pid"]
    odf["interval_id"] = odf["well_id"] + "#1"
    inv = []
    terrain = []
    for i, k in enumerate(keys):
        s = sites[k]
        w = app3a.get(k, {})
        form = w.get("formation", "")
        is_spring = s["kind"] == "spring"
        lc, lev, lconf = landscape(
            k, form, d10[i], cov["geomorphons"][i], is_spring, E[i] > MAYHILL_E
        )
        o = odf[odf["pid"] == k]
        tsv = o["timestamp"][o["timestamp"] != ""]
        notes = list(s["notes"])
        if w.get("ose_id") and w["ose_id"].lower() not in ("none", "unknown"):
            site_no = w["ose_id"]
        elif s.get("ose"):
            site_no = s["pid"]
        else:
            site_no = ""
        if k in STR_WELLS:
            notes.append(
                "report Fig. 21 short-term-response (STR) well: 'likely completed in' intermediate-scale perched aquifers"
            )
        if k in LTR_WELLS:
            notes.append(
                "report Fig. 22 long-term-response (LTR) well: deeper regional aquifer"
            )
        if k == "SM-0015":
            notes.append(
                "report: barometric fluctuations indicate confined conditions; App. 9: changed from confined to unconfined as levels rose"
            )
        if k == "SM-0049":
            notes.append(
                "App. 9: 'no correlation with atmospheric pressure variations, indicating that the well is completed in an unconfined aquifer'"
            )
        if any(x["site"] == k for x in app4):
            notes.append("continuous logger record in App. 4")
        if is_spring:
            notes.append(
                "spring: no depth-to-water rows (App. 6 lists springs with DTW 0 as water-table control points; not a measurement)"
            )
        rel_code = s.get("reliability", "")
        conf = {"C": "high", "U": "medium", "M": "medium", "L": "low"}.get(
            rel_code, "medium"
        )
        if s.get("ose"):
            conf = "low"
        if np.isfinite(s["alt_ft"]):
            elev_m = s["alt_ft"] * FT
        else:
            elev_m = np.nan
        inv.append(
            dict(
                source_id=SOURCE_ID,
                well_id=f"{SOURCE_ID}:{k}",
                interval_id=f"{SOURCE_ID}:{k}#1",
                well_name=k if not s.get("ose") else s["pid"],
                site_no=site_no,
                lat=round(float(lat[i]), 6),
                lon=round(float(lon[i]), 6),
                horiz_datum="NAD83",
                coord_source=s["coord_src"],
                coord_confidence=conf,
                elev_m=round(elev_m, 2) if np.isfinite(elev_m) else np.nan,
                elev_datum="unknown",
                elev_source=s["elev_src"],
                mp_height_m=round(w["mp_ft"] * FT, 3)
                if w and np.isfinite(w["mp_ft"])
                else np.nan,
                borehole_depth_m=np.nan,
                well_depth_m=round(w["depth_ft"] * FT, 2)
                if w and np.isfinite(w["depth_ft"])
                else np.nan,
                screen_top_m=np.nan,
                screen_bot_m=np.nan,
                geology=geology_text(form)
                if not is_spring
                else "spring discharge point",
                aquifer_setting="spring" if is_spring else aquifer_setting(form),
                state="NM",
                landscape_class=lc,
                landscape_evidence=lev,
                landscape_confidence=lconf,
                record_start=tsv.min()[:10] if len(tsv) else "",
                record_end=tsv.max()[:10] if len(tsv) else "",
                n_obs=int(len(o)),
                source_file=s["source_file"],
                notes="; ".join(notes)
                + f"; lat/lon transformed from UTM13 NAD83 E{s['easting']:.0f} N{s['northing']:.0f} (EPSG:26913->4326)",
                site_kind=s["kind"],
                formation_code=form,
            )
        )
        t = dict(well_id=f"{SOURCE_ID}:{k}", x5070=xa[i], y5070=ya[i])
        t.update({f"cov_{n}": cov[n][i] for n in cov})
        t.update(d10[i])
        terrain.append(t)
    idf = pd.DataFrame(inv)
    idf.to_csv(out / "inventory.csv", index=False)
    pd.DataFrame(terrain).to_csv(out / "terrain_context.csv", index=False)

    cols = [
        "source_id",
        "well_id",
        "interval_id",
        "timestamp",
        "orig_value",
        "orig_units",
        "orig_reference",
        "mp_offset_m",
        "dtw_bgs_m",
        "conversion",
        "flags",
        "censored",
        "source_file",
        "source_row",
        "obs_kind",
        "status_code",
        "method_code",
        "data_source_code",
        "obs_notes",
    ]
    odf = odf[cols]
    odf["source_row"] = odf["source_row"].astype(str)
    odf.to_parquet(out / "observations.parquet", index=False)

    # ---------------- daily medians ----------------
    d = odf[odf["timestamp"] != ""].copy()
    d["date"] = d["timestamp"].str[:10]
    excl = d["flags"].str.contains("pumping|missing|dry_censored", regex=True) | (
        d["censored"] != ""
    )
    good = d[~excl & d["dtw_bgs_m"].notna()]
    g = good.groupby(["source_id", "well_id", "interval_id", "date"])
    daily = g["dtw_bgs_m"].median().rename("dtw_bgs_m").to_frame()
    daily["n_used"] = g["dtw_bgs_m"].size()
    daily["obs_kinds"] = g["obs_kind"].agg(lambda s: "|".join(sorted(set(s))))
    daily["flags_any"] = g["flags"].agg(lambda s: flag_join(*s))
    daily = daily.reset_index()
    daily.to_parquet(out / "observations_daily.parquet", index=False)

    diag["counts"] = dict(
        inventory=len(idf),
        by_kind=idf["site_kind"].value_counts().to_dict(),
        obs_by_kind=odf["obs_kind"].value_counts().to_dict(),
        daily_rows=len(daily),
        app1_types={
            t: sum(1 for r in app1.values() if r["site_type"] == t)
            for t in sorted({r["site_type"] for r in app1.values()})
        },
        app3a_wells=len(app3a),
        app3b_rows=len(app3b),
        app6_rows=len(app6),
    )
    (out / "parse_diagnostics.json").write_text(json.dumps(diag, indent=2, default=str))

    # ---------------- manifest ----------------
    fl = json.loads(Path(fetch_log).read_text())
    man = dict(
        source_id=SOURCE_ID,
        title=(
            "Sacramento Mountains Hydrogeology Study: Final Technical Report, prepared for Otero Soil and "
            "Water Conservation District - NMBGMR Open-file Report 543"
        ),
        citation=(
            "Newton, B.T., Rawling, G.C., Timmons, S.S., Land, L., Johnson, P.S., Kludt, T.J., and Timmons, J.M., "
            "June 2012, Sacramento Mountains Hydrogeology Study, Final Technical Report, prepared for Otero Soil "
            "and Water Conservation District: New Mexico Bureau of Geology and Mineral Resources Open-file "
            "Report 543, Socorro, NM (authors and date from the report title page)"
        ),
        doi_or_url="https://geoinfo.nmt.edu/publications/openfile/details.cfml?Volume=543",
        landing_url="https://geoinfo.nmt.edu/publications/openfile/details.cfml?Volume=543",
        retrieval_date=date.today().isoformat(),
        release_version="OFR-543 (appendix PDFs dated 2021-02-01 in the zip; App. 4 .xls dated 2012-03-16)",
        license="NMBGMR publication; free download; no explicit license stated on landing page",
        files=fl["files"],
        status="acquired",
        unresolved=UNRESOLVED,
        notes=MANIFEST_NOTES,
    )
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(diag["counts"], indent=2, default=str))


UNRESOLVED = [
    "App. 2 (NMOSE well records) is 291 pages of scanned images with no text layer; not OCR'd, so screen "
    "intervals and borehole depths are not extracted (screen_top_m/screen_bot_m/borehole_depth_m empty).",
    "App. 4 logger reference is undocumented; logger DTW reads deeper than the App. 3B below-MP tape by a "
    "per-well constant (e.g. SM-0015 ~9.2 ft, SM-0026 ~2.2 ft, SM-0049 ~1.3 ft); converted by tying to manual "
    "visits, not by a stated offset.",
    "App. 4 holds 11 logger wells; the report text says 12 wells had recorders. The 12th is not in the release.",
    "SM-0035 logger series not convertible: no manual tie in App. 3B, 15,757 zero readings, ~30 ft unexplained "
    "drop after Aug 2006, and the App. 6 value differs ~7 ft; all dtw_bgs_m NaN.",
    "SM-0033 logger tied to a single App. 6 value with unknown measuring-point height (assumed 0 ft); estimated.",
    "SM-0136 logger ties scatter from -13 to +3.8 ft (sonic and F-coded manual reads); 12,569 of 22,140 "
    "readings left NaN, the remainder flagged suspect where drift exceeded 1 ft.",
    "Logger time zone / DST handling not stated (assumed local clock time, stored naive).",
    "Elevation vertical datum not stated; App. 1 says altitudes were taken from a 10 m DEM (elev_datum unknown).",
    "Measuring-point height blank in App. 3A for many wells; App. 6-only SM wells and 2 App. 3B rows assume "
    "MP = 0 ft and are flagged estimated.",
    "App. 6 OSE well-record levels (PN/RA/T ids, 35 wells) are undated static levels from drillers' records; "
    "assumed below ground surface; locations low confidence.",
    "No ridge/plateau-top well is confirmed by the source text; ridge and plateau classes rest on 10 m DEM "
    "metrics only.",
]

MANIFEST_NOTES = (
    "Manual water levels (App. 3B) are depth to water in feet below the noted measuring point; converted to "
    "metres below ground with the App. 3A measuring-point height. App. 6 is the control-point table behind the "
    "report's water-table map and mostly duplicates App. 3B (127 of 128 SM date/value pairs match exactly); only "
    "wells absent from App. 3B are kept from it. Logger data (App. 4 .xls) are 15-min then hourly; converted by "
    "per-visit tie offsets to the manual record, interpolated linearly between visits. OFR-543_complete.zip "
    "duplicates the appendix files and HR PDF and is kept only as a raw copy. Springs are inventoried but carry "
    "no observations. See NOTES.md."
)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
