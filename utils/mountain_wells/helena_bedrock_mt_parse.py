"""Parse USGS WRI 00-4212 (Thamke, Helena-area bedrock, MT, 1993-98) into the mountain-well schema.

Usage:
  uv run python -I utils/mountain_wells/helena_bedrock_mt_parse.py <raw_dir> <parsed_dir>

Stages (each re-run continues from whatever raw data exist):
  1. pdftotext -layout on the report -> <parsed_dir>/_work/report_layout.txt
  2. Table 4 headers (monitoring network: well number + location number) and Table 3 rows
     (inventory: well number, location number, area, altitude, depth) parsed from the OCR text layer,
     matched to NWIS station names in raw/nwis/site_catalog_bbox.rdb.
     -> <parsed_dir>/_work/nwis_sites.txt  (input to the fetch script --nwis-sites)
  3. NWIS sites matched to GWIC ids (local dump /data/ssd2/gwx/mt_gwic) by township/range/section/
     quarter, altitude, depth and completion date.
     -> <parsed_dir>/_work/gwic_ids.txt  (input to the fetch script --gwic-ids)
  4. With raw/nwis/field_measurements_*.json and raw/gwic/hydrograph_*.json present, write
     inventory.csv, observations.parquet, manifest.json (NOTES.md is hand-written).
"""

import base64
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_ID = "helena_bedrock_mt"
FT = 0.3048
GWIC_DIR = Path("/data/ssd2/gwx/mt_gwic")
TERRAIN = Path("/nas/handily/covariates/terrain")
# NWIS project numbers carried by the study's inventoried wells (from the site catalog; 191 sites)
STUDY_PROJECTS = ["463000250", "463016100"]

# ----------------------------------------------------------------------------------------------
# report text
# ----------------------------------------------------------------------------------------------
DIG = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "i": "1",
        "!": "1",
        "|": "1",
        "]": "1",
        "[": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
        "J": "3",
        "G": "6",
    }
)
LET = str.maketrans(
    {"0": "D", "8": "B", "6": "C", "(": "C", "O": "D", "Q": "D", "1": "A"}
)


def report_text(raw, work):
    out = work / "report_layout.txt"
    if not out.exists():
        subprocess.run(
            ["pdftotext", "-layout", str(raw / "WRI00-4212.pdf"), str(out)], check=True
        )
    return out.read_text().splitlines()


def fix_location(s):
    """Positional OCR repair of a Montana location number TTNRRWSSQQQQnn."""
    s = re.sub(r"\s+", "", s).replace("(Continued)", "").lstrip(":;.,")
    for a, b in (
        (".1", "3"),
        (".l", "3"),
        (".I", "3"),
        ("()", "0"),
        ("(l", "0"),
        ("(I", "0"),
        ("(t", "0"),
        ("(J", "0"),
        ("()", "0"),
        ("Il)", "10"),
        ("I)", "0"),
        ("IIJ", "10"),
        ("()'", "0"),
        ("0'", "0"),
    ):
        if a in s[:9]:
            s = s[:9].replace(a, b) + s[9:]
    s = re.sub(r"^[^0-9OIlD]+", "", s)
    s = re.sub(r"^(..)N[D0O]", r"\1N0", s)
    s = re.sub(r"^(..)N(..)W[D0O]", r"\1N\2W0", s)
    s = re.sub(r"^(..)ND", r"\1N0", s)
    m = re.match(r"^(..)N(..)W(..)(.*?)(..)$", s)
    if not m:
        return s
    t, r, sec, q, seq = m.groups()
    return (
        t.translate(DIG)
        + "N"
        + r.translate(DIG)
        + "W"
        + sec.translate(DIG)
        + q.translate(LET)
        + seq.translate(DIG)
    )


HDR = re.compile(
    r"^\s*W\S{1,3}\s*Number\s*:?\s*(\S+)\s*(\(Continued\))?\s+[Ll]oca\S*\s*Nu\s?m\s?be\S*\s*[:;]\s*(.+?)\s*(\(Continued\))?\s*$"
)


def table4_blocks(lines):
    i0 = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("Table 4. Records of water levels in selected")
    )
    i1 = next(i for i, ln in enumerate(lines) if re.match(r"^\s*Table 5\.", ln))
    blocks, cur = [], None
    for i in range(i0, i1):
        m = HDR.match(lines[i])
        if m:
            cur = {
                "wn_ocr": m.group(1),
                "cont": bool(m.group(2) or m.group(4)),
                "loc_ocr": m.group(3),
                "line": i + 1,
                "body": [],
            }
            blocks.append(cur)
        elif cur is not None and not lines[i].lstrip().startswith("Table 4"):
            cur["body"].append(lines[i])
    return blocks


AREA = {"NH": "NH", "SG": "SG", "WM": "WM", "SH": "SH"}
AREA_OCR = [
    (re.compile(r"^(NH|NI[-·~]?[HI]|Ni[il]|N~I|NB|NI~|Nil|NIl)$"), "NH"),
    (re.compile(r"^(SG|S0|SO|S\(;)$"), "SG"),
    (re.compile(r"^(WM|\\\\[l1]M|\\\\[l1]-?\.?M|\\11M|\\\\1M)$"), "WM"),
    (
        re.compile(r"^(SH|S[I1l][-·~]?[HIt]|51[1·]?|Sf!|SII|Sl-t|SI'I|5H|51,1|511)$"),
        "SH",
    ),
]
LOCTOK = re.compile(r"[0-9OIlJu\(\)\]\[!']{1,3}\s?[NM]\s?[0-9OIlJ\)\(\.]{1,3}\s?W")


def _looks_loc(tok):
    t = tok.replace(" ", "")
    return bool(re.search(r"[NM].{1,4}W", t[:9])) and len(t) >= 10


def table3_rows(lines):
    i0 = next(
        i for i, ln in enumerate(lines) if ln.startswith("Table 3. Records of wells")
    )
    i1 = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("Table 4. Records of water levels in selected")
    )
    rows = []
    for i in range(i0, i1):
        ln = lines[i]
        parts = re.split(r"\s{2,}", ln.strip())
        if not parts or not parts[0]:
            continue
        if _looks_loc(parts[0]) and len(parts[0].replace(" ", "")) <= 18:
            pre, loc, fields = "", parts[0], parts[1:]
        elif len(parts) > 1 and _looks_loc(parts[1]) and len(parts[0]) <= 5:
            pre, loc, fields = parts[0], parts[1], parts[2:]
        elif (
            len(parts) > 0
            and re.match(r"^\S{1,4}\s+\S", parts[0])
            and _looks_loc(parts[0].split(None, 1)[1])
        ):
            pre, loc = parts[0].split(None, 1)
            fields = parts[1:]
        else:
            continue
        if not re.search(r"\d", loc):
            continue
        area = None
        if fields:
            for rx, a in AREA_OCR:
                if rx.match(fields[0].replace(" ", "")):
                    area = a
                    fields = fields[1:]
                    break
        alt = None
        if fields:
            a = (
                fields[0]
                .replace(" ", "")
                .translate(
                    str.maketrans(
                        {
                            "O": "0",
                            "I": "1",
                            "l": "1",
                            "J": "3",
                            ")": "3",
                            "}": "3",
                            "\\": "3",
                            "U": "0",
                            "R": "8",
                            "S": "5",
                            "A": "4",
                            "~": "",
                            "_": ".",
                        }
                    )
                )
            )
            mm = re.match(r"^\.?(\d)[\.,](\d{3})$", a)
            if mm:
                alt = float(mm.group(1) + mm.group(2))
        wn = re.sub(
            r"[^0-9]",
            "",
            pre.translate(
                str.maketrans(
                    {
                        "I": "1",
                        "l": "1",
                        "J": "3",
                        ")": "3",
                        "}": "3",
                        "O": "0",
                        "&": "8",
                        "S": "5",
                    }
                )
            ),
        )
        rows.append(
            {
                "line": i + 1,
                "wn_ocr": pre,
                "wn_guess": wn,
                "loc_ocr": loc,
                "loc_fix": fix_location(loc),
                "area": area,
                "alt_ocr_ft": alt,
                "raw": ln.strip(),
            }
        )
    return pd.DataFrame(rows)


def read_rdb(p):
    d = pd.read_csv(p, sep="\t", comment="#", dtype=str)
    return d.iloc[1:].reset_index(drop=True)


# Manual resolutions (documented in NOTES.md): Table 4 header location numbers that do not exist in
# NWIS; resolved from the Table 3 row for the same well number (altitude, depth, construction date and
# inventory water-level date agree with the NWIS record).
MANUAL_T4 = {
    "129": (
        "464346112010301",
        "Table 4 header OCR '11N03W08BCBA01' not in NWIS; Table 3 row 129 (alt 3,895 ft, depth 208 ft, "
        "constructed 08-04-82, DTW measured 10-31-90) matches NWIS 11N03W08BBBD01 (depth 208, constructed 19820804, "
        "inventory 19901031)",
    ),
    "130": (
        "464346111592801",
        "Table 4 header OCR '11N03W09ABBB01' not in NWIS; Table 3 row 130 (11N03W09BDBA01, alt 3,865 ft, "
        "depth 240 ft, DTW 47.30 ft on 12-02-95) matches NWIS 11N03W09BDBA01 (depth 240, inventory 19951202)",
    ),
}


def _sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def align_rows(fixes, alts, cand_names, cand_alts, thresh=0.80, bonus=None):
    """Monotonic (order-preserving) alignment of OCR location numbers to sorted NWIS names.

    Table 3 is ordered by location number, as are sorted NWIS station names, so a Needleman-Wunsch
    style DP with free skips on the candidate side resolves OCR ambiguity using order. Pair score =
    string similarity (+0.1 if OCR altitude within 6 ft of NWIS altitude) - thresh; only positive
    pairs are taken.
    """
    n, m = len(fixes), len(cand_names)
    S = np.zeros((n, m))
    for i, (f, a) in enumerate(zip(fixes, alts)):
        for j, (c, ca) in enumerate(zip(cand_names, cand_alts)):
            sc = _sim(f, c)
            if a == a and a is not None and ca == ca and abs(a - ca) <= 6:
                sc += 0.10
            if bonus is not None:
                sc += bonus[i, j]
            S[i, j] = sc - thresh
    D = np.zeros((n + 1, m + 1))
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diag, 1 skip row, 2 skip cand
    for i in range(1, n + 1):
        D[i, 0] = D[i - 1, 0]
        bt[i, 0] = 1
        for j in range(1, m + 1):
            opts = (
                D[i - 1, j - 1] + max(S[i - 1, j - 1], -1e9)
                if S[i - 1, j - 1] > 0
                else -1e18,
                D[i - 1, j],
                D[i, j - 1],
            )
            k = int(np.argmax(opts))
            D[i, j], bt[i, j] = opts[k], k
    for j in range(1, m + 1):
        bt[0, j] = 2
    out = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        k = bt[i, j]
        if k == 0:
            out[i - 1] = (
                j - 1,
                S[i - 1, j - 1]
                + thresh
                - (bonus[i - 1, j - 1] if bonus is not None else 0.0),
            )
            i, j = i - 1, j - 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    return out


def match_wells(lines, cat, work):
    cat = cat.copy()
    cat["nm"] = cat.station_nm.str.replace(" ", "", regex=False).str.strip()
    cat["alt_ft"] = pd.to_numeric(cat.alt_va, errors="coerce")
    names = cat.nm
    # Table 4 (monitoring network)
    t4 = []
    for b in table4_blocks(lines):
        if b["cont"]:
            continue
        fix = fix_location(b["loc_ocr"])
        hit = cat.index[names == fix]
        if b["wn_ocr"] in MANUAL_T4 and len(hit) == 0:
            site, why = MANUAL_T4[b["wn_ocr"]]
            t4.append(
                {
                    **b,
                    "loc_fix": fix,
                    "cat_idx": cat.index[cat.site_no == site][0],
                    "t4_match": "manual: " + why,
                }
            )
        elif len(hit) == 1:
            t4.append({**b, "loc_fix": fix, "cat_idx": hit[0], "t4_match": "exact"})
        else:
            sc = names.apply(lambda c: _sim(fix, c)).sort_values(ascending=False)
            t4.append(
                {
                    **b,
                    "loc_fix": fix,
                    "cat_idx": sc.index[0] if sc.iloc[0] >= 0.9 else None,
                    "t4_match": f"fuzzy {sc.iloc[0]:.2f} (2nd {sc.iloc[1]:.2f})",
                }
            )
    t4 = pd.DataFrame(t4)
    # Table 3 (inventory), order-preserving alignment against NWIS names in the study townships
    t3 = table3_rows(lines)
    pool = cat[names.str.match(r"^(09|10|11|12)N0[2-6]W")].sort_values("nm")
    al = align_rows(
        list(t3.loc_fix), list(t3.alt_ocr_ft), list(pool.nm), list(pool.alt_ft)
    )
    t3["cat_idx"] = [pool.index[a[0]] if a else None for a in al]
    t3["t3_match"] = [
        (
            "exact"
            if a and a[1] >= 0.999
            else (f"aligned {a[1]:.2f}" if a else "unmatched")
        )
        for a in al
    ]
    t4.drop(columns=["body"]).to_csv(work / "table4_headers_matched.csv", index=False)
    t3.to_csv(work / "table3_rows_matched.csv", index=False)
    return t4, t3


def match_gwic(cat_sel, work):
    """Candidate GWIC ids per NWIS site from T/R/section, quarter-section consistency, altitude, depth,
    completion date, distance (GWIC NAD27 -> NAD83) and whether GWIC holds a USGS/LCWQPD-sourced SWL.
    All candidates scoring >= 5 are kept (GWIC often holds two records for one well: a driller-log
    record and a 7060xx record carrying the USGS measurements); they are confirmed later by
    water-level overlap with NWIS."""
    from pyproj import Transformer

    gs = pd.read_parquet(
        GWIC_DIR / "sites.parquet",
        columns=[
            "gwicid",
            "latitude",
            "longitude",
            "datum_latlon",
            "altitude",
            "township",
            "range",
            "section",
            "qsection",
            "county",
            "site_type",
        ],
    )
    bh = pd.read_parquet(GWIC_DIR / "boreholes.parquet").drop_duplicates("gwicid")
    gs = gs.merge(
        bh[["gwicid", "total_depth_ft_bgs", "date_completed"]], on="gwicid", how="left"
    )
    gs["done"] = pd.to_datetime(
        gs.date_completed, unit="ms", errors="coerce"
    ).dt.strftime("%Y%m%d")
    lt = pd.read_parquet(GWIC_DIR / "swl_latest.parquet", columns=["gwicid", "agency"])
    gs["usgs_swl"] = gs.gwicid.isin(lt.gwicid[lt.agency.isin(["USGS", "LCWQPD"])])
    t27 = Transformer.from_crs("EPSG:4267", "EPSG:4269", always_xy=True)
    m27 = gs.datum_latlon.eq("NAD27")
    x, y = t27.transform(gs.longitude[m27].values, gs.latitude[m27].values)
    gs.loc[m27, "longitude"], gs.loc[m27, "latitude"] = x, y
    out = []
    for _, r in cat_sel.iterrows():
        nm = r.nm
        t, rg, sec, q = nm[0:3], nm[3:6], int(nm[6:8]), nm[8:-2]
        c = gs[(gs.township == t) & (gs.range == rg) & (gs.section == sec)].copy()
        for _, g in c.iterrows():
            k, e = 0.0, []
            gq = g.qsection or ""
            if gq and q.startswith(gq):
                k += len(gq)
                e.append(f"qsec {gq}")
            elif gq:
                k -= 2
                e.append(f"qsec conflict GWIC {gq} vs NWIS {q}")
            if (
                g.altitude == g.altitude
                and r.alt_ft == r.alt_ft
                and abs(g.altitude - r.alt_ft) <= 5
            ):
                k += 2
                e.append(f"alt {g.altitude:.0f}")
            wd = pd.to_numeric(r.well_depth_va, errors="coerce")
            if g.total_depth_ft_bgs == g.total_depth_ft_bgs and wd == wd:
                if abs(g.total_depth_ft_bgs - wd) <= 1:
                    k += 3
                    e.append(f"depth {g.total_depth_ft_bgs:.0f}")
                else:
                    k -= 1
                    e.append(f"depth differs {g.total_depth_ft_bgs:.0f}")
            cd = (
                str(r.construction_dt) if r.construction_dt == r.construction_dt else ""
            )
            if g.done == g.done and len(cd) == 8 and g.done == cd:
                k += 3
                e.append(f"completed {cd}")
            dist = np.hypot(
                (g.latitude - r.lat) * 111e3,
                (g.longitude - r.lon) * 111e3 * np.cos(np.radians(r.lat)),
            )
            if dist <= 100:
                k += 1
            if g.usgs_swl:
                k += 2
                e.append("GWIC SWL from USGS/LCWQPD")
            e.append(f"dist {dist:.0f} m")
            if k >= 5:
                out.append(
                    {
                        "site_no": r.site_no,
                        "gwicid": int(g.gwicid),
                        "gwic_score": k,
                        "gwic_evidence": "; ".join(e),
                        "gwic_lat83": g.latitude,
                        "gwic_lon83": g.longitude,
                        "gwic_alt_ft": g.altitude,
                        "gwic_qsection": gq,
                        "gwic_depth_ft": g.total_depth_ft_bgs,
                    }
                )
    out = pd.DataFrame(out)
    out.to_csv(work / "gwic_candidates.csv", index=False)
    return out


# ----------------------------------------------------------------------------------------------
# inventory set (the report's 201 inventoried wells)
# ----------------------------------------------------------------------------------------------
# Table 3 numbers the 201 wells 1..201 in location-number order, so after the union of Table 4 matches,
# Table 3 alignments and the study's NWIS project numbers, the well number must equal the rank of the
# NWIS station name. Two union members break that order and two report wells are missing from it:
SET_DROP = {
    "464411112010902": "11N03W05CCBC02 carries study project 463016100 but is not in Table 3 (report well 125 is "
    "11N03W05CCBC01 and 126 is 11N03W06BBAA01); excluded",
    "463918112045501": "10N04W02BCBD01 carries project 463000250 but was inventoried 1999-05-25, after the study, "
    "and is not in Table 3; excluded",
}
SET_ADD = {
    "463615111565301": "report well 47 (Table 3 OCR '2bCAAC', altitude 3,918 ft, 420SPKN, depth 150 ft) = NWIS "
    "10N03W26CAAC01 (project 00208, altitude 3918, depth 150, 420SPKN)",
    "463932112043401": "report well 57 (Table 3 OCR '02BAIlD', altitude 4,180 ft) = NWIS 10N04W02BABD01 (project "
    "number stored malformed as '4630-16100', altitude 4180, 211SGVH, inventoried 1994-09-28)",
}

AQUIFER = {
    "110ALVM": "Quaternary alluvium",
    "123SDMS": "Oligocene sedimentary and volcanic rocks",
    "124PVIV": "Eocene volcanic/intrusive rocks (NWIS code 124PVIV)",
    "211BLDR": "Late Cretaceous intrusive rocks of the Boulder batholith",
    "211SGVH": "Late Cretaceous intrusive rocks of the Scratchgravel Hills",
    "211MAVL": "Late Cretaceous intrusive rocks of the Marysville stock",
    "211MRVL": "Late Cretaceous intrusive rocks of the Marysville stock",
    "211ELKM": "Late Cretaceous Elkhorn Mountains Volcanics",
    "220ELLS": "Jurassic Ellis Group",
    "320PSLV": "Pennsylvanian rocks",
    "320QDRN": "Pennsylvanian Quadrant Formation",
    "331MSNC": "Mississippian Mission Canyon Limestone",
    "331MDSN": "Mississippian Madison Group",
    "341JEFF": "Devonian Jefferson Formation",
    "341JFRS": "Devonian Jefferson Formation",
    "371CMBR": "Cambrian rocks",
    "371HMRK": "Cambrian Hasmark Formation",
    "374MGHR": "Cambrian Meagher Limestone",
    "374WLSY": "Cambrian Wolsey Shale",
    "420MSLD": "Middle Proterozoic Mount Shields Formation (Belt Supergroup)",
    "420SPRD": "Middle Proterozoic Shepard Formation (Belt Supergroup)",
    "420SSLP": "Middle Proterozoic Snowslip Formation (Belt Supergroup)",
    "420HELN": "Middle Proterozoic Helena Formation (Belt Supergroup)",
    "400HELN": "Middle Proterozoic Helena Formation (Belt Supergroup; NWIS code 400HELN)",
    "420EMPR": "Middle Proterozoic Empire Formation (Belt Supergroup)",
    "420SPKN": "Middle Proterozoic Spokane Formation (Belt Supergroup)",
    "420GRSN": "Middle Proterozoic Greyson Formation (Belt Supergroup)",
}
AREA_NAME = {
    "NH": "North Hills",
    "SG": "Scratchgravel Hills",
    "WM": "western mountains",
    "SH": "South Hills",
}
TOPO = {
    "H": "hilltop",
    "S": "hillside (slope)",
    "F": "flat",
    "U": "undulating",
    "V": "valley flat",
    "P": "pediment",
    "W": "upland draw",
    "T": "stream terrace",
    "G": "flood plain",
    "E": "dunes",
    "A": "alluvial fan",
    "D": "depression",
    "C": "stream channel",
}
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
COORD_CONF = {
    "H": "high",
    "1": "high",
    "5": "high",
    "S": "high",
    "R": "high",
    "F": "medium",
    "T": "low",
    "M": "low",
    "U": "low",
}
COORD_ACY = {
    "H": "0.01 s",
    "1": "0.1 s",
    "5": "0.5 s",
    "S": "1 s",
    "R": "3 s",
    "F": "5 s",
    "T": "10 s",
    "M": "1 min",
    "U": "unknown",
}
COORD_METH = {
    "M": "interpolated from map",
    "G": "GPS",
    "D": "differentially corrected GPS",
    "L": "LORAN",
    "N": "geographic name",
    "C": "GPS (unspecified)",
    "U": "unknown",
}
ALT_METH = {
    "M": "interpolated from topographic map",
    "L": "level or other surveyed method",
    "G": "GPS",
    "N": "precise GPS",
    "D": "DGPS",
}


def inventory_set(cat, t4, t3, work):
    t4_sites = set(cat.loc[t4.cat_idx, "site_no"])
    t3_sites = set(
        cat.loc[[i for i in t3.cat_idx if i is not None and i == i], "site_no"]
    )
    proj_sites = set(cat.site_no[cat.project_no.isin(STUDY_PROJECTS)])
    union = t4_sites | t3_sites | proj_sites
    sites = (union - set(SET_DROP)) | set(SET_ADD)
    sel = cat[cat.site_no.isin(sites)].copy()
    sel["nm"] = sel.station_nm.str.replace(" ", "", regex=False).str.strip()
    sel = sel.sort_values("nm").reset_index(drop=True)
    sel["well_no"] = np.arange(1, len(sel) + 1)
    for c, src in (
        ("alt_ft", "alt_va"),
        ("lat", "dec_lat_va"),
        ("lon", "dec_long_va"),
        ("well_depth_ft", "well_depth_va"),
        ("hole_depth_ft", "hole_depth_va"),
        ("alt_acy_ft", "alt_acy_va"),
    ):
        sel[c] = pd.to_numeric(sel[src], errors="coerce")
    # check: Table 4 OCR well numbers vs rank
    nm_by_idx = cat.station_nm.str.replace(" ", "", regex=False).str.strip()
    t4 = t4.copy()
    t4["nm"] = nm_by_idx.loc[t4.cat_idx].values
    t4 = t4.merge(sel[["nm", "site_no", "well_no"]], on="nm", how="left")
    agree = (t4.wn_ocr == t4.well_no.astype(str)).sum()
    print(
        f"union {len(union)} -> inventory {len(sel)} after {len(SET_DROP)} drops/{len(SET_ADD)} adds; "
        f"Table 4 OCR well no. = name rank for {agree}/{len(t4)} (rest are OCR digit errors)"
    )
    t4[["wn_ocr", "well_no", "site_no", "nm", "loc_ocr", "t4_match"]].to_csv(
        work / "table4_wells.csv", index=False
    )
    # Table 3 rows re-aligned to the final 201 (with a bonus when the OCR well number equals the rank)
    t3 = t3.copy()
    bonus = np.zeros((len(t3), len(sel)))
    for i, w in enumerate(t3.wn_guess):
        if isinstance(w, str) and w.isdigit():
            bonus[i, sel.index[sel.well_no == int(w)]] = 0.3
    al = align_rows(
        list(t3.loc_fix),
        list(t3.alt_ocr_ft),
        list(sel.nm),
        list(sel.alt_ft),
        thresh=0.70,
        bonus=bonus,
    )
    t3["well_no"] = [int(sel.well_no[a[0]]) if a else None for a in al]
    t3["t3_sim"] = [round(a[1], 2) if a else None for a in al]
    t3["wn_agree"] = [
        (isinstance(w, str) and n == n and n is not None and str(int(n)) == w)
        for w, n in zip(t3.wn_guess, t3.well_no)
    ]
    t3.to_csv(work / "table3_rows_final.csv", index=False)
    ok = t3[t3.well_no.notna() & ((t3.t3_sim >= 0.85) | t3.wn_agree)]
    ok = ok.drop_duplicates("well_no")
    sel = sel.merge(
        ok[["well_no", "area", "t3_sim", "wn_agree", "line"]].rename(
            columns={"line": "t3_line"}
        ),
        on="well_no",
        how="left",
    )
    print(
        f"Table 3: {len(t3)} OCR rows; {len(ok)} assigned to a well (sim>=0.85 or OCR well no. agrees); "
        f"area code for {sel.area.notna().sum()}"
    )
    sel["in_t4"] = sel.site_no.isin(set(t4.site_no))
    return sel, t4


# ----------------------------------------------------------------------------------------------
# observations
# ----------------------------------------------------------------------------------------------
def load_nwis(raw):
    rows = []
    for f in sorted((raw / "nwis").glob("field_measurements_*.json")):
        for k, ft in enumerate(json.loads(f.read_text())["features"]):
            p = ft["properties"]
            p["source_file"] = f"raw/{SOURCE_ID}/nwis/{f.name}"
            p["source_row"] = f"features[{k}]"
            rows.append(p)
    m = pd.DataFrame(rows)
    m["site_no"] = m.monitoring_location_id.str.replace("USGS-", "", regex=False)
    m["v"] = pd.to_numeric(m.value, errors="coerce")
    m["qual"] = m.qualifier.apply(lambda q: "|".join(q) if isinstance(q, list) else "")
    return m


def ms_bodies(lines, t4w):
    """Table 4 OCR body text per NWIS site (continuation blocks appended to their well)."""
    out, cur, k = {}, None, 0
    for b in table4_blocks(lines):
        if not b["cont"]:
            cur = t4w.site_no.iloc[k]
            k += 1
        if cur is not None:
            out.setdefault(cur, []).extend(b["body"])
    return {s: "\n".join(v) for s, v in out.items()}


def ms_code(body, v):
    if body is None or v != v:
        return None
    s = f"{abs(v):.2f}"
    m = re.search(
        r"(?<![\d.])"
        + (r"\+\s?" if v < 0 else "")
        + re.escape(s)
        + r"\s+([A-Za-z5]{1,2})(?![A-Za-z])",
        body,
    )
    if not m:
        return None
    c = m.group(1)
    return c[0].upper().replace("5", "S") + c[1:].upper()


MS_FLAG = {
    "P": "pumping",
    "R": "recovering",
    "D": "dry_censored",
    "F": "flowing",
    "O": "other:obstruction",
    "W": "other:destroyed",
}


def gwic_series(raw):
    rows = []
    for f in sorted((raw / "gwic").glob("hydrograph_*.json")):
        g = int(f.stem.split("_")[1])
        d = json.loads(f.read_text())
        if "graph" not in d:
            continue
        for ch in d["graph"]["doc"]["roots"][0]["attributes"]["children"]:
            for rd in ch["attributes"].get("renderers", []):
                ds = rd.get("attributes", {}).get("data_source", {})
                ents = (
                    ds.get("attributes", {}).get("data", {}).get("entries", [])
                    if isinstance(ds, dict)
                    else []
                )
                cols = {}
                for k, v in ents:
                    if isinstance(v, dict) and v.get("type") == "ndarray":
                        cols[k] = np.frombuffer(
                            base64.b64decode(v["array"]["data"]),
                            dtype=np.dtype(v["dtype"]).newbyteorder("<"),
                        )
                    elif isinstance(v, list):
                        cols[k] = np.asarray(v)
                if "date" in cols and "daily_average_orig" in cols:
                    dt = pd.to_datetime(cols["date"], unit="ms")
                    for i, (t, val) in enumerate(zip(dt, cols["daily_average_orig"])):
                        rows.append(
                            {
                                "gwicid": g,
                                "date": t.strftime("%Y-%m-%d"),
                                "swl_ft": float(val),
                                "source_file": f"raw/{SOURCE_ID}/gwic/{f.name}",
                                "source_row": f"daily_average_orig[{i}]",
                            }
                        )
    return pd.DataFrame(
        rows, columns=["gwicid", "date", "swl_ft", "source_file", "source_row"]
    )


def verify_gwic(gm, gw, nw, work):
    """Confirm candidate GWIC ids by same-date water-level agreement with the NWIS record: at least one
    identical value (|diff| < 0.006 ft) and either >= 80 % of shared dates identical or >= 20 identical
    values (later non-USGS measurements can differ by a constant measuring-point correction)."""
    n72 = nw[nw.parameter_code == "72019"].copy()
    n72["date"] = n72.time.str[:10]
    res = []
    for _, c in gm.iterrows():
        g = gw[gw.gwicid == c.gwicid]
        s = n72[n72.site_no == c.site_no]
        mm = g.merge(s[["date", "v"]], on="date", how="inner")
        eq = int((np.abs(mm.swl_ft - mm.v) < 0.006).sum())
        res.append(
            {
                **c.to_dict(),
                "gwic_n": len(g),
                "n_same_date": len(mm),
                "n_equal": eq,
                "verified": bool(eq >= 1 and (eq >= 0.8 * len(mm) or eq >= 20)),
            }
        )
    res = pd.DataFrame(res)
    res.to_csv(work / "gwic_verification.csv", index=False)
    return res


def build_obs(sel, nw, gw, ver, bodies):
    depth_m = dict(
        zip(
            sel.site_no,
            np.where(sel.well_depth_ft.notna(), sel.well_depth_ft, sel.hole_depth_ft)
            * FT,
        )
    )
    wid = {s: f"{SOURCE_ID}:{s}" for s in sel.site_no}
    nw = nw[nw.site_no.isin(wid)]
    vis = nw[nw.parameter_code == "72019"].copy()
    other = nw[~nw.field_visit_id.isin(vis.field_visit_id)]
    stats = {
        "nwis_records": len(nw),
        "nwis_visits": nw.field_visit_id.nunique(),
        "nwis_72019": len(vis),
        "visits_without_72019": other.field_visit_id.nunique(),
    }
    out, ms_hits, crosstab = [], 0, []
    for _, r in vis.iterrows():
        flags, cens = [], ""
        if r.month != r.month:
            ts = r.time[:4]
        elif r.day != r.day:
            ts = r.time[:7]
        elif r.time_of_day is None or r.time_of_day != r.time_of_day:
            ts = r.time[:10]
        else:
            ts = r.time
        q = r.qualifier if isinstance(r.qualifier, list) else []
        ms = ms_code(bodies.get(r.site_no), r.v) if r.time[:10] < "1998-06-01" else None
        if ms:
            ms_hits += 1
            crosstab.append((r.qual, ms))
        st = ms[1] if ms and len(ms) > 1 else None
        if st in MS_FLAG:
            flags.append(MS_FLAG[st])
        elif "Pumping" in q:
            flags.append("pumping")
        conv = f"NWIS 72019 ft below land surface x {FT}"
        if "Dry" in q:
            d = depth_m.get(r.site_no, np.nan)
            dtw, cens = d, "gt"
            flags = ["dry_censored"]
            conv = "dry: dtw_bgs_m set to NWIS well depth (well_depth_va, else hole_depth_va) x 0.3048, censored gt"
        elif "Flowing" in q and r.v != r.v:
            dtw, cens = 0.0, "lt"
            flags = ["flowing"]
            conv = "flowing, head not measured: dtw_bgs_m = 0 (water level at or above land surface), censored lt"
        else:
            dtw = r.v * FT
            if r.v < 0:
                flags.append("flowing")
                conv += "; negative = head above land surface"
        proc = r.observing_procedure or ""
        if "estim" in proc:
            flags.append("estimated")
        if "logs" in proc:
            flags.append("other:procedure_logs")
        if r.measuring_agency and r.measuring_agency != "USGS":
            flags.append(f"other:agency_{r.measuring_agency}")
        if r.approval_status and r.approval_status != "Approved":
            flags.append("provisional")
        if ms:
            conv += f"; report Table 4 MS code {ms}"
        if proc:
            conv += f"; {proc}"
        out.append(
            {
                "source_id": SOURCE_ID,
                "well_id": wid[r.site_no],
                "interval_id": wid[r.site_no] + "#1",
                "timestamp": ts,
                "orig_value": r.v,
                "orig_units": "ft",
                "orig_reference": "bgs",
                "mp_offset_m": np.nan,
                "dtw_bgs_m": dtw,
                "conversion": conv,
                "flags": "|".join(dict.fromkeys(flags)),
                "censored": cens,
                "source_file": r.source_file,
                "source_row": r.source_row,
                "_site": r.site_no,
                "_date": r.time[:10],
            }
        )
    obs = pd.DataFrame(out)
    stats["report_ms_matched"] = ms_hits
    stats["ms_by_nwis_qualifier"] = (
        pd.Series([f"{a} / {b}" for a, b in crosstab]).value_counts().to_dict()
    )
    # GWIC-only measurements of verified ids
    vv = ver[ver.verified]
    g = gw.merge(vv[["site_no", "gwicid"]], on="gwicid")
    key = set(zip(obs._site, obs._date))
    dup = (
        g.apply(lambda r: (r.site_no, r.date) in key, axis=1)
        if len(g)
        else pd.Series(dtype=bool)
    )
    stats["gwic_records_verified_ids"] = len(g)
    stats["gwic_same_date_as_nwis_dropped"] = int(dup.sum())
    g = g[~dup].drop_duplicates(["site_no", "date", "swl_ft"])
    # same value within 3 days of an NWIS measurement = same measurement with a transcribed date difference
    nv = obs[obs.orig_value.notna()][["_site", "_date", "orig_value"]]
    nv = nv.assign(d=pd.to_datetime(nv._date)).rename(columns={"_site": "site_no"})
    gg = g.assign(d=pd.to_datetime(g.date), _k=np.arange(len(g)))
    mm = gg.merge(nv, on="site_no")
    near = mm[
        (np.abs(mm.d_x - mm.d_y) <= pd.Timedelta("3D"))
        & (np.abs(mm.swl_ft - mm.orig_value) < 0.006)
    ]
    stats["gwic_same_value_within_3d_dropped"] = int(near._k.nunique())
    g = g[~gg._k.isin(near._k).values]
    # two verified GWIC records for one well: keep the first id's value on a shared date
    first = g.groupby("site_no").gwicid.transform("min")
    shared = g.duplicated(["site_no", "date"], keep=False) & (
        g.groupby(["site_no", "date"]).gwicid.transform("nunique") > 1
    )
    stats["gwic_cross_id_same_date_conflicts_dropped"] = int(
        (shared & (g.gwicid != first)).sum()
    )
    g = g[~(shared & (g.gwicid != first))]
    yr = g.date.str[:4]
    dense = g.groupby(["gwicid", yr]).date.transform("size") > 60
    stats["gwic_only_added"] = len(g)
    stats["gwic_only_logger_years"] = int(dense.sum())
    add = pd.DataFrame(
        {
            "source_id": SOURCE_ID,
            "well_id": g.site_no.map(wid),
            "interval_id": g.site_no.map(wid) + "#1",
            "timestamp": g.date,
            "orig_value": g.swl_ft,
            "orig_units": "ft",
            "orig_reference": "bgs",
            "mp_offset_m": np.nan,
            "dtw_bgs_m": g.swl_ft * FT,
            "conversion": "GWIC hydrograph daily_average_orig = SWL ft below ground surface (equals GWIC swl_ground_ft_bgs; "
            "equals NWIS 72019 on shared USGS dates) x 0.3048",
            "flags": np.where(
                dense, "other:gwic_only|other:logger_daily_mean", "other:gwic_only"
            ),
            "censored": "",
            "source_file": g.source_file,
            "source_row": g.source_row,
            "_site": g.site_no,
            "_date": g.date,
        }
    )
    obs = (
        pd.concat([obs, add], ignore_index=True)
        .sort_values(["well_id", "timestamp"])
        .reset_index(drop=True)
    )
    return obs, stats


# ----------------------------------------------------------------------------------------------
# inventory
# ----------------------------------------------------------------------------------------------
def sample_terrain(lat, lon):
    import rasterio
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True).transform(
        lon, lat
    )
    out = {}
    for name in ("elev48i0100a", "tpi_500m", "tpi_2km", "geomorphons"):
        with rasterio.open(TERRAIN / f"{name}.tif") as r:
            v = np.array([s[0] for s in r.sample(zip(x, y))], dtype=float)
            v[v == r.nodata] = np.nan
            out[name] = v
    return out


def landscape(topo, geo, tpi5, tpi2):
    tdesc = TOPO.get(topo, "not recorded") if topo == topo else "not recorded"
    g = GEOMORPHON.get(int(geo), "nodata") if geo == geo else "nodata"
    ev = (
        f"NWIS topo_cd={topo if topo == topo else 'blank'} ({tdesc}); 100 m DEM (elev48i0100a/geomorphons/"
        f"tpi_500m/tpi_2km, EPSG:5070): geomorphon={g}, TPI500={tpi5:+.2f}, TPI2km={tpi2:+.2f} (raster units)"
    )
    if topo == "H":
        return (
            "ridgetop_divide",
            ev,
            ("medium" if g in ("peak", "ridge", "shoulder", "spur") else "low"),
        )
    if topo == "S":
        return (
            "mountain_slope",
            ev,
            (
                "medium"
                if g in ("slope", "shoulder", "spur", "hollow", "footslope")
                else "low"
            ),
        )
    if topo in ("V", "W", "G", "C"):
        return (
            "mountain_valley_bottom",
            ev,
            ("medium" if g in ("valley", "hollow", "footslope", "pit") else "low"),
        )
    return "uncertain", ev, "low"


def build_inventory(sel, obs, ver, t4w):
    gd = GWIC_DIR
    comp = pd.read_parquet(gd / "completions.parquet")
    terr = sample_terrain(sel.lat.values, sel.lon.values)
    rec = obs.groupby("well_id").agg(
        record_start=("timestamp", "min"),
        record_end=("timestamp", "max"),
        n_obs=("timestamp", "size"),
    )
    rows = []
    for i, r in sel.iterrows():
        w = f"{SOURCE_ID}:{r.site_no}"
        vg = ver[(ver.site_no == r.site_no) & ver.verified]
        ug = ver[(ver.site_no == r.site_no) & ~ver.verified]
        cp = comp[comp.gwicid.isin(vg.gwicid)]
        st, sb, cnote = np.nan, np.nan, ""
        if len(cp) and cp.comp_from_ft.notna().any():
            st, sb = cp.comp_from_ft.min() * FT, cp.comp_to_ft.max() * FT
            cnote = (
                f"screen/perforations from GWIC completions (gwicid {','.join(map(str, sorted(cp.gwicid.unique())))}): "
                + "; ".join(
                    f"{a:g}-{b:g} ft {d}"
                    for a, b, d in zip(cp.comp_from_ft, cp.comp_to_ft, cp.description)
                )
            )
        aq = r.aqfr_cd if r.aqfr_cd == r.aqfr_cd else None
        setting = (
            "unknown"
            if aq is None
            else ("alluvium" if aq.startswith("110") else "bedrock")
        )
        topo = r.topo_cd
        lc, ev, conf = landscape(
            topo, terr["geomorphons"][i], terr["tpi_500m"][i], terr["tpi_2km"][i]
        )
        ev += (
            f"; DEM elev {terr['elev48i0100a'][i]:.0f} m vs NWIS {r.alt_ft * FT:.0f} m"
        )
        area = r.area if r.area == r.area else None
        notes = [
            f"report well no. {r.well_no} (Table 3 order; NWIS station name {r.station_nm.strip()})",
            f"monitoring network (Table 4): {'yes' if r.in_t4 else 'no'}",
        ]
        if area:
            notes.append(f"study area {area} ({AREA_NAME[area]}), Table 3 OCR")
        if r.site_no in SET_ADD:
            notes.append("inventory-set correction: " + SET_ADD[r.site_no])
        for k, (s, why) in MANUAL_T4.items():
            if s == r.site_no:
                notes.append(f"Table 4 header mismatch resolved: {why}")
        if aq:
            notes.append(f"NWIS aqfr_cd {aq}")
        if r.aqfr_type_cd == r.aqfr_type_cd:
            notes.append(
                f"NWIS aqfr_type_cd {r.aqfr_type_cd} (mixed confined/unconfined)"
            )
        if obs.loc[obs.well_id == w, "flags"].str.contains("flowing").any():
            notes.append(
                "flowing artesian (head above land surface): confined conditions"
            )
        if len(vg):
            notes.append(
                "GWIC verified: "
                + ", ".join(
                    f"{int(g)} ({e}/{n} same-date levels equal)"
                    for g, e, n in zip(vg.gwicid, vg.n_equal, vg.n_same_date)
                )
            )
        if len(ug):
            notes.append(
                "GWIC candidates not confirmed by water levels: "
                + ", ".join(map(str, ug.gwicid.astype(int)))
            )
        if cnote:
            notes.append(cnote)
        if r.site_no not in set(obs._site):
            notes.append("no NWIS or GWIC water levels returned")
        acy = r.coord_acy_cd
        rr = rec.loc[w] if w in rec.index else None
        rows.append(
            {
                "source_id": SOURCE_ID,
                "well_id": w,
                "interval_id": w + "#1",
                "well_name": r.station_nm.strip(),
                "site_no": r.site_no,
                "lat": r.lat,
                "lon": r.lon,
                "horiz_datum": "NAD83",
                "coord_source": f"NWIS dec_lat_va/dec_long_va (NAD83; original {r.coord_datum_cd}); method "
                f"{COORD_METH.get(r.coord_meth_cd, r.coord_meth_cd)}; accuracy ±{COORD_ACY.get(acy, acy)}",
                "coord_confidence": COORD_CONF.get(acy, "low"),
                "elev_m": r.alt_ft * FT,
                "elev_datum": r.alt_datum_cd,
                "elev_source": f"NWIS alt_va {r.alt_va} ft ({ALT_METH.get(r.alt_meth_cd, r.alt_meth_cd)}, ±{r.alt_acy_va} ft)",
                "mp_height_m": np.nan,
                "borehole_depth_m": r.hole_depth_ft * FT,
                "well_depth_m": r.well_depth_ft * FT,
                "screen_top_m": st,
                "screen_bot_m": sb,
                "geology": f"{AQUIFER.get(aq, aq)}" if aq else "",
                "aquifer_setting": setting,
                "state": "MT",
                "landscape_class": lc,
                "landscape_evidence": ev,
                "landscape_confidence": conf,
                "record_start": rr.record_start if rr is not None else "",
                "record_end": rr.record_end if rr is not None else "",
                "n_obs": int(rr.n_obs) if rr is not None else 0,
                "source_file": f"raw/{SOURCE_ID}/nwis/site_catalog_bbox.rdb; raw/{SOURCE_ID}/WRI00-4212.pdf (Tables 3-4)",
                "notes": "; ".join(notes),
                "_geo": GEOMORPHON.get(int(terr["geomorphons"][i]), "")
                if terr["geomorphons"][i] == terr["geomorphons"][i]
                else "",
                "_tpi5": terr["tpi_500m"][i],
                "_tpi2": terr["tpi_2km"][i],
                "_topo": topo,
                "_wn": r.well_no,
                "_area": area,
                "_t4": r.in_t4,
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
def main(raw_dir, parsed_dir):
    raw, parsed = Path(raw_dir), Path(parsed_dir)
    work = parsed / "_work"
    work.mkdir(parents=True, exist_ok=True)
    lines = report_text(raw, work)
    cat = read_rdb(raw / "nwis" / "site_catalog_bbox.rdb")
    t4, t3 = match_wells(lines, cat, work)
    sel, t4w = inventory_set(cat, t4, t3, work)
    (work / "nwis_sites.txt").write_text("\n".join(sel.site_no) + "\n")
    fm = sorted((raw / "nwis").glob("field_measurements_*.json"))
    have = set(load_nwis(raw).site_no) if fm else set()
    add = sorted(set(SET_ADD) - have)
    if add:
        (work / "nwis_sites_add.txt").write_text("\n".join(add) + "\n")
        print(
            "NWIS field measurements missing for",
            add,
            "-> fetch --nwis-sites",
            work / "nwis_sites_add.txt",
            "--nwis-tag add_",
        )
    gm = match_gwic(sel, work)
    ids = sorted(set(int(g) for g in gm.gwicid))
    (work / "gwic_ids.txt").write_text("\n".join(map(str, ids)) + "\n")
    print(
        f"GWIC candidates {len(ids)} ids for {gm.site_no.nunique()} / {len(sel)} NWIS sites"
    )
    if not fm or not list((raw / "gwic").glob("hydrograph_*.json")):
        print("fetch NWIS field measurements and GWIC hydrographs first")
        return
    nw = load_nwis(raw)
    gw = gwic_series(raw)
    ver = verify_gwic(gm, gw, nw, work)
    print(
        f"GWIC: {len(gw)} hydrograph values from {gw.gwicid.nunique()} ids; verified {ver.verified.sum()} ids "
        f"for {ver[ver.verified].site_no.nunique()} sites"
    )
    bodies = ms_bodies(lines, t4w)
    obs, stats = build_obs(sel, nw, gw, ver, bodies)
    inv = build_inventory(sel, obs, ver, t4w)
    obs = obs.drop(columns=["_site", "_date"])
    obs.to_parquet(parsed / "observations.parquet", index=False)
    inv.drop(columns=[c for c in inv.columns if c.startswith("_")]).to_csv(
        parsed / "inventory.csv", index=False
    )
    inv.to_csv(work / "inventory_diag.csv", index=False)
    (work / "obs_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps(stats, indent=2, default=str))
    write_manifest(raw, parsed, stats, inv, obs)


UNRESOLVED = [
    "Report PDF is a scan with a poor OCR text layer; Table 3/4 values were not transcribed. Observations come from "
    "NWIS (same values as Table 4 where checked) and MBMG GWIC; the OCR is used only to identify wells, assign study-area "
    "codes and recover Table 4 measurement-status codes.",
    "Study-area code (NH/SG/WM/SH) assigned for only the wells whose Table 3 OCR row aligned reliably; the rest are blank.",
    "Well 129 and 130 Table 4 header location numbers (11N03W08BCBA01, 11N03W09ABBB01) are not NWIS names; resolved to "
    "NWIS 11N03W08BBBD01 and 11N03W09BDBA01 from Table 3 attributes. GWIC 706052 uses quarter ABBB for well 130.",
    "NWIS qualifier 'Above|Pumping' is used for both report status P (pumping) and R (recently pumped); where the "
    "Table 4 code could not be recovered from the OCR these rows carry flag 'pumping'.",
    "Measuring-point heights are not carried: NWIS 72019 and GWIC hydrograph values are already depth below land surface. "
    "Later non-USGS (LCWQPD/MBMG) values in GWIC can differ from NWIS by a constant 0.3 ft (measuring-point correction); "
    "NWIS is kept on shared dates.",
    "GWIC candidates that did not reproduce NWIS levels were not used (e.g. 66332, constant -1.5 ft; 5834, -14 ft; "
    "892125, constant -0.10 ft over 40 dates, probably the same well with a different measuring-point correction). "
    "GWIC records with no shared dates (e.g. 196245, 706042, 138530, 66463) could not be confirmed and were not used; "
    "196245 (2003-2011, median 24.9 ft) is clearly a different well from NWIS 464424112042001 (median 70.0 ft).",
    "Confinement is not stated per well in the report or NWIS (aqfr_type_cd blank except one 'X'); one well (report "
    "well 15) is flowing artesian.",
    "Legacy NWIS waterservices gwlevels endpoint is decommissioned; levels were taken from the api.waterdata.usgs.gov "
    "field-measurements collection (2026-10-27 snapshot).",
]


def write_manifest(raw, parsed, stats, inv, obs):
    log = json.loads((parsed / "fetch_log.json").read_text())
    files = []
    for f in log["files"]:
        p = raw.parent.parent / f["path"]
        if p.exists():
            files.append(
                {
                    "path": f["path"],
                    "url": f["url"],
                    "sha256": f["sha256"],
                    "bytes": f["bytes"],
                }
            )
    m = {
        "source_id": SOURCE_ID,
        "title": "Hydrology of the Helena area bedrock, west-central Montana, 1993-98, with a section on geologic "
        "setting and a generalized bedrock geologic map",
        "citation": "Thamke, J.N., 2000, Hydrology of the Helena area bedrock, west-central Montana, 1993-98, with a "
        "section on geologic setting and a generalized bedrock geologic map: U.S. Geological Survey "
        "Water-Resources Investigations Report 2000-4212, 119 p., 3 pl.",
        "doi_or_url": "https://doi.org/10.3133/wri20004212",
        "landing_url": "https://pubs.usgs.gov/publication/wri20004212",
        "retrieval_date": "2026-10-27",
        "release_version": "WRI 2000-4212 (2000); NWIS field measurements via api.waterdata.usgs.gov OGC API "
        "(retrieved 2026-10-27); MBMG GWIC site reports/hydrographs (retrieved 2026-10-27)",
        "license": "USGS report and NWIS data: public domain (U.S. Government work). MBMG GWIC: public data, "
        "MBMG terms of use; site reports contain owner names (kept in raw only)",
        "files": files,
        "status": "acquired",
        "unresolved": [],
        "notes": "",
        "counts": {
            "wells": int(len(inv)),
            "monitoring_network_wells": int(inv._t4.sum()),
            "wells_with_obs": int((inv.n_obs > 0).sum()),
            "observations": int(len(obs)),
            **{k: v for k, v in stats.items() if not isinstance(v, dict)},
        },
        "fetch_failures": log.get("failures", []),
    }
    m["unresolved"] = UNRESOLVED
    m["notes"] = (
        "201 inventoried wells (Table 3) and the 112-well monthly network (Table 4) identified as NWIS sites; "
        "observations = NWIS discrete levels (1973-2019) plus MBMG GWIC hydrograph values of GWIC records "
        "confirmed against NWIS (to 2026, incl. transducer daily means). See NOTES.md."
    )
    (parsed / "manifest.json").write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
