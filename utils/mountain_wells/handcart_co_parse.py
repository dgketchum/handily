"""Parse USGS OFR 2006-1189 (Handcart Gulch / Webster Pass, CO) into the mountain_wells schema.

Usage:
  uv run python -I utils/mountain_wells/handcart_co_parse.py <raw_dir> <out_dir> [--terrain-dir DIR]

Inputs (unchanged downloads in raw_dir): HandcartWellData.xls, of06-1189_508.pdf, landing.html.
Outputs (out_dir): inventory.csv, observations.parquet, lithology.csv, manifest.json.
NOTES.md is written by hand alongside.
"""

import argparse
import hashlib
import json
import re
import struct
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xlrd
from xlrd import compdoc

SOURCE_ID = "handcart_co"
FT = 0.3048
XLS = "HandcartWellData.xls"
URLS = {
    "landing.html": "https://pubs.usgs.gov/of/2006/1189/",
    "of06-1189_508.pdf": "https://pubs.usgs.gov/of/2006/1189/downloads/pdf/of06-1189_508.pdf",
    "HandcartWellData.xls": "https://pubs.usgs.gov/of/2006/1189/downloads/xls/HandcartWellData.xls",
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

# Open / screened interval interpretation (ft bgs). Slotted PVC intervals are read from the
# worksheets; where the sheet gives no slotted interval the open-hole interval is inferred
# from casing depths (documented in NOTES.md).
INTERVALS_FT = {
    "WP1": (
        110.0,
        1103.0,
        "slotted PVC 110-1103 ft (blank 24-110); open borehole continues to 1220 ft",
    ),
    "WP2": (
        15.0,
        775.0,
        "slotted PVC 15-580, 590-620, 630-775 ft with blank 580-590 and 620-630 ft; open to 780 ft",
    ),
    "WP3": (
        96.0,
        298.0,
        "INFERRED open hole: no slotted PVC (na); blank PVC 0-45 ft, steel HW rod left to 96 ft; hole open to 298 ft",
    ),
    "WP4": (
        130.0,
        300.0,
        "INFERRED open hole: no slotted PVC (na); blank PVC 0-130 ft; HQ rod left in hole 300-601 ft; hole blocked at ~390-400 ft (stuck Kemmer bottle)",
    ),
    "HCBW1": (
        23.7,
        99.6,
        "sheet gives blank 0-23.7 ft and slotted '75.9' (read as slotted LENGTH -> 23.7-99.6 ft; TD 99.9 ft)",
    ),
    "HCSW1": (3.0, 13.0, "slotted PVC 3-13 ft"),
    "HCBW2": (38.5, 132.5, "slotted PVC 38.5-132.5 ft (blank 0-35.8 ft)"),
    "HCSW2": (5.0, 25.0, "slotted PVC 5-25 ft"),
    "HCBW3": (53.0, 172.1, "slotted PVC 53-172.1 ft (blank 0-53 ft)"),
    "HCFW3": (14.0, 30.0, "slotted PVC 14-30 ft"),
    "HCCW3": (0.0, 10.0, "slotted PVC 0-10 ft"),
    "HCBW4": (55.0, 150.0, "slotted PVC 55-150 ft (blank 0-55 ft)"),
    "HCFW5": (2.0, 22.5, "slotted PVC 2-22.5 ft"),
}

LANDSCAPE = {
    "WP1": (
        "ridgetop_divide",
        "high",
        "Sheet: 'WP1 (at webster pass)'. Report p.6: deep wells 'located in the upper part of the watershed, "
        "the highest one at Webster Pass on the Continental Divide at 12,100 feet'. A pass = saddle (col) on the "
        "divide, not a summit. Alpine also supported: report calls Handcart Gulch 'an alpine watershed along the "
        "Continental Divide'; WP1 (12,096 ft) is above WP2, the only WP well described as 'in forest' (treeline "
        "inference, not stated).",
    ),
    "WP2": (
        "mountain_slope",
        "medium",
        "Sheet: 'WP2 (at toe of rock glacier in forest)'; 11,505 ft, upper watershed below Webster Pass. "
        "'In forest' argues against alpine.",
    ),
    "WP3": (
        "alpine",
        "medium",
        "Sheet: 'WP3 (on rock glacier, south)'; Fig. 3 caption: upper road 'cuts across a major rock glacier' "
        "excavated to drill WP3 and WP4; report: 'rock glacier that shows evidence for recent motion' in NE "
        "watershed; 11,662 ft. Position is on a slope (rock glacier below Red Cone / divide), not a ridgetop. "
        "Above-treeline not stated explicitly.",
    ),
    "WP4": (
        "alpine",
        "medium",
        "Sheet: 'WP4 (on rock glacier, north)'; Fig. 3 caption (WP3/WP4 drilled from road across major rock "
        "glacier); 11,714 ft. Position is on a slope, not a ridgetop. Above-treeline not stated explicitly.",
    ),
}
HC_LANDSCAPE_EVID = (
    "Report p.6: nine shallow wells 'located adjacent to the trunk stream'; p.3: valley floor/trunk "
    "stream bed lined by ferricrete; p.9: artesian conditions in bedrock 'near the trunk stream'."
)
HC_SITE = {
    "HCBW1": "at confluence of N&S subdrainages",
    "HCSW1": "HC1 cluster (with HCBW1)",
    "HCBW2": "at central E toe rock glacier springs",
    "HCSW2": "HC2 cluster (with HCBW2)",
    "HCBW3": "at E toe of rock glacier",
    "HCFW3": "HC3 cluster",
    "HCCW3": "HC3 cluster",
    "HCBW4": "at N meadow site; 'set back about 200 feet from the stream' (report p.9)",
    "HCFW5": "at ferricrete flume site",
}
CONFINED = {"HCBW1", "HCBW2", "HCBW3", "HCFW5"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell(sh, r, c):
    if c >= sh.ncols:
        return None
    t = sh.cell_type(r, c)
    if t in (xlrd.XL_CELL_EMPTY, xlrd.XL_CELL_BLANK):
        return None
    return sh.cell_value(r, c)


def xdate(book, v):
    return xlrd.xldate_as_tuple(v, book.datemode)


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def textboxes(xls_path):
    """Return list of (substream_index, text) for BIFF8 TXO text boxes (lithologic logs live here)."""
    wb = compdoc.CompDoc(Path(xls_path).read_bytes()).get_named_stream("Workbook")
    recs, pos = [], 0
    while pos + 4 <= len(wb):
        rt, ln = struct.unpack("<HH", wb[pos : pos + 4])
        recs.append((rt, wb[pos + 4 : pos + 4 + ln]))
        pos += 4 + ln
    out, sub = [], -1
    for i, (rt, b) in enumerate(recs):
        if rt == 0x0809:
            sub += 1
        if rt == 0x01B6:
            cch = struct.unpack("<H", b[10:12])[0]
            j, txt = i + 1, ""
            while len(txt) < cch and j < len(recs) and recs[j][0] == 0x003C:
                cb = recs[j][1]
                txt += cb[1:].decode(
                    "utf-16-le" if cb[0] & 1 else "mac_roman", "replace"
                )
                j += 1
            out.append((sub, txt[:cch].replace("\r", "\n")))
    return out


def parse_lithology(xls_path):
    rows = []
    rng = re.compile(
        r"^\s*(~?[\d.]+)\s*(?:[-~–]\s*~?([\d.]+))?\s*(\(end of log\))?\s*feet\s*[-–]\s*(.*)$"
    )
    for sub, txt in textboxes(xls_path):
        lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
        head = lines[0]
        m = re.match(r"^(WP\s?\d|HC\s?[A-Z]{2}\s?\d|HC Adit)", head)
        name = m.group(1).replace(" ", "") if m else head
        for ln in lines[1:]:
            mm = rng.match(ln)
            if not mm:
                rows.append(
                    dict(
                        well=name,
                        top_ft=np.nan,
                        bot_ft=np.nan,
                        description=ln,
                        log_header=head,
                    )
                )
                continue
            top = float(mm.group(1).strip("~"))
            bot = float(mm.group(2).strip("~")) if mm.group(2) else np.nan
            rows.append(
                dict(
                    well=name,
                    top_ft=top,
                    bot_ft=bot,
                    description=mm.group(4).strip(),
                    log_header=head,
                )
            )
    df = pd.DataFrame(rows)
    df.insert(
        0,
        "well_id",
        [
            f"{SOURCE_ID}:{w}" if not w.startswith("HCAdit") else f"{SOURCE_ID}:HCAdit"
            for w in df["well"]
        ],
    )
    df["top_m"] = df["top_ft"] * FT
    df["bot_m"] = df["bot_ft"] * FT
    return df


def wp_construction(book):
    sh = book.sheet_by_name("WP Well Data")
    wells = {}
    for r in range(sh.nrows):
        v = cell(sh, r, 0)
        if isinstance(v, str) and re.match(r"^WP\d", v):
            name = v.split()[0]
            wells[name] = dict(
                row=r,
                label=v.strip(),
                utm_e=num(cell(sh, r, 1)),
                utm_n=num(cell(sh, r, 2)),
                loc_err_ft=num(cell(sh, r, 3)),
                lat_dms=cell(sh, r, 4),
                lon_dms=cell(sh, r, 5),
                elev_ft=num(cell(sh, r, 6)),
                permit=str(int(cell(sh, r, 7))),
                year=int(cell(sh, r, 11)),
                td_ft=num(cell(sh, r, 12)),
                open_ft=num(cell(sh, r, 22)),
                lith=" ".join(
                    str(cell(sh, rr, 9)) for rr in range(r, r + 3) if cell(sh, rr, 9)
                ),
                comments=" ".join(
                    str(cell(sh, rr, 34)) for rr in range(r, r + 4) if cell(sh, rr, 34)
                ),
                steel=str(cell(sh, r, 13))
                + (" " + str(cell(sh, r + 1, 13)) if cell(sh, r + 1, 13) else ""),
                test=cell(sh, r, 32),
                kind="bedrock well",
            )
    return wells


def hc_construction(book):
    sh = book.sheet_by_name("HC Well Data")
    wells = {}
    for r in range(sh.nrows):
        v = cell(sh, r, 0)
        if isinstance(v, str) and re.match(r"^HC(BW|SW|FW|CW)\d", v):
            name = v.split()[0]
            dd = cell(sh, r, 11)
            wells[name] = dict(
                row=r,
                label=v.strip(),
                utm_e=num(cell(sh, r, 1)),
                utm_n=num(cell(sh, r, 2)),
                loc_err_ft=num(cell(sh, r, 3)),
                lat_dms=cell(sh, r, 4),
                lon_dms=cell(sh, r, 5),
                elev_ft=num(cell(sh, r, 6)),
                permit=str(int(cell(sh, r, 7))),
                drilled=date(*xdate(book, dd)[:3]) if isinstance(dd, float) else None,
                td_ft=num(cell(sh, r, 13)),
                bedrock_ft=cell(sh, r, 14),
                lith=" ".join(
                    str(cell(sh, rr, 9)) for rr in range(r, r + 4) if cell(sh, rr, 9)
                ),
                comments=" ".join(
                    str(cell(sh, rr, 43)) for rr in range(r, r + 5) if cell(sh, rr, 43)
                ),
                kind=str(cell(sh, r + 1, 0) or ""),
            )
    return wells


def to_wgs84(e, n):
    import pyproj
    from pyproj import Transformer

    pyproj.network.set_network_enabled(
        True
    )  # fetch NADCON grid (us_noaa_conus.tif) if not cached
    t = Transformer.from_crs("EPSG:26713", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(e, n)
    desc = t.description
    if "unavailable" in desc:
        desc = "PROJ NAD27->WGS84 via NADCON grid us_noaa_conus.tif (PROJ network)"
    return lat, lon, desc


def sample_terrain(terrain_dir, lats, lons):
    import rasterio
    from pyproj import Transformer

    t = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs, ys = t.transform(lons, lats)
    out = {}
    for name in ["elev48i0100a", "tpi_500m", "tpi_2km", "geomorphons"]:
        p = Path(terrain_dir) / f"{name}.tif"
        if not p.exists():
            continue
        with rasterio.open(p) as src:
            out[name] = [v[0] for v in src.sample(list(zip(xs, ys)))]
    return out


def wp_observations(book, wells):
    sh = book.sheet_by_name("WP Water Level Data")
    obs, cur = [], None
    for r in range(sh.nrows):
        v0 = cell(sh, r, 0)
        if isinstance(v0, str) and re.match(r"^WP\d$", v0.strip()):
            cur = v0.strip()
        d = cell(sh, r, 2)
        if cur and sh.cell_type(r, 2) == xlrd.XL_CELL_DATE:
            btoc = float(cell(sh, r, 1))
            ts = date(*xdate(book, d)[:3]).isoformat()
            flags = []
            if cur == "WP1" and ts == "2004-07-14":
                flags += ["suspect", "recovering"]
            if cur == "WP3" and ts == "2004-07-27":
                flags.append("other:ice_plug_30-42ft_cleared_this_date")
            obs.append(
                dict(
                    well=cur,
                    timestamp=ts,
                    orig_value=btoc,
                    orig_units="ft",
                    orig_reference="below_mp",
                    mp_offset_m=0.0,
                    dtw_bgs_m=btoc * FT,
                    conversion="dtw_bgs_m = BTOC_ft*0.3048; MP offset 0 (wellheads buried within 6 in of ground "
                    "surface; source computes WSE = land-surface elev - BTOC, i.e. treats BTOC as bgs; "
                    "uncertainty <= +/-0.15 m)",
                    flags="|".join(flags),
                    censored="",
                    source_row=f"WP Water Level Data!R{r + 1}",
                    src_wse_ft=num(cell(sh, r, 3)),
                )
            )
    return obs


def hc_observations(book, wells):
    sh = book.sheet_by_name("HC Water Level Data")
    obs, cur = [], None
    for r in range(sh.nrows):
        v0 = cell(sh, r, 0)
        if isinstance(v0, str) and re.match(r"^HC", v0.strip()):
            cur = v0.strip()
        if cur is None or cur == "HCAdit" or sh.cell_type(r, 2) != xlrd.XL_CELL_DATE:
            continue
        d = date(*xdate(book, cell(sh, r, 2))[:3])
        ts = d.isoformat()
        tcell = cell(sh, r, 11)
        note_date = ""
        if sh.cell_type(r, 11) == xlrd.XL_CELL_DATE:
            tt = xdate(book, tcell)
            if tt[0] == 0:
                ts = datetime(d.year, d.month, d.day, tt[3], tt[4]).strftime(
                    "%Y-%m-%dT%H:%M"
                )
            elif date(*tt[:3]) != d:
                note_date = date(*tt[:3]).isoformat()
        static = cell(sh, r, 1)
        head = cell(sh, r, 10)
        flags, censored = [], ""
        if static == "flowing":
            flags.append("flowing")
            if isinstance(head, float):
                ov, ref, dtw = 0.0 - head, "bgs", 0.0 - head * FT
                conv = (
                    f"source artesian head +{head} ft above ground surface (AGS) via PVC riser; stored as "
                    f"orig_value=-{head} ft bgs; dtw_bgs_m = -head_ft*0.3048"
                )
            else:
                ov, ref, dtw, censored = np.nan, "bgs", 0.0, "lt"
                conv = "source reports 'flowing', head not measured; dtw_bgs_m=0 as upper bound (true level above land surface)"
        elif isinstance(static, float):
            ov, ref, dtw = static, "below_mp", static * FT
            conv = (
                "dtw_bgs_m = BTOC_ft*0.3048; MP offset 0 (wellheads buried within 6 in of ground surface; "
                "source WSE = land-surface elev - BTOC)"
            )
            if cell(sh, r, 7) == "yes" or (cur == "HCFW3" and static == 0.0):
                flags.append("flowing")
        elif static == "na" and isinstance(head, float):
            # first-row artesian records where static BTOC is 'na' but head AGS measured (0.0)
            ov, ref, dtw = 0.0 - head, "bgs", 0.0 - head * FT
            conv = f"static BTOC 'na'; measured artesian head {head} ft AGS; dtw_bgs_m = -head_ft*0.3048"
            if cell(sh, r, 7) == "yes":
                flags.append("flowing")
        else:
            continue
        w = wells[cur]
        if w.get("drilled") and (d - w["drilled"]).days < 0:
            flags.append("suspect")
            flags.append("other:date_precedes_drill_date")
        elif w.get("drilled") and (d - w["drilled"]).days <= 7:
            flags.append("other:within_7d_of_drilling")
        if cur in CONFINED:
            flags.append("other:artesian_confined_under_ferricrete")
        if cur == "HCFW5":
            flags.append("other:flow_up_outside_casing")
            wse = num(cell(sh, r, 3))
            if isinstance(head, float) and abs(wse - (w["elev_ft"] + head)) > 1:
                flags.append("other:source_wse_inconsistent")
        if note_date:
            flags.append(f"other:artesian_flow_measured_{note_date}")
        obs.append(
            dict(
                well=cur,
                timestamp=ts,
                orig_value=ov,
                orig_units="ft",
                orig_reference=ref,
                mp_offset_m=0.0 if ref == "below_mp" else np.nan,
                dtw_bgs_m=dtw,
                conversion=conv,
                flags="|".join(dict.fromkeys(flags)),
                censored=censored,
                source_row=f"HC Water Level Data!R{r + 1}",
                src_wse_ft=num(cell(sh, r, 3)),
            )
        )
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--terrain-dir", default="/nas/handily/covariates/terrain")
    a = ap.parse_args()
    raw, out = Path(a.raw_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xls = raw / XLS
    book = xlrd.open_workbook(str(xls))
    src_file = f"raw/{SOURCE_ID}/{XLS}"

    wp, hc = wp_construction(book), hc_construction(book)
    wells = {**wp, **hc}

    obs = wp_observations(book, wp) + hc_observations(book, hc)
    odf = pd.DataFrame(obs)
    odf.insert(0, "source_id", SOURCE_ID)
    odf.insert(1, "well_id", SOURCE_ID + ":" + odf["well"])
    odf.insert(2, "interval_id", odf["well_id"] + "#1")
    odf["source_file"] = src_file
    odf = odf[
        [
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
        ]
    ]
    odf.to_parquet(out / "observations.parquet", index=False)

    lith = parse_lithology(xls)
    lith.to_csv(out / "lithology.csv", index=False)

    names = list(wells)
    lats, lons, desc = [], [], None
    for n in names:
        la, lo, desc = to_wgs84(wells[n]["utm_e"], wells[n]["utm_n"])
        lats.append(la)
        lons.append(lo)
    terr = sample_terrain(a.terrain_dir, lats, lons)

    inv = []
    for i, n in enumerate(names):
        w = wells[n]
        wid = f"{SOURCE_ID}:{n}"
        o = odf[odf["well_id"] == wid]
        top, bot, idesc = INTERVALS_FT[n]
        is_wp = n.startswith("WP")
        if is_wp:
            cls, conf, evid = LANDSCAPE[n]
            setting = "bedrock"
            well_depth = w["open_ft"]
            bore = w["td_ft"]
        else:
            cls, conf = "mountain_valley_bottom", "high"
            evid = f"Sheet: '{w['label']}' ({HC_SITE[n]}). " + HC_LANDSCAPE_EVID
            k = w["kind"].lower()
            setting = "bedrock" if "bedrock well" == k.strip() else "colluvium"
            well_depth = w["td_ft"]
            bore = 161.9 if n == "HCBW4" else w["td_ft"]
        tnote = ""
        if terr:
            g = terr.get("geomorphons", [None] * len(names))[i]
            tnote = (
                f"DEM check (/nas/handily/covariates/terrain, 100 m, EPSG:5070 point sample): "
                f"elev={float(terr['elev48i0100a'][i]):.0f} m, TPI (WhiteboxTools, dimensionless, appears standardized) tpi_500m={float(terr['tpi_500m'][i]):.1f}, "
                f"tpi_2km={float(terr['tpi_2km'][i]):.1f}, geomorphon={GEOMORPHON.get(int(g), g)}"
            )
        coord_conf = "high" if not np.isnan(w["loc_err_ft"]) else "medium"
        coord_note = (
            f"UTM13N NAD27 E={w['utm_e']:.0f} N={w['utm_n']:.0f} (+/-{w['loc_err_ft']} ft); "
            f"sheet DMS NAD27 {w['lat_dms']} / {w['lon_dms']}; converted with '{desc}'"
        )
        if np.isnan(w["loc_err_ft"]):
            coord_note += "; UTM given = cluster centre (sheet: 'See HC location map'); exact offset in embedded sketch map not digitised"
        notes = [
            f"site: {w['label']}",
            coord_note,
            f"interval: {idesc}",
            f"COSE permit {w['permit']}",
        ]
        if is_wp:
            notes.append(
                f"ex-mineral exploration hole drilled {w['year']} by Mineral Systems Inc.; total drilled "
                f"{w['td_ft']:.0f} ft, depth open/accessible {w['open_ft']:.0f} ft; steel left in hole: {w['steel']}; "
                f"single-well test: {w['test']}"
            )
            if w["comments"]:
                notes.append("sheet comments: " + w["comments"])
        else:
            notes.append(
                f"{w['kind']}; drilled {w['drilled']}; depth to bedrock {w['bedrock_ft']} ft"
            )
            if w["comments"]:
                notes.append("sheet comments: " + w["comments"])
            if n in CONFINED:
                notes.append(
                    "CONFINED/artesian: report p.9 ferricrete 'behaves as a confining unit, resulting in "
                    "artesian conditions in the bedrock near the trunk stream'; heads are potentiometric"
                )
        if tnote:
            notes.append(tnote)
        geol = w["lith"]
        if not is_wp:
            geol = f"{geol}; screen in {setting}"
        inv.append(
            dict(
                source_id=SOURCE_ID,
                well_id=wid,
                interval_id=wid + "#1",
                well_name=n,
                site_no=f"COSE {w['permit']}",
                lat=lats[i],
                lon=lons[i],
                horiz_datum="WGS84",
                coord_source="table (UTM 13N NAD27, from 1:24,000 topo + handheld GPS)",
                coord_confidence=coord_conf,
                elev_m=w["elev_ft"] * FT,
                elev_datum="unknown",
                elev_source=f"source table 'mapped elevation' {w['elev_ft']:.0f} ft (1:24,000 topo)",
                mp_height_m=np.nan,
                borehole_depth_m=bore * FT,
                well_depth_m=well_depth * FT,
                screen_top_m=top * FT,
                screen_bot_m=bot * FT,
                geology=geol,
                aquifer_setting=setting,
                state="CO",
                landscape_class=cls,
                landscape_evidence=evid,
                landscape_confidence=conf,
                record_start=o["timestamp"].min()[:10] if len(o) else "",
                record_end=o["timestamp"].max()[:10] if len(o) else "",
                n_obs=len(o),
                source_file=src_file,
                notes=" | ".join(notes),
            )
        )
    pd.DataFrame(inv).to_csv(out / "inventory.csv", index=False)

    files = []
    for fn in ["landing.html", "of06-1189_508.pdf", XLS]:
        p = raw / fn
        files.append(
            dict(
                path=f"raw/{SOURCE_ID}/{fn}",
                url=URLS[fn],
                sha256=sha256(p),
                bytes=p.stat().st_size,
            )
        )
    manifest = dict(
        source_id=SOURCE_ID,
        title=(
            "Well Construction Information, Lithologic Logs, Water Level Data, and Overview of Research in "
            "Handcart Gulch, Colorado: An Alpine Watershed Affected by Metalliferous Hydrothermal Alteration"
        ),
        citation=(
            "Caine, J.S., Manning, A.H., Verplanck, P.L., Bove, D.J., Kahn, K.G., and Ge, S., 2006, U.S. "
            "Geological Survey Open-File Report 2006-1189, 14 p."
        ),
        doi_or_url="https://pubs.usgs.gov/of/2006/1189/",
        landing_url="https://pubs.usgs.gov/of/2006/1189/",
        retrieval_date="2026-10-27",
        release_version="Version 1.0, posted August 2006",
        license="U.S. Government work, public domain (report states permission needed for any copyrighted material)",
        files=files,
        status="acquired",
        unresolved=[
            "WP3/WP4 lithologic-log end depths (3500 vs 1750 ft) are swapped relative to the well-data total depths "
            "(WP3 1750, WP4 3500 ft); lithology labels match the table, depths do not. Not resolved.",
            "Report text says deep wells 'range in depth from 1,200 to 3,500 feet'; table total depths are 1573-3500 ft "
            "(1220 ft is WP1's open/accessible depth).",
            "WP3 and WP4 have no slotted interval; open intervals inferred from casing depths.",
            "HCBW1 slotted interval given as a single number 75.9 (read as length -> 23.7-99.6 ft).",
            "Vertical datum stated as 'North American Vertical Datum of 1927 (NAD 27)' (not a real vertical datum); "
            "elevations are from 1:24,000 topo maps (likely NGVD29) - elev_datum left 'unknown'.",
            "Measurement-point height not tabulated (wellheads buried within 6 in of ground surface); MP offset 0 used, as the source does.",
            "Exact positions of non-centre wells in HC clusters (HCSW2, HCFW3, HCCW3) are only on embedded sketch maps (not digitised).",
        ],
        notes="No logger data; all manual electric-tape readings or artesian heads. DOI not assigned (pre-DOI OFR).",
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        pd.DataFrame(inv)[
            [
                "well_name",
                "lat",
                "lon",
                "elev_m",
                "screen_top_m",
                "screen_bot_m",
                "landscape_class",
                "n_obs",
                "record_start",
                "record_end",
            ]
        ].to_string()
    )
    print(odf.groupby("well_id").size())
    for t in inv:
        print(t["well_name"], t["notes"].split(" | ")[-1])


if __name__ == "__main__":
    main()
