"""Parse CJCZO Marshall Gulch piezometer records (HydroShare a04392de2b3448d48836ad47571a1f48).

Inputs (unchanged downloads written by marshall_gulch_az_fetch.py) under <raw_dir>:
  MG_Piezometer_Groundwater/MG_Piezometer_Groundwater_<yyyy>.csv  30-min "Groundwater Depth" (cm)
  MG_Piezometer_Pressure/MG_Piezometer_Pressure_<yyyy>.csv        30-min absolute pressure + baro (kPa)
  *_metadata.hdr                                                  sensor depth (OffsetValue) + comments
  Catalina_Sites.csv, SCM_MG_PZO.kmz                              coordinates / elevations

The source "Groundwater Depth" is already depth below ground surface:
  GD_cm = sensor_depth_cm - (P_abs - P_baro) * 10.1972   (verified against the pressure files)
dtw_bgs_m = GD_cm / 100 for submerged sensors. Sensors out of water are censored 'gt' at sensor depth.

Writes inventory.csv, observations.parquet, observations_daily.parquet, manifest.json to <out_dir>.

Usage: uv run python -I utils/mountain_wells/marshall_gulch_az_parse.py <raw_dir> <out_dir> <fetch_log.json>
"""

import glob
import json
import math
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SID = "marshall_gulch_az"
RID = "a04392de2b3448d48836ad47571a1f48"
KPA_TO_CM = 10.1972  # cm H2O per kPa (rho_w = 1000 kg/m3, g = 9.80665)
RHO_AIR = 0.95  # kg/m3, ~77 kPa and ~10 C; used only for the documented-elevation bias estimate
WET_THRESH_CM = (
    2.0  # water column above the dry baseline needed to call a sensor submerged
)
ABOVE_GROUND_SUSPECT_CM = (
    -5.0
)  # GD below this (water >5 cm above ground in a soil tube) = artifact
MISMATCH_CM = 3.0  # |GD - GD recomputed from pressure files| above this = suspect
BASELINE_WINDOW = "61D"
BASELINE_Q = 0.25
SEGMENT_GAP = "3D"  # baseline is computed separately on data segments separated by gaps longer than this
CM_FRAC = (
    0.8  # common-mode artefact: >= this fraction of available sensors "wet" at once ...
)
CM_IQR_CM = (
    2.0  # ... with near-identical excess (IQR below max(this, 0.25*median excess))
)

TRANSIT_END = pd.Timestamp(
    "2007-05-04 11:00"
)  # loggers still being carried up (P_abs 92 -> 77 kPa)
PGD2_ADJ_START = pd.Timestamp(
    "2014-10-16 14:00"
)  # hdr: "computed with adjusted pressure (MG_PGD2raw + 0.528)"
PGD2_ADJ_KPA = 0.528
PSD2_ATM = (
    pd.Timestamp("2015-11-08 15:00"),
    pd.Timestamp("2016-02-18 13:30"),
)  # hdr: PSD2 used as ATM;
# bounds = first/last record with MG_ATM missing in the pressure files
PGD1_STEP = pd.Timestamp(
    "2015-11-08 12:00"
)  # undocumented ~-0.45 kPa step in MG_PGD1 pressure vs MG_ATM

BEDROCK = {"PG": "granite", "PS": "schist"}
POSITION = {"C": "convergent", "D": "divergent"}


def parse_hdr(path):
    txt = Path(path).read_text(errors="replace")
    cols = {}
    for m in re.finditer(
        r"SiteCode=([A-Za-z0-9_]+), Method=([^,]+),.*?OffsetValue=([-0-9.]+), OffsetValueUnits=(\w+), "
        r"OffsetDescription=([^.\n]+)",
        txt,
    ):
        cols[m.group(1)] = {
            "method": m.group(2),
            "offset": float(m.group(3)),
            "units": m.group(4),
            "desc": m.group(5).strip(),
        }
    com = re.search(r"COMMENTS\.(.*?)DEFAULT_PARAMETER", txt, re.S)
    tz = re.search(r"UTCOffset=([-0-9]+), TimeZone=(\w+)", txt)
    return cols, (com.group(1).strip() if com else ""), (tz.groups() if tz else None)


def load_series(raw, kind):
    frames = []
    for f in sorted(
        glob.glob(str(raw / f"MG_Piezometer_{kind}" / f"MG_Piezometer_{kind}_*.csv"))
    ):
        df = pd.read_csv(f, skiprows=[1, 2], header=0)
        df = df.rename(columns={"Sites": "dt_str"})
        df["source_file"] = f"raw/{SID}/MG_Piezometer_{kind}/{Path(f).name}"
        df["source_row"] = (
            np.arange(len(df)) + 4
        )  # 1-based line number in the CSV (3 header rows)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["dt"] = pd.to_datetime(df["dt_str"], format="%m/%d/%Y %H:%M")
    assert not df["dt"].duplicated().any(), kind
    return df.set_index("dt").sort_index()


def kmz_points(path):
    with zipfile.ZipFile(path) as z:
        kml = z.read([n for n in z.namelist() if n.endswith(".kml")][0]).decode(
            "utf-8", "replace"
        )
    pts = {}
    for m in re.finditer(r"<Placemark.*?</Placemark>", kml, re.S):
        p = m.group(0)
        code = re.search(r"SITECODE</td>\s*<td>([^<]+)<", p) or re.search(
            r"SITECODE\s*</td>\s*<td>\s*([^<\s]+)", p
        )
        c = re.search(r"<coordinates>\s*([-0-9.]+),([-0-9.]+)", p)
        if code and c:
            pts[code.group(1).strip()] = (float(c.group(2)), float(c.group(1)))
    return pts


def dist_m(a, b):
    dy = (a[0] - b[0]) * 111320.0
    dx = (a[1] - b[1]) * 111320.0 * math.cos(math.radians(a[0]))
    return math.hypot(dx, dy)


def main(raw_dir, out_dir, fetch_log):
    raw, out = Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- metadata: sensor depths must be identical across all yearly headers
    depth = {}
    comments = set()
    tzs = set()
    for h in sorted(glob.glob(str(raw / "MG_Piezometer_*" / "*_metadata.hdr"))):
        cols, com, tz = parse_hdr(h)
        comments.update(line.strip() for line in com.splitlines() if line.strip())
        tzs.add(tz)
        for s, c in cols.items():
            if s.startswith("MG_P") and c["method"] in ("PZO_U20", "PZO_U20_GWD"):
                assert (
                    c["desc"] == "Depth below the ground surface" and c["units"] == "cm"
                ), (h, s)
                assert depth.get(s, c["offset"]) == c["offset"], (h, s)
                depth[s] = c["offset"]
    assert tzs == {("-7", "MST")}, tzs
    sites = sorted(depth, key=lambda s: (s[:4], s))

    st = pd.read_csv(raw / "Catalina_Sites.csv").set_index("SiteCode")
    atm_elev = float(st.loc["MG_ATM", "Elevation_m"])
    kmz = kmz_points(raw / "SCM_MG_PZO.kmz")
    flume = {"granite": kmz.get("GR_FLUME"), "schist": kmz.get("SC_FLUME")}
    seep = kmz.get("S_SEEP")

    g = load_series(raw, "Groundwater")
    p = load_series(raw, "Pressure").replace(-9999, np.nan)

    # ---- pass 1: per-site series, QC recompute, invalid masks, first dry baseline
    per = {}
    for s in sites:
        d_cm = depth[s]
        gd = g[s].replace(-9999, np.nan)
        keep = gd.notna()
        df = pd.DataFrame(
            {
                "gd": gd[keep],
                "source_file": g.loc[keep, "source_file"],
                "source_row": g.loc[keep, "source_row"],
            }
        )
        pj = p.reindex(df.index)
        sens, atm = pj[s], pj["MG_ATM"]
        adj = np.zeros(len(df))
        if s == "MG_PGD2":
            # hdr: GD computed with MG_PGD2raw + 0.528 kPa since 10/16/2014 14:00; keep it only where it
            # reproduces the published GD (it ends at the 2016-05 sensor gap)
            cand = np.where(df.index >= PGD2_ADJ_START, PGD2_ADJ_KPA, 0.0)
            e_adj = (df["gd"] - (d_cm - (sens + cand - atm) * KPA_TO_CM)).abs()
            e_raw = (df["gd"] - (d_cm - (sens - atm) * KPA_TO_CM)).abs()
            adj = np.where(e_adj < e_raw, cand, 0.0)
        rec = d_cm - (sens + adj - atm) * KPA_TO_CM
        df["mism"] = ((df["gd"] - rec).abs() > MISMATCH_CM).to_numpy()
        df["adj_on"] = adj > 0
        df["transit"] = df.index < TRANSIT_END
        df["psd2_ref"] = (df.index >= PSD2_ATM[0]) & (df.index <= PSD2_ATM[1])
        df["h_app"] = d_cm - df["gd"]
        df["cm_art"] = False
        per[s] = df

    def baseline(df):
        ok = ~(
            df["transit"]
            | df["psd2_ref"]
            | df["cm_art"]
            | (df["gd"] < ABOVE_GROUND_SUSPECT_CM)
        )
        h = df.loc[ok, "h_app"]
        seg = (h.index.to_series().diff() > pd.Timedelta(SEGMENT_GAP)).cumsum()
        parts = []
        for _, hs in h.groupby(seg.values):
            reg = hs.resample("30min").asfreq()
            b = reg.rolling(BASELINE_WINDOW, center=True, min_periods=48).quantile(
                BASELINE_Q
            )
            parts.append(b.reindex(hs.index).fillna(hs.quantile(BASELINE_Q)))
        b = pd.concat(parts).reindex(df.index)
        return b.ffill().bfill(), seg.reindex(df.index).ffill().bfill()

    for s in sites:
        per[s]["base"], per[s]["segment"] = baseline(per[s])

    # ---- common-mode barometer artefacts: (almost) all sensors "rise" by the same amount at once
    exc = pd.DataFrame({s: per[s]["h_app"] - per[s]["base"] for s in sites})
    for s in sites:
        bad = per[s]["transit"] | per[s]["psd2_ref"]
        exc.loc[bad[bad].index, s] = np.nan
    n_av = exc.notna().sum(axis=1)
    frac = (exc > WET_THRESH_CM).sum(axis=1) / n_av.clip(lower=1)
    med = exc.median(axis=1)
    iqr = exc.quantile(0.75, axis=1) - exc.quantile(0.25, axis=1)
    cm = (n_av >= 8) & (frac >= CM_FRAC) & (iqr < np.maximum(CM_IQR_CM, 0.25 * med))
    cm_times = cm[cm].index
    for s in sites:
        per[s]["cm_art"] = per[s].index.isin(cm_times)
        per[s]["base"], per[s]["segment"] = baseline(per[s])

    obs_parts, inv_rows, daily_parts = [], [], []
    for s in sites:
        bed = BEDROCK[s[3:5]]
        pos = POSITION[s[5]]
        d_cm = depth[s]
        df = per[s]
        base = df["base"]
        h_true = df["h_app"] - base
        wet = h_true > WET_THRESH_CM
        dtw_c = (df["gd"] + base) / 100.0  # = (sensor_depth - h_true)/100
        transit, psd2_ref, cm_art = (
            df["transit"].to_numpy(),
            df["psd2_ref"].to_numpy(),
            df["cm_art"].to_numpy(),
        )
        above = (wet & (dtw_c < 0)).to_numpy()
        above_bad = (wet & (dtw_c < ABOVE_GROUND_SUSPECT_CM / 100.0)).to_numpy()
        mism, adj_on = df["mism"].to_numpy(), df["adj_on"].to_numpy()
        wet = wet.to_numpy()

        dtw = np.where(wet, dtw_c, d_cm / 100.0)
        cens = np.where(wet, "", "gt").astype(object)
        mp_off = np.where(wet, base / 100.0, np.nan)
        conv = np.where(
            wet,
            "dtw_bgs_m = (GroundwaterDepth_cm + b_cm)/100; b = site dry baseline of apparent water column "
            f"(rolling {BASELINE_WINDOW} q{BASELINE_Q:g}, = baro-elevation/sensor offset), stored in mp_offset_m",
            f"censored gt: sensor out of water (water column <= {WET_THRESH_CM:g} cm above dry baseline); "
            f"value = documented sensor depth {d_cm / 100:.3f} m",
        ).astype(object)
        bad = transit | psd2_ref | cm_art | above_bad
        flags = [["provisional"] for _ in range(len(df))]
        for i in range(len(df)):
            f = flags[i]
            if transit[i]:
                f += ["suspect", "other:logger_in_transit_pre_install"]
                continue
            if psd2_ref[i]:
                f += ["suspect", "other:baro_reference_was_in_tube_sensor_MG_PSD2"]
                continue
            if cm_art[i]:
                f += ["suspect", "other:common_mode_barometer_artifact"]
                continue
            f.append("other:dry_baseline_corrected" if wet[i] else "dry_censored")
            if above[i]:
                f.append("other:water_above_ground_surface")
            if above_bad[i]:
                f.append("suspect")
            if mism[i]:
                f += ["suspect", "other:gd_vs_pressure_file_mismatch_gt3cm"]
            if adj_on[i]:
                f += ["suspect", "other:source_pressure_adjusted_plus_0.528kPa"]
        segs = df["segment"].to_numpy()
        for i in np.flatnonzero(np.r_[False, segs[1:] != segs[:-1]]):
            flags[i].append("other:new_baseline_segment_after_gap")
        if s == "MG_PGD2":
            on = np.flatnonzero(adj_on)
            if len(on):
                flags[on[0]].append("instrument_change")
                nxt = np.flatnonzero(np.arange(len(df)) > on[-1])
                if len(nxt):
                    flags[nxt[0]].append("instrument_change")
        if s == "MG_PGD1":
            i = np.searchsorted(df.index.values, np.datetime64(PGD1_STEP))
            if i < len(df):
                flags[i].append("instrument_change")
        dtw = np.where(bad, np.nan, dtw)
        cens = np.where(bad, "", cens)
        mp_off = np.where(bad, np.nan, mp_off)
        conv = np.where(
            transit,
            "not_defensible: logger in transit before installation (P_abs 92->77 kPa)",
            conv,
        )
        conv = np.where(
            psd2_ref,
            "not_defensible: barometric reference replaced by in-tube sensor MG_PSD2 "
            "(11/8/2015-2/18/2016), its wetness unknown",
            conv,
        )
        conv = np.where(
            cm_art & ~transit & ~psd2_ref,
            "not_defensible: synchronous equal rise on >=80% of sensors = barometer artefact",
            conv,
        )
        conv = np.where(
            above_bad & ~transit & ~psd2_ref & ~cm_art,
            "not_defensible: water >5 cm above ground in soil tube (installation/flooding artifact)",
            conv,
        )
        fl = np.array(["|".join(dict.fromkeys(f)) for f in flags], dtype=object)

        well_id = f"{SID}:{s}"
        iid = f"{well_id}#1"
        ts = df.index.strftime("%Y-%m-%dT%H:%M") + "-07:00"
        o = pd.DataFrame(
            {
                "source_id": SID,
                "well_id": well_id,
                "interval_id": iid,
                "timestamp": ts,
                "orig_value": df["gd"].to_numpy(),
                "orig_units": "cm",
                "orig_reference": "bgs",
                "mp_offset_m": mp_off,
                "dtw_bgs_m": dtw,
                "conversion": conv,
                "flags": fl,
                "censored": cens,
                "source_file": df["source_file"].to_numpy(),
                "source_row": df["source_row"].astype(str).to_numpy(),
            }
        )
        obs_parts.append(o)

        # daily median (MST calendar day) over records with a usable value
        u = o[o["dtw_bgs_m"].notna()].copy()
        u["date"] = u["timestamp"].str[:10]
        u["is_c"] = u["censored"] == "gt"
        u["is_s"] = u["flags"].str.contains("suspect")
        agg = u.groupby("date").agg(
            dtw_bgs_m=("dtw_bgs_m", "median"),
            n_obs=("dtw_bgs_m", "size"),
            n_censored=("is_c", "sum"),
            n_suspect=("is_s", "sum"),
        )
        agg["censored"] = np.where(agg["n_censored"] * 2 >= agg["n_obs"], "gt", "")
        agg.loc[agg["censored"] == "gt", "dtw_bgs_m"] = d_cm / 100.0
        agg = agg.reset_index()
        agg.insert(0, "interval_id", iid)
        agg.insert(0, "well_id", well_id)
        agg.insert(0, "source_id", SID)
        daily_parts.append(agg)

        # inventory
        r = st.loc[s]
        lat, lon = float(r["Latitude"]), float(r["Longitude"])
        kp = kmz.get(s)
        kdiff = dist_m((lat, lon), kp) if kp else float("nan")
        z_sensor = float(r["Elevation_m"]) - d_cm / 100.0
        bias_doc = (
            RHO_AIR / 1000.0 * 100.0 * (z_sensor - atm_elev)
        )  # cm by which GD reads too deep
        dry_base = float(base[~wet].median()) if (~wet).any() else float("nan")
        near = []
        if flume[bed]:
            near.append(
                f"{dist_m((lat, lon), flume[bed]):.0f} m from the {bed} ZOB flume (KMZ)"
            )
        if bed == "schist" and seep:
            near.append(
                f"{dist_m((lat, lon), seep):.0f} m from the schist seep V-notch (KMZ S_SEEP)"
            )
        valid = o[o["dtw_bgs_m"].notna()]
        n_wet = int((valid["censored"] == "").sum())
        evidence = (
            f"Source site name '{r['SiteName'].strip()}' ({pos} position within a zero-order basin). CZO archive: "
            "'Two V shaped ZOBs (schist (SC) and granite (GC)) lie within the Marshall Gulch catchment. Both ZOBs "
            "are north oriented with relatively steep slopes (3.2 - 56.4 degrees)'; 'Elevation ranges from 2311 m "
            "to 2476 m for SC basin ... and from 2379 m to 2516 m for GC basin'. Site elev "
            f"{float(r['Elevation_m']):.1f} m; "
            + "; ".join(near)
            + ". Source does not call any site ridgetop or alpine."
        )
        low_outlet = (bed == "schist" and float(r["Elevation_m"]) <= 2320) or (
            bed == "granite" and float(r["Elevation_m"]) <= 2390
        )
        inv_rows.append(
            {
                "source_id": SID,
                "well_id": well_id,
                "interval_id": iid,
                "well_name": r["SiteName"].strip(),
                "site_no": s,
                "lat": lat,
                "lon": lon,
                "horiz_datum": "NAD83",
                "coord_source": "table (Catalina_Sites.csv); cross-checked vs SCM_MG_PZO.kmz",
                "coord_confidence": "high",
                "elev_m": float(r["Elevation_m"]),
                "elev_datum": "unknown",
                "elev_source": "Catalina_Sites.csv (VerticalDatum='MSL')",
                "mp_height_m": np.nan,
                "borehole_depth_m": np.nan,
                "well_depth_m": d_cm / 100.0,
                "screen_top_m": np.nan,
                "screen_bot_m": np.nan,
                "geology": f"soil over {bed} bedrock; sensor at soil-bedrock interface",
                "aquifer_setting": "soil-bedrock interface",
                "state": "AZ",
                "landscape_class": "mountain_slope",
                "landscape_evidence": evidence,
                "landscape_confidence": "medium" if low_outlet else "high",
                "record_start": o["timestamp"].iloc[0][:10],
                "record_end": o["timestamp"].iloc[-1][:10],
                "n_obs": len(o),
                "source_file": f"raw/{SID}/MG_Piezometer_Groundwater/MG_Piezometer_Groundwater_<yyyy>.csv",
                "notes": (
                    f"Bedrock {bed}; hillslope position {pos}. Sensor (HOBO U20, method PZO_U20) suspended at the "
                    f"soil-bedrock interface in a vertical plastic tube; sensor depth {d_cm:.1f} cm bgs (hdr OffsetValue), "
                    "used as well_depth_m; tube/borehole depth and screen not documented. "
                    f"n_obs={len(o)}, usable={len(valid)}, submerged(exact)={n_wet}, censored_dry={int((valid['censored'] == 'gt').sum())}, "
                    f"set NaN={int(o['dtw_bgs_m'].isna().sum())}. "
                    f"Baro reference MG_ATM at {atm_elev:.1f} m; documented-elevation bias estimate: source depth reads "
                    f"~{bias_doc:+.1f} cm too deep (rho_air {RHO_AIR}); empirical dry baseline of apparent water column "
                    f"{dry_base:+.1f} cm (negative = source depth reads deep); submerged dtw_bgs_m = source depth + baseline. "
                    f"KMZ vs table position difference {kdiff:.1f} m. NAD83 used as WGS84-equivalent (<~1 m)."
                ),
            }
        )

    obs = pd.concat(obs_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    inv = pd.DataFrame(inv_rows)
    obs.to_parquet(out / "observations.parquet", index=False)
    daily.to_parquet(out / "observations_daily.parquet", index=False)
    inv.to_csv(out / "inventory.csv", index=False)

    log = json.loads(Path(fetch_log).read_text())
    sysmeta = json.loads((raw / "sysmeta.json").read_text())
    manifest = {
        "source_id": SID,
        "title": sysmeta["resource_title"],
        "citation": "Troch, P., I. Heidbuechel, N. Abramson (2022). CJCZO -- Piezometer, Groundwater Depth -- "
        "Marshall Gulch -- (2007-2019), HydroShare, "
        "http://www.hydroshare.org/resource/a04392de2b3448d48836ad47571a1f48. Acknowledgment requested: "
        "'Logistical support and/or data were provided by the NSF-supported Jemez River Basin and Santa "
        "Catalina Mountains Critical Zone Observatory, EAR-0724958 and EAR-1331408.'",
        "doi_or_url": "http://www.hydroshare.org/resource/a04392de2b3448d48836ad47571a1f48 (no DOI; unpublished "
        "HydroShare resource, sysmeta doi=null, published=false)",
        "landing_url": f"https://www.hydroshare.org/resource/{RID}/",
        "retrieval_date": "2026-10-27",
        "release_version": f"HydroShare date_last_updated {sysmeta['date_last_updated']}; files modified 2022-07-05; "
        "data described as provisional",
        "license": "CC BY (Creative Commons Attribution), per scimeta rights statement",
        "files": [
            {k: f[k] for k in ("path", "url", "sha256", "bytes")} for f in log["files"]
        ],
        "status": "acquired",
        "unresolved": [
            "Linked methods PDF http://www.czo.arizona.edu/czo_data/MethodsSites/catalinas/GroundwaterTableMeasurementsMG.pdf "
            "not retrievable (host connection timeout; no Wayback snapshot).",
            "Tube/borehole depth, perforated interval and soil depth not documented; sensor depth used as well depth.",
            "Barometer (MG_ATM, 2340.8 m) sits 30 m above to 85 m below the sensors; the source depths are not "
            "corrected for this (bias up to ~10 cm). dtw_bgs_m applies an empirical dry-baseline correction "
            "(mp_offset_m); the documented-elevation estimate agrees within ~1-3 cm (inventory notes).",
            "MG_PGD1 pressure steps ~-0.45 kPa on 2015-11-08 12:00 (undocumented; flagged instrument_change).",
            "All sites 2015-11-08..2016-02-18 set NaN: baro reference was in-tube sensor MG_PSD2.",
            "MG_PGD2 sensor drift 2014-10-16..2016-05 (source applied a constant +0.528 kPa; baseline still drifts).",
            "Undocumented dry-baseline step at MG_PGC1 2016-2017 (~ -3 cm).",
            "Vertical datum given only as 'MSL'.",
        ],
        "notes": "15 groundwater-depth series (7 granite + 8 schist). The abstract's '15 locations (7 granite, 9 schist)' "
        "counts the schist seep V-notch logger S_SEEP (pressure only, 2007-05..2010-05, a surface-water "
        "reservoir sensor, method PZO_U20S) as a 9th schist location; it has no groundwater-depth series and "
        "is excluded. Hdr comments: "
        + " / ".join(sorted(c for c in comments if "@" not in c)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        inv[["well_id", "record_start", "record_end", "n_obs"]].to_string(index=False)
    )
    print("obs", len(obs), "daily", len(daily))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
