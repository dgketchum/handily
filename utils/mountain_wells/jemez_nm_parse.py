"""Parse CJCZO Jemez groundwater (HydroShare) into the mountain_wells shared schema.

Sources (raw files fetched by jemez_nm_fetch.py, unchanged):
  jemez_deep_nm : da61f854a1754e26a7c65de608bf5890  eight 2016-drilled wells, 2013 Burned (Mixed Conifer) ZOB
  jemez_piez_nm : 8d43148dba4c45fe8f65530d29aab7ae  Burn_Piez1-3, 2011 Burned ZOB
                  36419d60a7484844acab038b1fb84828  MC_Piez1-3 (SW_Piez1-3), 2013 Burned ZOB

Writes inventory.csv, observations.parquet, observations_daily.parquet, manifest.json under
<parsed_root>/<source_id>/ (NOTES.md is written by hand). With --dem, samples the two 10 m 3DEP
clips for a local landscape-position check.

Usage:
  uv run python -I utils/mountain_wells/jemez_nm_parse.py <raw_root> <parsed_root> \
      [--dem <mczob_dem.tif> <burn2011_dem.tif>]
"""

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEEP = "jemez_deep_nm"
PIEZ = "jemez_piez_nm"
R_DEEP = "da61f854a1754e26a7c65de608bf5890"
R_B11 = "8d43148dba4c45fe8f65530d29aab7ae"
R_MC = "36419d60a7484844acab038b1fb84828"
TZ = "-07:00"  # all files: UTCOffset=-7, TimeZone=MST (no DST)

DRY_TOL_M = 0.03  # water column <= 3 cm above sensor -> treated as sensor out of water (dry at sensor)
NEAR_TOL_M = 0.10  # 3-10 cm water column -> flagged near_sensor_floor
EXCURSION_M = 0.30  # |value - centred 1-day rolling median| -> transient excursion flag
INSTALL_END = pd.Timestamp(
    "2016-07-15"
)  # wells drilled June 2016; first ~3 weeks equilibrating

# White et al. (2019) HESS 23:4661, Table S1 (completed well dimensions, m below ground surface)
TABLE_S1 = {
    "1A": dict(casing_cm=2.54, screen=(38.7, 41.7), sensor="VWP @ 41.5", total=41.7),
    "1B": dict(casing_cm=2.54, screen=(31.7, 37.8), sensor="T @ 37.8", total=37.8),
    "2A": dict(casing_cm=5.08, screen=(42.8, 45.9), sensor="T @ 45.5", total=47.15),
    "2B": dict(casing_cm=5.08, screen=(34.7, 37.8), sensor="T @ 37.7", total=39.1),
    "2C": dict(casing_cm=5.08, screen=(24.7, 30.8), sensor="T @ 30.1", total=30.85),
    "2D": dict(casing_cm=5.08, screen=(3.7, 6.7), sensor="VWP @ 6.4", total=6.7),
    "3A": dict(casing_cm=2.54, screen=(13.8, 16.9), sensor="VWP @ 15.3", total=16.85),
    "3B": dict(casing_cm=5.08, screen=(6.8, 12.9), sensor="VWP @ 12.7", total=12.93),
}
ABSTRACT_DEPTH = {
    "1A": 42,
    "1B": 38,
    "2A": 46,
    "2B": 38,
    "2C": 31,
    "2D": 7,
    "3A": 18,
    "3B": 13,
}
DEEP_POS = {
    "1A": ("site 1", "SW-facing slope (east side of ZOB, upslope of convergent zone)"),
    "1B": ("site 1", "SW-facing slope (east side of ZOB, upslope of convergent zone)"),
    "2A": ("site 2", "SE-facing slope (west side of ZOB)"),
    "2B": ("site 2", "SE-facing slope (west side of ZOB)"),
    "2C": ("site 2", "SE-facing slope (west side of ZOB)"),
    "2D": ("site 2", "SE-facing slope (west side of ZOB)"),
    "3A": (
        "site 3",
        "above ZOB outlet, convergent zone (possible concealed fault zone)",
    ),
    "3B": (
        "site 3",
        "above ZOB outlet, convergent zone (possible concealed fault zone)",
    ),
}
DEEP_GEOL = {
    "site 1": "highly fractured welded tuff (max fracture density ~5000 m-2), Redondo Peak resurgent dome; "
    "White et al. 2019",
    "site 2": "highly weathered volcanic (caldera collapse) breccia in top 15 m over more-consolidated, "
    "less-fractured welded tuff (~1000 m-2); White et al. 2019",
    "site 3": "tuff in/next to a concealed fault zone in the ZOB convergent outlet; clay up to 50% below "
    "1.5 m; White et al. 2019",
}

PIEZ_META = {
    # site_code: (resource, file, csv column, source alias, position label, well depth m, logger)
    "MC_Piez1": (
        R_MC,
        "MCZOB_shallowWell1.csv",
        "GroundwaterDepth_SW1",
        "SW_Piez1",
        "low elevation",
        2.36,
        "non-vented In-Situ Level TROLL 300, baro-corrected with MC ZOB weather station",
    ),
    "MC_Piez2": (
        R_MC,
        "MCZOB_shallowWell2.csv",
        "GroundwaterDepth_SW2",
        "SW_Piez2",
        "middle elevation",
        2.34,
        "non-vented In-Situ Level TROLL 300, baro-corrected with MC ZOB weather station",
    ),
    "MC_Piez3": (
        R_MC,
        "MCZOB_shallowWell3.csv",
        "GroundwaterDepth_SW3",
        "SW_Piez3",
        "high elevation",
        3.63,
        "vented In-Situ Level TROLL 500",
    ),
    "Burn_Piez1": (
        R_B11,
        "BurnShallowWell1_highest.csv",
        "GroundwaterDepth",
        "Burn_Piez1",
        "highest elevation",
        3.2,
        "non-vented In-Situ Level TROLL 300, baro-corrected with 2011 Burned ZOB lowest weather station",
    ),
    "Burn_Piez2": (
        R_B11,
        "BurnShallowWell2_middle.csv",
        "GroundwaterDepth",
        "Burn_Piez2",
        "middle elevation",
        1.75,
        "vented In-Situ Level TROLL 500",
    ),
    "Burn_Piez3": (
        R_B11,
        "BurnShallowWell3_lowest.csv",
        "GroundwaterDepth",
        "Burn_Piez3",
        "lowest elevation",
        2.34,
        "vented In-Situ Level TROLL 500",
    ),
}

OBS_COLS = [
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


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_hdr(path):
    """Return {col_index: dict(var, units, site, offset)} from a CZO .hdr file (COLn lines)."""
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        m = re.match(
            r"COL(\d+)\.\s*label=VariableName,\s*Value=([^,]+),\s*Units=([^,]+),(.*)",
            line,
        )
        if not m:
            continue
        rest = m.group(4)
        site = re.search(r"SiteCode=(\S+)", rest)
        off = re.search(r"OffsetValue=([\d.]+)", rest)
        out[int(m.group(1))] = dict(
            var=m.group(2).strip(),
            units=m.group(3).strip(),
            site=site.group(1).rstrip(",") if site else None,
            offset=float(off.group(1)) if off else np.nan,
        )
    return out


def read_czo_csv(path):
    df = pd.read_csv(path, skiprows=[1], dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if c and not c.startswith("Unnamed")]]
    units = (
        pd.read_csv(path, skiprows=1, nrows=1, header=None, dtype=str).iloc[0].tolist()
    )
    t = pd.to_datetime(df.iloc[:, 0].str.strip(), format="%m/%d/%Y %H:%M")
    df.index = t
    df["_row"] = np.arange(len(df)) + 3  # 1-based file line: header, units, then data
    return df, [str(u).strip() for u in units]


def add_flag(fl, mask, flag):
    fl[mask] = [f"{a}|{flag}" if a else flag for a in fl[mask]]
    return fl


def excursion_mask(t, v, window="1D"):
    s = pd.Series(v, index=t)
    med = s.rolling(window, center=True, min_periods=1).median()
    return (s - med).abs().to_numpy() > EXCURSION_M


def build_series(
    source_id,
    well_id,
    interval_id,
    t,
    vals_m,
    orig_vals,
    orig_units,
    sensor_m,
    source_file,
    rows,
):
    """Common flagging. vals_m = depth below ground in m as reported (NaN removed already)."""
    n = len(t)
    fl = np.array(["provisional"] * n, dtype=object)
    cens = np.array([""] * n, dtype=object)
    dtw = vals_m.copy()
    conv = np.array(
        [
            "dtw_bgs_m = orig_value (source variable 'Groundwater Depth', m below surface)"
        ]
        * n,
        dtype=object,
    )
    if orig_units == "cm":
        conv[:] = (
            "dtw_bgs_m = orig_value / 100 (source variable 'Groundwater Depth', cm below surface)"
        )
    col = sensor_m - vals_m
    dry = col <= DRY_TOL_M
    near = (col > DRY_TOL_M) & (col <= NEAR_TOL_M)
    fl = add_flag(fl, dry, "dry_censored")
    cens[dry] = "gt"
    dtw[dry] = sensor_m[dry]
    conv[dry] = [
        f"censored gt: reading within {DRY_TOL_M} m of (or below) sensor depth {s:.2f} m; "
        f"water at/below sensor, dtw_bgs_m set to sensor depth"
        for s in sensor_m[dry]
    ]
    fl = add_flag(fl, near, "other:near_sensor_floor")
    exc = excursion_mask(t, vals_m) & ~dry
    fl = add_flag(fl, exc, "other:transient_excursion")
    return pd.DataFrame(
        dict(
            source_id=source_id,
            well_id=well_id,
            interval_id=interval_id,
            _t=t,
            orig_value=orig_vals,
            orig_units=orig_units,
            orig_reference="bgs",
            mp_offset_m=np.nan,
            dtw_bgs_m=dtw,
            conversion=conv,
            flags=fl,
            censored=cens,
            source_file=source_file,
            source_row=rows,
            _sensor=sensor_m,
        )
    )


# ----------------------------------------------------------------------------------------------- deep
def parse_deep(raw_root, log):
    rdir = Path(raw_root) / DEEP / R_DEEP
    pieces = []
    seen = {}  # (well, timestamp) -> value from VW files, for the 2020/2021 overlap
    for y in range(2016, 2022):
        f = rdir / f"groundwaterDepth_piezometers_{y}.csv"
        hdr = parse_hdr(rdir / f"groundwaterDepth_piezometers_metadata_{y}.hdr")
        df, _ = read_czo_csv(f)
        for i, c in enumerate(df.columns[:-1]):
            if not c.startswith("GroundwaterDepth_"):
                continue
            h = hdr[i + 1]
            w = c.replace("GroundwaterDepth_Well", "")
            assert h["site"] == f"MC_Well{w}" and h["var"] == "Groundwater Depth", (
                f,
                c,
                h,
            )
            v = pd.to_numeric(df[c]).replace(-9999, np.nan)
            ok = v.notna().to_numpy()
            t = df.index[ok]
            vv = v[ok].to_numpy()
            rows = df["_row"].to_numpy()[ok]
            keep = np.ones(len(t), bool)
            ndup, nmis = 0, 0
            for k, (tt, x) in enumerate(zip(t, vv)):
                key = (w, tt)
                if key in seen:
                    keep[k] = False
                    ndup += 1
                    nmis += int(abs(seen[key] - x) > 1e-9)
                else:
                    seen[key] = x
            if ndup:
                log.append(
                    f"{f.name} {c}: dropped {ndup} timestamps already in an earlier yearly file "
                    f"({nmis} with differing values)"
                )
            pieces.append(
                dict(
                    w=w,
                    t=t[keep],
                    v=vv[keep],
                    rows=rows[keep],
                    sensor=h["offset"],
                    file=f.name,
                    inst="Geokon 4500 vibrating-wire piezometer, 15 min",
                )
            )
    for tf in ["groundwaterDepth_troll_well1", "groundwaterDepth_troll_well2"]:
        hdr = parse_hdr(rdir / f"{tf}.hdr")
        df, _ = read_czo_csv(rdir / f"{tf}.csv")
        for i, c in enumerate(df.columns[:-1]):
            if not c.startswith("GroundwaterDepth_"):
                continue
            w = c.replace("GroundwaterDepth_Well", "")
            # troll_well1.hdr lists COLs in a different order than the CSV -> match by site + variable
            hs = [
                h
                for h in hdr.values()
                if h["site"] == f"MC_Well{w}" and h["var"] == "Groundwater Depth"
            ]
            assert len(hs) == 1, (tf, c, hdr)
            h = hs[0]
            if hdr.get(i + 1) is not h:
                log.append(
                    f"{tf}.hdr: COL order differs from CSV for {c}; matched by SiteCode+variable"
                )
            v = pd.to_numeric(df[c]).replace(-9999, np.nan)
            ok = v.notna().to_numpy()
            inst = (
                "In-Situ Level TROLL 400 (non-vented), 6 h"
                if tf.endswith("1")
                else "In-Situ Rugged TROLL 100 (non-vented), 6 h"
            )
            pieces.append(
                dict(
                    w=w,
                    t=df.index[ok],
                    v=v[ok].to_numpy(),
                    rows=df["_row"].to_numpy()[ok],
                    sensor=h["offset"],
                    file=f"{tf}.csv",
                    inst=inst,
                )
            )
    out = []
    seg_info = {}
    for w in TABLE_S1:
        segs = [p for p in pieces if p["w"] == w and len(p["t"])]
        if not segs:
            continue
        well_id = f"{DEEP}:MC_Well{w}"
        frames = []
        for p in segs:
            sf = f"raw/{DEEP}/{R_DEEP}/{p['file']}"
            fr = build_series(
                DEEP,
                well_id,
                f"{well_id}#1",
                p["t"],
                p["v"].astype(float),
                p["v"].astype(float),
                "m",
                np.full(len(p["t"]), p["sensor"]),
                sf,
                p["rows"].astype(str),
            )
            fr["_inst"] = p["inst"]
            frames.append(fr)
        d = pd.concat(frames).sort_values("_t").reset_index(drop=True)
        # instrument change: first 24 h after the logger type changes
        inst = d["_inst"].to_numpy()
        chg = np.where(inst[1:] != inst[:-1])[0] + 1
        m = np.zeros(len(d), bool)
        for k in chg:
            m |= (d["_t"] >= d["_t"].iloc[k]) & (
                d["_t"] < d["_t"].iloc[k] + pd.Timedelta("1D")
            )
        d["flags"] = add_flag(d["flags"].to_numpy(), m, "instrument_change")
        m = (d["_t"] < INSTALL_END).to_numpy() & (d["censored"] == "").to_numpy()
        d["flags"] = add_flag(
            d["flags"].to_numpy(), m, "other:post_installation_equilibration"
        )
        if w == "2B":
            m = (d["_inst"].str.startswith("Geokon") & (d["censored"] == "")).to_numpy()
            d["flags"] = add_flag(
                d["flags"].to_numpy(),
                m,
                "suspect|other:post_drilling_drainage_to_floor",
            )
        if w == "2D":
            d["flags"] = add_flag(
                d["flags"].to_numpy(), np.ones(len(d), bool), "perched"
            )
        if w in ("3A", "3B"):
            m = (d["censored"] == "").to_numpy()
            d["flags"] = add_flag(
                d["flags"].to_numpy(),
                m,
                "suspect|other:site3_reported_no_water_White2019",
            )
        seg_info[w] = [
            (p["inst"], p["file"], p["sensor"], p["t"].min(), p["t"].max(), len(p["t"]))
            for p in segs
        ]
        out.append(d)
    return pd.concat(out).reset_index(drop=True), seg_info


# ----------------------------------------------------------------------------------------------- piez
def parse_piez(raw_root, log):
    out = []
    for code, (rid, fname, col, alias, pos, depth, logger) in PIEZ_META.items():
        f = Path(raw_root) / PIEZ / rid / fname
        hdr = parse_hdr(f.with_name(fname.replace(".csv", "_metadata.hdr")))
        df, units = read_czo_csv(f)
        ci = list(df.columns).index(col)
        hs = [h for h in hdr.values() if h["var"] == "Groundwater Depth"]
        assert len(hs) == 1 and abs(hs[0]["offset"] - depth) < 1e-6, (fname, hdr)
        h = hs[0]
        if hdr.get(ci + 1) is not h:
            log.append(
                f"{fname}: .hdr COL order differs from CSV column order; matched by variable name"
            )
        csv_units = units[ci]
        assert csv_units == "cm", (fname, csv_units)
        if h["units"] != csv_units:
            log.append(
                f"{fname}: .hdr says Units={h['units']} but CSV units row says {csv_units}; values "
                f"(e.g. max {pd.to_numeric(df[col]).max():.1f}) only fit cm -> cm used"
            )
        v = pd.to_numeric(df[col]).replace(-9999, np.nan)
        ok = v.notna().to_numpy()
        t = df.index[ok]
        vv = v[ok].to_numpy().astype(float)
        well_id = f"{PIEZ}:{code}"
        sf = f"raw/{PIEZ}/{rid}/{fname}"
        d = build_series(
            PIEZ,
            well_id,
            f"{well_id}#1",
            t,
            vv / 100.0,
            vv,
            "cm",
            np.full(len(t), depth),
            sf,
            df["_row"].to_numpy()[ok].astype(str),
        )
        if code == "MC_Piez1":
            z = (d["orig_value"] == 0).to_numpy()
            d.loc[z, "censored"] = "lt"
            d.loc[z, "dtw_bgs_m"] = 0.0
            d.loc[z, "conversion"] = (
                "censored lt: value 0 cm (no negatives in record) = water at or above ground "
                "surface, likely clipped at 0"
            )
            d["flags"] = add_flag(
                d["flags"].to_numpy(), z, "other:at_or_above_surface_clipped_0"
            )
            post = (d["_t"] > pd.Timestamp("2014-08-01")).to_numpy()
            d["flags"] = add_flag(
                d["flags"].to_numpy(), post, "other:post_gap_level_shift_unverified"
            )
        out.append(d)
    return pd.concat(out).reset_index(drop=True)


# ----------------------------------------------------------------------------------------------- daily
def daily(obs):
    o = obs.copy()
    o["date"] = o["_t"].dt.strftime("%Y-%m-%d")
    o["susp"] = o["flags"].str.contains("suspect")
    rows = []
    for (wid, iid, date), g in o.groupby(["well_id", "interval_id", "date"], sort=True):
        clean = g[(g.censored == "") & ~g.susp]
        cg = g[(g.censored != "") & ~g.susp]
        sus = g[g.susp & (g.censored == "")]
        if len(clean):
            val, cen, basis = clean.dtw_bgs_m.median(), "", "median_uncensored"
            if len(cg) > len(clean):
                cen = cg.censored.mode().iloc[0]
                val = cg.dtw_bgs_m.max() if cen == "gt" else cg.dtw_bgs_m.min()
                basis = "majority_censored"
        elif len(cg):
            cen = cg.censored.mode().iloc[0]
            val = cg.dtw_bgs_m.max() if cen == "gt" else cg.dtw_bgs_m.min()
            basis = "all_censored"
        elif len(sus):
            val, cen, basis = sus.dtw_bgs_m.median(), "", "median_suspect_only"
        else:
            cg = g[g.censored != ""]
            cen = cg.censored.mode().iloc[0]
            val = cg.dtw_bgs_m.max() if cen == "gt" else cg.dtw_bgs_m.min()
            basis = "all_censored_suspect"
        fl = sorted({x for s in g["flags"] for x in s.split("|") if x})
        rows.append(
            dict(
                source_id=g.source_id.iloc[0],
                well_id=wid,
                interval_id=iid,
                date=date,
                dtw_bgs_m=val,
                censored=cen,
                basis=basis,
                n_obs=len(g),
                n_uncensored_clean=len(clean),
                n_censored=int((g.censored != "").sum()),
                n_suspect=int(g.susp.sum()),
                dtw_min_m=g.dtw_bgs_m.min(),
                dtw_max_m=g.dtw_bgs_m.max(),
                flags="|".join(fl),
            )
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- DEM
def dem_metrics(dem_path, x, y):
    import rasterio
    from scipy import ndimage

    with rasterio.open(dem_path) as r:
        z = r.read(1).astype(float)
        tr = r.transform
    res = tr.a
    zs = ndimage.uniform_filter(z, 3)
    gy, gx = np.gradient(zs, -res, res)  # row index increases southward
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = (
        np.degrees(np.arctan2(-gx, -gy)) + 360
    ) % 360  # direction of steepest descent, deg from N

    col = int((x - tr.c) / res)
    row = int((tr.f - y) / res)
    out = dict(
        dem_elev_m=z[row, col], slope_deg=slope[row, col], aspect_deg=aspect[row, col]
    )
    for rad in (100, 300):
        k = int(rad / res)
        win = z[row - k : row + k + 1, col - k : col + k + 1]
        yy, xx = np.mgrid[-k : k + 1, -k : k + 1]
        fp = (xx**2 + yy**2) <= k * k
        vals = win[fp]
        out[f"tpi_{rad}m"] = z[row, col] - vals.mean()
        out[f"elev_pct_{rad}m"] = (vals < z[row, col]).mean() * 100
    k = int(500 / res)
    win = z[row - k : row + k + 1, col - k : col + k + 1]
    out["relief_above_500m"] = win.max() - z[row, col]
    out["relief_below_500m"] = z[row, col] - win.min()
    return out


# ----------------------------------------------------------------------------------------------- main
def manifest_files(raw_root, sid, urls):
    files = []
    for p in sorted((Path(raw_root) / sid).rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(Path(raw_root).parent))
            files.append(
                dict(
                    path=rel,
                    url=urls.get(rel, ""),
                    sha256=sha256(p),
                    bytes=p.stat().st_size,
                )
            )
    return files


def url_map(raw_root, sid, rids):
    m = {}
    for rid in rids:
        base = f"raw/{sid}/{rid}/"
        m[base + "sysmeta.json"] = (
            f"https://www.hydroshare.org/hsapi/resource/{rid}/sysmeta/"
        )
        m[base + "scimeta.xml"] = (
            f"https://www.hydroshare.org/hsapi/resource/{rid}/scimeta/"
        )
        m[base + "files.json"] = (
            f"https://www.hydroshare.org/hsapi/resource/{rid}/files/"
        )
        m[base + "landing.html"] = f"https://www.hydroshare.org/resource/{rid}/"
        listing = json.loads((Path(raw_root) / sid / rid / "files.json").read_text())
        for f in listing["results"]:
            rel = f["url"].split("/data/contents/", 1)[1]
            m[base + rel] = (
                f"https://www.hydroshare.org/hsapi/resource/{rid}/files/{rel}/"
            )
    for name in ["hess-23-4661-2019.pdf", "hess-23-4661-2019-supplement.pdf"]:
        m[f"raw/{sid}/white_etal_2019_hess/{name}"] = (
            f"https://hess.copernicus.org/articles/23/4661/2019/{name}"
        )
    return m


def finish_obs(obs):
    obs = obs.sort_values(["interval_id", "_t"]).reset_index(drop=True)
    obs["timestamp"] = obs["_t"].dt.strftime("%Y-%m-%dT%H:%M") + TZ
    return obs


def main(raw_root, parsed_root, dems=None):
    raw_root, parsed_root = Path(raw_root), Path(parsed_root)
    sites = pd.read_csv(raw_root / DEEP / R_DEEP / "Jemez_Sites.csv").set_index(
        "SiteCode"
    )
    log = []

    # ---------------- deep
    deep, seg = parse_deep(raw_root, log)
    deep = finish_obs(deep)
    inv = []
    for w, segs in seg.items():
        code = f"MC_Well{w}"
        s = sites.loc[code]
        g = deep[deep.well_id == f"{DEEP}:{code}"]
        ts = TABLE_S1[w]
        site, pos = DEEP_POS[w]
        segtxt = "; ".join(
            f"{a} [{f}] sensor {sd} m bgs, {t0:%Y-%m-%d}..{t1:%Y-%m-%d} n={n}"
            for a, f, sd, t0, t1, n in segs
        )
        unc = g[(g.censored == "") & ~g["flags"].str.contains("suspect")]
        notes = (
            f"Position: {pos} ({site}). SiteName in Jemez_Sites.csv = '{s.SiteName}'"
            + (
                " (3A/3B names swapped in SiteName; SiteCode + sensor depth match Table S1, SiteCode used)"
                if w in ("3A", "3B")
                else ""
            )
            + f". Casing {ts['casing_cm']} cm, Table S1 sensor {ts['sensor']} m bgs; abstract depth "
            f"{ABSTRACT_DEPTH[w]} m vs Table S1 total depth {ts['total']} m. Segments: {segtxt}. "
            f"n censored(gt, at/below sensor)={int((g.censored == 'gt').sum())}, "
            f"uncensored non-suspect dtw range {unc.dtw_bgs_m.min():.2f}-{unc.dtw_bgs_m.max():.2f} m. "
            "Horizontal datum NAD83 used as WGS84 (<~1 m)."
        )
        if w in ("3A", "3B"):
            notes += " White et al. 2019: 'site 3 wells did not produce water' during their study period."
        if w == "2D":
            notes += (
                " White et al. 2019: 2D 'is likely a perched aquifer' in collapse breccia, separated from deeper "
                "groundwater by a suspected low-permeability confining layer (aquitard)."
            )
        if w in ("2A", "2B", "2C"):
            notes += (
                " White et al. 2019: deeper site 2 wells lie below a 'suspected confining layer' (acting 'more "
                "of an aquitard than aquiclude') beneath the perched 2D aquifer -> head may be semi-confined."
            )
        if w == "1A":
            notes += " Median level (~37.3 m) is ~1.4 m above the screen top (38.7 m)."
        inv.append(
            dict(
                source_id=DEEP,
                well_id=f"{DEEP}:{code}",
                interval_id=f"{DEEP}:{code}#1",
                well_name=f"Well {w}",
                site_no=code,
                lat=s.Latitude,
                lon=s.Longitude,
                horiz_datum="NAD83",
                coord_source="table (Jemez_Sites.csv in HydroShare resource)",
                coord_confidence="high",
                elev_m=float(s.Elevation_m),
                elev_datum=s.VerticalDatum,
                elev_source="Jemez_Sites.csv",
                mp_height_m=np.nan,
                borehole_depth_m=ts["total"],
                well_depth_m=ts["total"],
                screen_top_m=ts["screen"][0],
                screen_bot_m=ts["screen"][1],
                geology=DEEP_GEOL[site],
                aquifer_setting="perched" if w == "2D" else "bedrock",
                state="NM",
                landscape_class="mountain_valley_bottom"
                if site == "site 3"
                else "mountain_slope",
                landscape_evidence="",
                landscape_confidence="",
                record_start=g._t.min().strftime("%Y-%m-%d"),
                record_end=g._t.max().strftime("%Y-%m-%d"),
                n_obs=len(g),
                source_file="|".join(sorted(g.source_file.unique())),
                notes=notes,
                _x=s.LocalX,
                _y=s.LocalY,
                _dem=0,
            )
        )
    inv_deep = pd.DataFrame(inv)

    # ---------------- piez
    piez = finish_obs(parse_piez(raw_root, log))
    inv = []
    for code, (rid, fname, col, alias, pos, depth, logger) in PIEZ_META.items():
        s = sites.loc[code]
        g = piez[piez.well_id == f"{PIEZ}:{code}"]
        unc = g[g.censored == ""]
        zob = (
            "2013 Burned ZOB (formerly Mixed Conifer ZOB)"
            if code.startswith("MC")
            else "2011 Burned ZOB"
        )
        notes = (
            f"{zob} shallow well, {pos}; abstract name {alias}, file {fname}, SiteName '{s.SiteName}'. "
            f"Logger: {logger}; sensor at {depth} m bgs (= stated well depth). "
            f"n censored gt={int((g.censored == 'gt').sum())}, lt={int((g.censored == 'lt').sum())}; "
            f"uncensored dtw range {unc.dtw_bgs_m.min():.2f}-{unc.dtw_bgs_m.max():.2f} m. "
            "Screen interval not reported. Horizontal datum NAD83 used as WGS84."
        )
        inv.append(
            dict(
                source_id=PIEZ,
                well_id=f"{PIEZ}:{code}",
                interval_id=f"{PIEZ}:{code}#1",
                well_name=alias,
                site_no=code,
                lat=s.Latitude,
                lon=s.Longitude,
                horiz_datum="NAD83",
                coord_source="table (Jemez_Sites.csv in HydroShare resource)",
                coord_confidence="high",
                elev_m=float(s.Elevation_m),
                elev_datum=s.VerticalDatum,
                elev_source="Jemez_Sites.csv",
                mp_height_m=np.nan,
                borehole_depth_m=np.nan,
                well_depth_m=depth,
                screen_top_m=np.nan,
                screen_bot_m=np.nan,
                geology=(
                    "not stated in piezometer resource; ZOB on Redondo Peak resurgent dome (Bandelier Tuff, "
                    "rhyolite, collapse breccia; White et al. 2019)"
                )
                if code.startswith("MC")
                else "not stated in source (Valles Caldera / Jemez Mountains volcanic terrain)",
                aquifer_setting="unknown",
                state="NM",
                landscape_class="",
                landscape_evidence="",
                landscape_confidence="",
                record_start=g._t.min().strftime("%Y-%m-%d"),
                record_end=g._t.max().strftime("%Y-%m-%d"),
                n_obs=len(g),
                source_file=f"raw/{PIEZ}/{rid}/{fname}",
                notes=notes,
                _x=s.LocalX,
                _y=s.LocalY,
                _dem=0 if code.startswith("MC") else 1,
            )
        )
    inv_piez = pd.DataFrame(inv)

    # ---------------- landscape (source first; DEM numbers appended as supporting evidence)
    for inv_df in (inv_deep, inv_piez):
        metrics = []
        for _, r in inv_df.iterrows():
            metrics.append(dem_metrics(dems[r._dem], r._x, r._y) if dems else {})
        inv_df["_m"] = metrics

    def demtxt(m):
        if not m:
            return ""
        return (
            f" DEM check (USGS 3DEP 10 m, UTM13N clip, at source LocalX/LocalY): elev {m['dem_elev_m']:.0f} m, "
            f"slope {m['slope_deg']:.0f} deg, aspect {m['aspect_deg']:.0f} deg, TPI100 {m['tpi_100m']:+.1f} m, "
            f"TPI300 {m['tpi_300m']:+.1f} m, elev percentile within 300 m {m['elev_pct_300m']:.0f}%, "
            f"terrain within 500 m rises {m['relief_above_500m']:.0f} m above / falls "
            f"{m['relief_below_500m']:.0f} m below the site."
        )

    for i, r in inv_deep.iterrows():
        w = r.well_name.split()[-1]
        site, pos = DEEP_POS[w]
        if site == "site 3":
            ev = (
                "Source abstract: wells 3A and 3B are 'above the zero order basin (ZOB) outlet'; White et al. "
                "2019: site 3 wells are 'in the convergent zone', which 'just above the ZOB flume, is "
                "characterized by boggy land'. Not ridgetop. Convergent hollow at ZOB outlet -> nearest class "
                "mountain_valley_bottom."
            )
            conf = "medium"
        else:
            ev = (
                f"Source abstract: well on the {'south-west' if site == 'site 1' else 'south-east'} facing slope "
                "of the ZOB; White et al. 2019: 'The ZOB consists of SE- and SW-facing slopes that are separated "
                "by a convergent zone' on the eastern side of Redondo Peak. Source does not call the site a "
                "ridge or divide; not ridgetop."
            )
            conf = "high"
        inv_deep.loc[i, "landscape_evidence"] = ev + demtxt(r._m)
        inv_deep.loc[i, "landscape_confidence"] = conf
    for i, r in inv_piez.iterrows():
        code = r.site_no
        pos = PIEZ_META[code][4]
        if code == "MC_Piez1":
            cls, conf = "mountain_valley_bottom", "medium"
            ev = (
                "Source: 2013 Burned ZOB shallow well at the 'low elevation' position. Coordinates/elevation "
                "(2991 m) coincide with wells 3A/3B 'above the ZOB outlet' in the boggy convergent zone (White "
                "et al. 2019); record shows water at the surface in spring. Not ridgetop."
            )
        elif code.startswith("MC"):
            cls, conf = "mountain_slope", "low"
            ev = (
                f"Source: 2013 Burned ZOB shallow well at the '{pos}' position only; no slope/axis statement. "
                "Coordinates lie on the N-S line between the SW-facing (site 1) and SE-facing (site 2) slopes, "
                "i.e. probably the ZOB convergent axis upslope of the outlet (unchanneled headwater hollow on "
                "the Redondo Peak flank). Not ridgetop; class could equally be a hollow/valley-axis position."
            )
        else:
            cls, conf = "mountain_slope", "low"
            ev = (
                f"Source: 2011 Burned ZOB shallow well at the '{pos}' position (low/middle/highest elevation "
                "within a zero-order basin); no statement of ridge, slope or axis. Not ridgetop per source."
            )
        inv_piez.loc[i, "landscape_class"] = cls
        inv_piez.loc[i, "landscape_evidence"] = ev + demtxt(r._m)
        inv_piez.loc[i, "landscape_confidence"] = conf

    # ---------------- write
    meta = {
        DEEP: dict(rids=[R_DEEP], obs=deep, inv=inv_deep),
        PIEZ: dict(rids=[R_B11, R_MC], obs=piez, inv=inv_piez),
    }
    for sid, d in meta.items():
        out = parsed_root / sid
        out.mkdir(parents=True, exist_ok=True)
        inv_df = d["inv"]
        dem_cols = pd.DataFrame(list(inv_df["_m"])) if dems else pd.DataFrame()
        inv_df.drop(columns=[c for c in inv_df.columns if c.startswith("_")]).to_csv(
            out / "inventory.csv", index=False
        )
        if dems:
            dem_cols.insert(0, "interval_id", inv_df.interval_id.values)
            dem_cols.round(2).to_csv(out / "dem_check_3dep10m.csv", index=False)
        obs = d["obs"]
        obs[OBS_COLS].to_parquet(out / "observations.parquet", index=False)
        daily(obs).to_parquet(out / "observations_daily.parquet", index=False)
        sysm = [
            json.loads((raw_root / sid / rid / "sysmeta.json").read_text())
            for rid in d["rids"]
        ]
        cites = []
        for rid in d["rids"]:
            x = (raw_root / sid / rid / "scimeta.xml").read_text()
            m = re.search(
                r"<dcterms:bibliographicCitation>(.*?)</dcterms:bibliographicCitation>",
                x,
                re.S,
            )
            cites.append(m.group(1).strip() if m else "")
        man = dict(
            source_id=sid,
            title=" | ".join(s["resource_title"] for s in sysm),
            citation=" | ".join(cites)
            + (
                " ; construction/geology: White, A., et al. (2019) Distinct stores and the routing of water in the "
                "deep critical zone of a snow-dominated volcanic catchment. HESS 23, 4661-4683, "
                "doi:10.5194/hess-23-4661-2019"
                if sid == DEEP
                else ""
            ),
            doi_or_url=" | ".join(s["resource_url"] for s in sysm)
            + " (no DOI minted; resources not formally published)",
            landing_url=" | ".join(
                f"https://www.hydroshare.org/resource/{rid}/" for rid in d["rids"]
            ),
            retrieval_date="2026-10-27",
            release_version=" | ".join(
                f"{s['resource_id']}: date_last_updated {s['date_last_updated']}, "
                f"published={s['published']}, immutable={s['immutable']}"
                for s in sysm
            ),
            license="CC BY 4.0 (HydroShare rights statement); CZO acknowledgment requested (NSF EAR-1331408)",
            files=manifest_files(raw_root, sid, url_map(raw_root, sid, d["rids"])),
            status="acquired",
            unresolved=[],
            notes="",
        )
        man_path = out / "manifest.json"
        if man_path.exists():  # keep hand-written unresolved/notes
            old = json.loads(man_path.read_text())
            man["unresolved"] = old.get("unresolved", [])
            man["notes"] = old.get("notes", "")
        man_path.write_text(json.dumps(man, indent=2))
        print(sid, "intervals", len(inv_df), "obs", len(obs))
    for line in log:
        print("LOG:", line)


if __name__ == "__main__":
    dems = None
    if "--dem" in sys.argv:
        k = sys.argv.index("--dem")
        dems = sys.argv[k + 1 : k + 3]
    main(sys.argv[1], sys.argv[2], dems)
