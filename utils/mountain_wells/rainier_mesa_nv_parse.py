"""Parse USGS DS 190 (Rainier Mesa, NTS water levels 1957-2005) plus post-2005 NWIS extension.

Usage (repo root):
  uv run --with access-parser --with xlrd python -I utils/mountain_wells/rainier_mesa_nv_parse.py \
      <raw_dir> <out_dir> <fetch_log_json>

Inputs (all under <raw_dir>, unchanged): appendixes.zip (Access MDB + XLS), table1.html,
nwis/field_measurements_*.csv, nwis/monitoring_locations_*.csv, nnss_explorer/site_data/*.json.
The zip is extracted to a temporary directory outside <raw_dir>.

Outputs (<out_dir>): inventory.csv, observations.parquet, manifest.json, parse_stats.json.
NOTES.md is written separately by hand from parse_stats.json.
"""

import glob
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC = "rainier_mesa_nv"
FT = 0.3048
MDB = "ds_190.app1_WL_database.mdb"
XLS = "ds_190.app1_WL_flags.xls"
TERRAIN = "/nas/handily/covariates/terrain/"

# Landscape class per borehole/site (source-first; DEM numbers appended at run time).
LANDSCAPE = {
    "Hagestad_1": (
        "mesa_plateau_top",
        "high",
        "DS190: 'Rainier Mesa is a volcanic plateau at an altitude of about 7,500 ft'; LS 7,485 ft; "
        "well drained by the U12e.07 drift beneath the mesa",
    ),
    "ER-12-3": (
        "mesa_plateau_top",
        "high",
        "DS190: regional water table 'as great as 3,100 ft on top of the mesa' (ER-12-3 main DTW ~3,112-3,116 ft); "
        "LS 7,385 ft on the ~7,500 ft plateau",
    ),
    "U-12q": (
        "mesa_plateau_top",
        "high",
        "LS 7,413 ft on the ~7,500 ft Rainier Mesa plateau (DS190 text, fig. 1)",
    ),
    "UE-12n_15A": (
        "mesa_plateau_top",
        "high",
        "LS 7,369 ft on the ~7,500 ft Rainier Mesa plateau (DS190 text, fig. 1)",
    ),
    "U-12e.06-1_R/C": (
        "mesa_plateau_top",
        "high",
        "LS 7,573 ft on the ~7,500 ft Rainier Mesa plateau (DS190 text, fig. 1)",
    ),
    "U-12g.06_PS_1V": (
        "mesa_plateau_top",
        "high",
        "LS 7,626 ft on the ~7,500 ft Rainier Mesa plateau; post-shot hole (DS190 text, fig. 1)",
    ),
    "U-12e.M1_UG": (
        "mesa_plateau_top",
        "medium",
        "DS190 narrative: 'spudded in the floor of the main drift of the Rainier Mesa U12e tunnel complex, about 1,370 ft "
        "below land surface'; depths are referenced to the mesa-top land surface (LS 7,539 ft) but the wellhead is in the tunnel",
    ),
    "U-12e.03-1": (
        "mesa_plateau_top",
        "medium",
        "DS190 narrative: 'spudded under Rainier Mesa in the floor of the U12e.03 personnel tunnel, about 1,395 ft below "
        "land surface'; depths referenced to mesa-top land surface (LS 7,545 ft), wellhead in tunnel",
    ),
    "ER-12-4": (
        "mesa_plateau_top",
        "low",
        "Aqueduct Mesa (NE lobe of Rainier Mesa, DS190 fig. 1 label) at LS 6,883 ft; position on the mesa top "
        "inferred from figure + DEM, not stated in text",
    ),
    "U-12t.04_CH_1": (
        "mesa_plateau_top",
        "low",
        "Aqueduct Mesa area (DS190 fig. 1) at LS 6,796 ft; drilled near U-12t tunnel; mesa-top position inferred from DEM",
    ),
    "UE-12t_6": (
        "mesa_plateau_top",
        "low",
        "Aqueduct Mesa area (DS190 fig. 1) at LS 6,907 ft; DS190 narrative: 'shallow, perched system underlying a "
        "recharge area'; mesa-top position inferred from DEM",
    ),
    "UE-12t_7": (
        "mesa_plateau_top",
        "low",
        "Aqueduct Mesa area (DS190 fig. 1) at LS 6,961 ft; mesa-top position inferred from DEM",
    ),
    "U-12s": (
        "mountain_valley_bottom",
        "low",
        "Gold Meadows (DS190 fig. 1 label; narrative 'granite of the Gold Meadows stock ... perched system underlying a "
        "recharge area'), an upland basin N of Rainier Mesa at LS 6,794 ft; DEM hollow/negative TPI",
    ),
    "Dolomite_Hill_Hole": (
        "uncertain",
        "low",
        "Named 'Dolomite Hill' (Table 1 hole name) at LS 6,399 ft east of the mesa; source gives no landscape position; "
        "DEM suggests a local ridge",
    ),
    "ER-12-1": (
        "mountain_valley_bottom",
        "low",
        "LS 5,817 ft, ~1,700 ft below the Rainier Mesa plateau to the east (off the mesa); DS190 text: regional WT depth "
        "'about 1,400 ft off of the mesa'; DEM valley form",
    ),
    "ER-19-1": (
        "mountain_valley_bottom",
        "low",
        "LS 6,140 ft west of Rainier Mesa, off the mesa; DEM valley form (Stockade Wash area, DS190 fig. 1)",
    ),
    "TW-1": (
        "mountain_valley_bottom",
        "medium",
        "Stockade Wash (DS190 fig. 1 label) SW of Rainier Mesa at LS 6,156 ft; DEM valley form",
    ),
    "WW-8": (
        "mountain_valley_bottom",
        "low",
        "Water-supply well at LS 5,695 ft W of Rainier Mesa (Ammonia Tanks quadrangle, hydrographic area 227B); "
        "DEM valley/pit form; source gives no landscape position",
    ),
}
TUNNEL_LANDSCAPE = (
    "uncertain",
    "low",
    "Impounded water inside a sealed tunnel drift beneath Rainier/Aqueduct Mesa; LS altitude is a map value "
    "(+/-20 ft) for ground above the drift; not a natural water table",
)

# DS190 attribute -> schema flag
DET_FLAG = {
    "Suspected perched water": "perched",
    "Pumping/recovery": "recovering",
    "Injection/recovery": "recovering",
    "Equilibration": "recovering",
    "Questionable accuracy": "suspect",
    "Erratic/Unstable": "suspect",
    "Maximum estimate": "estimated",
    "Minimum estimate": "estimated",
    "Packer test": "other:packer_test",
    "Testing area": "other:testing_area",
    "Tunnel effect": "other:tunnel_effect",
    "Well-construction effect": "other:well_construction",
    "Suspected unsaturated": "other:suspected_unsaturated",
    "Obstruction": "other:obstruction",
    "Dry": "dry_censored",
    "Undeveloped": "other:undeveloped",
    "Anomalous - high": "suspect",
    "Anomalous - low": "suspect",
}
GEN_FLAG = {
    "Suspect": ["suspect"],
    "Nonstatic level": ["other:nonstatic"],
    "Steady state--LOCAL": ["perched"],
    "Transient--LOCAL": ["perched", "other:transient"],
    "Transient--REGIONAL": ["other:transient"],
    "Localized conditions": ["other:localized"],
}
STATUS_FLAG = {
    "Recently pumped": "recovering",
    "Injector site": "other:injector_site",
    "Pumping": "pumping",
}
METH_EST = {"Reported", "Geophysical log", "Pressure-gage", "Unknown"}
# NNSS explorer detCond -> flag (post-2005 NWIS rows)
EXPL_DET = {
    "Perched/semi-perched": "perched",
    "Pumping effect": "pumping",
    "Pumping recovery": "recovering",
    "Injection recovery": "recovering",
    "Equilibration": "recovering",
    "Questionable accuracy": "suspect",
    "Erratic/Unstable": "suspect",
    "Testing area": "other:testing_area",
    "Tunnel effect": "other:tunnel_effect",
    "Well-construction effect": "other:well_construction",
    "Packer test": "other:packer_test",
}
# aquifer-setting overrides from DS190 narratives
SETTING_OVERRIDE = {
    "371122116122202": "perched",  # U-12e.03-1 (682 ft): 'likely reflects perched water in fractures'
    "371028116123002": "unknown",  # U-12g.06 PS 1V: dry at 1,472 ft; no water found
}
# NWIS-only intervals -> DS190 interval of the same zone (screen/geology borrowed)
NEW_PARENT = {
    "371043116142104": "371043116142102",
    "371043116142105": "371043116142103",
}
EXPL_GEN = {
    "Suspect": ["suspect"],
    "Nonstatic": ["other:nonstatic"],
    "Transient": ["other:transient"],
}


def clean_name(n):
    n = re.sub(r"\s+", " ", n.strip())
    n = re.sub(r"^(TW|WW|U|UE)\s*-\s*", r"\1-", n)
    return n


def borehole_of(name, site):
    if site.startswith("9999"):
        return clean_name(name).replace(" ", "_")
    n = clean_name(name)
    n = re.sub(r"\s*\(.*\)\s*\w*$", "", n)
    n = re.sub(r" (main|piezometer)$", "", n)
    n = re.sub(r"^ER-19-1-\d.*", "ER-19-1", n)
    return n.replace(" ", "_")


def flags_join(fl):
    out = []
    for f in fl:
        if f and f not in out:
            out.append(f)
    return "|".join(out)


def read_mdb(path):
    from access_parser import AccessParser

    db = AccessParser(str(path))
    tabs = {}
    for t in [
        "RM_sitefile",
        "RM_gw_lev",
        "RM_wl_attrib_gen",
        "RM_wl_attrib_det",
        "RM_wl_rmks",
        "RM_gw_mpnt",
        "RM_gw_cons",
        "RM_gw_open",
        "RM_gw_aqfr",
        "RM_hyd_narrative",
    ]:
        df = pd.DataFrame(db.parse_table(t))
        if "USGS_site_id" in df:
            df["USGS_site_id"] = df["USGS_site_id"].astype(str).str.strip()
        tabs[t] = df
    return tabs


def dkey(ts):
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")


def main(raw_dir, out_dir, log_path):
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = {}
    tmp = tempfile.TemporaryDirectory()
    with zipfile.ZipFile(raw / "appendixes.zip") as z:
        for m in z.namelist():
            if Path(m).name in (MDB, XLS):
                tgt = Path(tmp.name) / Path(m).name
                tgt.write_bytes(z.read(m))
    T = read_mdb(Path(tmp.name) / MDB)
    site = T["RM_sitefile"].copy()
    lev = T["RM_gw_lev"].copy()
    lev["lev_dtm"] = pd.to_datetime(lev["lev_dtm"])
    lev["k"] = lev["lev_dtm"].map(dkey)
    stats["ds190_sites"] = int(len(site))
    stats["ds190_level_rows"] = int(len(lev))

    # XLS cross-check
    try:
        x = pd.read_excel(
            Path(tmp.name) / XLS, sheet_name="Water-level conditions", header=27
        )
        x = x[x["USGS Site ID"].notna()]
        x["sid"] = (
            x["USGS Site ID"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )
        stats["xls_rows"] = int(len(x))
        xd = x.groupby("sid").size()
        md = lev.groupby("USGS_site_id").size()
        stats["xls_mdb_per_site_count_mismatch"] = {
            s: [int(md.get(s, 0)), int(xd.get(s, 0))]
            for s in set(md.index) | set(xd.index)
            if md.get(s, 0) != xd.get(s, 0)
        }
        xv = pd.to_numeric(x["WL depth (ft)"], errors="coerce").sum()
        stats["xls_minus_mdb_sum_depth_ft"] = round(float(xv - lev["lev_va"].sum()), 3)
    except Exception as e:  # cross-check only
        stats["xls_crosscheck_error"] = repr(e)

    # attributes keyed by site + datetime
    gen = T["RM_wl_attrib_gen"].copy()
    gen["k"] = pd.to_datetime(gen["lev_dtm"]).map(dkey)
    gen_map = gen.groupby(["USGS_site_id", "k"])["wl_attrib_gen"].apply(list).to_dict()
    det = T["RM_wl_attrib_det"].copy()
    det["k"] = pd.to_datetime(det["lev_dtm"]).map(dkey)
    det_map = det.groupby(["USGS_site_id", "k"])["wl_attrib_det"].apply(list).to_dict()
    rmk = T["RM_wl_rmks"].copy()
    rmk["k"] = pd.to_datetime(rmk["wl_rmks_dtm"]).map(dkey)
    rmk_map = (
        rmk.groupby(["USGS_site_id", "k"])["wl_rmks_tx"]
        .apply(lambda s: " / ".join(map(str, s)))
        .to_dict()
    )
    unknown_det = (
        set(det["wl_attrib_det"])
        - set(DET_FLAG)
        - {
            "Elevated",
            "Rising trend",
            "Declining trend",
            "Consistent",
            "Uncertain",
            "Limited data",
            "Abrupt change",
        }
    )
    stats["det_attributes_not_mapped_to_flags"] = sorted(unknown_det)

    site["bh"] = [
        borehole_of(n, s) for n, s in zip(site.USGS_local_nm, site.USGS_site_id)
    ]
    site["tunnel"] = site.USGS_site_id.str.startswith("9999")

    # Table 1 open intervals
    t1 = pd.read_html(raw / "table1.html")[0]
    t1.columns = [
        "name",
        "site_id",
        "nts_hole",
        "lat",
        "lon",
        "ls_alt_ft",
        "depth_drilled_ft",
        "top_ft",
        "bot_ft",
        "units",
    ]
    t1["site_id"] = (
        t1["site_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    )
    t1 = t1[t1.site_id.str.fullmatch(r"\d{15}")]
    t1 = t1.set_index("site_id")
    for c in ["top_ft", "bot_ft", "depth_drilled_ft"]:
        t1[c] = pd.to_numeric(t1[c].astype(str).str.replace(",", ""), errors="coerce")

    cons = T["RM_gw_cons"].set_index("USGS_site_id")
    aq = T["RM_gw_aqfr"].set_index("USGS_site_id")
    nar = T["RM_hyd_narrative"].set_index("USGS_site_id")["hyd_narrative"]
    mp = T["RM_gw_mpnt"].copy()

    # ---------------- DS190 observations ----------------
    rows = []
    for i, r in lev.reset_index(drop=True).iterrows():
        sid = r.USGS_site_id
        key = (sid, r.k)
        g = [a for a in gen_map.get(key, []) if isinstance(a, str) and a]
        d = det_map.get(key, [])
        fl = []
        for a in g:
            fl += GEN_FLAG.get(a, [])
        for a in d:
            fl.append(DET_FLAG.get(a, ""))
        st = r.lev_status
        if st in STATUS_FLAG:
            fl.append(STATUS_FLAG[st])
        if r.lev_meth in METH_EST:
            fl.append("estimated")
        if sid.startswith("9999"):
            fl += ["other:tunnel_impounded_water", "perched"]
        if sid[:13] in ("3711061161230", "3711221161222"):
            fl.append("other:in_tunnel_hole")
        rm = rmk_map.get(key, "")
        val = r.lev_va
        cen = ""
        dtw = np.nan
        conv = "dtw_bgs_m = orig_value * 0.3048 (DS190 lev_va is depth below land surface, ft)"
        if pd.isna(val):
            if st == "Dry":
                m = re.search(r"No fluid @ ([\d,]+)", rm)
                val = float(m.group(1).replace(",", "")) if m else np.nan
                cen = "gt"
                fl.append("dry_censored")
                conv = "dry: orig_value = depth sounded without fluid (from remark), ft; dtw_bgs_m = orig_value*0.3048 as lower bound"
                dtw = val * FT
            else:
                fl.append("missing")
                if st == "Obstruction":
                    fl.append("other:obstruction")
                conv = "not_defensible: no water level recorded (" + str(st) + ")"
        else:
            dtw = val * FT
            if "Maximum estimate" in d:
                cen = "gt"  # max water-level altitude -> DTW at least this deep
            elif "Minimum estimate" in d:
                cen = "lt"
        if "greater than" in rm.lower() and not cen:
            cen = "gt"
        ts = r.lev_dtm
        tstr = (
            ts.strftime("%Y-%m-%d")
            if (ts.hour == 0 and ts.minute == 0)
            else ts.strftime("%Y-%m-%dT%H:%M")
        )
        if g or d or rm:
            note = "; ".join(
                filter(
                    None,
                    [
                        "gen=" + ",".join(g) if g else "",
                        "det=" + ",".join(d) if d else "",
                        rm,
                    ],
                )
            )

        else:
            note = ""
        rows.append(
            dict(
                site_no=sid,
                timestamp=tstr,
                orig_value=float(val) if pd.notna(val) else np.nan,
                orig_units="ft",
                orig_reference="bgs",
                mp_offset_m=np.nan,
                dtw_bgs_m=dtw,
                conversion=conv,
                flags=flags_join(fl),
                censored=cen,
                source_file=f"raw/{SRC}/appendixes.zip:{MDB}#RM_gw_lev",
                source_row=f"RM_gw_lev:{i}",
                _date=ts.strftime("%Y-%m-%d"),
                _attr=note,
                _src="ds190",
            )
        )
    ds = pd.DataFrame(rows)
    stats["ds190_censored"] = ds.censored.value_counts().to_dict()
    stats["ds190_no_value_rows"] = int(ds.orig_value.isna().sum())

    # ---------------- NWIS extension ----------------
    fm = pd.concat(
        [
            pd.read_csv(p, dtype={"monitoring_location_id": str})
            for p in sorted(glob.glob(str(raw / "nwis/field_measurements_*.csv")))
        ]
    )
    fm_src = {}
    for p in sorted(glob.glob(str(raw / "nwis/field_measurements_*.csv"))):
        for j, v in enumerate(
            pd.read_csv(p, usecols=["field_visit_id", "parameter_code"]).itertuples()
        ):
            if v.parameter_code == 72019:
                fm_src[v.field_visit_id] = (f"raw/{SRC}/nwis/{Path(p).name}", j)
    stats["nwis_rows_all_params"] = int(len(fm))
    fm = fm[fm.parameter_code == 72019].copy()
    stats["nwis_72019_rows"] = int(len(fm))
    fm["site_no"] = fm.monitoring_location_id.str.replace("USGS-", "")
    fm["t"] = pd.to_datetime(fm["time"], utc=True)
    fm["dateonly"] = fm["time_of_day"].astype(str).str.startswith("12:00:00")
    loc = fm["t"].dt.tz_convert("America/Los_Angeles")
    fm["ldate"] = np.where(
        fm.dateonly, fm["t"].dt.strftime("%Y-%m-%d"), loc.dt.strftime("%Y-%m-%d")
    )
    fm["q"] = fm.qualifier.fillna("").astype(str)

    # explorer interpreted conditions
    ex = []
    for p in sorted(glob.glob(str(raw / "nnss_explorer/site_data/*.json"))):
        d = json.loads(Path(p).read_text())
        for rec in d.get("data", []):
            ex.append(
                dict(
                    site_no=Path(p).stem,
                    dt=rec["datetime"],
                    v=rec["wlDepth"],
                    gen=rec.get("genCond"),
                    det=rec.get("detCond"),
                )
            )
    ex = pd.DataFrame(ex)
    ex["v"] = pd.to_numeric(ex.v, errors="coerce")
    ex["d"] = ex.dt.str[:10]
    ex_map = {
        (a, b, round(c, 2)): (g, dd)
        for a, b, c, g, dd in zip(ex.site_no, ex.d, ex.v, ex.gen, ex.det)
    }
    stats["explorer_rows"] = int(len(ex))

    ds_idx = ds.groupby(["site_no", "_date"])["orig_value"].apply(list).to_dict()
    nrows = []
    dup = conflict = notin = post = 0
    for r in fm.itertuples():
        cand = ds_idx.get((r.site_no, r.ldate), [])
        is_pre = r.ldate < "2006-01-01"
        diffs = [abs(c - r.value) for c in cand if pd.notna(c)]
        if is_pre and ((diffs and min(diffs) <= 0.05) or (pd.isna(r.value) and cand)):
            dup += 1
            continue
        fl = []
        if "Pumping" in r.q:
            fl.append("pumping")
        if "Above" in r.q:
            fl.append("other:nwis_qualifier_above")
        cen = ""
        val = r.value
        conv = (
            "dtw_bgs_m = orig_value * 0.3048 (NWIS 72019 depth below land surface, ft)"
        )
        dtw = val * FT if pd.notna(val) else np.nan
        if "Dry" in r.q:
            fl.append("dry_censored")
            cen = "gt"
            if pd.isna(val):
                fl.append("missing")
                conv = "not_defensible: NWIS dry, no sounded depth reported"
        if is_pre:
            if cand:
                conflict += 1
                fl += ["suspect", "other:nwis_differs_from_ds190_same_date"]
            else:
                notin += 1
                fl.append("other:nwis_not_in_ds190")
        else:
            post += 1
        e = (
            ex_map.get((r.site_no, r.t.strftime("%Y-%m-%d"), round(val, 2)))
            if pd.notna(val)
            else None
        )
        attr = ""
        if e:
            g, dd = e
            fl += EXPL_GEN.get(g, [])
            for a in str(dd or "").split(";"):
                a = a.strip()
                if a in EXPL_DET:
                    fl.append(EXPL_DET[a])
            attr = f"explorer gen={g}; det={dd}"
        tstr = (
            r.t.strftime("%Y-%m-%d")
            if r.dateonly
            else r.t.strftime("%Y-%m-%dT%H:%M:%SZ")
        )
        sf, sr = fm_src.get(r.field_visit_id, ("", ""))
        nrows.append(
            dict(
                site_no=r.site_no,
                timestamp=tstr,
                orig_value=float(val) if pd.notna(val) else np.nan,
                orig_units="ft",
                orig_reference="bgs",
                mp_offset_m=np.nan,
                dtw_bgs_m=dtw,
                conversion=conv,
                flags=flags_join(fl),
                censored=cen,
                source_file=sf,
                source_row=f"field_visit_id={r.field_visit_id}",
                _date=r.ldate,
                _attr=attr,
                _src="nwis",
                _post=not is_pre,
                _expl=bool(e),
            )
        )
    nw = pd.DataFrame(nrows)
    stats["nwis_dropped_duplicate_of_ds190"] = dup
    stats["nwis_pre2006_kept_not_in_ds190"] = notin
    stats["nwis_pre2006_kept_conflicting_same_date"] = conflict
    stats["nwis_post2005_kept"] = post
    stats["nwis_kept_with_explorer_attrs"] = int(nw._expl.sum())
    stats["nwis_post2005_with_explorer_attrs"] = int((nw._expl & nw._post).sum())
    stats["nwis_kept_not_in_ds190_by_site"] = (
        nw[~nw._post].groupby("site_no").size().to_dict()
    )

    ml = pd.concat(
        [
            pd.read_csv(p, dtype=str)
            for p in sorted(glob.glob(str(raw / "nwis/monitoring_locations_*.csv")))
        ]
    )
    ml = ml.drop_duplicates("monitoring_location_number").set_index(
        "monitoring_location_number"
    )
    new_sites = sorted(set(nw.site_no) - set(site.USGS_site_id))
    stats["nwis_new_intervals"] = new_sites

    obs = pd.concat([ds, nw.drop(columns=["_post", "_expl"])], ignore_index=True)

    # ---------------- inventory ----------------
    inv_sites = []
    for r in site.itertuples():
        inv_sites.append(
            dict(site_no=r.USGS_site_id, name=r.USGS_local_nm, bh=r.bh, new=False)
        )
    for s in new_sites:
        nm = re.sub(
            r"^.*?\s{2,}\d+\s{2,}", "", ml.loc[s, "monitoring_location_name"]
        ).strip()
        bh = borehole_of(nm, s)
        inv_sites.append(dict(site_no=s, name=nm, bh=bh, new=True))
    inv_sites = pd.DataFrame(inv_sites)

    # order intervals in a borehole by open-interval top then site number
    def top_of(s):
        if s in t1.index:
            return t1.loc[s, "top_ft"]
        return np.nan

    inv_sites["top"] = inv_sites.site_no.map(top_of)
    inv_sites = inv_sites.sort_values(["bh", "new", "top", "site_no"])
    inv_sites["n"] = inv_sites.groupby("bh").cumcount() + 1
    inv_sites["well_id"] = SRC + ":" + inv_sites.bh
    inv_sites["interval_id"] = inv_sites.well_id + "#" + inv_sites.n.astype(str)
    id_map = inv_sites.set_index("site_no")[["well_id", "interval_id"]]
    obs = obs.merge(id_map, left_on="site_no", right_index=True, how="left")
    assert obs.interval_id.notna().all()

    # DEM sampling
    dem = {}
    try:
        import rasterio
        from pyproj import Transformer

        sx = site.drop_duplicates("USGS_site_id").set_index("USGS_site_id")
        tr = Transformer.from_crs(4269, 5070, always_xy=True)
        pts = {
            s: tr.transform(sx.loc[s, "dec_long_va"], sx.loc[s, "dec_lat_va"])
            for s in sx.index
        }
        for f in ["elev48i0100a", "tpi_500m", "tpi_2km", "geomorphons"]:
            with rasterio.open(TERRAIN + f + ".tif") as rr:
                for s, v in zip(pts, rr.sample(list(pts.values()))):
                    dem.setdefault(s, {})[f] = float(v[0])
                if f == "elev48i0100a":
                    for s, (xx, yy) in pts.items():
                        w = rasterio.windows.from_bounds(
                            xx - 1500, yy - 1500, xx + 1500, yy + 1500, rr.transform
                        )
                        a = rr.read(1, window=w).astype(float)
                        dem[s]["max3km"] = float(a.max())
    except Exception as e:
        stats["dem_error"] = repr(e)
    GEOM = {
        1: "flat",
        2: "summit",
        3: "ridge",
        4: "shoulder",
        5: "spur",
        6: "slope",
        7: "hollow",
        8: "footslope",
        9: "valley",
        10: "depression",
    }

    inv = []
    for r in inv_sites.itertuples():
        s = r.site_no
        o = obs[obs.site_no == s]
        sr = site[site.USGS_site_id == s]
        base = s if len(sr) else None
        if (
            base is None
        ):  # new NWIS interval: borrow borehole coords/alt from an existing interval
            base = site[site.bh == r.bh].USGS_site_id.iloc[0]
            sr = site[site.USGS_site_id == base]
        sr = sr.iloc[0]
        tunnel = s.startswith("9999")
        cmeth = sr.coord_meth
        cconf = (
            "high"
            if (cmeth == "Survey" and "Hndrth" in str(sr.coord_acy))
            else "medium"
        )
        if tunnel:
            cconf = "medium"
        mps = mp[mp.USGS_site_id == s].sort_values("mpnt_seq_nu")
        mph = float(mps.mpnt_ht_va.iloc[-1]) * FT if len(mps) else np.nan
        mpnote = "; ".join(
            f"MP{int(m.mpnt_seq_nu)} {m.mpnt_ht_va} ft ({m.mpnt_ds}, {str(m.mpnt_begin_dtm)[:10]}-{str(m.mpnt_end_dtm)[:10]})"
            for m in mps.itertuples()
        )
        if s in cons.index:
            c = cons.loc[s]
            bd, wd = c.hole_depth_va, c.well_depth_va
        elif not r.new:
            bd = wd = np.nan
        else:
            bd = float(ml.loc[s, "hole_constructed_depth"])
            wd = float(ml.loc[s, "well_constructed_depth"])
        zs = NEW_PARENT.get(s, s)
        if zs in t1.index:
            st_, sb_ = t1.loc[zs, "top_ft"], t1.loc[zs, "bot_ft"]
        else:
            m = re.search(r"\((\d+)-(\d+) ft\)", r.name)
            st_, sb_ = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
        geol = ""
        if zs in aq.index:
            a = aq.loc[zs]
            geol = "; ".join(
                str(v)
                for v in [a.aqfr, a.aqfr_era, a.aqfr_system, a.aqfr_series]
                if pd.notna(v) and str(v) not in ("", "nan")
            )
            if zs in t1.index:
                geol += f"; Table 1 contributing units {t1.loc[zs, 'units']}"
        # aquifer setting
        fls = o["flags"].fillna("")
        perched_frac = fls.str.contains("perched").mean() if len(o) else 0
        ntxt = str(nar.get(s, ""))
        if tunnel:
            setting = "perched"
        elif (
            "unsaturated" in ntxt
            or fls.str.contains("suspected_unsaturated").mean() > 0.5
        ):
            setting = "unknown"
        elif perched_frac >= 0.5 or re.search(
            r"\bperched (aquifer|volcanic|system|ground water|water table)|probably are perched|assumed to be perched|reflects perched|likely reflects perched",
            ntxt,
        ):
            setting = "perched"
        else:
            setting = "bedrock"
        setting = SETTING_OVERRIDE.get(s, setting)
        lc, lconf, lev_txt = TUNNEL_LANDSCAPE if tunnel else LANDSCAPE[r.bh]
        dm = dem.get(base, {})
        if dm:
            lev_txt += (
                f" | DEM check (elev48i0100a 100 m, EPSG:5070, point sample): DEM {dm['elev48i0100a']:.0f} m vs source LS "
                f"{sr.alt_va * FT:.0f} m NGVD29; max within 1.5 km {dm['max3km']:.0f} m; tpi_500m {dm['tpi_500m']:.2f}, "
                f"tpi_2km {dm['tpi_2km']:.2f} (raster units); geomorphon {GEOM.get(int(dm['geomorphons']), dm['geomorphons'])}"
            )
        notes = []
        if r.new:
            notes.append(
                f"Interval added from NWIS (not in DS190): NWIS name '{ml.loc[s, 'monitoring_location_name'].strip()}'; coords/LS altitude taken from DS190 site {base}"
            )
        else:
            notes.append(
                f"NTS hole name: {t1.loc[s, 'nts_hole'] if s in t1.index else ''}; site use {sr.site_use_1}; original coords NAD27 DMS {str(sr.lat_va).strip()} {str(sr.long_va).strip()}; alt method {sr.alt_meth}"
            )
        if mpnote:
            notes.append(
                "MP history (DTW already referenced to land surface): " + mpnote
            )
        if tunnel:
            notes.append(
                f"DS190 pseudo site ID {s} (not an NWIS site); tunnel impounded-water site; DTW is depth below map ground surface above the drift"
            )
        if s in NEW_PARENT:
            notes.append(
                f"Same zone as DS190 interval {NEW_PARENT[s]}; screen and geology copied from it (second access tube)"
            )
        if r.bh in ("U-12e.M1_UG", "U-12e.03-1"):
            notes.append(
                "Hole drilled from tunnel floor; depths/open interval converted by DS190 to below mesa land surface"
            )
        ntx = ntxt.replace("\n", " ")
        if ntx:
            notes.append("DS190 narrative: " + ntx[:600])
        inv.append(
            dict(
                source_id=SRC,
                well_id=r.well_id,
                interval_id=r.interval_id,
                well_name=r.name,
                site_no=s if not tunnel else "",
                lat=float(sr.dec_lat_va),
                lon=float(sr.dec_long_va),
                horiz_datum="NAD83",
                coord_source="table (DS190 RM_sitefile dec_lat_va/dec_long_va; method "
                + str(cmeth)
                + ")",
                coord_confidence=cconf,
                elev_m=round(float(sr.alt_va) * FT, 2),
                elev_datum="NGVD29",
                elev_source=f"DS190 RM_sitefile alt_va ({sr.alt_meth})",
                mp_height_m=round(mph, 3) if pd.notna(mph) else np.nan,
                borehole_depth_m=round(float(bd) * FT, 2) if pd.notna(bd) else np.nan,
                well_depth_m=round(float(wd) * FT, 2) if pd.notna(wd) else np.nan,
                screen_top_m=round(float(st_) * FT, 2) if pd.notna(st_) else np.nan,
                screen_bot_m=round(float(sb_) * FT, 2) if pd.notna(sb_) else np.nan,
                geology=geol,
                aquifer_setting=setting,
                state="NV",
                landscape_class=lc,
                landscape_evidence=lev_txt,
                landscape_confidence=lconf,
                record_start=o._date.min() if len(o) else "",
                record_end=o._date.max() if len(o) else "",
                n_obs=int(len(o)),
                source_file=f"raw/{SRC}/appendixes.zip:{MDB}"
                if not r.new
                else f"raw/{SRC}/nwis/",
                notes=" | ".join(notes),
            )
        )
    inv = pd.DataFrame(inv)
    inv.to_csv(out / "inventory.csv", index=False)

    keep = [
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
    obs["source_id"] = SRC
    obs["source_row"] = obs.source_row.astype(str)
    # attribute text goes into flags only via mapped tokens; keep raw attribute text as extra column
    obs = obs.rename(columns={"_attr": "source_attributes"})
    obs = obs.sort_values(["interval_id", "timestamp"]).reset_index(drop=True)
    obs[keep + ["source_attributes"]].to_parquet(
        out / "observations.parquet", index=False
    )

    stats["n_obs_total"] = int(len(obs))
    stats["n_obs_by_origin"] = obs._src.value_counts().to_dict()
    stats["n_intervals"] = int(len(inv))
    stats["n_wells"] = int(inv.well_id.nunique())
    stats["landscape_counts_intervals"] = inv.landscape_class.value_counts().to_dict()
    stats["landscape_counts_wells"] = (
        inv.drop_duplicates("well_id").landscape_class.value_counts().to_dict()
    )
    stats["aquifer_setting_counts"] = inv.aquifer_setting.value_counts().to_dict()
    stats["date_range"] = [obs.timestamp.min(), obs.timestamp.max()]
    stats["flag_token_counts"] = (
        pd.Series([t for f in obs["flags"].fillna("") for t in f.split("|") if t])
        .value_counts()
        .to_dict()
    )
    (out / "parse_stats.json").write_text(json.dumps(stats, indent=2, default=str))

    # manifest
    log = json.loads(Path(log_path).read_text())
    files = sorted(log["files"].values(), key=lambda d: d["path"])
    unresolved = [
        "DS190 Table 2 measurement counts are double the database counts for 5 intervals (TW-1 1615-4206, U-12s 1480, ER-12-1 1641-1846, ER-19-1-3, TW-1 1615-3300); MDB/XLS (1,227 rows) used as authoritative",
        "Legacy NWIS gwlevels service decommissioned (HTTP 301 to notice); extension pulled from USGS Water Data OGC API field-measurements instead",
        "NNSS explorer get-site-data returned HTTP 500 for 7 in-tunnel hole intervals (U-12e.M1 UG x4, U-12e.03-1 x3); no post-2005 data expected there",
        "DS190 measurement times are local (Pacific) with no stated time zone or DST convention; matched NWIS rows differ by 0 h or 1 h",
        "Pre-2006 NWIS rows absent from DS190 (mostly TW-1 3700-4206 ft 1962 test-period rows) kept with flag other:nwis_not_in_ds190; Fenelon's reason for excluding them is not documented",
        "Landscape class for Aqueduct Mesa / Gold Meadows / off-mesa holes inferred from figure 1 labels + DEM, not from explicit text",
        "pubs.usgs.gov figure.html index page 404 (figure1-3.html retrieved)",
    ]
    man = dict(
        source_id=SRC,
        title="Database of ground-water levels in the vicinity of Rainier Mesa, Nevada Test Site, Nye County, Nevada, 1957-2005",
        citation="Fenelon, J.M., 2006 (v1.1 2007), Database of ground-water levels in the vicinity of Rainier Mesa, Nevada Test Site, Nye County, Nevada, 1957-2005: U.S. Geological Survey Data Series 190, version 1.1",
        doi_or_url="https://pubs.usgs.gov/ds/2006/190/",
        landing_url="https://pubs.usgs.gov/ds/2006/190/",
        retrieval_date="2026-10-27",
        release_version="1.10 (October 10, 2007); NWIS extension via api.waterdata.usgs.gov OGC field-measurements retrieved 2026-10-27",
        license="US Government work, public domain",
        files=[
            dict(path=f["path"], url=f["url"], sha256=f["sha256"], bytes=f["bytes"])
            for f in files
        ],
        status="acquired",
        unresolved=unresolved,
        notes="DS190 Access MDB is authoritative for 1957-2005; NWIS 72019 (depth below land surface, ft) rows added where not duplicating DS190 (same site + local date + |diff|<=0.05 ft). NNSS explorer interpreted conditions attached to NWIS rows by site+UTC date+value. Fetch failures: "
        + json.dumps(
            log.get("failures", [])[:1]
            and [{k: f[k] for k in ("name", "http")} for f in log["failures"]]
        ),
    )
    (out / "manifest.json").write_text(json.dumps(man, indent=2))
    tmp.cleanup()
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
