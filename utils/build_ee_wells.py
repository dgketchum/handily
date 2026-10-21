"""Build a water-table wells CSV for the handily EE app.

Two footprint modes select which wells go into an app table:

* ``--state <ST>`` -- one state's wells (a statewide app unit, e.g. MT, NM).
* ``--basins-file <path>`` -- wells inside the union of a HUC8 basin list (a
  regional app unit, e.g. UCRB = the on-disk region-14 HUC8s). Optionally drop
  whole states with ``--exclude-states NM,MT`` so wells already carried by an
  existing statewide table are not drawn twice.

Both modes filter to water-table wells only (confinement_class in {unconfined,
unconfined_marginal} -- confined wells measure a potentiometric surface, not
the water table, and never enter the app), map columns to the app schema, and
bake the DTW color ramp into a ``fillcolor`` column. The color must be
pre-baked: computing the ramp bin per-feature in the EE app timed out map
tiles at 150k+ wells.

``--well-class monitoring`` additionally restricts to the production GNN
training population (mirrors build_conus_graph_inputs.py --well-class, the
monitoring-only lineage) -- output is tagged ``wells_train`` and belongs under
the ``conus/wells_train/<unit>`` assets so the app can draw the assimilated
population against all wells. Note the model matches assimilated wells partly
by construction; the un-assimilated remainder is the independent check.

``--bundle-dir <graph bundle> --basins-file <huc8 list>`` is the third mode:
the training population read straight from the bundle the shipped arm was
trained on (``query_nodes.parquet`` real-well rows inside the basin list), so
the layer shows exactly the labels the model saw -- the monitoring networks
plus the appended state-agency wells (NV NDWR / NM OSE admitted halves), which
are also render-time sources. Output is tagged ``wells_train`` like the
``--well-class`` mode; ``role`` carries the bundle ``well_class`` (monitoring /
ndwr / ose).

Upload the CSV with::

    earthengine upload table \
        --asset_id=projects/ssebop-montana/assets/handily/conus/wells/<unit> \
        --x_column longitude --y_column latitude --crs EPSG:4326 \
        gs://ucrc-wtd/handily/<unit>/handily_wells_<unit>.csv

(``.../conus/wells_train/<unit>`` and ``handily_wells_train_<unit>.csv`` for
``--well-class`` builds.) Tables under the conus/wells folders do NOT inherit
the folder ACL -- run ``earthengine acl set`` on the new table after ingest.
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from handily.io import load_huc

WELLS_PARQUET = "/data/ssd2/gwx/products/current/wells.geoparquet"
OUT_ROOT = "/data/ssd2/handily/share/ee_app"

WATER_TABLE_CLASSES = ("unconfined", "unconfined_marginal")

# DTW ramp: identical stops/colors to the app script (handily_ee_app_conus.js)
# and the QGIS project -- a well dot that blends into the DTW raster behind it
# means good model agreement.
DTW_STOPS = [0.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 100.0]
DTW_COLORS = [
    "2b83ba",
    "6dbedc",
    "abdda4",
    "d9ef8b",
    "ffffbf",
    "fee08b",
    "fdae61",
    "f46d43",
    "d73027",
    "d7191c",
]

COLUMNS = {
    "canonical_id": "canonical_id",
    "source": "source",
    "mean_dtw": "dtw_m",
    "confinement_class": "confine",
    "aquifer": "aquifer",
    "well_depth": "well_dep_m",
    "obs_count": "obs_n",
    "well_use": "well_use",
    "well_class": "well_class",
    "por_start": "por_start",
    "por_end": "por_end",
    "longitude": "longitude",
    "latitude": "latitude",
}


def ramp_hex(values_m, stops=DTW_STOPS, colors=DTW_COLORS):
    """Piecewise-linear interpolation of the ramp; clamps beyond the ends."""
    channels = [np.array([int(c[i : i + 2], 16) for c in colors]) for i in (0, 2, 4)]
    rgb = [np.interp(values_m, stops, ch).round().astype(int) for ch in channels]
    return ["{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in zip(*rgb)]


def _finalize(df, label, out_csv):
    """Map to the app schema, bake the ramp, write the CSV, report percentiles.

    ``df`` carries the raw parquet columns (COLUMNS keys) and no ``state``
    column; the output schema is identical across both selection modes.
    """
    df = df.rename(columns=COLUMNS)
    df = df[df["dtw_m"].notna() & df["longitude"].notna() & df["latitude"].notna()]
    df["fillcolor"] = ramp_hex(df["dtw_m"].to_numpy())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"{label}: {len(df)} water-table wells -> {out_csv}")
    print(
        f"dtw_m percentiles (m): "
        f"{df['dtw_m'].quantile([0, 0.25, 0.5, 0.75, 1]).round(1).to_dict()}"
    )
    return df


def build_state(state, wells_parquet, out_csv, well_classes=()):
    """Water-table wells for one state (statewide app unit)."""
    df = pd.read_parquet(wells_parquet, columns=list(COLUMNS) + ["state"])
    keep = (df["state"] == state) & df["confinement_class"].isin(WATER_TABLE_CLASSES)
    if well_classes:
        keep &= df["well_class"].isin(well_classes)
    df = df.loc[keep].drop(columns=["state"])
    return _finalize(df, state, out_csv)


def build_bundle(bundle_dir, basins, out_csv, label="bundle"):
    """Training wells of a graph bundle inside a HUC8 basin list (wells_train).

    Reads the real-well rows of ``query_nodes.parquet`` (water-stage pseudo-rows
    dropped), keeps those whose ``huc8`` is in ``basins``, and maps them to the
    app schema. ``dtw_m`` is the bundle label depth (land surface minus observed
    water-table elevation, the same ``mean_dtw`` the trainer scores); EPSG:5070
    coordinates are converted to lon/lat for the table upload.
    """
    cols = [
        "canonical_id",
        "source",
        "well_class",
        "confinement_class",
        "huc8",
        "x5070",
        "y5070",
        "mean_dtw",
        "z_surf_well_m",
        "wte_obs_m",
        "is_water_pseudo",
    ]
    q = pd.read_parquet(Path(bundle_dir) / "query_nodes.parquet", columns=cols)
    q = q[~q["is_water_pseudo"].astype(bool)]
    codes = {str(b).zfill(8) for b in basins}
    q = q[q["huc8"].astype(str).str.zfill(8).isin(codes)].copy()
    if q.empty:
        raise SystemExit(f"{label}: no bundle wells inside the {len(codes)} basins")
    # the bundle label is z_surf - wte_obs; mean_dtw is that same quantity
    d = q["z_surf_well_m"] - q["wte_obs_m"]
    if not np.allclose(d, q["mean_dtw"], atol=1e-6, equal_nan=True):
        raise SystemExit(f"{label}: mean_dtw != z_surf - wte_obs in the bundle")
    lon, lat = (
        gpd.GeoSeries(gpd.points_from_xy(q["x5070"], q["y5070"]), crs=5070)
        .to_crs(4326)
        .pipe(lambda g: (g.x.to_numpy(), g.y.to_numpy()))
    )
    df = pd.DataFrame(
        {
            "canonical_id": q["canonical_id"].to_numpy(),
            "source": q["source"].to_numpy(),
            "mean_dtw": q["mean_dtw"].to_numpy(),
            "confinement_class": q["confinement_class"].to_numpy(),
            "well_class": q["well_class"].to_numpy(),
            "role": q["well_class"].to_numpy(),
            "longitude": lon,
            "latitude": lat,
        }
    )
    print(f"{label}: bundle rows by role {df['role'].value_counts().to_dict()}")
    return _finalize(df, label, out_csv)


def build_footprint(
    basins,
    wells_parquet,
    out_csv,
    exclude_states=(),
    label="footprint",
    well_classes=(),
):
    """Water-table wells inside the union of a HUC8 basin list (regional unit).

    Clips to the dissolved EPSG:5070 union of the WBD HUC8 polygons named in
    ``basins`` (via ``load_huc(8)``); ``exclude_states`` drops whole states
    first (two-letter codes) so wells already carried by a statewide app table
    are not drawn twice.
    """
    codes = [str(b) for b in basins]
    hucs = load_huc(8, columns=[])
    sel = hucs[hucs["huc8"].astype(str).isin(codes)]
    missing = sorted(set(codes) - set(sel["huc8"].astype(str)))
    if missing:
        raise SystemExit(f"HUC8 codes not found in WBD: {missing}")
    union = sel.geometry.union_all()
    exclude = {s.strip().upper() for s in exclude_states if s.strip()}

    df = pd.read_parquet(wells_parquet, columns=list(COLUMNS) + ["state"])
    df = df[df["confinement_class"].isin(WATER_TABLE_CLASSES)]
    if well_classes:
        df = df[df["well_class"].isin(well_classes)]
    if exclude:
        df = df[~df["state"].str.upper().isin(exclude)]
    df = df[df["longitude"].notna() & df["latitude"].notna()]
    # Bounding-box prefilter (lon/lat) shrinks the exact within-union test to
    # the regional footprint before the pointwise geometry op.
    minx, miny, maxx, maxy = gpd.GeoSeries([union], crs=5070).to_crs(4326).total_bounds
    df = df[df["longitude"].between(minx, maxx) & df["latitude"].between(miny, maxy)]
    pts = gpd.GeoSeries(
        gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=4326,
        index=df.index,
    ).to_crs(5070)
    df = df[pts.within(union).to_numpy()].drop(columns=["state"])
    return _finalize(df, label, out_csv)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--state", help="two-letter state code, e.g. NM")
    mode.add_argument(
        "--basins-file",
        help="text file of HUC8 codes (one per line); wells clipped to their union",
    )
    mode.add_argument(
        "--bundle-dir",
        help="graph bundle directory; wells_train from its query_nodes.parquet "
        "real-well rows inside --bundle-basins (training labels of that bundle)",
    )
    ap.add_argument(
        "--bundle-basins",
        default=None,
        help="HUC8 list (one per line) restricting --bundle-dir rows to a footprint",
    )
    ap.add_argument(
        "--exclude-states",
        default="",
        help="comma-separated state codes to drop in footprint mode, e.g. NM,MT",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="unit tag for output naming/logging in footprint mode (e.g. ucrb)",
    )
    ap.add_argument(
        "--well-class",
        default="",
        help="comma-separated GWX well_class filter (e.g. 'monitoring' = the "
        "production GNN training population); tags the output wells_train",
    )
    ap.add_argument("--wells", default=WELLS_PARQUET)
    ap.add_argument(
        "--out",
        default=None,
        help=f"output CSV (default {OUT_ROOT}/<unit>/handily_<stem>_<unit>.csv)",
    )
    args = ap.parse_args()

    well_classes = tuple(s.strip() for s in args.well_class.split(",") if s.strip())
    stem = "wells_train" if well_classes else "wells"

    if args.bundle_dir:
        if not args.bundle_basins:
            raise SystemExit("--bundle-dir needs --bundle-basins")
        basins = [
            ln.strip()
            for ln in Path(args.bundle_basins).read_text().splitlines()
            if ln.strip()
        ]
        label = (args.label or Path(args.bundle_basins).stem).lower()
        out = (
            Path(args.out)
            if args.out
            else Path(OUT_ROOT) / label / f"handily_wells_train_{label}.csv"
        )
        build_bundle(args.bundle_dir, basins, out, label=label)
    elif args.state:
        unit = args.state.upper()
        out = (
            Path(args.out)
            if args.out
            else Path(OUT_ROOT) / unit.lower() / f"handily_{stem}_{unit.lower()}.csv"
        )
        build_state(unit, args.wells, out, well_classes=well_classes)
    else:
        basins = [
            ln.strip()
            for ln in Path(args.basins_file).read_text().splitlines()
            if ln.strip()
        ]
        label = (args.label or Path(args.basins_file).stem).lower()
        exclude = [s for s in args.exclude_states.split(",") if s.strip()]
        out = (
            Path(args.out)
            if args.out
            else Path(OUT_ROOT) / label / f"handily_{stem}_{label}.csv"
        )
        build_footprint(
            basins,
            args.wells,
            out,
            exclude_states=exclude,
            label=label,
            well_classes=well_classes,
        )
