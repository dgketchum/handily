"""Rank NM HUC8 basins by suitability for estimating shallow groundwater in
agricultural regions: density of shallow, non-NWIS water-level wells co-located
with OpenET irrigated fields.

Inputs (current GWX build + shared NAS layers):
  - wells   : /data/ssd2/gwx/products/current/wells.geoparquet, NM non-NWIS
              sources (nm_ose driller + nm_sta telemetry). NWIS-independence also
              flagged via canonical_id (not shared with any nwis row).
  - HUC8    : NM WBDHU8 (85 basins, names).
  - fields  : OpenET field polygons, NM layer (EPSG:5071, OPENET_ID + crop hist).

"Shallow" = mean_dtw in (0, 15] m (shallow water table relevant to irrigated ag).
"Near field" = within 500 m of an OpenET field (matches the v6 field-neighborhood
definition). Ranking favors basins with many shallow non-NWIS wells sitting in/near
irrigated fields, with substantial OpenET field area.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pyarrow.parquet as pq

PROD = "/data/ssd2/gwx/products/current/wells.geoparquet"
HUC8 = "/mnt/mco_nas1/dgketchum/boundaries/wbd/NHD_H_New_Mexico_State_Shape/Shape/WBDHU8.shp"
FIELDS = "/mnt/mco_nas1/dgketchum/openET/OpenET_GeoDatabase_5071/NM.shp"
OUT = Path("/data/ssd2/handily/nm/regional/ag_shallow_scan")
NM_SOURCES = ["nm_ose", "nm_sta"]
SHALLOW_M = 15.0
NEAR_M = 500.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # --- wells: NM non-NWIS, with NWIS-independence flag ---
    df = pq.read_table(
        PROD, columns=["source", "longitude", "latitude", "mean_dtw", "canonical_id"]
    ).to_pandas()
    nwis_cids = set(df.loc[df["source"] == "nwis", "canonical_id"].dropna())
    nm = df[df["source"].isin(NM_SOURCES)].copy()
    nm["nwis_indep"] = ~nm["canonical_id"].isin(nwis_cids)
    print(
        f"NM non-NWIS-source wells: {len(nm)} "
        f"(mean_dtw finite {nm['mean_dtw'].notna().sum()}, "
        f"NWIS-independent by canonical_id {int(nm['nwis_indep'].sum())})"
    )
    wells = gpd.GeoDataFrame(
        nm, geometry=gpd.points_from_xy(nm.longitude, nm.latitude), crs="EPSG:4326"
    )

    # --- HUC8 + OpenET fields (work in the fields' projected CRS for areas/dist) ---
    huc = gpd.read_file(HUC8)[["huc8", "name", "geometry"]]
    fields = gpd.read_file(FIELDS)
    crs = fields.crs
    wells = wells.to_crs(crs)
    huc = huc.to_crs(crs)
    fields["area_m2"] = fields.geometry.area
    huc["huc_area_km2"] = huc.geometry.area / 1e6
    print(
        f"OpenET NM fields: {len(fields)} polygons, {fields['area_m2'].sum() / 1e6:,.0f} km2"
    )

    # --- assign wells + fields to HUC8 ---
    wells = gpd.sjoin(wells, huc, how="inner", predicate="within").drop(
        columns="index_right"
    )
    fc = fields.copy()
    fc["geometry"] = fields.representative_point()
    fc = gpd.sjoin(fc, huc[["huc8", "geometry"]], how="inner", predicate="within").drop(
        columns="index_right"
    )

    # --- shallow + near-field ---
    sh = wells[(wells["mean_dtw"] > 0) & (wells["mean_dtw"] <= SHALLOW_M)].copy()
    nf = gpd.sjoin_nearest(
        sh, fields[["geometry"]], max_distance=NEAR_M, distance_col="d"
    )
    near = nf[nf["index_right"].notna()].index.unique()
    sh["near_field"] = sh.index.isin(near)
    print(
        f"shallow (<= {SHALLOW_M:g} m) non-NWIS wells: {len(sh)}; "
        f"within {NEAR_M:g} m of an OpenET field: {int(sh['near_field'].sum())}"
    )

    # --- per-HUC8 aggregation ---
    agg = huc[["huc8", "name", "huc_area_km2"]].copy()
    agg = agg.merge(
        wells.groupby("huc8").size().rename("n_wells").reset_index(),
        on="huc8",
        how="left",
    )
    agg = agg.merge(
        sh.groupby("huc8").size().rename("n_shallow").reset_index(),
        on="huc8",
        how="left",
    )
    agg = agg.merge(
        sh[sh["near_field"]]
        .groupby("huc8")
        .size()
        .rename("n_shallow_near_field")
        .reset_index(),
        on="huc8",
        how="left",
    )
    agg = agg.merge(
        sh.groupby("huc8")["mean_dtw"]
        .median()
        .round(1)
        .rename("shallow_dtw_med")
        .reset_index(),
        on="huc8",
        how="left",
    )
    agg = agg.merge(
        (fc.groupby("huc8")["area_m2"].sum() / 1e6)
        .round(1)
        .rename("field_area_km2")
        .reset_index(),
        on="huc8",
        how="left",
    )
    agg = agg.merge(
        fc.groupby("huc8").size().rename("field_count").reset_index(),
        on="huc8",
        how="left",
    )
    for c in [
        "n_wells",
        "n_shallow",
        "n_shallow_near_field",
        "field_area_km2",
        "field_count",
    ]:
        agg[c] = agg[c].fillna(0)
    agg["irrig_frac_pct"] = (100 * agg["field_area_km2"] / agg["huc_area_km2"]).round(1)
    agg = agg.sort_values(
        ["n_shallow_near_field", "field_area_km2"], ascending=False
    ).reset_index(drop=True)

    agg.to_csv(OUT / "nm_ag_shallow_huc8_ranked.csv", index=False)
    sh[sh["near_field"]].drop(
        columns=[c for c in ("index_right",) if c in sh]
    ).to_parquet(OUT / "nm_shallow_nonnwis_wells_near_fields.geoparquet")
    # HUC8 polygons carrying the stats, for QGIS choropleth
    huc.merge(
        agg.drop(columns=["name", "huc_area_km2"]), on="huc8", how="left"
    ).to_file(OUT / "nm_ag_shallow_huc8.fgb", driver="FlatGeobuf")

    cols = [
        "huc8",
        "name",
        "n_shallow_near_field",
        "n_shallow",
        "shallow_dtw_med",
        "field_area_km2",
        "irrig_frac_pct",
        "field_count",
    ]
    print("\n=== TOP 18 NM HUC8s for shallow GW in ag regions ===")
    with __import__("pandas").option_context(
        "display.width", 160, "display.max_columns", 20
    ):
        print(agg[cols].head(18).to_string(index=False))
    print(f"\nwrote {OUT}/nm_ag_shallow_huc8_ranked.csv")


if __name__ == "__main__":
    main()
