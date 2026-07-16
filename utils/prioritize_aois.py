#!/usr/bin/env python
"""Add stream_km, n_fields, field_km2, and priority attributes to an AOI shapefile.

Computes per-AOI:
  - stream_km:  total length of perennial NHD flowlines (FCODE 46006, 55800) in km
  - n_fields:   count of intersecting irrigation fields
  - field_km2:  total area of intersecting fields in km²
  - priority:   stream_km × field_km2 (higher = more likely REM matters)

Usage:
    uv run python utils/prioritize_aois.py \
        --aoi-shp /data/ssd2/handily/nv/aois/nv_aois.shp \
        --fields /nas/Nevada/.../Nevada_Agricultural_Field_Boundaries_20250214_5071_GFID.shp \
        --flowlines /nas/boundaries/wbd/NHD_H_Nevada_State_Shape/Shape/NHDFlowline_filtered.fgb
"""

import argparse

import geopandas as gpd
import numpy as np


PERENNIAL_FCODES = {46006, 55800}


def prioritize(aoi_path, fields_path, flowlines_path):
    aois = gpd.read_file(aoi_path)
    print(f"AOIs: {len(aois)}")

    # Load flowlines and filter to perennial
    print(f"Loading flowlines: {flowlines_path}")
    flow = gpd.read_file(flowlines_path)
    fcode_col = next((c for c in flow.columns if c.lower() == "fcode"), None)
    if fcode_col:
        flow = flow[flow[fcode_col].isin(PERENNIAL_FCODES)].copy()
    print(f"Perennial flowlines: {len(flow)}")

    # Project everything to EPSG:5070 for metric calculations
    aois_proj = aois.to_crs("EPSG:5070")
    flow_proj = flow.to_crs("EPSG:5070")

    print(f"Loading fields: {fields_path}")
    fields = gpd.read_file(fields_path)
    fields_proj = fields.to_crs("EPSG:5070")
    print(f"Fields: {len(fields_proj)}")

    # Build spatial indices
    flow_sindex = flow_proj.sindex
    fields_sindex = fields_proj.sindex

    stream_km = np.zeros(len(aois), dtype=float)
    n_fields = np.zeros(len(aois), dtype=int)
    field_km2 = np.zeros(len(aois), dtype=float)

    for i, row in aois_proj.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Stream length: intersect flowlines with AOI, sum clipped lengths
        flow_hits = list(flow_sindex.intersection(geom.bounds))
        if flow_hits:
            candidates = flow_proj.iloc[flow_hits]
            clipped = gpd.clip(
                candidates, gpd.GeoDataFrame(geometry=[geom], crs="EPSG:5070")
            )
            stream_km[i] = clipped.geometry.length.sum() / 1000.0

        # Fields: count and area
        field_hits = list(fields_sindex.intersection(geom.bounds))
        if field_hits:
            candidates = fields_proj.iloc[field_hits]
            within = candidates[candidates.intersects(geom)]
            n_fields[i] = len(within)
            field_km2[i] = within.geometry.area.sum() / 1e6

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(aois)} AOIs processed")

    aois["stream_km"] = np.round(stream_km, 2)
    aois["n_fields"] = n_fields
    aois["field_km2"] = np.round(field_km2, 2)
    aois["priority"] = np.round(stream_km * field_km2, 2)

    aois.to_file(aoi_path)
    print(f"Updated: {aoi_path}")
    print("Top 10 by priority:")
    top = aois.nlargest(10, "priority")[
        ["aoi_id", "stream_km", "n_fields", "field_km2", "priority"]
    ]
    print(top.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Add priority attributes to AOI shapefile"
    )
    parser.add_argument("--aoi-shp", required=True, help="AOI shapefile to update")
    parser.add_argument("--fields", required=True, help="Irrigation fields shapefile")
    parser.add_argument("--flowlines", required=True, help="NHD flowlines FlatGeobuf")
    args = parser.parse_args()
    prioritize(args.aoi_shp, args.fields, args.flowlines)


if __name__ == "__main__":
    main()
