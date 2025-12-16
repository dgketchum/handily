import os
import math
import geopandas as gpd
from shapely.geometry import box as shapely_box, LineString
from shapely.ops import split


def _fishnet_chunks(geom_aea, max_km2):
    max_area_m2 = float(max_km2) * 1_000_000.0
    cell = math.sqrt(max_area_m2)
    minx, miny, maxx, maxy = geom_aea.bounds
    xs = int(math.ceil((maxx - minx) / cell))
    ys = int(math.ceil((maxy - miny) / cell))
    parts = []
    for ix in range(xs):
        for iy in range(ys):
            x0 = minx + ix * cell
            y0 = miny + iy * cell
            x1 = min(x0 + cell, maxx)
            y1 = min(y0 + cell, maxy)
            tile = shapely_box(x0, y0, x1, y1)
            if not tile.intersects(geom_aea):
                continue
            inter = tile.intersection(geom_aea)
            if inter.is_empty:
                continue
            parts.append(inter)
    return parts


def _geometry_area_m2(geom):
    if geom.is_empty:
        return 0.0
    series = gpd.GeoSeries([geom], crs="EPSG:4326")
    return float(series.to_crs("EPSG:5070").area.iloc[0])


def _rectangle_area_m2(rect):
    return _geometry_area_m2(rect)


def _split_polygon_to_max(poly, max_area_m2):
    out = []
    stack = [poly]
    while stack:
        g = stack.pop()
        if g.area <= max_area_m2:
            out.append(g)
            continue
        minx, miny, maxx, maxy = g.bounds
        width = maxx - minx
        height = maxy - miny
        if width >= height:
            mid = (minx + maxx) / 2.0
            line = LineString([(mid, miny - 1.0), (mid, maxy + 1.0)])
        else:
            mid = (miny + maxy) / 2.0
            line = LineString([(minx - 1.0, mid), (maxx + 1.0, mid)])
        parts = split(g, line)
        if len(parts.geoms) == 1:  # GeometryCollection
            if width >= height:
                mid2 = (miny + maxy) / 2.0
                line2 = LineString([(minx - 1.0, mid2), (maxx + 1.0, mid2)])
            else:
                mid2 = (minx + maxx) / 2.0
                line2 = LineString([(mid2, miny - 1.0), (mid2, maxy + 1.0)])
            parts = split(g, line2)
        for p in parts.geoms:
            stack.append(p)
    return out


def build_centroid_buffer_aois(fields_path,
                               max_km2,
                               buffer_m=1000,
                               bounds_wsen=None,
                               simplify_tolerance_m=None):
    if bounds_wsen is not None:
        w, s, e, n = bounds_wsen
        fields = gpd.read_file(fields_path, bbox=(w, s, e, n))
    else:
        fields = gpd.read_file(fields_path)
    fields = fields[~fields.geometry.is_empty & fields.geometry.notnull()].copy()
    fields_aea = fields.to_crs("EPSG:5070")

    cents = fields_aea.geometry.centroid
    disks = cents.buffer(float(buffer_m))
    dissolved = gpd.GeoSeries(disks, crs="EPSG:5070").unary_union
    if dissolved.is_empty:
        raise ValueError("Buffered centroids produced an empty geometry")
    if simplify_tolerance_m is not None and float(simplify_tolerance_m) > 0:
        dissolved = dissolved.simplify(float(simplify_tolerance_m), preserve_topology=True)
    aoi_ll_geom = gpd.GeoSeries([dissolved], crs="EPSG:5070").to_crs(4326).iloc[0]

    max_area_m2 = float(max_km2) * 1_000_000.0
    w, s, e, n = aoi_ll_geom.bounds
    stack = [(w, s, e, n)]
    tiles = []
    eps = 1e-9

    while stack:
        w_, s_, e_, n_ = stack.pop()
        if e_ - w_ <= 0 or n_ - s_ <= 0:
            continue
        rect = shapely_box(w_, s_, e_, n_)
        if not rect.intersects(aoi_ll_geom):
            continue
        inter_geom = rect.intersection(aoi_ll_geom)
        if _geometry_area_m2(inter_geom) <= 0:
            continue
        rect_area = _rectangle_area_m2(rect)
        if rect_area <= max_area_m2:
            tiles.append(rect)
            continue
        width = e_ - w_
        height = n_ - s_
        if width >= height and width > eps:
            mid = (w_ + e_) / 2.0
            if mid == w_ or mid == e_:
                tiles.append(rect)
                continue
            stack.append((mid, s_, e_, n_))
            stack.append((w_, s_, mid, n_))
        elif height > eps:
            mid = (s_ + n_) / 2.0
            if mid == s_ or mid == n_:
                tiles.append(rect)
                continue
            stack.append((w_, mid, e_, n_))
            stack.append((w_, s_, e_, mid))
        else:
            tiles.append(rect)

    aoi_tiles = gpd.GeoDataFrame(geometry=gpd.GeoSeries(tiles, crs="EPSG:4326"))
    aoi_tiles = aoi_tiles.reset_index(drop=True)
    aoi_tiles["aoi_id"] = aoi_tiles.index.astype(int)
    return aoi_tiles


def build_envelope_aois(fields_path,
                        max_km2,
                        buffer_m=0,
                        bounds_wsen=None,
                        simplify_tolerance_m=None):
    if bounds_wsen is not None:
        w, s, e, n = bounds_wsen
        fields = gpd.read_file(fields_path, bbox=(w, s, e, n))
    else:
        fields = gpd.read_file(fields_path)
    fields = fields[~fields.geometry.is_empty & fields.geometry.notnull()].copy()
    fields_aea = fields.to_crs("EPSG:5070")

    minx, miny, maxx, maxy = fields_aea.total_bounds
    pad = float(buffer_m)
    env = shapely_box(minx - pad, miny - pad, maxx + pad, maxy + pad)

    if simplify_tolerance_m is not None and float(simplify_tolerance_m) > 0:
        env = env.simplify(float(simplify_tolerance_m), preserve_topology=True)

    parts = _fishnet_chunks(env, max_km2)
    aoi_ll = gpd.GeoDataFrame(geometry=gpd.GeoSeries(parts, crs="EPSG:5070").to_crs(4326))
    aoi_ll = aoi_ll.reset_index(drop=True)
    aoi_ll["aoi_id"] = aoi_ll.index.astype(int)
    return aoi_ll


def build_field_water_aois(fields_path,
                           flowlines_local_dir,
                           max_km2,
                           buffer_m=100,
                           bounds_wsen=None,
                           simplify_tolerance_m=None,
                           min_intersection_area_m2=1000):
    if bounds_wsen is not None:
        w, s, e, n = bounds_wsen
        fields = gpd.read_file(fields_path, bbox=(w, s, e, n))
    else:
        fields = gpd.read_file(fields_path)
    fields = fields[~fields.geometry.is_empty & fields.geometry.notnull()].copy()
    fields_aea = fields.to_crs("EPSG:5070")

    if bounds_wsen is None:
        hull = fields.to_crs(4326).geometry.unary_union.envelope
        aoi_ll = gpd.GeoDataFrame([{}], geometry=[hull], crs="EPSG:4326")
    else:
        aoi_ll = gpd.GeoDataFrame([{}], geometry=[shapely_box(*bounds_wsen)], crs="EPSG:4326")

    flow = get_flowlines_within_aoi(aoi_ll, local_flowlines_dir=flowlines_local_dir)
    flow_aea = flow.to_crs("EPSG:5070")

    buf = flow_aea.buffer(float(buffer_m))
    buf_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(buf, crs=flow_aea.crs))

    hits = gpd.overlay(fields_aea, buf_gdf, how="intersection")
    hits["__area__"] = hits.geometry.area
    hits = hits[hits["__area__"] >= float(min_intersection_area_m2)].copy()

    dissolved = gpd.GeoDataFrame(geometry=[hits.unary_union], crs=hits.crs)
    dissolved = dissolved.explode(index_parts=False, ignore_index=True)

    aoi_parts = []
    for g in dissolved.geometry:
        if g.is_empty:
            continue
        if g.area <= float(max_km2) * 1_000_000.0:
            aoi_parts.append(g)
        else:
            aoi_parts.extend(_fishnet_chunks(g, max_km2))

    if simplify_tolerance_m is not None:
        aoi_parts = [g.simplify(float(simplify_tolerance_m), preserve_topology=True) for g in aoi_parts]
        aoi_parts = [g for g in aoi_parts if not g.is_empty]

    aoi_ll = gpd.GeoDataFrame(geometry=gpd.GeoSeries(aoi_parts, crs="EPSG:5070").to_crs(4326))
    aoi_ll = aoi_ll.reset_index(drop=True)
    aoi_ll["aoi_id"] = aoi_ll.index.astype(int)
    return aoi_ll


def write_aois_shapefile(aoi_gdf, shp_path):
    out_dir = os.path.dirname(os.path.abspath(shp_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    aoi_gdf.to_file(shp_path)
    print(f'wrote {shp_path}')
    return shp_path


if __name__ == "__main__":
    fields_path = "~/data/IrrigationGIS/Montana/statewide_irrigation_dataset/statewide_irrigation_dataset_15FEB2024.shp"
    max_km2 = 300
    buffer_m = 1000
    simplify_tolerance_m = None
    shp_out_dir = "/home/dgketchum/data/IrrigationGIS/handily/outputs/testing"
    shp_out_path = os.path.join(shp_out_dir, "ndwi_aois.shp")

    aoi_gdf = build_centroid_buffer_aois(
        fields_path=fields_path,
        max_km2=max_km2,
        buffer_m=buffer_m,
        bounds_wsen= (-113.8, 45., -112.27, 46.),
        simplify_tolerance_m=simplify_tolerance_m,
    )
    write_aois_shapefile(aoi_gdf, shp_out_path)
# ========================= EOF ====================================================================
