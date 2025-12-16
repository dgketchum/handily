import os
import json
import logging
import math

import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

LOGGER = logging.getLogger("handily.viz")


def _to_wgs84_geojson(gdf):
    """
    Convert a GeoDataFrame to WGS84 (EPSG:4326) and return a GeoJSON string.
    Leaflet expects lon/lat in degrees.
    """
    if gdf is None or len(gdf) == 0:
        empty = json.dumps({"type": "FeatureCollection", "features": []})
        return empty
    gj = gdf.to_crs("EPSG:4326").to_json()
    return gj


def write_interactive_map(results, out_html, initial_threshold=2.0):
    """
    Write a self-contained HTML (Leaflet) for interactive debugging of REM thresholds.

    Layers:
    - AOI boundary
    - NHD flowlines (optionally filtered upstream)
    - Fields polygons colored by rem_mean compared to a user-controlled threshold

    The slider changes the partitioning classification (rem_mean < threshold) client-side,
    updating styling and counts without re-running Python.
    """
    aoi = results.get("aoi")
    flowlines = results.get("flowlines")
    fields_stats = results.get("fields_stats")

    if fields_stats is None or "rem_mean" not in fields_stats.columns:
        raise ValueError("fields_stats with 'rem_mean' is required in results for visualization.")

    aoi_gj = _to_wgs84_geojson(aoi)
    flow_geo = flowlines[['ObjectID', 'visibility', 'geometry']]
    flow_gj = _to_wgs84_geojson(flow_geo)
    fields_gj = _to_wgs84_geojson(fields_stats)

    # Optional raster overlays (REM/DEM) as PNGs with WGS84 bounds
    # Using single-image overlays keeps things simple without a tile pyramid.
    out_dir = os.path.dirname(os.path.abspath(out_html))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Bake XYZ tiles for REM/DEM with selectable color ramps
    rem_da = results.get("rem")
    dem_da = results.get("dem")

    rem_schemes = ["blue-red", "viridis", "terrain", "grayscale"]
    dem_schemes = ["grayscale", "terrain", "viridis"]

    rem_tiles_js = "const remLayers = {}\n"
    dem_tiles_js = "const demLayers = {}\n"

    if rem_da is not None:
        rem_tiles_dir = os.path.join(out_dir, "tiles", "rem")
        _bake_xyz_tiles(rem_da, rem_tiles_dir, mode="rem", schemes=rem_schemes, zmin=9, zmax=14)
        # Build JS dictionary
        parts = []
        for s in rem_schemes:
            url = f"tiles/rem/{s}/{{{{z}}}}/{{{{x}}}}/{{{{y}}}}.png"
            parts.append(f"  '{s}': L.tileLayer('{url}', {{ maxZoom: 22, opacity: 0.6 }})")
        rem_tiles_js = "const remLayers = {\n" + ",\n".join(parts) + "\n};\n"

    if dem_da is not None:
        dem_tiles_dir = os.path.join(out_dir, "tiles", "dem")
        _bake_xyz_tiles(dem_da, dem_tiles_dir, mode="dem", schemes=dem_schemes, zmin=9, zmax=14)
        parts = []
        for s in dem_schemes:
            url = f"tiles/dem/{s}/{{{{z}}}}/{{{{x}}}}/{{{{y}}}}.png"
            parts.append(f"  '{s}': L.tileLayer('{url}', {{ maxZoom: 22, opacity: 0.5 }})")
        dem_tiles_js = "const demLayers = {\n" + ",\n".join(parts) + "\n};\n"

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>HAND/REM Debug Map</title>
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <style>
    html, body, #map {{ height: 100%; margin: 0; padding: 0; }}
    .control-panel {{ position: absolute; top: 10px; left: 10px; z-index: 1000; background: #fff; padding: 8px 10px; border-radius: 4px; box-shadow: 0 1px 4px rgba(0,0,0,0.3); }}
    .legend {{ position: absolute; bottom: 10px; left: 10px; background: #fff; padding: 6px 8px; border-radius: 4px; font: 12px/14px Arial, sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,0.3); }}
    .legend .swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; vertical-align: middle; }}
  </style>
  <base target=\"_self\">
	  <meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'self' https://unpkg.com https://*.tile.openstreetmap.org; script-src 'self' 'unsafe-inline' https://unpkg.com; style-src 'self' 'unsafe-inline' https://unpkg.com;\" />
	  <!-- CSP keeps things tidy while allowing Leaflet and OSM tiles. -->
</head>
<body>
  <div id=\"map\"></div>
  <div class=\"control-panel\"> 
    <div><strong>REM Threshold (m)</strong></div>
    <input id=\"thr\" type=\"range\" min=\"0\" max=\"5\" step=\"0.1\" value=\"{float(initial_threshold)}\" />
    <span id=\"thrval\">{float(initial_threshold):.1f}</span>
    <div id=\"counts\" style=\"margin-top:6px\"></div>
    <div style=\"margin-top:6px\">REM Tiles <input type=\"checkbox\" id=\"toggleRem\" checked> 
      <select id=\"remRamp\"> 
        <option value=\"blue-red\">blue-red</option>
        <option value=\"viridis\">viridis</option>
        <option value=\"terrain\">terrain</option>
        <option value=\"grayscale\">grayscale</option>
      </select>
    </div>
    <div style=\"margin-top:6px\">DEM Tiles <input type=\"checkbox\" id=\"toggleDem\"> 
      <select id=\"demRamp\"> 
        <option value=\"grayscale\">grayscale</option>
        <option value=\"terrain\">terrain</option>
        <option value=\"viridis\">viridis</option>
      </select>
    </div>
  </div>
  <div class=\"legend\">
    <div><span class=\"swatch\" style=\"background:#d7191c\"></span> Partitioned (rem_mean &lt; threshold)</div>
    <div><span class=\"swatch\" style=\"background:#2c7bb6\"></span> Not partitioned</div>
  </div>
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <script>
    const AOI = {aoi_gj};
    const FLOW = {flow_gj};
    const FIELDS = {fields_gj};

    const map = L.map('map');
    // Multiple free basemaps with toggle
    const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19, attribution: '&copy; OpenStreetMap' }});
    const esri = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{x}}/{{y}}', {{ maxZoom: 19, attribution: 'Tiles &copy; Esri' }});
    const usgsTopo = L.tileLayer('https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{{z}}/{{x}}/{{y}}', {{ maxZoom: 19, attribution: 'USGS' }});
    osm.addTo(map);

    const aoiStyle = {{ color: '#222', weight: 2, fill: false }};
    const flowStyle = {{ color: '#008800', weight: 1 }};

    const aoiLayer = L.geoJSON(AOI, {{ style: aoiStyle }}).addTo(map);
    const flowLayer = L.geoJSON(FLOW, {{ style: flowStyle }}).addTo(map);

    const thr = document.getElementById('thr');
    const thrval = document.getElementById('thrval');
    const counts = document.getElementById('counts');
    const remRampSel = document.getElementById('remRamp');
    const demRampSel = document.getElementById('demRamp');
    const toggleRem = document.getElementById('toggleRem');
    const toggleDem = document.getElementById('toggleDem');

    function styleForFeature(f, t) {{
      const m = f.properties && typeof f.properties.rem_mean === 'number' ? f.properties.rem_mean : null;
      const part = (m !== null) && (m < t);
      return {{
        color: '#555',
        weight: 0.7,
        fill: true,
        fillOpacity: 0.5,
        fillColor: part ? '#d7191c' : '#2c7bb6'
      }};
    }}

    function onEachField(f, layer) {{
      const m = f.properties && typeof f.properties.rem_mean === 'number' ? f.properties.rem_mean : null;
      const txt = 'rem_mean: ' + (m === null ? 'n/a' : m.toFixed(2));
      layer.bindPopup(txt);
    }}

    let fieldsLayer = L.geoJSON(FIELDS, {{ style: (f)=>styleForFeature(f, parseFloat(thr.value)), onEachFeature: onEachField }}).addTo(map);

    function updateCounts(t) {{
      const feats = FIELDS.features || [];
      let tot = 0, part = 0;
      for (let i=0; i<feats.length; i++) {{
        const m = feats[i].properties && typeof feats[i].properties.rem_mean === 'number' ? feats[i].properties.rem_mean : null;
        if (m !== null) {{
          tot += 1;
          if (m < t) part += 1;
        }}
      }}
      counts.innerText = 'Fields: ' + part + ' / ' + tot + ' partitioned';
    }}

    function refresh() {{
      const t = parseFloat(thr.value);
      thrval.innerText = t.toFixed(1);
      fieldsLayer.setStyle((f)=>styleForFeature(f, t));
      updateCounts(t);
    }}

    thr.addEventListener('input', refresh);

    // Tile layers baked from REM/DEM (injected from Python)
{rem_tiles_js}
{dem_tiles_js}

    let activeRem = null;
    let activeDem = null;

    function setRemLayer(name) {{
      if (typeof remLayers === 'undefined' || !remLayers[name]) return;
      if (activeRem) {{ try {{ map.removeLayer(activeRem); }} catch(e) {{}} }}
      activeRem = remLayers[name];
      if (toggleRem.checked) activeRem.addTo(map);
    }}
    function setDemLayer(name) {{
      if (typeof demLayers === 'undefined' || !demLayers[name]) return;
      if (activeDem) {{ try {{ map.removeLayer(activeDem); }} catch(e) {{}} }}
      activeDem = demLayers[name];
      if (toggleDem.checked) activeDem.addTo(map);
    }}

    remRampSel.addEventListener('change', () => setRemLayer(remRampSel.value));
    demRampSel.addEventListener('change', () => setDemLayer(demRampSel.value));
    toggleRem.addEventListener('change', () => {{ if (activeRem) {{ if (toggleRem.checked) activeRem.addTo(map); else map.removeLayer(activeRem); }} }});
    toggleDem.addEventListener('change', () => {{ if (activeDem) {{ if (toggleDem.checked) activeDem.addTo(map); else map.removeLayer(activeDem); }} }});

    // Fit map to AOI or fields
    let fitBounds = null;
    try {{ fitBounds = aoiLayer.getBounds(); }} catch (e) {{}}
    if (!fitBounds || !fitBounds.isValid()) {{
      try {{ fitBounds = fieldsLayer.getBounds(); }} catch (e) {{}}
    }}
    if (fitBounds && fitBounds.isValid()) {{
      map.fitBounds(fitBounds.pad(0.05));
    }} else {{
      map.setView([45.0, -112.5], 9);
    }}

    // Base/overlay layer control (vector overlays)
    const baseLayers = {{ 'OpenStreetMap': osm, 'Esri World Imagery': esri, 'USGS Topo': usgsTopo }};
    const overlays = {{ 'AOI': aoiLayer, 'Flowlines': flowLayer, 'Fields': fieldsLayer }};
    L.control.layers(baseLayers, overlays, {{ collapsed: true }}).addTo(map);

    // Initial UI update
    updateCounts(parseFloat(thr.value));

    // Initialize tile layers
    if (typeof remLayers !== 'undefined' && remLayers['blue-red']) {{
      setRemLayer('blue-red');
    }}
    if (typeof demLayers !== 'undefined' && demLayers['grayscale']) {{
      // leave DEM off by default; toggle if checkbox is checked
      if (toggleDem.checked) setDemLayer('grayscale');
    }}
  </script>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html


# ----- Helpers: raster overlays -----

def _export_overlay_png(da, out_png, mode="rem"):
    """
    Colorize an xarray DataArray (DEM/REM) to a PNG with RGBA bands and return WGS84 bounds.
    The PNG is georeferenced implicitly via Leaflet bounds (no world file required).
    """
    if getattr(da, "rio", None) is None or da.rio.crs is None:
        raise ValueError("DataArray must have a valid CRS via rioxarray.")

    data = np.asarray(da.data)
    data = data.astype("float32")
    mask = np.isfinite(data)

    if not np.any(mask):
        raise ValueError("Overlay source has no finite values.")

    vals = data[mask]
    if str(mode) == "rem":
        vmin = 0.0
        vmax_guess = float(np.nanpercentile(vals, 99.0)) if vals.size > 100 else float(np.nanmax(vals))
        vmax = max(1.0, min(vmax_guess, 10.0))  # keep dynamic range reasonable for REM
        rgba = _colormap_linear(data, vmin, vmax, scheme="blue-red", mask=mask)
    else:
        # DEM grayscale by percentile stretch
        vmin = float(np.nanpercentile(vals, 2.0)) if vals.size > 100 else float(np.nanmin(vals))
        vmax = float(np.nanpercentile(vals, 98.0)) if vals.size > 100 else float(np.nanmax(vals))
        if vmax <= vmin:
            vmax = vmin + 1.0
        rgba = _colormap_linear(data, vmin, vmax, scheme="grayscale", mask=mask)

    # Write PNG (RGBA)
    h, w = rgba.shape[1], rgba.shape[2]
    profile = {
        "driver": "PNG",
        "width": w,
        "height": h,
        "count": 4,
        "dtype": "uint8",
    }
    with rasterio.open(out_png, "w", **profile) as dst:
        dst.write(rgba[0], 1)
        dst.write(rgba[1], 2)
        dst.write(rgba[2], 3)
        dst.write(rgba[3], 4)

    # Bounds in WGS84 for Leaflet
    minx, miny, maxx, maxy = da.rio.bounds()
    g = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=da.rio.crs)
    bnd = g.to_crs("EPSG:4326").total_bounds  # [minx, miny, maxx, maxy]
    south, west, north, east = float(bnd[1]), float(bnd[0]), float(bnd[3]), float(bnd[2])
    return (south, west), (north, east)


def _colormap_linear(arr, vmin, vmax, scheme="blue-red", mask=None):
    """
    Map data in [vmin, vmax] to RGBA uint8 using simple ramps.
    - 'blue-red': blue (#2c7bb6) -> red (#d7191c)
    - 'grayscale': 0..255 gray
    - 'terrain': dark green -> yellow -> brown -> white
    - 'viridis': approximate using key stops
    Returns array shaped (4, H, W).
    """
    a = np.asarray(arr, dtype="float32")
    if mask is None:
        mask = np.isfinite(a)
    norm = (a - float(vmin)) / max(1e-6, float(vmax) - float(vmin))
    norm = np.clip(norm, 0.0, 1.0)

    h, w = a.shape[-2], a.shape[-1]
    r = np.zeros((h, w), dtype="float32")
    g = np.zeros((h, w), dtype="float32")
    b = np.zeros((h, w), dtype="float32")
    alpha = np.zeros((h, w), dtype="float32")

    def lerp(a0, a1, t):
        return a0 + (a1 - a0) * t

    def apply_stops(stops, t):
        # stops: list of (pos in 0..1, (r,g,b)) ascending
        if t <= stops[0][0]:
            return stops[0][1]
        if t >= stops[-1][0]:
            return stops[-1][1]
        for i in range(len(stops) - 1):
            p0, c0 = stops[i]
            p1, c1 = stops[i + 1]
            if t >= p0 and t <= p1:
                tt = (t - p0) / max(1e-6, (p1 - p0))
                return (
                    lerp(c0[0], c1[0], tt),
                    lerp(c0[1], c1[1], tt),
                    lerp(c0[2], c1[2], tt),
                )
        return stops[-1][1]

    if scheme == "blue-red":
        stops = [
            (0.0, (44, 123, 182)),  # #2c7bb6
            (1.0, (215, 25, 28)),  # #d7191c
        ]
        alpha_val = 200.0
    elif scheme == "grayscale":
        # handled separately below
        stops = None
        alpha_val = 180.0
    elif scheme == "terrain":
        stops = [
            (0.0, (26, 102, 26)),  # dark green
            (0.35, (190, 190, 60)),  # yellowish
            (0.7, (160, 82, 45)),  # brown
            (1.0, (245, 245, 245)),  # near white
        ]
        alpha_val = 180.0
    elif scheme == "viridis":
        stops = [
            (0.0, (68, 1, 84)),  # #440154
            (0.33, (49, 104, 142)),  # #31688e
            (0.66, (53, 183, 121)),  # #35b779
            (1.0, (253, 231, 37)),  # #fde725
        ]
        alpha_val = 180.0
    else:
        # default to grayscale
        stops = None
        alpha_val = 180.0

    if stops is None and scheme == "grayscale":
        gray = 255.0 * norm
        r = gray
        g = gray
        b = gray
    else:
        # vectorized stop interpolation
        # approximate by binning t into 256 steps for performance
        tvals = norm
        rmap = np.zeros_like(tvals)
        gmap = np.zeros_like(tvals)
        bmap = np.zeros_like(tvals)
        # evaluate by mapping each pixel independently; vectorization via np.nditer could be heavy
        # tradeoff: compute 256-step LUT
        lut_r = np.zeros(256, dtype="float32")
        lut_g = np.zeros(256, dtype="float32")
        lut_b = np.zeros(256, dtype="float32")
        for i in range(256):
            ti = i / 255.0
            rr, gg, bb = apply_stops(stops, ti)
            lut_r[i] = rr
            lut_g[i] = gg
            lut_b[i] = bb
        idx = (tvals * 255.0).astype("int32")
        idx = np.clip(idx, 0, 255)
        r = lut_r[idx]
        g = lut_g[idx]
        b = lut_b[idx]

    alpha[mask] = alpha_val

    r = r.astype("uint8")
    g = g.astype("uint8")
    b = b.astype("uint8")
    a8 = alpha.astype("uint8")
    rgba = np.stack([r, g, b, a8], axis=0)
    return rgba


def _compute_vminmax(da, mode):
    data = np.asarray(da.data).astype("float32")
    mask = np.isfinite(data)
    vals = data[mask]
    if vals.size == 0:
        return 0.0, 1.0
    if str(mode) == "rem":
        vmin = 0.0
        vmax_guess = float(np.nanpercentile(vals, 99.0)) if vals.size > 100 else float(np.nanmax(vals))
        vmax = max(1.0, min(vmax_guess, 10.0))
        return vmin, vmax
    # DEM
    vmin = float(np.nanpercentile(vals, 2.0)) if vals.size > 100 else float(np.nanmin(vals))
    vmax = float(np.nanpercentile(vals, 98.0)) if vals.size > 100 else float(np.nanmax(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _lonlat_bounds_from_tile(z, x, y):
    n = float(1 << int(z))
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1.0) / n * 360.0 - 180.0

    def lat_from_y(ty):
        t = math.pi * (1.0 - 2.0 * (ty / n))
        return math.degrees(math.atan(math.sinh(t)))

    lat_n = lat_from_y(float(y))
    lat_s = lat_from_y(float(y + 1))
    return lon_w, lat_s, lon_e, lat_n


def _merc_from_lonlat(lon, lat):
    # Web Mercator
    max_lat = 85.05112878
    lat = max(-max_lat, min(max_lat, float(lat)))
    x = 6378137.0 * math.radians(float(lon))
    y = 6378137.0 * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def _bake_xyz_tiles(da, out_root, mode, schemes, zmin=9, zmax=14):
    # Compute global scaling for consistent colors
    vmin, vmax = _compute_vminmax(da, mode)
    data = np.asarray(da.data).astype("float32")
    src_transform = da.rio.transform()
    src_crs = da.rio.crs
    nd = da.rio.nodata

    # Bounds in lon/lat for tile coverage
    minx, miny, maxx, maxy = da.rio.bounds()
    g = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=src_crs)
    lonmin, latmin, lonmax, latmax = g.to_crs("EPSG:4326").total_bounds

    for z in range(int(zmin), int(zmax) + 1):
        n = 1 << z
        # x range
        x0 = int(math.floor((lonmin + 180.0) / 360.0 * n))
        x1 = int(math.floor((lonmax + 180.0) / 360.0 * n))
        x0 = max(0, min(n - 1, x0))
        x1 = max(0, min(n - 1, x1))

        # y range (note: y increases southward)
        def y_from_lat(lat):
            lat = max(-85.05112878, min(85.05112878, float(lat)))
            lat_rad = math.radians(lat)
            return int(math.floor((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n))

        y0 = y_from_lat(latmax)
        y1 = y_from_lat(latmin)
        y0 = max(0, min(n - 1, y0))
        y1 = max(0, min(n - 1, y1))

        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                lon_w, lat_s, lon_e, lat_n = _lonlat_bounds_from_tile(z, x, y)
                minxm, minym = _merc_from_lonlat(lon_w, lat_s)
                maxxm, maxym = _merc_from_lonlat(lon_e, lat_n)
                dst_transform = from_bounds(minxm, minym, maxxm, maxym, 256, 256)
                dst = np.full((256, 256), np.nan, dtype="float32")
                reproject(
                    source=data,
                    destination=dst,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    src_nodata=nd if nd is not None else np.nan,
                    dst_transform=dst_transform,
                    dst_crs="EPSG:3857",
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )

                mask = np.isfinite(dst)
                for scheme in schemes:
                    rgba = _colormap_linear(dst, vmin, vmax, scheme=scheme, mask=mask)
                    out_png = os.path.join(out_root, scheme, str(z), str(x), f"{y}.png")
                    out_dir = os.path.dirname(out_png)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    profile = {
                        "driver": "PNG",
                        "width": 256,
                        "height": 256,
                        "count": 4,
                        "dtype": "uint8",
                    }
                    with rasterio.open(out_png, "w", **profile) as dst_img:
                        dst_img.write(rgba[0], 1)
                        dst_img.write(rgba[1], 2)
                        dst_img.write(rgba[2], 3)
                        dst_img.write(rgba[3], 4)

# EOF
