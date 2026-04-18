"""Paired-reach side strips using the overall DEM aspect.

For two groups of reaches whose frames stay within some maximum separation of
each other, :func:`paired_sides_aspect_strips`:

1. composes each side into a single ordered LineString (frames and snapped
   thalwegs), inserting straight bridges through internal gaps,
2. clips each composite frame to the sub-segment where distance to the other
   composite frame is <= ``max_frame_sep_m`` (defaults to 2 km),
3. transfers that along-extent to each composite snapped thalweg by projecting
   the clipped-frame endpoints onto the thalweg,
4. closes a polygon from the two clipped composite thalwegs plus straight
   end-connectors — so every thalweg anchor sits on the polygon boundary,
5. computes the overall downhill direction of the DEM inside that polygon,
6. emits strips whose orientation is normal to the aspect (i.e. along
   contours), anchored at regularly-spaced stations on the *individual* reach
   thalwegs of both sides, clipped to the polygon.

:func:`paired_reach_aspect_strips` is a thin wrapper for the single-reach
case on each side.

Unlike the reach-local cross-section logic in :mod:`handily.rem_frame`, strip
orientation here is decoupled from local frame tangents; it is a single vector
derived from the bulk DEM gradient inside the shared polygon.

The module-level driver (:func:`main`) runs this on reaches 7 vs 8 (braided
bight) and on reaches (2, 6, 12, 10) vs (9) (open-ended inter-reach zone) of
the NV AOI 0773 debug subset and writes the outputs to
``/data/ssd2/handily/nv/aoi_0773/experimental/``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

DEBUG_DIR = Path("/data/ssd2/handily/nv/aoi_0773/debug_subset")
OUT_DIR = Path("/data/ssd2/handily/nv/aoi_0773/experimental")
TARGET_REACHES = (7, 8)
TARGET_SIDES = ([2, 6, 12, 10], [9])

MAX_FRAME_SEP_M = 2000.0
SAMPLE_SPACING_M = 10.0
STRIP_SPACING_M = 20.0
MAX_STRIP_LEN_M = 3000.0
RIDGE_PROMINENCE_M = 10.0
DESCEND_STOP_M = 5.0
RAY_STEP_M = 2.0

# Snap config: NHD as prior, more permissive than debug_subset_0773
RESNAP_MAX_OFFSET_M = 50.0
RESNAP_SEARCH_SPACING_M = 3.0
RESNAP_STATION_SPACING_M = 5.0
RESNAP_W_ELEV = 0.45
RESNAP_W_WATER = 0.4
RESNAP_W_PRIOR = 0.15
RESNAP_W_TRANSITION = 1.0
RESNAP_SMOOTHING_M = 100.0

OPEN_EDGE_ENDPOINT_TOL_M = 5.0
OPEN_EDGE_MIN_STABLE_LEN_M = 100.0
OPEN_EDGE_JUNCTION_TOL_M = 1.0
OPEN_EDGE_TOUCH_TOL_M = 1.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _remove_loops(line: LineString, tol: float = 1.0) -> LineString:
    """Remove self-intersection loops from a LineString.

    Walks the coordinate sequence and detects where a later segment crosses
    an earlier one. When a crossing is found, the loop between the two
    segments is excised and the walk continues from the crossing point.
    """
    coords = np.array(line.coords, dtype=np.float64)[:, :2]
    n = len(coords)
    if n < 4:
        return line

    # Walk forward, emitting cleaned coords
    out = [coords[0].copy()]
    i = 0
    while i < n - 1:
        # Current segment: coords[i] → coords[i+1]
        ax, ay = coords[i]
        bx, by = coords[i + 1]
        # Check if any later segment crosses this one
        found_cross = False
        for j in range(i + 2, n - 1):
            cx, cy = coords[j]
            dx, dy = coords[j + 1]
            # Line-line intersection test
            denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
            if abs(denom) < 1e-12:
                continue
            t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom
            u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom
            if 0.0 < t < 1.0 and 0.0 < u < 1.0:
                # Intersection point — skip the loop from i+1..j
                ix_x = ax + t * (bx - ax)
                ix_y = ay + t * (by - ay)
                out.append(np.array([ix_x, ix_y]))
                i = j + 1
                if i < n:
                    out.append(coords[i].copy())
                found_cross = True
                break
        if not found_cross:
            out.append(coords[i + 1].copy())
            i += 1

    if len(out) < 2:
        return line
    return LineString(out)


def _clip_line_near_other(
    line: LineString,
    other: LineString,
    max_dist_m: float,
    sample_spacing_m: float,
) -> LineString | None:
    """Return the longest sub-segment of ``line`` whose distance to ``other``
    stays <= ``max_dist_m``. Samples ``line`` at ``sample_spacing_m``.
    """
    total = float(line.length)
    if total <= 0.0:
        return None
    n = max(2, int(np.ceil(total / sample_spacing_m)) + 1)
    s_vals = np.linspace(0.0, total, n)
    close = np.array(
        [line.interpolate(float(s)).distance(other) <= max_dist_m for s in s_vals]
    )
    if not np.any(close):
        return None
    best = (0, 0)
    i = 0
    while i < n:
        if close[i]:
            j = i
            while j + 1 < n and close[j + 1]:
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j + 1
        else:
            i += 1
    i0, i1 = best
    if i1 <= i0:
        return None
    return substring(line, float(s_vals[i0]), float(s_vals[i1]))


def _compose_ordered_line(
    geoms_in_order: list[LineString], junction_tol_m: float = 1.0
) -> tuple[LineString, list[LineString]]:
    """Concatenate LineStrings in the given order into a single LineString.

    Each subsequent geometry is auto-flipped so its start is closer to the
    running end than its end is. Gaps larger than ``junction_tol_m`` are
    closed with a straight-line bridge vertex; the bridge segments are
    returned separately for reference/visualization. Input order is
    respected (no connectivity-based reordering is performed).
    """
    if not geoms_in_order:
        raise ValueError("_compose_ordered_line requires at least one geometry")

    first_coords = [(float(c[0]), float(c[1])) for c in geoms_in_order[0].coords]
    coords: list[tuple[float, float]] = list(first_coords)
    bridges: list[LineString] = []
    prev_end = np.asarray(coords[-1], dtype=np.float64)

    for g in geoms_in_order[1:]:
        g_coords = [(float(c[0]), float(c[1])) for c in g.coords]
        s_xy = np.asarray(g_coords[0], dtype=np.float64)
        e_xy = np.asarray(g_coords[-1], dtype=np.float64)
        if float(np.linalg.norm(e_xy - prev_end)) < float(
            np.linalg.norm(s_xy - prev_end)
        ):
            g_coords = g_coords[::-1]
            s_xy, e_xy = e_xy, s_xy
        gap = float(np.linalg.norm(s_xy - prev_end))
        if gap > junction_tol_m:
            bridges.append(LineString([tuple(prev_end.tolist()), tuple(s_xy.tolist())]))
            coords.extend(g_coords)
        else:
            coords.extend(g_coords[1:])
        prev_end = e_xy

    return LineString(coords), bridges


def _transfer_clip_to_thalweg(
    clipped_frame: LineString, thalweg: LineString
) -> LineString | None:
    """Project the endpoints of a frame-clipped sub-segment onto ``thalweg``
    and return the corresponding thalweg sub-segment. Frame and thalweg share
    station indices, so the projected s-values closely track the along-reach
    range of the clip.
    """
    if thalweg.length <= 0:
        return None
    coords = list(clipped_frame.coords)
    s_a = float(thalweg.project(Point(coords[0])))
    s_b = float(thalweg.project(Point(coords[-1])))
    s_lo, s_hi = sorted((s_a, s_b))
    if s_hi - s_lo <= 1e-6:
        return None
    return substring(thalweg, s_lo, s_hi)


def _polygon_from_adjacent_lines(line_a: LineString, line_b: LineString) -> Polygon:
    """Close a polygon by walking ``line_a`` start→end then ``line_b`` oriented
    so its endpoint nearest ``line_a``'s end connects first.
    """
    a_coords = [tuple(c[:2]) for c in line_a.coords]
    b_coords = [tuple(c[:2]) for c in line_b.coords]
    a_end = np.asarray(a_coords[-1], dtype=np.float64)
    b0 = np.asarray(b_coords[0], dtype=np.float64)
    b1 = np.asarray(b_coords[-1], dtype=np.float64)
    if float(np.linalg.norm(a_end - b1)) <= float(np.linalg.norm(a_end - b0)):
        ring = a_coords + b_coords[::-1] + [a_coords[0]]
    else:
        ring = a_coords + b_coords + [a_coords[0]]
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
    return poly


def _compute_downhill_aspect(dem_da: xr.DataArray, polygon: Polygon) -> np.ndarray:
    """Return the mean unit downhill-direction vector of the DEM inside
    ``polygon`` in DEM-CRS coordinates (x, y).
    """
    clipped = dem_da.rio.clip(
        [polygon], crs=dem_da.rio.crs, drop=True, all_touched=True
    )
    x_vals = clipped.x.values.astype(np.float64)
    y_vals = clipped.y.values.astype(np.float64)
    if x_vals.size < 2 or y_vals.size < 2:
        raise ValueError("Polygon clip produced insufficient DEM pixels")

    arr = np.asarray(clipped.values, dtype=np.float64)
    nodata = clipped.rio.nodata
    if nodata is not None and np.isfinite(nodata):
        arr = np.where(arr == nodata, np.nan, arr)

    dx = abs(float(x_vals[1] - x_vals[0]))
    dy = abs(float(y_vals[1] - y_vals[0]))
    y_descending = y_vals[0] > y_vals[-1]

    # np.gradient returns (d/drow, d/dcol); row index increases toward y[-1].
    dz_drow, dz_dcol = np.gradient(arr, dy, dx)
    dz_dx = dz_dcol
    dz_dy = -dz_drow if y_descending else dz_drow

    down_x = -dz_dx
    down_y = -dz_dy
    valid = np.isfinite(down_x) & np.isfinite(down_y)
    if not np.any(valid):
        raise ValueError("No finite DEM gradient inside polygon")

    mean_x = float(np.mean(down_x[valid]))
    mean_y = float(np.mean(down_y[valid]))
    norm = float(np.hypot(mean_x, mean_y))
    if norm <= 1e-9:
        raise ValueError("Downhill direction is degenerate (zero mean gradient)")
    return np.array([mean_x / norm, mean_y / norm], dtype=np.float64)


def _intersection_points(geom) -> list[tuple[float, float]]:
    """Return all distinct point-like coordinates from a shapely intersection
    result (Point, MultiPoint, LineString, MultiLineString, GeometryCollection).
    """
    if geom is None or geom.is_empty:
        return []
    gt = geom.geom_type
    if gt == "Point":
        return [(float(geom.x), float(geom.y))]
    if gt == "MultiPoint":
        return [(float(p.x), float(p.y)) for p in geom.geoms]
    if gt == "LineString":
        return [(float(c[0]), float(c[1])) for c in geom.coords]
    if gt in ("MultiLineString", "GeometryCollection"):
        out: list[tuple[float, float]] = []
        for part in geom.geoms:
            out.extend(_intersection_points(part))
        return out
    return []


def _truncate_strip_at_obstacles(
    strip: LineString,
    anchor_pt: Point,
    obstacle_geoms: list[LineString],
    obstacle_reach_ids: list[int],
    anchor_reach_id: int,
    reach_tree,
    eps_m: float = 1e-3,
) -> LineString | None:
    """Truncate a polygon-clipped strip at the first non-anchor snapped-reach
    crossing on each side of the anchor.

    Returns the sub-strip between the nearest plus-side and nearest minus-side
    crossings (or the polygon-boundary ends if no crossing exists on that
    side). Returns None if the truncated strip degenerates.
    """
    s_anchor = float(strip.project(anchor_pt))
    total = float(strip.length)
    s_minus_cut = 0.0
    s_plus_cut = total

    candidate_idxs = (
        reach_tree.query(strip)
        if reach_tree is not None
        else range(len(obstacle_geoms))
    )
    for idx in candidate_idxs:
        rid = obstacle_reach_ids[idx]
        if rid == anchor_reach_id:
            continue
        other = obstacle_geoms[idx]
        ix = strip.intersection(other)
        if ix.is_empty:
            continue
        for px, py in _intersection_points(ix):
            s_pt = float(strip.project(Point(px, py)))
            if s_pt > s_anchor + eps_m:
                if s_pt < s_plus_cut:
                    s_plus_cut = s_pt
            elif s_pt < s_anchor - eps_m:
                if s_pt > s_minus_cut:
                    s_minus_cut = s_pt

    if s_plus_cut - s_minus_cut <= eps_m:
        return None
    if s_minus_cut <= eps_m and s_plus_cut >= total - eps_m:
        return strip
    truncated = substring(strip, s_minus_cut, s_plus_cut)
    if not isinstance(truncated, LineString) or truncated.length <= eps_m:
        return None
    return truncated


def _place_strips_along_anchor(
    anchor_line: LineString,
    strip_dir: np.ndarray,
    polygon: Polygon,
    spacing_m: float,
    max_strip_len_m: float,
    reach_id: int,
    obstacle_geoms: list[LineString] | None = None,
    obstacle_reach_ids: list[int] | None = None,
    reach_tree=None,
) -> list[dict]:
    """Walk ``anchor_line`` at ``spacing_m`` and cast a bidirectional ray in
    ``strip_dir``, clipped to ``polygon`` and truncated at the nearest
    non-self snapped-reach crossing on each side.
    """
    total = float(anchor_line.length)
    if total <= 0.0:
        return []
    s_vals = np.arange(spacing_m * 0.5, total + 1e-6, spacing_m)
    if len(s_vals) == 0:
        s_vals = np.array([total * 0.5])

    # Anchors outside the polygon boundary produce disconnected strip pieces;
    # skip them. Small tolerance absorbs float noise on boundary points.
    boundary_tol = 1e-3
    obstacle_geoms = obstacle_geoms or []
    obstacle_reach_ids = obstacle_reach_ids or []

    rows: list[dict] = []
    for i, s in enumerate(s_vals):
        anchor_pt = anchor_line.interpolate(float(s))
        anchor_xy = np.array([anchor_pt.x, anchor_pt.y], dtype=np.float64)
        if polygon.distance(anchor_pt) > boundary_tol:
            continue
        p_plus = anchor_xy + max_strip_len_m * strip_dir
        p_minus = anchor_xy - max_strip_len_m * strip_dir
        full = LineString([tuple(p_minus), tuple(anchor_xy), tuple(p_plus)])
        ix = full.intersection(polygon)
        if ix.is_empty:
            continue
        if ix.geom_type == "MultiLineString":
            ix = min(ix.geoms, key=lambda g: g.distance(anchor_pt))
        if ix.geom_type != "LineString":
            continue
        if obstacle_geoms:
            ix = _truncate_strip_at_obstacles(
                ix,
                anchor_pt,
                obstacle_geoms,
                obstacle_reach_ids,
                int(reach_id),
                reach_tree,
            )
            if ix is None:
                continue
        rows.append(
            {
                "reach_id": int(reach_id),
                "station_id": int(i),
                "s_m": float(s),
                "anchor_x": float(anchor_xy[0]),
                "anchor_y": float(anchor_xy[1]),
                "length_m": float(ix.length),
                "geometry": ix,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


@dataclass
class PairedStripResult:
    polygon: Polygon
    aspect_down: np.ndarray  # unit downhill vector (x, y)
    strip_dir: np.ndarray  # unit strip direction (perpendicular to aspect)
    clipped_frame_a: LineString
    clipped_frame_b: LineString
    clipped_thalweg_a: LineString
    clipped_thalweg_b: LineString
    bridges_a: list[LineString]
    bridges_b: list[LineString]
    strips: gpd.GeoDataFrame


def paired_sides_aspect_strips(
    frame_gdf: gpd.GeoDataFrame,
    snapped_gdf: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
    side_a_reaches: list[int],
    side_b_reaches: list[int],
    max_frame_sep_m: float = MAX_FRAME_SEP_M,
    sample_spacing_m: float = SAMPLE_SPACING_M,
    strip_spacing_m: float = STRIP_SPACING_M,
    max_strip_len_m: float = MAX_STRIP_LEN_M,
    junction_tol_m: float = 1.0,
) -> PairedStripResult | None:
    """Build aspect-normal side strips between two groups of reaches.

    Each side's reaches are concatenated in the given order (with straight
    bridges across internal gaps) into a single composite LineString for both
    the smoothed frame and the snapped thalweg. The 2-km separation test runs
    on the composite frames; anchors are placed at per-reach stations on the
    original snapped thalwegs, and anchors falling outside the resulting
    polygon are skipped.

    Parameters
    ----------
    frame_gdf, snapped_gdf
        GeoDataFrames with a ``reach_id`` column and a single LineString per
        reach. Must share the same CRS.
    dem_da
        DEM as an xarray DataArray with rioxarray-attached CRS matching the
        vector data.
    side_a_reaches, side_b_reaches
        Ordered lists of reach IDs forming each bounding side.
    max_frame_sep_m
        Only the portion of the composite frames whose mutual distance is
        <= this value contributes to the polygon (defaults to 2 km).
    junction_tol_m
        Maximum endpoint-to-endpoint gap treated as a real junction during
        side composition; larger gaps insert a straight-line bridge.

    Returns
    -------
    PairedStripResult or None
        ``None`` if any requested reach is missing, no composite frame segment
        stays within ``max_frame_sep_m``, or the polygon/aspect is degenerate.
    """
    if not side_a_reaches or not side_b_reaches:
        return None

    def lookup(gdf: gpd.GeoDataFrame, rid: int) -> LineString | None:
        row = gdf[gdf["reach_id"] == rid]
        return row.iloc[0].geometry if not row.empty else None

    frames_a = [lookup(frame_gdf, r) for r in side_a_reaches]
    frames_b = [lookup(frame_gdf, r) for r in side_b_reaches]
    snaps_a = [lookup(snapped_gdf, r) for r in side_a_reaches]
    snaps_b = [lookup(snapped_gdf, r) for r in side_b_reaches]
    if any(g is None for g in frames_a + frames_b + snaps_a + snaps_b):
        return None

    comp_frame_a, bridges_frame_a = _compose_ordered_line(frames_a, junction_tol_m)
    comp_frame_b, bridges_frame_b = _compose_ordered_line(frames_b, junction_tol_m)
    comp_snap_a, bridges_snap_a = _compose_ordered_line(snaps_a, junction_tol_m)
    comp_snap_b, bridges_snap_b = _compose_ordered_line(snaps_b, junction_tol_m)

    clip_frame_a = _clip_line_near_other(
        comp_frame_a, comp_frame_b, max_frame_sep_m, sample_spacing_m
    )
    clip_frame_b = _clip_line_near_other(
        comp_frame_b, comp_frame_a, max_frame_sep_m, sample_spacing_m
    )
    if clip_frame_a is None or clip_frame_b is None:
        return None

    clip_tw_a = _transfer_clip_to_thalweg(clip_frame_a, comp_snap_a)
    clip_tw_b = _transfer_clip_to_thalweg(clip_frame_b, comp_snap_b)
    if clip_tw_a is None or clip_tw_b is None:
        return None

    polygon = _polygon_from_adjacent_lines(clip_tw_a, clip_tw_b)
    if polygon.is_empty or polygon.area <= 0:
        return None

    aspect_down = _compute_downhill_aspect(dem_da, polygon)
    strip_dir = np.array(
        [-aspect_down[1], aspect_down[0]], dtype=np.float64
    )  # rotate +90°

    # Build obstacle set from every snapped reach (interior reaches like a
    # braided parallel channel would otherwise be crossed by strips).
    from shapely import STRtree

    obstacle_geoms: list[LineString] = []
    obstacle_reach_ids: list[int] = []
    for _, row in snapped_gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        obstacle_geoms.append(g)
        obstacle_reach_ids.append(int(row["reach_id"]))
    reach_tree = STRtree(obstacle_geoms) if obstacle_geoms else None

    rows: list[dict] = []
    for rid, snap_line in zip(side_a_reaches, snaps_a):
        rows.extend(
            _place_strips_along_anchor(
                snap_line,
                strip_dir,
                polygon,
                strip_spacing_m,
                max_strip_len_m,
                rid,
                obstacle_geoms=obstacle_geoms,
                obstacle_reach_ids=obstacle_reach_ids,
                reach_tree=reach_tree,
            )
        )
    for rid, snap_line in zip(side_b_reaches, snaps_b):
        rows.extend(
            _place_strips_along_anchor(
                snap_line,
                strip_dir,
                polygon,
                strip_spacing_m,
                max_strip_len_m,
                rid,
                obstacle_geoms=obstacle_geoms,
                obstacle_reach_ids=obstacle_reach_ids,
                reach_tree=reach_tree,
            )
        )
    if not rows:
        return None

    strips_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=frame_gdf.crs)
    return PairedStripResult(
        polygon=polygon,
        aspect_down=aspect_down,
        strip_dir=strip_dir,
        clipped_frame_a=clip_frame_a,
        clipped_frame_b=clip_frame_b,
        clipped_thalweg_a=clip_tw_a,
        clipped_thalweg_b=clip_tw_b,
        bridges_a=bridges_snap_a,
        bridges_b=bridges_snap_b,
        strips=strips_gdf,
    )


def paired_reach_aspect_strips(
    frame_gdf: gpd.GeoDataFrame,
    snapped_gdf: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
    reach_a: int,
    reach_b: int,
    max_frame_sep_m: float = MAX_FRAME_SEP_M,
    sample_spacing_m: float = SAMPLE_SPACING_M,
    strip_spacing_m: float = STRIP_SPACING_M,
    max_strip_len_m: float = MAX_STRIP_LEN_M,
) -> PairedStripResult | None:
    """Single-reach-per-side wrapper for :func:`paired_sides_aspect_strips`."""
    return paired_sides_aspect_strips(
        frame_gdf,
        snapped_gdf,
        dem_da,
        side_a_reaches=[reach_a],
        side_b_reaches=[reach_b],
        max_frame_sep_m=max_frame_sep_m,
        sample_spacing_m=sample_spacing_m,
        strip_spacing_m=strip_spacing_m,
        max_strip_len_m=max_strip_len_m,
    )


# ---------------------------------------------------------------------------
# Network polygonize + edge classification
# ---------------------------------------------------------------------------


@dataclass
class NetworkFaces:
    """Result of polygonizing the snapped-reach network."""

    polygons: list[Polygon]
    closure_edges: list[LineString]
    reach_side_map: dict[tuple[int, str], int | None]
    polygon_aspects: dict[int, np.ndarray]
    min_area_m2: float = 0.0


def polygonize_reach_network(
    snapped_gdf: gpd.GeoDataFrame,
    dem_da: xr.DataArray | None = None,
    closure_max_dist_m: float = 2000.0,
    junction_tol_m: float = 2.0,
    min_polygon_area_m2: float = 100.0,
) -> tuple[list[Polygon], list[LineString]]:
    """Polygonize the snapped-reach network with closure edges.

    Steps:
    1. Find degree-1 endpoints (dangles) in the reach graph.
    2. Greedily pair nearby dangles (< ``closure_max_dist_m``) and connect
       with straight-line closure edges.
    3. Connect remaining unpaired dangles to the DEM extent boundary.
    4. Add DEM extent boundary itself as an outer closure edge.
    5. Node everything via ``unary_union`` and ``polygonize``.
    6. Drop sliver polygons below ``min_polygon_area_m2``.

    Returns ``(polygons, closure_edges)``.
    """
    from shapely.ops import polygonize, unary_union
    from scipy.spatial import cKDTree

    reach_geoms = [g for g in snapped_gdf.geometry if g is not None and not g.is_empty]
    if not reach_geoms:
        return [], []

    # --- Collect endpoints and find dangles (degree-1 nodes) ---
    ep_xy: list[np.ndarray] = []
    for g in reach_geoms:
        c = list(g.coords)
        ep_xy.append(np.asarray(c[0][:2], dtype=np.float64))
        ep_xy.append(np.asarray(c[-1][:2], dtype=np.float64))
    pts = np.array(ep_xy)
    tree = cKDTree(pts)

    parent = list(range(len(pts)))

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in tree.query_pairs(junction_tol_m):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    group_count = Counter(_find(i) for i in range(len(pts)))
    dangle_xy = [pts[i] for i in range(len(pts)) if group_count[_find(i)] == 1]

    # --- Pair nearby dangles with greedy nearest-neighbor ---
    closure_edges: list[LineString] = []
    used_dangles: set[int] = set()
    if len(dangle_xy) >= 2:
        d_arr = np.array(dangle_xy)
        dtree = cKDTree(d_arr)
        dd, ii = dtree.query(d_arr, k=min(len(d_arr), 2))
        pair_list = sorted(
            (float(dd[i, 1]), i, int(ii[i, 1])) for i in range(len(d_arr))
        )
        for d, a, b in pair_list:
            if d > closure_max_dist_m:
                break
            if a in used_dangles or b in used_dangles:
                continue
            closure_edges.append(LineString([tuple(d_arr[a]), tuple(d_arr[b])]))
            used_dangles.add(a)
            used_dangles.add(b)

    # --- Connect remaining dangles to DEM boundary ---
    dem_boundary = None
    if dem_da is not None:
        xmin = float(dem_da.x.values.min())
        xmax = float(dem_da.x.values.max())
        ymin = float(dem_da.y.values.min())
        ymax = float(dem_da.y.values.max())
        dem_boundary = Polygon(
            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        ).boundary
        for i, dxy in enumerate(dangle_xy):
            if i in used_dangles:
                continue
            dp = Point(dxy)
            nearest = dem_boundary.interpolate(dem_boundary.project(dp))
            if dp.distance(nearest) < closure_max_dist_m:
                closure_edges.append(LineString([tuple(dxy), (nearest.x, nearest.y)]))

    # --- Polygonize ---
    all_edges: list = list(reach_geoms) + closure_edges
    if dem_boundary is not None:
        all_edges.append(dem_boundary)
    noded = unary_union(all_edges)
    polys = [p for p in polygonize(noded) if p.area >= min_polygon_area_m2]
    return polys, closure_edges


def classify_reach_sides(
    snapped_gdf: gpd.GeoDataFrame,
    polygons: list[Polygon],
    n_probes: int = 5,
    probe_offset_m: float = 5.0,
) -> dict[tuple[int, str], int | None]:
    """Determine which polygon (if any) is on each side of each reach.

    For each ``(reach_id, side)`` with ``side`` in ``{"left", "right"}``
    relative to the reach's start→end direction, the function casts
    ``n_probes`` offset points along the reach and takes the majority-vote
    polygon hit (or ``None`` for the exterior).
    """
    from shapely import STRtree

    result: dict[tuple[int, str], int | None] = {}
    if not polygons:
        for _, row in snapped_gdf.iterrows():
            rid = int(row["reach_id"])
            result[(rid, "left")] = None
            result[(rid, "right")] = None
        return result

    poly_tree = STRtree(polygons)

    for _, row in snapped_gdf.iterrows():
        rid = int(row["reach_id"])
        geom = row.geometry
        if geom is None or geom.is_empty or geom.length < 1e-6:
            result[(rid, "left")] = None
            result[(rid, "right")] = None
            continue

        total = float(geom.length)
        probe_s = np.linspace(total * 0.1, total * 0.9, n_probes)
        eps = min(1.0, total * 0.01)

        for side_label, sign in [("left", 1.0), ("right", -1.0)]:
            votes: Counter = Counter()
            for s in probe_s:
                p0 = geom.interpolate(max(float(s) - eps, 0.0))
                p1 = geom.interpolate(min(float(s) + eps, total))
                tx = p1.x - p0.x
                ty = p1.y - p0.y
                tn = float(np.hypot(tx, ty))
                if tn < 1e-9:
                    continue
                nx = -ty / tn * sign
                ny = tx / tn * sign
                base = geom.interpolate(float(s))
                probe_pt = Point(
                    base.x + probe_offset_m * nx,
                    base.y + probe_offset_m * ny,
                )
                hits = poly_tree.query(probe_pt, predicate="within")
                if len(hits) > 0:
                    votes[int(hits[0])] += 1
                else:
                    votes[-1] += 1  # exterior sentinel

            if not votes:
                result[(rid, side_label)] = None
            else:
                winner = votes.most_common(1)[0][0]
                result[(rid, side_label)] = None if winner == -1 else winner

    return result


def compute_polygon_aspects(
    polygons: list[Polygon],
    dem_da: xr.DataArray,
) -> dict[int, np.ndarray]:
    """Compute the mean unit downhill vector for each polygon."""
    aspects: dict[int, np.ndarray] = {}
    for i, poly in enumerate(polygons):
        try:
            aspects[i] = _compute_downhill_aspect(dem_da, poly)
        except (ValueError, Exception):
            aspects[i] = np.array([np.nan, np.nan], dtype=np.float64)
    return aspects


def build_network_faces(
    snapped_gdf: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
    closure_max_dist_m: float = 2000.0,
    min_polygon_area_m2: float = 100.0,
) -> NetworkFaces:
    """End-to-end: polygonize, classify sides, compute per-polygon aspects."""
    polygons, closure_edges = polygonize_reach_network(
        snapped_gdf,
        dem_da,
        closure_max_dist_m=closure_max_dist_m,
        min_polygon_area_m2=min_polygon_area_m2,
    )
    reach_side_map = classify_reach_sides(snapped_gdf, polygons)
    aspects = compute_polygon_aspects(polygons, dem_da)
    return NetworkFaces(
        polygons=polygons,
        closure_edges=closure_edges,
        reach_side_map=reach_side_map,
        polygon_aspects=aspects,
        min_area_m2=min_polygon_area_m2,
    )


# ---------------------------------------------------------------------------
# Network-face strip generation
# ---------------------------------------------------------------------------


def _build_dem_interp(dem_da: xr.DataArray) -> RegularGridInterpolator:
    y = dem_da.y.values.astype(np.float64)
    x = dem_da.x.values.astype(np.float64)
    arr = np.asarray(dem_da.values, dtype=np.float64)
    if y[0] > y[-1]:
        y = y[::-1]
        arr = arr[::-1, :]
    return RegularGridInterpolator((y, x), arr, bounds_error=False, fill_value=np.nan)


def _ridge_stop_ray(
    anchor_xy: np.ndarray,
    direction: np.ndarray,
    dem_interp: RegularGridInterpolator,
    max_dist_m: float,
    ridge_prom_m: float,
    descend_stop_m: float,
    step_m: float = RAY_STEP_M,
) -> LineString | None:
    """Cast a ray from *anchor_xy* in *direction*, return a LineString that
    terminates where the ridge-and-descend rule fires or at *max_dist_m*.
    """
    n = int(max_dist_m / step_m) + 1
    ts = np.arange(n, dtype=np.float64) * step_m
    pts_xy = anchor_xy[None, :] + ts[:, None] * direction[None, :]
    elev = dem_interp(np.column_stack([pts_xy[:, 1], pts_xy[:, 0]]))
    if not np.isfinite(elev[0]):
        return None
    base = float(elev[0])
    running_max = base
    ridge_found = False
    peak_val = base
    stop_d = float(ts[-1])
    for i in range(1, n):
        v = float(elev[i])
        if np.isnan(v):
            stop_d = float(ts[i])
            break
        if v > running_max:
            running_max = v
        if not ridge_found and (running_max - base) >= ridge_prom_m:
            ridge_found = True
            peak_val = running_max
        if ridge_found and (peak_val - v) >= descend_stop_m:
            stop_d = float(ts[i])
            break
        if ridge_found and v > peak_val:
            peak_val = v
    if stop_d <= 1e-3:
        return None
    end = anchor_xy + stop_d * direction
    return LineString([tuple(anchor_xy), tuple(end)])


def _truncate_one_sided(
    strip: LineString,
    anchor_rid: int,
    obstacle_geoms: list[LineString],
    obstacle_rids: list[int],
    reach_tree,
    eps_m: float = 1e-3,
) -> LineString | None:
    """Truncate a one-sided strip (anchor at coords[0]) at the first
    non-self snapped-reach crossing.
    """
    total = float(strip.length)
    s_cut = total
    candidates = (
        reach_tree.query(strip)
        if reach_tree is not None
        else range(len(obstacle_geoms))
    )
    for idx in candidates:
        if obstacle_rids[idx] == anchor_rid:
            continue
        ix = strip.intersection(obstacle_geoms[idx])
        if ix.is_empty:
            continue
        for px, py in _intersection_points(ix):
            s_pt = float(strip.project(Point(px, py)))
            if s_pt > eps_m and s_pt < s_cut:
                s_cut = s_pt
    if s_cut <= eps_m:
        return None
    if s_cut >= total - eps_m:
        return strip
    trunc = substring(strip, 0.0, s_cut)
    return trunc if isinstance(trunc, LineString) and trunc.length > eps_m else None


def _build_reach_chains(
    snapped_gdf: gpd.GeoDataFrame,
    junction_tol_m: float = 1.0,
) -> list[list[tuple[int, bool]]]:
    """Partition all reaches into maximal non-branching chains.

    Returns a list of chains, each chain a list of ``(reach_id, reversed_flag)``
    in traversal order.  A chain ends at leaves (degree-1 junctions) or
    branches (degree >= 3).
    """
    from scipy.spatial import cKDTree

    # -- collect endpoints for every reach --
    rids: list[int] = []
    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for _, row in snapped_gdf.iterrows():
        g = row.geometry
        if g is None or g.is_empty:
            continue
        coords = np.array(g.coords)
        rids.append(int(row["reach_id"]))
        starts.append(coords[0, :2].copy())
        ends.append(coords[-1, :2].copy())

    n = len(rids)
    if n == 0:
        return []

    # -- union-find on endpoints within junction_tol_m --
    # 2N points: index 2*i = start of reach i, 2*i+1 = end of reach i
    pts = np.vstack(starts + ends)  # shape (2N, 2)
    parent = list(range(2 * n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    tree = cKDTree(pts)
    pairs = tree.query_pairs(junction_tol_m)
    for a, b in pairs:
        union(a, b)

    # -- count how many reach-endpoints meet at each junction cluster --
    junction_degree: dict[int, int] = Counter()
    for i in range(2 * n):
        junction_degree[find(i)] += 1

    # -- build adjacency: for each reach, which other reaches share a junction --
    # endpoint_cluster[i] = (cluster_of_start, cluster_of_end)
    ep_cluster = [(find(2 * i), find(2 * i + 1)) for i in range(n)]
    # cluster -> list of (reach_index, is_end_side)
    cluster_reaches: dict[int, list[tuple[int, bool]]] = {}
    for i in range(n):
        cs, ce = ep_cluster[i]
        cluster_reaches.setdefault(cs, []).append((i, False))  # start side
        cluster_reaches.setdefault(ce, []).append((i, True))  # end side

    # -- greedy chain walk --
    chained: set[int] = set()
    chains: list[list[tuple[int, bool]]] = []

    for seed_idx in range(n):
        if seed_idx in chained:
            continue
        chain: list[tuple[int, bool]] = [(rids[seed_idx], False)]
        chained.add(seed_idx)

        # Extend forward from the end of the chain
        def _extend(direction: str) -> None:
            while True:
                if direction == "forward":
                    last_rid, last_rev = chain[-1]
                    li = rids.index(last_rid)
                    # free end = end if not reversed, start if reversed
                    free_cluster = (
                        ep_cluster[li][1] if not last_rev else ep_cluster[li][0]
                    )
                else:
                    last_rid, last_rev = chain[0]
                    li = rids.index(last_rid)
                    # free end = start if not reversed, end if reversed
                    free_cluster = (
                        ep_cluster[li][0] if not last_rev else ep_cluster[li][1]
                    )

                # Only extend through degree-2 junctions
                if junction_degree[free_cluster] != 2:
                    break

                # Find the other reach at this junction
                candidates = cluster_reaches[free_cluster]
                found = False
                for ci, is_end_side in candidates:
                    if ci in chained:
                        continue
                    # Determine orientation: if the candidate's end_side matches
                    # the free cluster, the candidate connects via that end
                    if is_end_side:
                        # candidate's end touches our free end -> reversed
                        reversed_flag = True
                    else:
                        # candidate's start touches our free end -> not reversed
                        reversed_flag = False
                    if direction == "forward":
                        chain.append((rids[ci], reversed_flag))
                    else:
                        # Attach to start: if candidate's start touches our free
                        # start, it's not reversed; if its end touches, reversed.
                        # But direction=="backward" means we're extending the
                        # start of the chain, so flip the logic:
                        if is_end_side:
                            chain.insert(0, (rids[ci], False))
                        else:
                            chain.insert(0, (rids[ci], True))
                    chained.add(ci)
                    found = True
                    break
                if not found:
                    break

        _extend("forward")
        _extend("backward")
        chains.append(chain)

    return chains


@dataclass
class _OverrideChainStation:
    reach_id: int
    station_id: int
    chain_pos: int
    s_m_reach: float
    base_xy: np.ndarray
    tangent: np.ndarray
    base_elev_m: float
    is_seed: bool


@dataclass
class _OpenEdgeAnchorCandidate:
    poly_id: int
    side_label: str
    reach_id: int
    station_id: int
    open_end: str
    unstable_run_len: int
    row: dict


@dataclass
class _OpenEdgeFamilyPlan:
    poly_id: int
    side_label: str
    chain_order: list[tuple[int, bool]]
    anchor_reach_id: int
    anchor_station_id: int
    score: tuple[int, float, float, int]


def _west_normal(tangent: np.ndarray) -> np.ndarray:
    left = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    return left if left[0] < 0.0 else -left


def _short_arc_interp(a0: float, a1: float, frac: float) -> float:
    diff = a1 - a0
    if diff > np.pi:
        diff -= 2.0 * np.pi
    elif diff < -np.pi:
        diff += 2.0 * np.pi
    return a0 + frac * diff


def _truncate_against_strip_blockers(
    strip: LineString,
    anchor_xy: np.ndarray,
    blockers: list[LineString],
    margin_m: float = 0.1,
) -> LineString | None:
    max_d = float(strip.length)
    best_d = max_d
    for other in blockers:
        if other is None or other.is_empty:
            continue
        ix = strip.intersection(other)
        if ix.is_empty:
            continue
        for px, py in _intersection_points(ix):
            d = float(np.hypot(px - anchor_xy[0], py - anchor_xy[1]))
            if 1e-6 < d < best_d:
                best_d = d
    if best_d < max_d:
        best_d = max(best_d - margin_m, 0.0)
    if best_d <= 1e-3:
        return None
    if best_d >= max_d - 1e-3:
        return strip
    trunc = substring(strip, 0.0, best_d)
    return trunc if isinstance(trunc, LineString) and trunc.length > 1e-3 else None


def _dem_boundary(dem_da: xr.DataArray) -> LineString:
    xmin = float(dem_da.x.values.min())
    xmax = float(dem_da.x.values.max())
    ymin = float(dem_da.y.values.min())
    ymax = float(dem_da.y.values.max())
    return Polygon(
        [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    ).boundary


def _find_open_edge_polygon_ids(
    faces: NetworkFaces,
    dem_da: xr.DataArray,
    touch_tol_m: float = OPEN_EDGE_TOUCH_TOL_M,
) -> set[int]:
    open_poly_ids: set[int] = set()
    dem_boundary = _dem_boundary(dem_da)
    for poly_id, poly in enumerate(faces.polygons):
        if poly is None or poly.is_empty:
            continue
        probe = poly.buffer(touch_tol_m)
        if probe.intersects(dem_boundary):
            open_poly_ids.add(poly_id)
            continue
        if any(probe.intersects(edge) for edge in faces.closure_edges):
            open_poly_ids.add(poly_id)
    return open_poly_ids


def _frame_endpoint_xy(line: LineString, which: str) -> np.ndarray:
    coords = np.array(line.coords, dtype=np.float64)
    if which == "start":
        return coords[0, :2].copy()
    return coords[-1, :2].copy()


def _frame_endpoint_tangent(line: LineString, which: str) -> np.ndarray:
    total = float(line.length)
    if total <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    eps = min(5.0, total * 0.05)
    if which == "start":
        p0 = line.interpolate(0.0)
        p1 = line.interpolate(min(eps, total))
    else:
        p0 = line.interpolate(max(total - eps, 0.0))
        p1 = line.interpolate(total)
    vec = np.array([p1.x - p0.x, p1.y - p0.y], dtype=np.float64)
    nrm = float(np.hypot(vec[0], vec[1]))
    if nrm <= 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    return vec / nrm


def _build_polygon_side_graph(
    reach_ids: set[int],
    frame_by_rid: dict[int, LineString],
    junction_tol_m: float = OPEN_EDGE_JUNCTION_TOL_M,
) -> dict[int, dict[str, list[tuple[int, str]]]]:
    graph: dict[int, dict[str, list[tuple[int, str]]]] = {
        int(rid): {"start": [], "end": []} for rid in reach_ids if rid in frame_by_rid
    }
    rids = sorted(graph)
    if not rids:
        return graph
    endpoints = {
        rid: {
            "start": _frame_endpoint_xy(frame_by_rid[rid], "start"),
            "end": _frame_endpoint_xy(frame_by_rid[rid], "end"),
        }
        for rid in rids
    }
    for i, rid_a in enumerate(rids):
        for rid_b in rids[i + 1 :]:
            for side_a in ("start", "end"):
                xy_a = endpoints[rid_a][side_a]
                for side_b in ("start", "end"):
                    xy_b = endpoints[rid_b][side_b]
                    if float(np.hypot(xy_a[0] - xy_b[0], xy_a[1] - xy_b[1])) <= junction_tol_m:
                        graph[rid_a][side_a].append((rid_b, side_b))
                        graph[rid_b][side_b].append((rid_a, side_a))
    return graph


def _graph_components(
    graph: dict[int, dict[str, list[tuple[int, str]]]],
) -> list[set[int]]:
    remaining = set(graph)
    comps: list[set[int]] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        comp = {seed}
        while stack:
            rid = stack.pop()
            neighbors = {
                nbr for side in ("start", "end") for nbr, _ in graph[rid][side]
            }
            for nbr in neighbors:
                if nbr not in remaining:
                    continue
                remaining.remove(nbr)
                comp.add(nbr)
                stack.append(nbr)
        comps.append(comp)
    return comps


def _reach_length(reach_id: int, snap_by_rid: dict[int, LineString]) -> float:
    geom = snap_by_rid.get(int(reach_id))
    return float(geom.length) if geom is not None and not geom.is_empty else 0.0


def _travel_exit_dir(frame_by_rid: dict[int, LineString], reach_id: int, reversed_flag: bool) -> np.ndarray:
    line = frame_by_rid[int(reach_id)]
    if reversed_flag:
        return -_frame_endpoint_tangent(line, "start")
    return _frame_endpoint_tangent(line, "end")


def _travel_entry_dir(frame_by_rid: dict[int, LineString], reach_id: int, reversed_flag: bool) -> np.ndarray:
    line = frame_by_rid[int(reach_id)]
    if reversed_flag:
        return -_frame_endpoint_tangent(line, "end")
    return _frame_endpoint_tangent(line, "start")


def _build_override_chain_stations(
    chain_order: list[tuple[int, bool]],
    frame_by_rid: dict[int, LineString],
    snap_by_rid: dict[int, LineString],
    dem_interp: RegularGridInterpolator,
    strip_spacing_m: float,
    seed_reaches: set[int],
) -> list[_OverrideChainStation]:
    stations: list[_OverrideChainStation] = []
    chain_pos = 0
    for rid, reversed_flag in chain_order:
        snap_geom = snap_by_rid.get(rid)
        frame_geom = frame_by_rid.get(rid)
        if snap_geom is None or snap_geom.is_empty:
            continue
        if frame_geom is None or frame_geom.is_empty:
            continue

        total = float(snap_geom.length)
        s_vals = np.arange(strip_spacing_m * 0.5, total + 1e-6, strip_spacing_m)
        if len(s_vals) == 0:
            continue

        n_stations = len(s_vals)
        center_idx = n_stations // 2
        native_idx = list(range(n_stations))
        if reversed_flag:
            native_idx = native_idx[::-1]

        for native_i in native_idx:
            s = float(s_vals[native_i])
            anchor_pt = snap_geom.interpolate(s)
            anchor_xy = np.array([anchor_pt.x, anchor_pt.y], dtype=np.float64)

            frame_s = float(frame_geom.project(anchor_pt))
            eps = min(1.0, frame_geom.length * 0.01)
            p0 = frame_geom.interpolate(max(frame_s - eps, 0.0))
            p1 = frame_geom.interpolate(min(frame_s + eps, frame_geom.length))
            tangent = np.array([p1.x - p0.x, p1.y - p0.y], dtype=np.float64)
            tn = float(np.hypot(tangent[0], tangent[1]))
            if tn < 1e-9:
                continue
            tangent /= tn
            if reversed_flag:
                tangent = -tangent

            base_elev = dem_interp(np.array([[anchor_xy[1], anchor_xy[0]]], dtype=np.float64))
            elev = float(base_elev[0]) if np.isfinite(base_elev[0]) else float("nan")
            stations.append(
                _OverrideChainStation(
                    reach_id=int(rid),
                    station_id=int(native_i),
                    chain_pos=chain_pos,
                    s_m_reach=float(s),
                    base_xy=anchor_xy,
                    tangent=tangent,
                    base_elev_m=elev,
                    is_seed=(rid in seed_reaches and native_i == center_idx),
                )
            )
            chain_pos += 1
    return stations


def _nearest_nonself_endpoint_reach(
    endpoint: Point,
    anchor_rid: int,
    obstacle_geoms: list[LineString],
    obstacle_rids: list[int],
    reach_tree,
    tol_m: float,
) -> int | None:
    candidates = (
        reach_tree.query(endpoint.buffer(tol_m))
        if reach_tree is not None
        else range(len(obstacle_geoms))
    )
    best_rid: int | None = None
    best_d = float("inf")
    for idx in candidates:
        rid = obstacle_rids[idx]
        if rid == anchor_rid:
            continue
        d = endpoint.distance(obstacle_geoms[idx])
        if d <= tol_m and d < best_d:
            best_d = d
            best_rid = rid
    return best_rid


def _find_open_edge_anchor_candidates(
    strips: gpd.GeoDataFrame,
    poly_id: int,
    reach_id: int,
    side_label: str,
    obstacle_geoms: list[LineString],
    obstacle_rids: list[int],
    reach_tree,
    max_strip_len_m: float,
    min_len_m: float,
    endpoint_tol_m: float,
) -> list[_OpenEdgeAnchorCandidate]:
    sub = strips[
        (strips["reach_id"] == int(reach_id))
        & (strips["side"] == str(side_label))
        & (strips["strip_type"] == "interreach")
    ].sort_values("station_id")
    if sub.empty:
        return []

    rows = []
    for _, row in sub.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        end_pt = Point(geom.coords[-1][:2])
        hit_rid = _nearest_nonself_endpoint_reach(
            end_pt,
            int(reach_id),
            obstacle_geoms,
            obstacle_rids,
            reach_tree,
            endpoint_tol_m,
        )
        length_m = float(row["length_m"])
        is_stable = (
            hit_rid is not None
            and length_m >= min_len_m
            and length_m < 0.95 * max_strip_len_m
        )
        rows.append(
            {
                "row": row.to_dict(),
                "station_id": int(row["station_id"]),
                "length_m": length_m,
                "hit_rid": hit_rid,
                "is_stable": is_stable,
            }
        )
    if not rows:
        return []

    candidates: list[_OpenEdgeAnchorCandidate] = []

    low = 0
    while low < len(rows) and not rows[low]["is_stable"]:
        low += 1
    if 0 < low < len(rows):
        candidates.append(
            _OpenEdgeAnchorCandidate(
                poly_id=int(poly_id),
                side_label=str(side_label),
                reach_id=int(reach_id),
                station_id=int(rows[low]["station_id"]),
                open_end="start",
                unstable_run_len=int(low),
                row=rows[low]["row"],
            )
        )

    high = len(rows) - 1
    while high >= 0 and not rows[high]["is_stable"]:
        high -= 1
    if 0 <= high < len(rows) - 1:
        candidates.append(
            _OpenEdgeAnchorCandidate(
                poly_id=int(poly_id),
                side_label=str(side_label),
                reach_id=int(reach_id),
                station_id=int(rows[high]["station_id"]),
                open_end="end",
                unstable_run_len=int(len(rows) - 1 - high),
                row=rows[high]["row"],
            )
        )

    return candidates


def _best_family_chain_from_anchor(
    candidate: _OpenEdgeAnchorCandidate,
    component_reach_ids: set[int],
    graph: dict[int, dict[str, list[tuple[int, str]]]],
    frame_by_rid: dict[int, LineString],
    snap_by_rid: dict[int, LineString],
) -> tuple[list[tuple[int, bool]], tuple[int, float, float, int]]:
    anchor_rev = candidate.open_end == "start"
    anchor_exit_side = "start" if anchor_rev else "end"

    def _dfs(
        rid: int,
        reversed_flag: bool,
        visited: set[int],
    ) -> tuple[list[tuple[int, bool]], tuple[int, float, float, int]]:
        base_score = (1, _reach_length(rid, snap_by_rid), 0.0, 0)
        best_path = [(int(rid), bool(reversed_flag))]
        best_score = base_score
        exit_side = "start" if reversed_flag else "end"
        curr_exit_dir = _travel_exit_dir(frame_by_rid, rid, reversed_flag)
        for nbr, nbr_side in graph.get(int(rid), {}).get(exit_side, []):
            if nbr not in component_reach_ids or nbr in visited:
                continue
            nbr_rev = nbr_side == "end"
            nbr_entry_dir = _travel_entry_dir(frame_by_rid, nbr, nbr_rev)
            continuity = float(np.dot(curr_exit_dir, nbr_entry_dir))
            child_path, child_score = _dfs(nbr, nbr_rev, visited | {nbr})
            score = (
                1 + child_score[0],
                _reach_length(rid, snap_by_rid) + child_score[1],
                continuity + child_score[2],
                candidate.unstable_run_len,
            )
            if score > best_score:
                best_score = score
                best_path = [(int(rid), bool(reversed_flag))] + child_path
        return best_path, best_score

    if candidate.reach_id not in component_reach_ids:
        return [], (0, 0.0, 0.0, candidate.unstable_run_len)
    if candidate.reach_id not in graph:
        return [], (0, 0.0, 0.0, candidate.unstable_run_len)
    path, score = _dfs(candidate.reach_id, anchor_rev, {candidate.reach_id})
    return path, score


def _select_best_open_edge_family(
    strips: gpd.GeoDataFrame,
    poly_id: int,
    side_label: str,
    component_reach_ids: set[int],
    graph: dict[int, dict[str, list[tuple[int, str]]]],
    frame_by_rid: dict[int, LineString],
    snap_by_rid: dict[int, LineString],
    obstacle_geoms: list[LineString],
    obstacle_rids: list[int],
    reach_tree,
    max_strip_len_m: float,
    min_len_m: float,
    endpoint_tol_m: float,
) -> _OpenEdgeFamilyPlan | None:
    best_plan: _OpenEdgeFamilyPlan | None = None
    for rid in sorted(component_reach_ids):
        candidates = _find_open_edge_anchor_candidates(
            strips,
            poly_id,
            rid,
            side_label,
            obstacle_geoms,
            obstacle_rids,
            reach_tree,
            max_strip_len_m=max_strip_len_m,
            min_len_m=min_len_m,
            endpoint_tol_m=endpoint_tol_m,
        )
        for candidate in candidates:
            chain_order, score = _best_family_chain_from_anchor(
                candidate,
                component_reach_ids,
                graph,
                frame_by_rid,
                snap_by_rid,
            )
            if len(chain_order) < 2:
                continue
            plan = _OpenEdgeFamilyPlan(
                poly_id=int(poly_id),
                side_label=str(side_label),
                chain_order=chain_order,
                anchor_reach_id=int(candidate.reach_id),
                anchor_station_id=int(candidate.station_id),
                score=score,
            )
            if best_plan is None or plan.score > best_plan.score:
                best_plan = plan
    return best_plan


def _build_family_override_rows(
    strips: gpd.GeoDataFrame,
    family: _OpenEdgeFamilyPlan,
    frame_by_rid: dict[int, LineString],
    snap_by_rid: dict[int, LineString],
    dem_interp: RegularGridInterpolator,
    obstacle_geoms: list[LineString],
    obstacle_rids: list[int],
    reach_tree,
    strip_spacing_m: float,
    max_strip_len_m: float,
    ridge_prominence_m: float,
    descend_stop_m: float,
) -> tuple[list[dict], set[tuple[int, int, str]]] | None:
    side_label = family.side_label
    anchor_rows = strips[
        (strips["reach_id"] == int(family.anchor_reach_id))
        & (strips["side"] == side_label)
        & (strips["station_id"] == int(family.anchor_station_id))
    ]
    if anchor_rows.empty:
        return None
    anchor_row = anchor_rows.iloc[0].to_dict()
    anchor_geom = anchor_row["geometry"]
    anchor_coords = np.array(anchor_geom.coords, dtype=np.float64)
    anchor_vec = anchor_coords[-1, :2] - anchor_coords[0, :2]
    anchor_norm = float(np.hypot(anchor_vec[0], anchor_vec[1]))
    if anchor_norm <= 1e-9:
        return None
    anchor_angle = float(np.arctan2(anchor_vec[1], anchor_vec[0]))

    chain = _build_override_chain_stations(
        family.chain_order,
        frame_by_rid,
        snap_by_rid,
        dem_interp,
        strip_spacing_m,
        seed_reaches={rid for rid, _ in family.chain_order[1:]},
    )
    if not chain:
        return None

    pos_by_key = {(st.reach_id, st.station_id): i for i, st in enumerate(chain)}
    anchor_pos = pos_by_key.get((int(family.anchor_reach_id), int(family.anchor_station_id)))
    if anchor_pos is None:
        return None

    anchor_specs: list[tuple[int, float, str]] = [
        (anchor_pos, anchor_angle, "interreach_anchor")
    ]
    for pos, st in enumerate(chain):
        if st.is_seed:
            nrm = _west_normal(st.tangent)
            ang = float(np.arctan2(nrm[1], nrm[0]))
            anchor_specs.append((pos, ang, "reach_seed"))
    anchor_specs.sort(key=lambda x: x[0])
    if len(anchor_specs) < 2:
        return None

    angles = np.full(len(chain), np.nan, dtype=np.float64)
    anchor_kind = [""] * len(chain)
    for pos, _, kind in anchor_specs:
        anchor_kind[pos] = kind
    for j in range(len(anchor_specs) - 1):
        p0, a0, _ = anchor_specs[j]
        p1, a1, _ = anchor_specs[j + 1]
        span = p1 - p0
        if span <= 0:
            continue
        for p in range(p0, p1 + 1):
            frac = (p - p0) / span
            angles[p] = _short_arc_interp(a0, a1, frac)
    p_last, a_last, _ = anchor_specs[-1]
    angles[p_last:] = a_last

    override_keys = {
        (int(st.reach_id), int(st.station_id), str(side_label))
        for pos, st in enumerate(chain)
        if pos > anchor_pos
    }
    if not override_keys:
        return None

    preserved_blockers: list[LineString] = []
    protected = strips[
        (strips["side"] == side_label)
        & (~strips.apply(
            lambda r: (int(r["reach_id"]), int(r["station_id"]), str(r["side"])) in override_keys,
            axis=1,
        ))
    ]
    for _, row in protected.iterrows():
        geom = row.geometry
        if geom is not None and not geom.is_empty:
            preserved_blockers.append(geom)

    blockers = list(preserved_blockers)
    rows: list[dict] = []
    for pos in range(anchor_pos + 1, len(chain)):
        st = chain[pos]
        angle = angles[pos]
        if not np.isfinite(angle):
            continue
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
        strip = _ridge_stop_ray(
            st.base_xy,
            direction,
            dem_interp,
            max_strip_len_m,
            ridge_prominence_m,
            descend_stop_m,
        )
        if strip is None:
            continue
        strip = _truncate_one_sided(
            strip, st.reach_id, obstacle_geoms, obstacle_rids, reach_tree
        )
        if strip is None or strip.length < 1e-3:
            continue
        strip = _truncate_against_strip_blockers(strip, st.base_xy, blockers)
        if strip is None or strip.length < 1e-3:
            continue
        rows.append(
            {
                "reach_id": int(st.reach_id),
                "station_id": int(st.station_id),
                "side": side_label,
                "strip_type": "side",
                "poly_id": -1,
                "anchor_x": float(st.base_xy[0]),
                "anchor_y": float(st.base_xy[1]),
                "length_m": float(strip.length),
                "angle_deg": float(np.degrees(angle)),
                "is_seed": bool(st.is_seed),
                "anchor_kind": anchor_kind[pos],
                "geometry": strip,
            }
        )
        blockers.append(strip)
    return rows, override_keys


def _apply_open_edge_strip_overrides(
    strips: gpd.GeoDataFrame,
    faces: NetworkFaces,
    frame_by_rid: dict[int, LineString],
    snap_by_rid: dict[int, LineString],
    dem_da: xr.DataArray,
    dem_interp: RegularGridInterpolator,
    obstacle_geoms: list[LineString],
    obstacle_rids: list[int],
    reach_tree,
    strip_spacing_m: float,
    max_strip_len_m: float,
    ridge_prominence_m: float,
    descend_stop_m: float,
) -> gpd.GeoDataFrame:
    open_poly_ids = _find_open_edge_polygon_ids(faces, dem_da)
    if not open_poly_ids:
        return strips

    all_override_rows: list[dict] = []
    all_override_keys: set[tuple[int, int, str]] = set()

    for poly_id in sorted(open_poly_ids):
        side_to_reaches: dict[str, set[int]] = {}
        for (rid, side_label), pidx in faces.reach_side_map.items():
            if pidx != poly_id:
                continue
            side_to_reaches.setdefault(str(side_label), set()).add(int(rid))

        for side_label, reach_ids in sorted(side_to_reaches.items()):
            if len(reach_ids) < 2:
                continue
            graph = _build_polygon_side_graph(
                reach_ids,
                frame_by_rid,
                junction_tol_m=OPEN_EDGE_JUNCTION_TOL_M,
            )
            for component in _graph_components(graph):
                if len(component) < 2:
                    continue
                family = _select_best_open_edge_family(
                    strips,
                    int(poly_id),
                    side_label,
                    component,
                    graph,
                    frame_by_rid,
                    snap_by_rid,
                    obstacle_geoms,
                    obstacle_rids,
                    reach_tree,
                    max_strip_len_m=max_strip_len_m,
                    min_len_m=OPEN_EDGE_MIN_STABLE_LEN_M,
                    endpoint_tol_m=OPEN_EDGE_ENDPOINT_TOL_M,
                )
                if family is None:
                    continue
                built = _build_family_override_rows(
                    strips,
                    family,
                    frame_by_rid,
                    snap_by_rid,
                    dem_interp,
                    obstacle_geoms,
                    obstacle_rids,
                    reach_tree,
                    strip_spacing_m,
                    max_strip_len_m,
                    ridge_prominence_m,
                    descend_stop_m,
                )
                if built is None:
                    continue
                override_rows, override_keys = built
                if not override_rows:
                    continue
                all_override_rows.extend(override_rows)
                all_override_keys |= override_keys

    if not all_override_rows:
        return strips

    keep_mask = ~strips.apply(
        lambda r: (int(r["reach_id"]), int(r["station_id"]), str(r["side"])) in all_override_keys,
        axis=1,
    )
    preserved_rows = strips.loc[keep_mask].to_dict("records")
    return gpd.GeoDataFrame(
        preserved_rows + all_override_rows,
        geometry="geometry",
        crs=strips.crs,
    )


def _generate_chain_side_strips(
    chain: list[tuple[int, bool]],
    side_label: str,
    faces: NetworkFaces,
    frame_by_rid: dict[int, LineString],
    snap_by_rid: dict[int, LineString],
    dem_interp: RegularGridInterpolator,
    obstacle_geoms: list[LineString],
    obstacle_rids: list[int],
    reach_tree,
    strip_spacing_m: float,
    max_strip_len_m: float,
    ridge_prominence_m: float,
    descend_stop_m: float,
) -> list[dict]:
    """Process one chain on one side: build stations, compute angles with
    seed-and-interpolate blending, then generate strips center-out with
    intra-chain truncation.
    """
    sign = 1.0 if side_label == "left" else -1.0

    # ---- Phase A: build station list along chain ----
    stations: list[dict] = []
    for rid, reversed_flag in chain:
        snap_geom = snap_by_rid.get(rid)
        frame_geom = frame_by_rid.get(rid)
        if snap_geom is None or snap_geom.is_empty:
            continue
        if frame_geom is None or frame_geom.is_empty:
            continue

        total = float(snap_geom.length)
        s_vals = np.arange(strip_spacing_m * 0.5, total + 1e-6, strip_spacing_m)
        if len(s_vals) == 0:
            continue

        n_stations = len(s_vals)
        center_idx = n_stations // 2
        indices = list(range(n_stations))
        if reversed_flag:
            indices = indices[::-1]

        for native_i in indices:
            s = float(s_vals[native_i])
            anchor_pt = snap_geom.interpolate(s)
            anchor_xy = np.array([anchor_pt.x, anchor_pt.y], dtype=np.float64)

            frame_s = float(frame_geom.project(anchor_pt))
            eps = min(1.0, frame_geom.length * 0.01)
            p0 = frame_geom.interpolate(max(frame_s - eps, 0.0))
            p1 = frame_geom.interpolate(min(frame_s + eps, frame_geom.length))
            tx, ty = p1.x - p0.x, p1.y - p0.y
            tn = float(np.hypot(tx, ty))
            if tn < 1e-9:
                continue
            tx /= tn
            ty /= tn
            left_normal = np.array([-ty, tx], dtype=np.float64)
            local_dir = sign * left_normal

            poly_idx = faces.reach_side_map.get((rid, side_label))
            is_interreach = poly_idx is not None

            stations.append(
                {
                    "rid": rid,
                    "station_id": native_i,
                    "anchor_xy": anchor_xy,
                    "local_dir": local_dir,
                    "poly_idx": poly_idx,
                    "is_interreach": is_interreach,
                    "is_reach_center": (native_i == center_idx),
                }
            )

    if not stations:
        return []

    n_st = len(stations)

    # ---- Phase B: compute raw angles ----
    raw_angles = np.zeros(n_st, dtype=np.float64)
    for k, st in enumerate(stations):
        if st["is_interreach"]:
            aspect = faces.polygon_aspects.get(st["poly_idx"])
            if aspect is None or not np.isfinite(aspect[0]):
                raw_angles[k] = float(
                    np.arctan2(st["local_dir"][1], st["local_dir"][0])
                )
                continue
            strip_dir = np.array([-aspect[1], aspect[0]], dtype=np.float64)
            if float(np.dot(strip_dir, st["local_dir"])) < 0:
                strip_dir = -strip_dir
            raw_angles[k] = float(np.arctan2(strip_dir[1], strip_dir[0]))
        else:
            raw_angles[k] = float(np.arctan2(st["local_dir"][1], st["local_dir"][0]))

    # ---- Phase C: seed identification + interpolation (exterior only) ----
    angles = raw_angles.copy()

    # Identify seeds: boundary interreach stations and exterior reach centers
    seed_positions: list[int] = []
    seed_angles: list[float] = []

    for k in range(n_st):
        is_seed = False
        # (a) Last interreach before an exterior run
        if (
            stations[k]["is_interreach"]
            and k + 1 < n_st
            and not stations[k + 1]["is_interreach"]
        ):
            is_seed = True
        # (b) First interreach after an exterior run
        if (
            stations[k]["is_interreach"]
            and k - 1 >= 0
            and not stations[k - 1]["is_interreach"]
        ):
            is_seed = True
        # (c) Exterior reach center
        if not stations[k]["is_interreach"] and stations[k]["is_reach_center"]:
            is_seed = True
        if is_seed:
            seed_positions.append(k)
            seed_angles.append(float(raw_angles[k]))

    if seed_positions:
        # Shorter-arc unwrap
        for j in range(1, len(seed_angles)):
            diff = seed_angles[j] - seed_angles[j - 1]
            if diff > np.pi:
                seed_angles[j] -= 2 * np.pi
            elif diff < -np.pi:
                seed_angles[j] += 2 * np.pi

        # Interpolate only exterior stations between seeds
        # Hold before first seed
        for k in range(0, seed_positions[0]):
            if not stations[k]["is_interreach"]:
                angles[k] = seed_angles[0]

        # Interpolate between consecutive seeds
        for j in range(len(seed_positions) - 1):
            p0, p1 = seed_positions[j], seed_positions[j + 1]
            a0, a1 = seed_angles[j], seed_angles[j + 1]
            span = p1 - p0
            if span <= 0:
                continue
            for k in range(p0, p1 + 1):
                if not stations[k]["is_interreach"]:
                    frac = (k - p0) / span
                    angles[k] = a0 + frac * (a1 - a0)

        # Hold after last seed
        for k in range(seed_positions[-1], n_st):
            if not stations[k]["is_interreach"]:
                angles[k] = seed_angles[-1]

    # ---- Phase D: generate strip geometry (center-out) ----
    # Pass 1: compute max-extent strips per station
    max_strips: list[LineString | None] = [None] * n_st
    for k, st in enumerate(stations):
        direction = np.array([np.cos(angles[k]), np.sin(angles[k])], dtype=np.float64)
        anchor_xy = st["anchor_xy"]

        if st["is_interreach"]:
            poly = faces.polygons[st["poly_idx"]]
            end_xy = anchor_xy + max_strip_len_m * direction
            ray = LineString([tuple(anchor_xy), tuple(end_xy)])
            ix = ray.intersection(poly)
            if ix.is_empty:
                continue
            if ix.geom_type == "MultiLineString":
                anchor_pt = Point(anchor_xy)
                ix = min(ix.geoms, key=lambda g: g.distance(anchor_pt))
            if ix.geom_type != "LineString":
                continue
            strip = _truncate_one_sided(
                ix, st["rid"], obstacle_geoms, obstacle_rids, reach_tree
            )
        else:
            strip = _ridge_stop_ray(
                anchor_xy,
                direction,
                dem_interp,
                max_strip_len_m,
                ridge_prominence_m,
                descend_stop_m,
            )
            if strip is not None:
                strip = _truncate_one_sided(
                    strip, st["rid"], obstacle_geoms, obstacle_rids, reach_tree
                )

        if strip is not None and strip.length >= 1e-3:
            max_strips[k] = strip

    # Pass 2: center-out placement with intra-chain truncation
    final_strips: list[LineString | None] = [None] * n_st
    center = n_st // 2
    placed: list[int] = []

    def _place(idx: int) -> None:
        s = max_strips[idx]
        if s is None:
            return
        # Truncate against previously-placed strips in this chain-side
        if placed:
            anchor_xy = stations[idx]["anchor_xy"]
            max_d = float(s.length)
            best_d = max_d
            for pi in placed:
                other = final_strips[pi]
                if other is None:
                    continue
                ix = s.intersection(other)
                if ix.is_empty:
                    continue
                for pt in _intersection_points(ix):
                    d = float(np.hypot(pt[0] - anchor_xy[0], pt[1] - anchor_xy[1]))
                    if 1e-6 < d < best_d:
                        best_d = d
            if best_d < max_d:
                best_d = max(best_d - 0.1, 0.0)
            if best_d <= 1e-3:
                return
            if best_d < max_d - 1e-3:
                s = substring(s, 0.0, best_d)
                if not isinstance(s, LineString) or s.length < 1e-3:
                    return
        final_strips[idx] = s

    _place(center)
    placed.append(center)
    up = center - 1
    dn = center + 1
    while up >= 0 or dn < n_st:
        if up >= 0:
            _place(up)
            placed.append(up)
            up -= 1
        if dn < n_st:
            _place(dn)
            placed.append(dn)
            dn += 1

    # Build output rows
    rows: list[dict] = []
    for k, st in enumerate(stations):
        strip = final_strips[k]
        if strip is None:
            continue
        rows.append(
            {
                "reach_id": st["rid"],
                "station_id": st["station_id"],
                "side": side_label,
                "strip_type": "interreach" if st["is_interreach"] else "side",
                "poly_id": st["poly_idx"] if st["poly_idx"] is not None else -1,
                "anchor_x": float(st["anchor_xy"][0]),
                "anchor_y": float(st["anchor_xy"][1]),
                "length_m": float(strip.length),
                "geometry": strip,
            }
        )
    return rows


def generate_network_strips(
    snapped_gdf: gpd.GeoDataFrame,
    frame_gdf: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
    faces: NetworkFaces,
    strip_spacing_m: float = STRIP_SPACING_M,
    max_strip_len_m: float = MAX_STRIP_LEN_M,
    ridge_prominence_m: float = RIDGE_PROMINENCE_M,
    descend_stop_m: float = DESCEND_STOP_M,
) -> gpd.GeoDataFrame:
    """Generate all strips for every (reach, station, side).

    Reaches are partitioned into maximal non-branching chains.  For each
    chain-side, angles are blended via seed-and-interpolate (interreach
    boundary seeds + exterior reach-center seeds) and strips are placed
    center-out with intra-chain truncation.
    """
    from shapely import STRtree

    dem_interp = _build_dem_interp(dem_da)

    # Obstacle tree over all snapped reaches
    obstacle_geoms: list[LineString] = []
    obstacle_rids: list[int] = []
    for _, row in snapped_gdf.iterrows():
        g = row.geometry
        if g is not None and not g.is_empty:
            obstacle_geoms.append(g)
            obstacle_rids.append(int(row["reach_id"]))
    reach_tree = STRtree(obstacle_geoms) if obstacle_geoms else None

    # Pre-index geometries by reach_id
    frame_by_rid: dict[int, LineString] = {}
    for _, row in frame_gdf.iterrows():
        frame_by_rid[int(row["reach_id"])] = row.geometry
    snap_by_rid: dict[int, LineString] = {}
    for _, row in snapped_gdf.iterrows():
        g = row.geometry
        if g is not None and not g.is_empty:
            snap_by_rid[int(row["reach_id"])] = g

    chains = _build_reach_chains(snapped_gdf)
    rows: list[dict] = []
    for chain in chains:
        for side_label in ("left", "right"):
            rows += _generate_chain_side_strips(
                chain,
                side_label,
                faces,
                frame_by_rid,
                snap_by_rid,
                dem_interp,
                obstacle_geoms,
                obstacle_rids,
                reach_tree,
                strip_spacing_m,
                max_strip_len_m,
                ridge_prominence_m,
                descend_stop_m,
            )

    strips = (
        gpd.GeoDataFrame(rows, geometry="geometry", crs=snapped_gdf.crs)
        if rows
        else gpd.GeoDataFrame(
            columns=[
                "reach_id",
                "station_id",
                "side",
                "strip_type",
                "poly_id",
                "length_m",
                "geometry",
            ],
            geometry="geometry",
            crs=snapped_gdf.crs,
        )
    )
    if strips.empty:
        return strips
    return _apply_open_edge_strip_overrides(
        strips,
        faces,
        frame_by_rid,
        snap_by_rid,
        dem_da,
        dem_interp,
        obstacle_geoms,
        obstacle_rids,
        reach_tree,
        strip_spacing_m,
        max_strip_len_m,
        ridge_prominence_m,
        descend_stop_m,
    )


# ---------------------------------------------------------------------------
# Elevation attachment + wedge construction + rasterization
# ---------------------------------------------------------------------------


def attach_strip_elevations(
    strips: gpd.GeoDataFrame,
    snapped_gdf: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
) -> gpd.GeoDataFrame:
    """Add ``base_elev_m`` and ``endpoint_elev_m`` columns to strips.

    - ``base_elev_m``: DEM at the anchor (thalweg station).
    - ``endpoint_elev_m``:
      - *interreach*: thalweg elevation of the nearest non-self snapped reach
        at the projected far-endpoint position.
      - *side*: equals ``base_elev_m`` (flat one-sided water surface).
    """
    from shapely import STRtree

    strips = strips.copy()
    dem_interp = _build_dem_interp(dem_da)

    # Base elevation (vectorized)
    ax = strips["anchor_x"].values.astype(np.float64)
    ay = strips["anchor_y"].values.astype(np.float64)
    strips["base_elev_m"] = dem_interp(np.column_stack([ay, ax]))

    # Far-endpoint coordinates
    end_x = np.array([g.coords[-1][0] for g in strips.geometry], dtype=np.float64)
    end_y = np.array([g.coords[-1][1] for g in strips.geometry], dtype=np.float64)
    strips["endpoint_x"] = end_x
    strips["endpoint_y"] = end_y

    endpoint_elev = strips["base_elev_m"].values.copy()

    ir_mask = strips["strip_type"].values == "interreach"
    if ir_mask.any():
        geom_list: list[LineString] = []
        rid_list: list[int] = []
        for _, r in snapped_gdf.iterrows():
            g = r.geometry
            if g is not None and not g.is_empty:
                geom_list.append(g)
                rid_list.append(int(r["reach_id"]))
        tree = STRtree(geom_list)

        ir_idx = np.where(ir_mask)[0]
        proj_x = np.full(len(ir_idx), np.nan)
        proj_y = np.full(len(ir_idx), np.nan)

        for k, i in enumerate(ir_idx):
            rid = int(strips.iat[i, strips.columns.get_loc("reach_id")])
            ep = Point(end_x[i], end_y[i])
            nearest = int(tree.nearest(ep))
            if rid_list[nearest] == rid:
                # Self — search wider
                candidates = tree.query(ep.buffer(500.0), predicate="intersects")
                best_d = float("inf")
                chosen = -1
                for ci in candidates:
                    if rid_list[ci] == rid:
                        continue
                    d = ep.distance(geom_list[ci])
                    if d < best_d:
                        best_d = d
                        chosen = ci
                if chosen < 0:
                    continue
                nearest = chosen
            other = geom_list[nearest]
            proj_pt = other.interpolate(other.project(ep))
            proj_x[k] = proj_pt.x
            proj_y[k] = proj_pt.y

        valid = np.isfinite(proj_x)
        if valid.any():
            elev = dem_interp(np.column_stack([proj_y[valid], proj_x[valid]]))
            endpoint_elev[ir_idx[valid]] = elev

    strips["endpoint_elev_m"] = endpoint_elev

    # Smooth elevations along each (reach, side) to remove DEM noise
    from scipy.ndimage import gaussian_filter1d

    smooth_sigma = 5.0  # stations (~100m at 20m spacing)
    for col in ("base_elev_m", "endpoint_elev_m"):
        vals = strips[col].values.copy()
        for _, grp in strips.groupby(["reach_id", "side"]):
            if len(grp) < 3:
                continue
            idx = grp.sort_values("station_id").index
            raw = vals[idx]
            vals[idx] = gaussian_filter1d(raw, sigma=smooth_sigma, mode="nearest")
        strips[col] = vals

    return strips


def build_wedge_polygons(
    strips: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Build quadrilateral wedge polygons from consecutive same-(reach, side)
    strip pairs. Each wedge carries the four corner elevations needed for
    bilinear water-surface interpolation.
    """
    wedges: list[dict] = []
    for (rid, side), grp in strips.groupby(["reach_id", "side"]):
        grp = grp.sort_values("station_id").reset_index(drop=True)
        n = len(grp)
        for j in range(n - 1):
            s0 = grp.iloc[j]
            s1 = grp.iloc[j + 1]
            a0 = (float(s0["anchor_x"]), float(s0["anchor_y"]))
            a1 = (float(s1["anchor_x"]), float(s1["anchor_y"]))
            e0 = tuple(s0.geometry.coords[-1][:2])
            e1 = tuple(s1.geometry.coords[-1][:2])
            quad = Polygon([a0, e0, e1, a1, a0])
            if not quad.is_valid:
                quad = quad.buffer(0)
                if quad.geom_type == "MultiPolygon":
                    quad = max(quad.geoms, key=lambda p: p.area)
            if quad.is_empty or quad.area < 1e-3:
                continue
            wedges.append(
                {
                    "reach_id": int(rid),
                    "side": str(side),
                    "station_start": int(s0["station_id"]),
                    "station_end": int(s1["station_id"]),
                    "strip_type": str(s0["strip_type"]),
                    "poly_id": int(s0["poly_id"]),
                    "base_elev_0": float(s0["base_elev_m"]),
                    "base_elev_1": float(s1["base_elev_m"]),
                    "ep_elev_0": float(s0["endpoint_elev_m"]),
                    "ep_elev_1": float(s1["endpoint_elev_m"]),
                    "geometry": quad,
                }
            )
    if not wedges:
        return gpd.GeoDataFrame(
            columns=[
                "reach_id",
                "side",
                "station_start",
                "station_end",
                "strip_type",
                "poly_id",
                "base_elev_0",
                "base_elev_1",
                "ep_elev_0",
                "ep_elev_1",
                "geometry",
            ],
            geometry="geometry",
            crs=strips.crs,
        )
    return gpd.GeoDataFrame(wedges, geometry="geometry", crs=strips.crs)


def _burn_strip_to_raster(
    strip_geom: LineString,
    base_elev: float,
    endpoint_elev: float,
    ws: np.ndarray,
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    y_desc: bool,
) -> None:
    """Walk a strip line at sub-pixel resolution and burn interpolated
    water-surface elevation into *ws* (in-place). Pure numpy — no shapely
    calls in the inner loop.
    """
    line_coords = np.array(strip_geom.coords, dtype=np.float64)[:, :2]
    n_verts = len(line_coords)
    if n_verts < 2:
        return
    # Cumulative arc-length along the polyline vertices
    seg_dx = np.diff(line_coords[:, 0])
    seg_dy = np.diff(line_coords[:, 1])
    seg_len = np.sqrt(seg_dx**2 + seg_dy**2)
    cum_len = np.empty(n_verts, dtype=np.float64)
    cum_len[0] = 0.0
    np.cumsum(seg_len, out=cum_len[1:])
    total = cum_len[-1]
    if total < 1e-6:
        return

    res_x = abs(float(x_vals[1] - x_vals[0]))
    res_y = abs(float(y_vals[1] - y_vals[0]))
    step = min(res_x, res_y) * 0.5
    n_samples = max(2, int(np.ceil(total / step)) + 1)
    ts = np.linspace(0.0, total, n_samples)

    # Interpolate (x, y) along the polyline at each sample distance
    seg_idx = np.searchsorted(cum_len, ts, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, n_verts - 2)
    local_t = np.where(
        seg_len[seg_idx] > 1e-12,
        (ts - cum_len[seg_idx]) / seg_len[seg_idx],
        0.0,
    )
    sx = line_coords[seg_idx, 0] + local_t * seg_dx[seg_idx]
    sy = line_coords[seg_idx, 1] + local_t * seg_dy[seg_idx]

    # Interpolate elevation linearly along the strip
    t_frac = ts / total
    elevs = base_elev + t_frac * (endpoint_elev - base_elev)

    # Map to pixel indices
    cols = np.round((sx - x_vals[0]) / res_x).astype(np.intp)
    if y_desc:
        rows = np.round((y_vals[0] - sy) / res_y).astype(np.intp)
    else:
        rows = np.round((sy - y_vals[0]) / res_y).astype(np.intp)

    ny, nx = ws.shape
    valid = (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
    ws[rows[valid], cols[valid]] = elevs[valid]


def rasterize_water_surface(
    wedges: gpd.GeoDataFrame,
    strips: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
    snapped_gdf: gpd.GeoDataFrame | None = None,
    faces: NetworkFaces | None = None,
    strip_spacing_m: float = STRIP_SPACING_M,
) -> xr.DataArray:
    """Rasterize the wedge-based water surface onto the DEM grid.

    **Side wedges** receive bilinear interpolation from the four corner
    elevations (anchor → ridge).

    **Interreach strips** are burned directly into the raster with
    linearly interpolated thalweg-to-thalweg elevations, then gaps
    between burned cross sections are filled with a NaN-aware Gaussian
    filter within each interreach polygon mask.

    When multiple side wedges cover the same pixel, the one from the nearest
    thalweg (smallest anchor distance) wins.
    """
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds
    from scipy.ndimage import gaussian_filter

    y_vals = dem_da.y.values.astype(np.float64)
    x_vals = dem_da.x.values.astype(np.float64)
    ny, nx = len(y_vals), len(x_vals)
    res_x = abs(float(x_vals[1] - x_vals[0]))
    res_y = abs(float(y_vals[1] - y_vals[0]))
    y_desc = y_vals[0] > y_vals[-1]

    # Output arrays
    ws = np.full((ny, nx), np.nan, dtype=np.float64)
    best_dist = np.full((ny, nx), np.inf, dtype=np.float64)

    # Transform: pixel (col, row) → map coords
    if y_desc:
        transform = from_bounds(
            float(x_vals[0]) - res_x / 2,
            float(y_vals[-1]) - res_y / 2,
            float(x_vals[-1]) + res_x / 2,
            float(y_vals[0]) + res_y / 2,
            nx,
            ny,
        )
    else:
        transform = from_bounds(
            float(x_vals[0]) - res_x / 2,
            float(y_vals[0]) - res_y / 2,
            float(x_vals[-1]) + res_x / 2,
            float(y_vals[-1]) + res_y / 2,
            nx,
            ny,
        )

    # Pixel coordinate grids (needed for side wedge bilinear)
    col_idx = np.arange(nx)
    row_idx = np.arange(ny)
    cc, rr = np.meshgrid(col_idx, row_idx)
    px_x = x_vals[cc]
    px_y = y_vals[rr]

    # --- Phase 1: interreach — burn cross sections, Gaussian fill ---
    if faces is not None:
        ir_strips = strips[strips["strip_type"] == "interreach"]
        if not ir_strips.empty:
            # Burn all interreach strips into a sparse raster
            ir_sparse = np.full((ny, nx), np.nan, dtype=np.float64)
            for _, s in ir_strips.iterrows():
                g = s.geometry
                be = float(s["base_elev_m"])
                ee = float(s["endpoint_elev_m"])
                if (
                    g is None
                    or g.is_empty
                    or not np.isfinite(be)
                    or not np.isfinite(ee)
                ):
                    continue
                _burn_strip_to_raster(g, be, ee, ir_sparse, x_vals, y_vals, y_desc)

            n_burned = np.isfinite(ir_sparse).sum()

            # Gaussian sigma: half the strip spacing in pixels
            sigma = strip_spacing_m / (2.0 * res_x)

            # Per-polygon: mask, fill gaps from burned cross sections
            poly_ids = sorted(ir_strips["poly_id"].unique())
            for poly_id in poly_ids:
                poly_id = int(poly_id)
                if poly_id < 0 or poly_id >= len(faces.polygons):
                    continue
                poly = faces.polygons[poly_id]

                mask = rio_rasterize(
                    [(poly, 1)],
                    out_shape=(ny, nx),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    all_touched=True,
                ).astype(bool)
                if not mask.any():
                    continue

                # Extract sub-raster within the polygon bounding box
                rr_m, cc_m = np.where(mask)
                r0, r1 = int(rr_m.min()), int(rr_m.max()) + 1
                c0, c1 = int(cc_m.min()), int(cc_m.max()) + 1
                sub_mask = mask[r0:r1, c0:c1]
                sub_sparse = ir_sparse[r0:r1, c0:c1].copy()

                # NaN-aware Gaussian: blur values and weights separately
                burned = np.isfinite(sub_sparse) & sub_mask
                vals = np.where(burned, sub_sparse, 0.0)
                wts = burned.astype(np.float64)
                vals_smooth = gaussian_filter(vals, sigma)
                wts_smooth = gaussian_filter(wts, sigma)
                wts_smooth = np.maximum(wts_smooth, 1e-12)
                filled = vals_smooth / wts_smooth

                # Write filled values into ws where polygon mask is active
                write = sub_mask
                ws[r0:r1, c0:c1][write] = filled[write]
                # Interreach gets priority 0 (always wins over side wedges
                # for pixels inside the polygon)
                best_dist[r0:r1, c0:c1][write] = 0.0

            print(
                f"    interreach: {n_burned} burned pixels, "
                f"{len(poly_ids)} polygons filled (sigma={sigma:.1f}px)"
            )

    # --- Phase 2: side wedges (bilinear per wedge) ---
    side_wedges = wedges[wedges["strip_type"] != "interreach"]
    for idx, wedge in side_wedges.iterrows():
        geom = wedge.geometry
        if geom is None or geom.is_empty:
            continue

        mask = rio_rasterize(
            [(geom, 1)],
            out_shape=(ny, nx),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        ).astype(bool)
        if not mask.any():
            continue

        rows_hit, cols_hit = np.where(mask)
        hit_x = px_x[rows_hit, cols_hit]
        hit_y = px_y[rows_hit, cols_hit]

        ring = np.array(wedge.geometry.exterior.coords, dtype=np.float64)
        anchor_0 = ring[0, :2]
        endpt_0 = ring[1, :2]
        endpt_1 = ring[2, :2]
        anchor_1 = ring[3, :2]

        be0 = float(wedge["base_elev_0"])
        be1 = float(wedge["base_elev_1"])
        ee0 = float(wedge["ep_elev_0"])
        ee1 = float(wedge["ep_elev_1"])

        if not all(np.isfinite([be0, be1, ee0, ee1])):
            continue

        along = anchor_1 - anchor_0
        along_len = float(np.linalg.norm(along))
        if along_len < 1e-6:
            continue
        along_unit = along / along_len

        dx = hit_x - anchor_0[0]
        dy = hit_y - anchor_0[1]
        t_along = np.clip((dx * along_unit[0] + dy * along_unit[1]) / along_len, 0, 1)

        anch_x = anchor_0[0] + t_along * along[0]
        anch_y = anchor_0[1] + t_along * along[1]
        ep_x = endpt_0[0] + t_along * (endpt_1[0] - endpt_0[0])
        ep_y = endpt_0[1] + t_along * (endpt_1[1] - endpt_0[1])

        cross_dx = ep_x - anch_x
        cross_dy = ep_y - anch_y
        cross_len = np.sqrt(cross_dx**2 + cross_dy**2)
        cross_len = np.maximum(cross_len, 1e-6)
        pixel_dx = hit_x - anch_x
        pixel_dy = hit_y - anch_y
        t_cross = np.clip(
            (pixel_dx * cross_dx + pixel_dy * cross_dy) / (cross_len**2),
            0,
            1,
        )
        ws_anchor = be0 + t_along * (be1 - be0)
        ws_endpt = ee0 + t_along * (ee1 - ee0)
        ws_pixel = ws_anchor + t_cross * (ws_endpt - ws_anchor)

        d_anch = np.sqrt((hit_x - anch_x) ** 2 + (hit_y - anch_y) ** 2)

        better = d_anch < best_dist[rows_hit, cols_hit]
        write_rows = rows_hit[better]
        write_cols = cols_hit[better]
        ws[write_rows, write_cols] = ws_pixel[better]
        best_dist[write_rows, write_cols] = d_anch[better]

    ws_da = xr.DataArray(
        ws,
        dims=dem_da.dims,
        coords=dem_da.coords,
        attrs={"long_name": "water_surface_elevation", "units": "m"},
    )
    ws_da = ws_da.rio.write_crs(dem_da.rio.crs)
    ws_da = ws_da.rio.write_transform(transform)
    ws_da = ws_da.rio.write_nodata(np.nan)
    return ws_da


def compute_rem(dem_da: xr.DataArray, water_surface_da: xr.DataArray) -> xr.DataArray:
    """REM = max(DEM − water_surface, 0), NaN where no water surface."""
    dem = np.asarray(dem_da.values, dtype=np.float64)
    ws = np.asarray(water_surface_da.values, dtype=np.float64)
    rem = np.maximum(dem - ws, 0.0)
    rem[np.isnan(ws)] = np.nan
    rem_da = xr.DataArray(
        rem,
        dims=dem_da.dims,
        coords=dem_da.coords,
        attrs={"long_name": "relative_elevation_model", "units": "m"},
    )
    rem_da = rem_da.rio.write_crs(dem_da.rio.crs)
    rem_da = rem_da.rio.write_nodata(np.nan)
    return rem_da


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _write_result(
    result: PairedStripResult,
    crs,
    tag: str,
    side_a_reaches: list[int],
    side_b_reaches: list[int],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.strips.to_file(OUT_DIR / f"paired_strips_{tag}.fgb", driver="FlatGeobuf")
    gpd.GeoDataFrame(
        {
            "side_a": ["|".join(str(r) for r in side_a_reaches)],
            "side_b": ["|".join(str(r) for r in side_b_reaches)],
        },
        geometry=[result.polygon],
        crs=crs,
    ).to_file(OUT_DIR / f"paired_polygon_{tag}.fgb", driver="FlatGeobuf")
    gpd.GeoDataFrame(
        {"side": ["a", "b"]},
        geometry=[result.clipped_frame_a, result.clipped_frame_b],
        crs=crs,
    ).to_file(OUT_DIR / f"paired_clipped_frames_{tag}.fgb", driver="FlatGeobuf")
    gpd.GeoDataFrame(
        {"side": ["a", "b"]},
        geometry=[result.clipped_thalweg_a, result.clipped_thalweg_b],
        crs=crs,
    ).to_file(OUT_DIR / f"paired_clipped_thalwegs_{tag}.fgb", driver="FlatGeobuf")
    bridges = list(result.bridges_a) + list(result.bridges_b)
    if bridges:
        sides = (["a"] * len(result.bridges_a)) + (["b"] * len(result.bridges_b))
        gpd.GeoDataFrame({"side": sides}, geometry=bridges, crs=crs).to_file(
            OUT_DIR / f"paired_bridges_{tag}.fgb", driver="FlatGeobuf"
        )


def _run_case(
    frame_gdf,
    snapped_gdf,
    dem_da,
    side_a: list[int],
    side_b: list[int],
    tag: str,
) -> None:
    print(f"\n== {tag}  side_a={side_a}  side_b={side_b} ==")
    result = paired_sides_aspect_strips(
        frame_gdf,
        snapped_gdf,
        dem_da,
        side_a_reaches=side_a,
        side_b_reaches=side_b,
    )
    if result is None:
        print("  no paired strips produced")
        return
    print(
        f"  aspect down: ({result.aspect_down[0]:+.3f}, {result.aspect_down[1]:+.3f})"
    )
    print(f"  strip dir:   ({result.strip_dir[0]:+.3f}, {result.strip_dir[1]:+.3f})")
    print(f"  polygon area: {result.polygon.area:.0f} m^2")
    print(f"  bridges: side_a={len(result.bridges_a)} side_b={len(result.bridges_b)}")
    print(f"  strips: {len(result.strips)}")
    _write_result(result, frame_gdf.crs, tag, side_a, side_b)
    print(f"  wrote outputs to {OUT_DIR} (tag={tag})")


def _write_network_faces(
    faces: NetworkFaces,
    snapped_gdf: gpd.GeoDataFrame,
    crs,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Polygon layer
    if faces.polygons:
        poly_rows = []
        for i, p in enumerate(faces.polygons):
            asp = faces.polygon_aspects.get(i)
            ax = float(asp[0]) if asp is not None and np.isfinite(asp[0]) else np.nan
            ay = float(asp[1]) if asp is not None and np.isfinite(asp[1]) else np.nan
            # Count distinct reaches touching this polygon
            reach_ids = set()
            for (rid, _side), pidx in faces.reach_side_map.items():
                if pidx == i:
                    reach_ids.add(rid)
            poly_rows.append(
                {
                    "poly_id": i,
                    "area_m2": round(p.area, 1),
                    "n_reaches": len(reach_ids),
                    "reach_ids": "|".join(str(r) for r in sorted(reach_ids)),
                    "aspect_x": ax,
                    "aspect_y": ay,
                    "geometry": p,
                }
            )
        gpd.GeoDataFrame(poly_rows, geometry="geometry", crs=crs).to_file(
            OUT_DIR / "network_polygons.fgb", driver="FlatGeobuf"
        )

    # Closure edges
    if faces.closure_edges:
        gpd.GeoDataFrame(
            {"edge_id": list(range(len(faces.closure_edges)))},
            geometry=faces.closure_edges,
            crs=crs,
        ).to_file(OUT_DIR / "network_closure_edges.fgb", driver="FlatGeobuf")

    # Per-reach-side classification as midpoint arrows
    side_rows = []
    for _, row in snapped_gdf.iterrows():
        rid = int(row["reach_id"])
        geom = row.geometry
        if geom is None or geom.is_empty or geom.length < 1e-6:
            continue
        mid = geom.interpolate(0.5, normalized=True)
        eps = min(1.0, geom.length * 0.01)
        p0 = geom.interpolate(max(geom.length * 0.5 - eps, 0.0))
        p1 = geom.interpolate(min(geom.length * 0.5 + eps, geom.length))
        tx, ty = p1.x - p0.x, p1.y - p0.y
        tn = float(np.hypot(tx, ty))
        if tn < 1e-9:
            continue
        for side_label, sign in [("left", 1.0), ("right", -1.0)]:
            nx = -ty / tn * sign
            ny = tx / tn * sign
            arrow_end = Point(mid.x + 30.0 * nx, mid.y + 30.0 * ny)
            pidx = faces.reach_side_map.get((rid, side_label))
            side_rows.append(
                {
                    "reach_id": rid,
                    "side": side_label,
                    "poly_id": pidx if pidx is not None else -1,
                    "label": (
                        f"r{rid}_{side_label}→p{pidx}"
                        if pidx is not None
                        else f"r{rid}_{side_label}→ext"
                    ),
                    "geometry": LineString(
                        [(mid.x, mid.y), (arrow_end.x, arrow_end.y)]
                    ),
                }
            )
    if side_rows:
        gpd.GeoDataFrame(side_rows, geometry="geometry", crs=crs).to_file(
            OUT_DIR / "reach_side_classification.fgb", driver="FlatGeobuf"
        )


def resnap_reaches(
    flowlines: gpd.GeoDataFrame,
    dem_da: xr.DataArray,
    ndwi_da: xr.DataArray,
    max_offset_m: float = RESNAP_MAX_OFFSET_M,
    search_spacing_m: float = RESNAP_SEARCH_SPACING_M,
    station_spacing_m: float = RESNAP_STATION_SPACING_M,
    w_elev: float = RESNAP_W_ELEV,
    w_water: float = RESNAP_W_WATER,
    w_prior: float = RESNAP_W_PRIOR,
    w_transition: float = RESNAP_W_TRANSITION,
    smoothing_m: float = RESNAP_SMOOTHING_M,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Snap NHD flowlines to the DEM channel bottom.

    Splits the flowlines into reaches, then DP-snaps each reach using the
    NHD geometry as the prior. Returns ``(snapped_gdf, frame_gdf)``.
    """
    from handily.config import HandilyConfig
    from handily.rem_frame import (
        SnappedReach,
        _build_dem_interpolator,
        _enforce_junction_topology,
        build_smoothed_frame,
        snap_reach_to_thalweg,
        split_confirmed_flowlines_into_reaches,
    )

    config = HandilyConfig(
        out_dir="",
        flowlines_local_dir="",
        ndwi_dir="",
        stac_dir="",
        fields_path="",
        rem_frame_station_spacing_m=station_spacing_m,
        rem_snap_max_offset_m=max_offset_m,
        rem_snap_search_spacing_m=search_spacing_m,
        rem_snap_w_elev=w_elev,
        rem_snap_w_water=w_water,
        rem_snap_w_prior=w_prior,
        rem_snap_w_transition=w_transition,
        rem_water_support_mode="continuous_index",
        ndwi_threshold=0.25,
        rem_support_corridor_half_width_m=15.0,
        rem_support_corridor_half_length_m=10.0,
    )

    reaches = split_confirmed_flowlines_into_reaches(flowlines)
    print(f"  {len(reaches)} reaches from {len(flowlines)} flowlines")

    dem_interp = _build_dem_interpolator(dem_da)
    ndwi_interp = _build_dem_interpolator(ndwi_da)

    snapped_list: list[SnappedReach] = []
    for _, row in reaches.iterrows():
        rid = int(row["reach_id"])
        prior_geom = row.geometry
        if (
            prior_geom is None
            or prior_geom.is_empty
            or prior_geom.length < 2 * station_spacing_m
        ):
            print(f"  reach {rid}: too short, keeping original")
            coords = list(prior_geom.coords)
            st_rows = []
            for si, c in enumerate(coords):
                st_rows.append(
                    {
                        "station_id": si,
                        "s_m": 0.0,
                        "x_prior": float(c[0]),
                        "y_prior": float(c[1]),
                        "x_snap": float(c[0]),
                        "y_snap": float(c[1]),
                        "snap_offset_m": 0.0,
                        "thalweg_elev_m": 0.0,
                        "water_support_frac": 0.0,
                        "water_support_mean": 0.0,
                        "water_hit": False,
                        "geometry": Point(c[0], c[1]),
                    }
                )
            snapped_list.append(
                SnappedReach(
                    reach_id=rid,
                    prior_geom=prior_geom,
                    snapped_geom=prior_geom,
                    stations=gpd.GeoDataFrame(
                        st_rows, geometry="geometry", crs=dem_da.rio.crs
                    ),
                    confidence=0.0,
                )
            )
            continue

        result = snap_reach_to_thalweg(
            prior_geom,
            dem_da,
            ndwi_da,
            config,
            dem_interp=dem_interp,
            ndwi_interp=ndwi_interp,
        )
        result.reach_id = rid
        offset = float(np.mean(np.abs(result.stations["snap_offset_m"].values)))
        print(
            f"  reach {rid}: mean_offset={offset:.1f}m, "
            f"water_hit={result.confidence * 100:.0f}%"
        )
        snapped_list.append(result)

    # Remove meander loops (self-intersections) from snapped lines
    for s in snapped_list:
        s.snapped_geom = _remove_loops(s.snapped_geom)
        if not s.stations.empty:
            coords = np.array(s.snapped_geom.coords, dtype=np.float64)[:, :2]
            n = len(coords)
            n_st = len(s.stations)
            if n != n_st:
                cum = np.zeros(n)
                cum[1:] = np.cumsum(
                    np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
                )
                new_rows = []
                for i in range(n):
                    new_rows.append(
                        {
                            "station_id": i,
                            "s_m": float(cum[i]),
                            "x_prior": float(coords[i, 0]),
                            "y_prior": float(coords[i, 1]),
                            "x_snap": float(coords[i, 0]),
                            "y_snap": float(coords[i, 1]),
                            "snap_offset_m": 0.0,
                            "thalweg_elev_m": float(
                                dem_interp(np.array([[coords[i, 1], coords[i, 0]]]))[0]
                            ),
                            "water_support_frac": 0.0,
                            "water_support_mean": 0.0,
                            "water_hit": False,
                            "geometry": Point(coords[i, 0], coords[i, 1]),
                        }
                    )
                s.stations = gpd.GeoDataFrame(
                    new_rows,
                    geometry="geometry",
                    crs=dem_da.rio.crs,
                )

    # Enforce junction topology (shared endpoints)
    snapped_list = _enforce_junction_topology(reaches, snapped_list)

    # Build output GeoDataFrames
    snap_rows = []
    frame_rows = []
    for s in snapped_list:
        snap_rows.append(
            {
                "reach_id": s.reach_id,
                "geometry": s.snapped_geom,
            }
        )
        fr = build_smoothed_frame(s, smoothing_m)
        frame_rows.append(
            {
                "reach_id": fr.reach_id,
                "geometry": fr.frame_geom,
            }
        )

    crs = dem_da.rio.crs
    snapped = gpd.GeoDataFrame(snap_rows, geometry="geometry", crs=crs)
    frames = gpd.GeoDataFrame(frame_rows, geometry="geometry", crs=crs)
    return snapped, frames


def _load_debug_flowlines() -> gpd.GeoDataFrame:
    """Load NHD flowlines for the debug subset (same as debug_subset_0773.py)."""
    from handily.nhd import REM_EXCLUDED_FCODES, get_fcode_column

    aoi_dir = Path("/data/ssd2/handily/nv/aoi_0773")
    annotated = gpd.read_file(aoi_dir / "flowlines_annotated_ndwi025.fgb")
    reachable = annotated[annotated["reachable_from_seed"]].copy()
    fcode_col = get_fcode_column(reachable)
    if fcode_col and REM_EXCLUDED_FCODES:
        reachable = reachable[~reachable[fcode_col].isin(REM_EXCLUDED_FCODES)].copy()

    # Clip to the debug polygon
    W, E = -115.484908, -115.46031387
    N, S = 40.788723, 40.75029785
    clip_poly = Polygon([(W, N), (E, N), (E, S), (W, S), (W, N)])
    clip_gdf = gpd.GeoDataFrame(geometry=[clip_poly], crs="EPSG:4326").to_crs(
        "EPSG:5070"
    )
    clip_geom = clip_gdf.geometry.iloc[0]
    flowlines = reachable[reachable.intersects(clip_geom)].copy()
    print(f"  {len(flowlines)} reachable flowlines in debug clip")
    return flowlines


def main() -> None:
    dem_da = rioxarray.open_rasterio(DEBUG_DIR / "dem.tif").squeeze("band", drop=True)
    ndwi_da = rioxarray.open_rasterio(DEBUG_DIR / "ndwi.tif").squeeze("band", drop=True)

    # --- Snap NHD flowlines to DEM channel bottom ---
    print("\n== Loading NHD flowlines ==")
    flowlines = _load_debug_flowlines()

    print("\n== Snapping to DEM channel ==")
    snapped_gdf, frame_gdf = resnap_reaches(flowlines, dem_da, ndwi_da)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    snapped_gdf.to_file(OUT_DIR / "resnapped.fgb", driver="FlatGeobuf")
    frame_gdf.to_file(OUT_DIR / "resnapped_frames.fgb", driver="FlatGeobuf")
    print(f"  wrote resnapped.fgb + resnapped_frames.fgb to {OUT_DIR}")

    # --- Network faces analysis ---
    print("\n== Network polygonize + classify ==")
    faces = build_network_faces(snapped_gdf, dem_da)
    print(f"  polygons: {len(faces.polygons)} (min_area={faces.min_area_m2} m^2)")
    print(f"  closure edges: {len(faces.closure_edges)}")
    for i, p in enumerate(faces.polygons):
        asp = faces.polygon_aspects.get(i)
        rids = sorted(
            {rid for (rid, _), pidx in faces.reach_side_map.items() if pidx == i}
        )
        ax = (
            f"({asp[0]:+.2f},{asp[1]:+.2f})"
            if asp is not None and np.isfinite(asp[0])
            else "n/a"
        )
        print(f"  poly {i}: {p.area:.0f} m^2, {len(rids)} reaches {rids}, aspect={ax}")

    # Summarize side classification
    n_inter = sum(1 for v in faces.reach_side_map.values() if v is not None)
    n_ext = sum(1 for v in faces.reach_side_map.values() if v is None)
    print(f"  reach-sides: {n_inter} inter-reach, {n_ext} exterior")
    _write_network_faces(faces, snapped_gdf, snapped_gdf.crs)
    print(f"  wrote network layers to {OUT_DIR}")

    # --- Generate strips for all reaches ---
    print("\n== Generating network strips ==")
    strips = generate_network_strips(snapped_gdf, frame_gdf, dem_da, faces)
    n_ir = (strips["strip_type"] == "interreach").sum()
    n_side = (strips["strip_type"] == "side").sum()
    print(f"  total: {len(strips)} strips ({n_ir} interreach, {n_side} side)")
    for st, grp in strips.groupby("strip_type"):
        ls = grp["length_m"]
        print(
            f"  {st}: n={len(grp)}, len min/med/max = {ls.min():.1f}/{ls.median():.1f}/{ls.max():.1f}"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    strips.to_file(OUT_DIR / "network_cross_sections.fgb", driver="FlatGeobuf")
    print(f"  wrote {OUT_DIR / 'network_cross_sections.fgb'}")
    if "is_seed" in strips.columns:
        seed_mask = strips["is_seed"].eq(True)
        if seed_mask.any():
            strips.loc[seed_mask].copy().to_file(
                OUT_DIR / "network_seeds.fgb", driver="FlatGeobuf"
            )
            print(f"  wrote {OUT_DIR / 'network_seeds.fgb'}")

    # --- Elevation attachment ---
    print("\n== Attaching strip elevations ==")
    strips = attach_strip_elevations(strips, snapped_gdf, dem_da)
    be = strips["base_elev_m"]
    ee = strips["endpoint_elev_m"]
    print(f"  base_elev: {be.min():.1f} / {be.median():.1f} / {be.max():.1f}")
    print(f"  endpoint_elev: {ee.min():.1f} / {ee.median():.1f} / {ee.max():.1f}")

    # --- Wedge construction ---
    print("\n== Building wedge polygons ==")
    wedges = build_wedge_polygons(strips)
    print(f"  wedges: {len(wedges)}")
    wedges.to_file(OUT_DIR / "network_wedges.fgb", driver="FlatGeobuf")
    print(f"  wrote {OUT_DIR / 'network_wedges.fgb'}")

    # --- Rasterize water surface + REM ---
    print("\n== Rasterizing water surface ==")
    ws_da = rasterize_water_surface(wedges, strips, dem_da, snapped_gdf, faces)
    ws_valid = np.isfinite(ws_da.values).sum()
    ws_total = ws_da.values.size
    print(
        f"  water surface: {ws_valid}/{ws_total} pixels ({100 * ws_valid / ws_total:.1f}%)"
    )
    ws_da.rio.to_raster(str(OUT_DIR / "network_water_surface.tif"))

    rem_da = compute_rem(dem_da, ws_da)
    rem_da.rio.to_raster(str(OUT_DIR / "network_rem.tif"))
    print(f"  wrote network_water_surface.tif + network_rem.tif to {OUT_DIR}")


if __name__ == "__main__":
    main()
