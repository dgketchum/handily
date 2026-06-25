"""Step-by-step figure series for the FAC depth-to-water REM pipeline.

Renders one captioned PNG per pipeline stage over a New-Mexico AOI (the Mesilla
Valley reach of the Rio Grande near Las Cruces) straight from the intermediate
outputs already on disk -- no pipeline re-run. Built for a NM audience to walk
the FAC workflow: terrain + channel network -> flow accumulation -> NDVI wet
seed -> upstream propagation -> aspect-normal strips -> channel-head solve ->
sparse burn / IDW fill -> final REM -> validation vs wells.

Colormap conventions mirror ``src/handily/qgis.py`` (REM 0-10 m Spectral-style,
NDVI RdYlGn) so the figures match what shows up in QGIS. The validation panel
reuses ``utils/gwx_wells.resid_stats`` for the n/MAD/bias/RMSE annotation.

The 10 m REM/IDW/sparse rasters are multi-GB; every raster read is a windowed
read over the AOI bbox only (~2080x1270 px), never a full read.

Usage:
    uv run python utils/figure_fac_workflow.py \
        --base-dir /data/ssd2/handily/nm/regional/mesilla \
        --version nm_mesilla_v6_corridor \
        --out-dir /data/ssd2/handily/nm/regional/mesilla/figures/workflow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from pyproj import Transformer
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gwx_wells import resid_stats  # noqa: E402

# AOI: NW 32.23778,-106.83794 -> SE 32.12362,-106.61710 (lon_w, lat_s, lon_e, lat_n)
DEFAULT_BBOX = (-106.83794, 32.12362, -106.61710, 32.23778)

# All figures are drawn in EPSG:5070 (the raster CRS / locator reprojection).
PLOT_CRS = "EPSG:5070"

# State boundaries (TIGER, WGS84) for the locator inset.
STATES_SHP = "/nas/boundaries/us_states_tiger_wgs.shp"
# Southwestern context states around NM, plus orientation cities (lon, lat).
LOCATOR_STATES = ("NM", "AZ", "TX", "CO", "UT", "OK", "KS")
NM_CITIES = {"Albuquerque": (-106.65, 35.08), "Las Cruces": (-106.78, 32.31)}

# REM / depth colormap matching qgis.py pseudocolor stops (vmin blue -> mid
# yellow -> vmax red): shallow/wet blue, deep/dry red.
REM_CMAP = LinearSegmentedColormap.from_list(
    "rem_spectral",
    [
        (0.0, (44 / 255, 123 / 255, 182 / 255)),
        (0.5, (255 / 255, 255 / 255, 191 / 255)),
        (1.0, (215 / 255, 25 / 255, 28 / 255)),
    ],
)


def bbox_5070(bbox_wgs84: tuple[float, float, float, float], dst_crs) -> tuple:
    """Reproject a (lon_w, lat_s, lon_e, lat_n) WGS84 bbox to ``dst_crs`` bounds."""
    lon_w, lat_s, lon_e, lat_n = bbox_wgs84
    return transform_bounds("EPSG:4326", dst_crs, lon_w, lat_s, lon_e, lat_n)


def read_window(path: str, bounds: tuple[float, float, float, float]):
    """Windowed read of band 1 over ``bounds`` (raster CRS). nodata/fill -> NaN.

    Returns (data, extent) where extent is [xmin, xmax, ymin, ymax] for imshow.
    """
    left, bottom, right, top = bounds
    with rasterio.open(path) as src:
        window = from_bounds(left, bottom, right, top, src.transform)
        arr = src.read(1, window=window, masked=True)
        wt = src.window_transform(window)
        nod = src.nodata
    data = arr.filled(np.nan).astype("float64")
    if nod is not None and np.isfinite(nod):
        data[data == nod] = np.nan
    data[np.abs(data) > 1e29] = np.nan
    h, w = data.shape
    x0, y0 = wt.c, wt.f
    extent = [x0, x0 + w * wt.a, y0 + h * wt.e, y0]
    return data, extent


def hillshade(dem: np.ndarray, dx: float = 10.0, vert_exag: float = 2.5) -> np.ndarray:
    """Grayscale hillshade from a DEM window (NaN filled to the local minimum)."""
    z = np.where(np.isfinite(dem), dem, np.nanmin(dem))
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(z, vert_exag=vert_exag, dx=dx, dy=dx)


def new_map_fig(extent, title: str, caption: str, figsize=(11, 8)):
    """Map figure scaffold: equal aspect, no ticks, title, italic caption."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.10)
    fig.text(
        0.5,
        0.035,
        caption,
        ha="center",
        va="center",
        fontsize=9.5,
        style="italic",
        wrap=True,
    )
    return fig, ax


def add_scalebar(ax, extent, km: float = 5.0) -> None:
    dx, dy = extent[1] - extent[0], extent[3] - extent[2]
    x0 = extent[0] + 0.06 * dx
    y0 = extent[2] + 0.06 * dy
    ax.plot(
        [x0, x0 + km * 1000.0], [y0, y0], color="black", lw=3, solid_capstyle="butt"
    )
    ax.text(x0 + km * 500.0, y0 + 0.018 * dy, f"{km:g} km", ha="center", fontsize=8.5)


def true_north_angle(x: float, y: float, crs: str = PLOT_CRS) -> float:
    """Angle (deg, CCW from +x/east) of true north at projected point (x, y).

    EPSG:5070 is a conic projection; west of its -96 deg central meridian, grid
    north (straight up) is rotated from true north by the meridian-convergence
    angle (~6 deg over New Mexico). Measure it numerically by projecting a point
    slightly north and taking the bearing of the displacement.
    """
    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    to_proj = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon, lat = to_geo.transform(x, y)
    x2, y2 = to_proj.transform(lon, lat + 0.05)
    return float(np.degrees(np.arctan2(y2 - y, x2 - x)))


def add_north(ax, extent, crs: str = PLOT_CRS) -> None:
    """North arrow aligned to *true* north at the map center (not grid-up)."""
    dx, dy = extent[1] - extent[0], extent[3] - extent[2]
    cx, cy = (extent[0] + extent[1]) / 2, (extent[2] + extent[3]) / 2
    ang = np.radians(true_north_angle(cx, cy, crs))
    ux, uy = (
        np.cos(ang),
        np.sin(ang),
    )  # data units are equal (equal aspect) -> on-screen angle preserved
    tx, ty = extent[1] - 0.10 * dx, extent[3] - 0.17 * dy
    length = 0.10 * dy
    hx, hy = tx + ux * length, ty + uy * length
    ax.annotate(
        "",
        xy=(hx, hy),
        xytext=(tx, ty),
        arrowprops=dict(arrowstyle="-|>", lw=1.6, color="black"),
    )
    ax.text(
        hx + ux * 0.022 * dy,
        hy + uy * 0.022 * dy,
        "N",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )


def show_hillshade(ax, hs, extent) -> None:
    ax.imshow(hs, extent=extent, cmap="gray", origin="upper", zorder=0)


def strahler_lw(strahler) -> np.ndarray:
    return 0.25 + 0.32 * np.clip(np.asarray(strahler, dtype="float64"), 1, 7)


def save(fig, out_dir: Path, name: str, dpi: int) -> Path:
    out = out_dir / name
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig0_locator(ctx):
    """Standalone NM locator/inset: AOI red box within the state for orientation."""
    states = gpd.read_file(STATES_SHP).to_crs(5070)
    nbr = states[states["STUSPS"].isin(LOCATOR_STATES)]
    nm = states[states["STUSPS"] == "NM"]

    minx, miny, maxx, maxy = nm.total_bounds
    pad = 45000.0
    extent = [minx - pad, maxx + pad, miny - pad, maxy + pad]
    window = box(extent[0], extent[2], extent[1], extent[3])

    fig, ax = new_map_fig(
        extent,
        "Study area location — Mesilla Valley, New Mexico",
        "The FAC-REM workflow figures cover the Mesilla Valley reach of the Rio Grande "
        "in south-central New Mexico (red box), upstream of El Paso.",
        figsize=(7.5, 8.5),
    )
    nbr.plot(ax=ax, color="#ece9e2", edgecolor="white", linewidth=0.8, zorder=1)
    nm.plot(ax=ax, color="#d9c7a6", edgecolor="#5b4a2f", linewidth=1.8, zorder=2)

    # Neighbor abbreviations placed inside the visible window for orientation.
    for _, row in gpd.clip(nbr[nbr["STUSPS"] != "NM"], window).iterrows():
        pt = row.geometry.representative_point()
        ax.text(
            pt.x,
            pt.y,
            row["STUSPS"],
            fontsize=9,
            color="#9a9488",
            ha="center",
            va="center",
            zorder=3,
        )

    b = ctx["bounds"]  # AOI bounds in EPSG:5070 (left, bottom, right, top)
    ax.add_patch(
        Rectangle(
            (b[0], b[1]),
            b[2] - b[0],
            b[3] - b[1],
            facecolor="red",
            alpha=0.45,
            edgecolor="red",
            linewidth=1.6,
            zorder=5,
        )
    )
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    ax.annotate(
        "Mesilla Valley AOI",
        xy=(cx, cy),
        xytext=(cx - 235000, cy - 25000),
        fontsize=10,
        fontweight="bold",
        color="darkred",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.3),
        zorder=6,
    )

    cities = gpd.GeoDataFrame(
        {"name": list(NM_CITIES)},
        geometry=gpd.points_from_xy(*zip(*NM_CITIES.values())),
        crs="EPSG:4326",
    ).to_crs(5070)
    for _, c in cities.iterrows():
        ax.plot(c.geometry.x, c.geometry.y, "o", color="black", ms=4, zorder=6)
        ax.text(
            c.geometry.x + 9000,
            c.geometry.y,
            c["name"],
            fontsize=8.5,
            va="center",
            zorder=6,
        )

    add_scalebar(ax, extent, km=100)
    add_north(ax, extent)
    return save(fig, ctx["out_dir"], "fig0_nm_locator.png", ctx["dpi"])


def fig1_study_area(ctx):
    fig, ax = new_map_fig(
        ctx["dem_extent"],
        "Mesilla Valley, NM — terrain and FAC channel network",
        "3DEP 10 m terrain (hillshade) with the FAC-derived, Strahler-ordered drainage "
        "network. The perennial Rio Grande threads the valley; ephemeral arroyos drain the flanking desert.",
    )
    show_hillshade(ax, ctx["hs"], ctx["dem_extent"])
    s = ctx["streams"]
    s.plot(ax=ax, color="#1f4e9c", linewidth=strahler_lw(s["strahler"]), zorder=3)
    add_scalebar(ax, ctx["dem_extent"])
    add_north(ax, ctx["dem_extent"])
    return save(fig, ctx["out_dir"], "fig1_study_area.png", ctx["dpi"])


def fig2_flow_accum(ctx):
    fac, extent = read_window(ctx["paths"]["fac"], ctx["bounds"])
    logfac = np.log10(np.where(fac >= 1, fac, np.nan))
    fig, ax = new_map_fig(
        extent,
        "Step 1 — D8 flow accumulation defines the channels",
        "Upslope contributing area (log10 cells) routed by D8 flow direction. Flow concentrates "
        "into valley threads; thresholding the accumulation grid extracts the channel network (the “FAC”).",
    )
    show_hillshade(ax, ctx["hs"], ctx["dem_extent"])
    im = ax.imshow(
        logfac, extent=extent, cmap="cubehelix_r", origin="upper", alpha=0.85, zorder=2
    )
    ctx["streams"].plot(ax=ax, color="black", linewidth=0.4, alpha=0.6, zorder=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cb.set_label("log10 flow accumulation (cells)", fontsize=9)
    add_scalebar(ax, extent)
    add_north(ax, extent)
    return save(fig, ctx["out_dir"], "fig2_flow_accum.png", ctx["dpi"])


def fig3_ndvi_seed(ctx):
    ndvi, extent = read_window(ctx["paths"]["ndvi"], ctx["bounds"])
    fig, ax = new_map_fig(
        extent,
        "Step 2 — Greenness becomes a per-reach wet seed",
        "3-year NAIP median NDVI (brown→green) flags persistently green riparian and irrigated "
        "ground. A logistic on the along-reach NDVI quantile sets each reach's wet-seed strength (0–1).",
    )
    im = ax.imshow(
        ndvi,
        extent=extent,
        cmap="RdYlGn",
        origin="upper",
        vmin=-0.2,
        vmax=0.6,
        zorder=1,
    )
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cb.set_label("NAIP NDVI (3-yr median)", fontsize=9)
    heads = ctx["heads"]
    heads.plot(
        ax=ax,
        column="seed_strength",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        linewidth=1.4,
        zorder=4,
        legend=True,
        legend_kwds={"label": "reach seed strength", "shrink": 0.5, "pad": 0.08},
    )
    add_scalebar(ax, extent)
    add_north(ax, extent)
    return save(fig, ctx["out_dir"], "fig3_ndvi_seed.png", ctx["dpi"])


def fig4_propagation(ctx):
    fig, ax = new_map_fig(
        ctx["dem_extent"],
        "Step 3 — Wet influence propagates upstream; arroyos go dry",
        "Wet seeds propagate up the flow graph with distance/elevation decay (topo pin weight). "
        "Reaches far above any wet anchor lose influence — the desert-arroyo dry regime.",
    )
    show_hillshade(ax, ctx["hs"], ctx["dem_extent"])
    ctx["heads"].plot(
        ax=ax,
        column="topo_pin_weight",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidth=1.4,
        zorder=3,
        legend=True,
        legend_kwds={
            "label": "upstream wet influence (topo pin weight)",
            "shrink": 0.5,
            "pad": 0.08,
        },
    )
    add_scalebar(ax, ctx["dem_extent"])
    add_north(ax, ctx["dem_extent"])
    return save(fig, ctx["out_dir"], "fig4_propagation.png", ctx["dpi"])


def fig5_cross_sections(ctx):
    strips = gpd.read_file(ctx["paths"]["strips"], bbox=ctx["bounds"])
    # Subsample for legibility: every 3rd station keeps the cross-section pattern
    # visible without a solid mat of lines.
    if "station_id" in strips.columns:
        strips = strips[strips["station_id"] % 3 == 0]
    fig, ax = new_map_fig(
        ctx["dem_extent"],
        "Step 4 — Aspect-normal valley cross-sections",
        "Rays cast perpendicular to the smoothed valley slope at stations along each reach, "
        "terminating at the next channel (interreach), the AOI edge, or a bounded anchor (naked). "
        "Strips sample the cross-valley terrain that the water surface is fit to.",
    )
    show_hillshade(ax, ctx["hs"], ctx["dem_extent"])
    strips.plot(
        ax=ax,
        column="hit_type",
        categorical=True,
        cmap="Set1",
        linewidth=0.5,
        zorder=3,
        legend=True,
        legend_kwds={"title": "strip termination", "loc": "lower right", "fontsize": 8},
    )
    ctx["streams"].plot(ax=ax, color="black", linewidth=0.6, zorder=4)
    add_scalebar(ax, ctx["dem_extent"])
    add_north(ax, ctx["dem_extent"])
    return save(fig, ctx["out_dir"], "fig5_cross_sections.png", ctx["dpi"])


def fig6_head_depth(ctx):
    heads = ctx["heads"]
    vmax = float(np.nanpercentile(heads["head_depth_m"], 95)) or 1.0
    fig, ax = new_map_fig(
        ctx["dem_extent"],
        "Step 5 — Channel-head solve sets water-surface depth below bed",
        "Per-reach residual-depth relaxation: the water surface sits just below the bed in the wet "
        "valley and sags deep under dry arroyos, with a smoothness constraint along the network.",
    )
    show_hillshade(ax, ctx["hs"], ctx["dem_extent"])
    heads.plot(
        ax=ax,
        column="head_depth_m",
        cmap=REM_CMAP,
        vmin=0.0,
        vmax=vmax,
        linewidth=1.5,
        zorder=3,
        legend=True,
        legend_kwds={"label": "head depth below bed (m)", "shrink": 0.5, "pad": 0.08},
    )
    add_scalebar(ax, ctx["dem_extent"])
    add_north(ax, ctx["dem_extent"])
    return save(fig, ctx["out_dir"], "fig6_head_depth.png", ctx["dpi"])


def fig7_sparse_vs_idw(ctx):
    sparse, ext_s = read_window(ctx["paths"]["sparse"], ctx["bounds"])
    idw, ext_i = read_window(ctx["paths"]["idw"], ctx["bounds"])
    finite = idw[np.isfinite(idw)]
    vmin, vmax = np.nanpercentile(finite, [2, 98])
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, data, ext, sub in (
        (axes[0], sparse, ext_s, "Sparse burn (strips only)"),
        (axes[1], idw, ext_i, "IDW-filled water surface"),
    ):
        ax.imshow(
            ctx["hs"], extent=ctx["dem_extent"], cmap="gray", origin="upper", zorder=0
        )
        im = ax.imshow(
            data, extent=ext, cmap="viridis", norm=norm, origin="upper", zorder=2
        )
        ax.set_title(sub, fontsize=12, fontweight="bold")
        ax.set_xlim(ctx["dem_extent"][0], ctx["dem_extent"][1])
        ax.set_ylim(ctx["dem_extent"][2], ctx["dem_extent"][3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
    cb.set_label("water-surface elevation (m)", fontsize=9)
    fig.suptitle(
        "Step 6 — Sparse strip burns interpolated to a continuous water surface",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.04,
        "Each strip burns its bed-minus-head-depth water surface to a sparse grid (left, ~15–30% coverage); "
        "inverse-distance weighting fills it to a continuous surface (right).",
        ha="center",
        fontsize=9.5,
        style="italic",
        wrap=True,
    )
    fig.subplots_adjust(left=0.02, right=0.93, top=0.92, bottom=0.10, wspace=0.04)
    return save(fig, ctx["out_dir"], "fig7_sparse_vs_idw.png", ctx["dpi"])


def fig8_final_rem(ctx):
    rem, extent = read_window(ctx["paths"]["rem"], ctx["bounds"])
    fig, ax = new_map_fig(
        extent,
        "Step 7 — Final REM = depth to water table",
        "REM = DEM − interpolated water surface. Display clamped 0–10 m to resolve the valley; "
        "uplands saturate red (deep). This is the FAC depth-to-water prior.",
    )
    show_hillshade(ax, ctx["hs"], ctx["dem_extent"])
    im = ax.imshow(
        rem,
        extent=extent,
        cmap=REM_CMAP,
        origin="upper",
        vmin=0.0,
        vmax=10.0,
        alpha=0.78,
        zorder=2,
    )
    ctx["streams"].plot(ax=ax, color="black", linewidth=0.4, alpha=0.5, zorder=3)
    cb = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.01, extend="max")
    cb.set_label("depth to water (m)", fontsize=9)
    add_scalebar(ax, extent)
    add_north(ax, extent)
    return save(fig, ctx["out_dir"], "fig8_final_rem.png", ctx["dpi"])


def fig9_validation(ctx):
    wells = gpd.read_file(ctx["paths"]["residuals"])
    obs = wells["mean_dtw"].to_numpy(dtype="float64")
    fac = wells["pred_FAC"].to_numpy(dtype="float64")
    ma = wells["pred_Ma"].to_numpy(dtype="float64")
    st_fac = resid_stats(fac, obs)
    st_ma = resid_stats(ma, obs)

    lim = float(np.nanpercentile(np.concatenate([obs, fac, ma]), 98))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, pred, st, name, color in (
        (axes[0], fac, st_fac, "FAC depth-to-water", "#1f77b4"),
        (axes[1], ma, st_ma, "Ma et al. (benchmark)", "#d62728"),
    ):
        m = np.isfinite(obs) & np.isfinite(pred)
        ax.scatter(obs[m], pred[m], s=10, color=color, alpha=0.45, edgecolors="none")
        ax.plot([0, lim], [0, lim], color="black", lw=1.1, label="1:1")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("Observed depth to water (m, well median)")
        ax.set_ylabel("Predicted depth to water (m)")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.04,
            0.96,
            f"n={st['n']}\nMAD={st['mad_m']:.2f} m\nbias={st['bias_m']:.2f} m\n"
            f"median resid={st['median_residual_m']:.2f} m\nRMSE={st['rmse_m']:.2f} m",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85),
        )
    fig.suptitle(
        "Validation — FAC vs Ma against GWX unconfined wells (Mesilla)",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "FAC tracks the shallow valley/riparian water table; it currently trails Ma on a deep "
        "arroyo/upland tail (points high above the 1:1 line). Confined wells excluded.",
        ha="center",
        fontsize=9.5,
        style="italic",
        wrap=True,
    )
    fig.subplots_adjust(left=0.07, right=0.97, top=0.90, bottom=0.13, wspace=0.22)
    return save(fig, ctx["out_dir"], "fig9_validation.png", ctx["dpi"])


FIGURES = {
    0: fig0_locator,
    1: fig1_study_area,
    2: fig2_flow_accum,
    3: fig3_ndvi_seed,
    4: fig4_propagation,
    5: fig5_cross_sections,
    6: fig6_head_depth,
    7: fig7_sparse_vs_idw,
    8: fig8_final_rem,
    9: fig9_validation,
}


def build_paths(base: Path, version: str) -> dict:
    rem = base / "rem" / version
    val = "fac_v6_corridor_gwx" if "v6" in version else "fac_v5_arid_gwx"
    return {
        "dem": str(base / "dem_10m.tif"),
        "fac": str(base / "flow_accumulation.tif"),
        "streams": str(base / "streams_regional.fgb"),
        "ndvi": str(base / "evidence" / "naip" / "naip_ndvi_3yr_10m.tif"),
        "heads": str(rem / "fac_channel_heads.fgb"),
        "strips": str(rem / "fac_normals_cross_sections.fgb"),
        "sparse": str(rem / "fac_head_depth_sparse_10m.tif"),
        "idw": str(rem / "fac_rem_water_surface_10m.tif"),
        "rem": str(rem / "fac_head_depth_rem_10m.tif"),
        "residuals": str(base / "validation" / val / "fac_well_residuals.fgb"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", default="/data/ssd2/handily/nm/regional/mesilla")
    ap.add_argument("--version", default="nm_mesilla_v6_corridor")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        metavar=("LON_W", "LAT_S", "LON_E", "LAT_N"),
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument(
        "--only", default=None, help="comma list of figure numbers, e.g. 1,8"
    )
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.out_dir) if args.out_dir else base / "figures" / "workflow"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = build_paths(base, args.version)
    bbox = tuple(args.bbox) if args.bbox else DEFAULT_BBOX

    with rasterio.open(paths["dem"]) as src:
        crs = src.crs
    bounds = bbox_5070(bbox, crs)

    print(f"AOI bbox (WGS84): {bbox}")
    print(f"AOI bounds ({crs}): {tuple(round(b, 1) for b in bounds)}")

    which = [int(x) for x in args.only.split(",")] if args.only else sorted(FIGURES)

    ctx = {
        "paths": paths,
        "bounds": bounds,
        "out_dir": out_dir,
        "dpi": args.dpi,
    }

    # The locator (fig 0) needs only the AOI bounds; load the heavy AOI backdrop
    # and vectors once, but only when a map/data figure is actually requested.
    if any(n >= 1 for n in which):
        dem, dem_extent = read_window(paths["dem"], bounds)
        ctx["dem"] = dem
        ctx["dem_extent"] = dem_extent
        ctx["hs"] = hillshade(dem)
        ctx["streams"] = gpd.read_file(paths["streams"], bbox=bounds)
        ctx["heads"] = gpd.read_file(paths["heads"], bbox=bounds)
    for n in which:
        print(f"Figure {n}: {FIGURES[n].__name__}")
        FIGURES[n](ctx)
    print(f"Done. {len(which)} figure(s) in {out_dir}")


if __name__ == "__main__":
    main()
