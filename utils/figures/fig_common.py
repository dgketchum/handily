"""Shared style, colors, geometry and drawing primitives for the figure series.

Run as::

    uv run python -m utils.figures.fig_common

which prints the public constants (there is nothing else to execute here).

Conventions fixed for the whole series
--------------------------------------
* **Array orientation.** Every 2-D world array is indexed ``arr[j, i]`` where ``j``
  indexes *y* ascending (row 0 = south edge) and ``i`` indexes *x* ascending
  (column 0 = west edge). Draw maps with ``origin="lower"`` and
  ``extent=EXTENT`` (metres). Never flip an array; flip the axis if you must.
* **Units.** All lengths in metres, all elevations/depths in metres. Distances
  along a section are metres, usually plotted as km with the axis title saying so.
* **Residual sign.** ``residual = predicted DTW - observed DTW``; positive means
  the prediction is too deep.
* **Well ledger glyph.** Filled marker / solid black stick = the label was visible
  to the surface being drawn. Hollow marker (white face, black edge) / white stick
  with a black outline = the label was hidden. The observed water level is a short
  horizontal tick on the stick.
* **Panel letters** are lowercase ``(a)`` set *inside* the axes at the top-left.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

STYLE_PATH = Path.home() / "code" / "style" / "journal_figures.mplstyle"
FIG_ROOT = Path("/data/ssd2/handily/figures")
TOY_DIR = FIG_ROOT / "toy"
TOY_WORLD_NPZ = TOY_DIR / "toy_world.npz"
TOY_WELLS_PARQUET = TOY_DIR / "toy_wells.parquet"

# --------------------------------------------------------------------------
# Sizing
# --------------------------------------------------------------------------

MM = 1 / 25.4  # multiply millimetres by this to get inches
WIDTH_1COL = 90 * MM
WIDTH_15COL = 140 * MM
WIDTH_2COL = 190 * MM

# --------------------------------------------------------------------------
# Color assignments (Okabe-Ito; guide section 7)
# --------------------------------------------------------------------------

COLORS = {
    "truth": "#000000",  # true water table (toy only)
    "pred": "#E69F00",  # handily prediction
    "r": "#0072B2",  # regional base R
    "fac": "#009E73",  # FAC-REM
    "deep": "#D55E00",  # deep prior
    "dd": "#CC79A7",  # drilled-depth prior
    "mirror": "#808080",  # 3 m mirror (grey dashed, context)
    "ma": "#808080",  # Ma-like direct-DTW interpolation (grey solid, context)
    "land": "#000000",  # land surface hairline in 2-D sections
}

#: Uniform pale sand used for the translucent land surface in the 3-D register.
LAND_3D = "#e8dcc0"
LAND_3D_OPACITY = 0.35

#: Line styles that go with COLORS for the two grey context series.
LINESTYLES = {"mirror": (0, (2.5, 1.5)), "ma": "-"}

CMAPS = {
    "dtw": "Blues_r",  # dark = shallow; colorbar reads "Depth to water (m)"
    "wte": "viridis",  # water-table elevation
    "resid": "RdBu",  # white at 0, symmetric limits
    "gate": "Greys",  # gate weights, 0-1, one shared sequential ramp
}

# --------------------------------------------------------------------------
# Geometry shared by every figure
# --------------------------------------------------------------------------

VE = 4.0  #: vertical exaggeration, sections and 3-D
#: The exaggeration statement that must appear on every exaggerated section and
#: on the 3-D scale bar. Use :func:`annotate_ve`.
VE_LABEL = f"{VE:g}× vertical"

CELL = 100.0  #: toy world cell size (m)
NX, NY = 240, 160  #: toy world grid shape (columns, rows)
EXTENT = (0.0, NX * CELL, 0.0, NY * CELL)  #: (xmin, xmax, ymin, ymax) in metres

#: A-A' runs west to east along the trunk valley; B-B' crosses it through the
#: densest well cluster. (x0, y0, x1, y1) in metres. Fixed for the whole series.
SECTION_A = (500.0, 6100.0, 23500.0, 6100.0)
SECTION_B = (8000.0, 500.0, 8000.0, 15500.0)

SECTION_LABELS = {"A": ("A", "A′"), "B": ("B", "B′")}


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------


def use_style() -> None:
    """Load the shared journal mplstyle and enforce TrueType (type 42) fonts."""
    if not STYLE_PATH.exists():
        raise FileNotFoundError(f"shared mplstyle not found: {STYLE_PATH}")
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.style.use(str(STYLE_PATH))
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["savefig.dpi"] = 300


# --------------------------------------------------------------------------
# Sampling along a section line
# --------------------------------------------------------------------------


def _bilinear(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear sample ``arr[j, i]`` at world coordinates ``xs``/``ys`` (metres).

    Cell centres sit at ``(i + 0.5) * CELL``. Queries outside the domain are
    clamped to the edge cell centres (the section lines are inset from the
    boundary, so this never fires for the series' own geometry).
    """
    ny, nx = arr.shape
    fi = np.clip(xs / CELL - 0.5, 0.0, nx - 1.0)
    fj = np.clip(ys / CELL - 0.5, 0.0, ny - 1.0)
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    wx = fi - i0
    wy = fj - j0
    a = arr.astype(float)
    return (
        a[j0, i0] * (1 - wx) * (1 - wy)
        + a[j0, i1] * wx * (1 - wy)
        + a[j1, i0] * (1 - wx) * wy
        + a[j1, i1] * wx * wy
    )


def profile_along(line, grid_arrays: dict, n: int = 400) -> dict:
    """Sample any set of 2-D world arrays along a straight section line.

    Parameters
    ----------
    line : tuple
        ``(x0, y0, x1, y1)`` in metres, e.g. :data:`SECTION_A`.
    grid_arrays : dict
        ``{name: 2-D array}``; every array must be on the world grid.
    n : int
        Number of samples along the line.

    Returns
    -------
    dict
        ``{"d": distance along the line (m), "x": x (m), "y": y (m), **values}``
        where each ``name`` maps to a length-``n`` float array. Boolean inputs
        are sampled bilinearly as floats (threshold at 0.5 to get a mask back).
    """
    x0, y0, x1, y1 = (float(v) for v in line)
    t = np.linspace(0.0, 1.0, int(n))
    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)
    d = t * float(np.hypot(x1 - x0, y1 - y0))
    out = {"d": d, "x": xs, "y": ys}
    for name, arr in grid_arrays.items():
        arr = np.asarray(arr)
        if arr.shape != (NY, NX):
            raise ValueError(f"{name}: expected shape {(NY, NX)}, got {arr.shape}")
        out[name] = _bilinear(arr, xs, ys)
    return out


def wells_near_line(wells_df, line, buffer_m: float):
    """Project wells onto a section line and keep those within ``buffer_m``.

    Returns a copy of the rows within the buffer *and* within the segment's span,
    with two added columns: ``d_along`` (m from the line's first endpoint) and
    ``d_offset`` (signed perpendicular offset in m, positive to the left of the
    direction of travel). Sorted by ``d_along``.
    """
    x0, y0, x1, y1 = (float(v) for v in line)
    dx, dy = x1 - x0, y1 - y0
    length = float(np.hypot(dx, dy))
    ux, uy = dx / length, dy / length
    px = wells_df["x"].to_numpy(float) - x0
    py = wells_df["y"].to_numpy(float) - y0
    along = px * ux + py * uy
    offset = -px * uy + py * ux
    keep = (np.abs(offset) <= buffer_m) & (along >= 0.0) & (along <= length)
    out = wells_df.loc[keep].copy()
    out["d_along"] = along[keep]
    out["d_offset"] = offset[keep]
    return out.sort_values("d_along")


# --------------------------------------------------------------------------
# The visibility ledger
# --------------------------------------------------------------------------


def draw_wells_section(
    ax,
    wells,
    visible_mask,
    x_col: str = "d_along",
    top_col: str = "dem",
    level_col: str = "wte_obs",
    depth_col: str = "drilled_depth",
    x_scale: float = 1e-3,
    tick_half_width: float = 0.18,
    lw: float = 0.6,
):
    """Draw well sticks with the ledger glyph on a section axes.

    A stick runs from the land surface (``top_col``) down to the well bottom
    (``top_col - depth_col``), and the observed water level (``level_col``) is a
    short horizontal tick across it.

    ``visible_mask`` is a boolean array aligned with ``wells``: True = the label
    was visible to the surface drawn on this axes (solid black stick, filled
    marker), False = hidden (white stick with a black outline, hollow marker).

    ``x_scale`` converts the ``x_col`` metres to the axes' x units (default km);
    ``tick_half_width`` is in those same axes units.
    """
    vis = np.asarray(visible_mask, dtype=bool)
    if vis.shape[0] != len(wells):
        raise ValueError("visible_mask must align with wells")
    xs = wells[x_col].to_numpy(float) * x_scale
    top = wells[top_col].to_numpy(float)
    lev = wells[level_col].to_numpy(float)
    bot = top - wells[depth_col].to_numpy(float)
    for xi, ti, li, bi, vi in zip(xs, top, lev, bot, vis):
        if vi:
            ax.plot(
                [xi, xi],
                [ti, bi],
                color="black",
                lw=lw,
                solid_capstyle="butt",
                zorder=6,
            )
        else:
            ax.plot(
                [xi, xi],
                [ti, bi],
                color="black",
                lw=lw + 0.7,
                solid_capstyle="butt",
                zorder=5,
            )
            ax.plot(
                [xi, xi],
                [ti, bi],
                color="white",
                lw=lw,
                solid_capstyle="butt",
                zorder=6,
            )
        ax.plot(
            [xi - tick_half_width, xi + tick_half_width],
            [li, li],
            color="black",
            lw=lw + 0.2,
            solid_capstyle="butt",
            zorder=7,
        )
        ax.plot(
            [xi],
            [li],
            marker="o",
            ms=2.2,
            mfc="black" if vi else "white",
            mec="black",
            mew=0.5,
            ls="none",
            zorder=8,
        )


def draw_wells_map(
    ax, wells, visible_mask, ms: float = 2.2, mew: float = 0.5, zorder: int = 6
):
    """Plot wells on a map axes with the ledger glyph (filled = visible)."""
    vis = np.asarray(visible_mask, dtype=bool)
    if vis.shape[0] != len(wells):
        raise ValueError("visible_mask must align with wells")
    x = wells["x"].to_numpy(float)
    y = wells["y"].to_numpy(float)
    ax.plot(
        x[~vis],
        y[~vis],
        marker="o",
        ls="none",
        ms=ms,
        mfc="white",
        mec="black",
        mew=mew,
        zorder=zorder,
    )
    ax.plot(
        x[vis],
        y[vis],
        marker="o",
        ls="none",
        ms=ms,
        mfc="black",
        mec="black",
        mew=mew,
        zorder=zorder + 1,
    )


def well_ledger_handles(visible_label="label visible", hidden_label="label hidden"):
    """Legend handles for the ledger glyph, for a framed legend."""
    return [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=2.6,
            mfc="black",
            mec="black",
            mew=0.5,
            label=visible_label,
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=2.6,
            mfc="white",
            mec="black",
            mew=0.5,
            label=hidden_label,
        ),
    ]


def draw_section_lines(
    ax, color="black", lw: float = 0.7, fontsize: float = 6.0, pad: float = 350.0
):
    """Draw A-A' and B-B' on a map axes in world coordinates, with end labels."""
    for key, line in (("A", SECTION_A), ("B", SECTION_B)):
        x0, y0, x1, y1 = line
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=(0, (4, 2)), zorder=9)
        lab0, lab1 = SECTION_LABELS[key]
        dx, dy = x1 - x0, y1 - y0
        n = np.hypot(dx, dy)
        ux, uy = dx / n, dy / n
        ax.annotate(
            lab0,
            (x0 - ux * pad, y0 - uy * pad),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            zorder=10,
        )
        ax.annotate(
            lab1,
            (x1 + ux * pad, y1 + uy * pad),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            zorder=10,
        )


# --------------------------------------------------------------------------
# Axes furniture
# --------------------------------------------------------------------------


def add_ticks(ax, x_step=None, y_step=None):
    """Set round major tick sets at the given steps, in the axes' own units."""
    if x_step is not None:
        lo, hi = ax.get_xlim()
        ax.set_xticks(np.arange(np.ceil(lo / x_step) * x_step, hi + 1e-9, x_step))
        ax.set_xlim(lo, hi)
    if y_step is not None:
        lo, hi = ax.get_ylim()
        ax.set_yticks(np.arange(np.ceil(lo / y_step) * y_step, hi + 1e-9, y_step))
        ax.set_ylim(lo, hi)


def panel_letter(
    ax, letter: str, x: float = 0.025, y: float = 0.975, fontsize: float = 8.0, **kw
):
    """Lowercase ``(a)`` inside the axes at the top-left (guide section 4)."""
    return ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color="black",
        **kw,
    )


def annotate_ve(ax, x: float = 0.985, y: float = 0.03, fontsize: float = 6.0, **kw):
    """Print the exaggeration statement (:data:`VE_LABEL`) in a section's corner.

    Every section drawn with :data:`VE` applied to its aspect must carry it.
    """
    return ax.text(
        x,
        y,
        VE_LABEL,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize,
        color="black",
        **kw,
    )


def map_axes(ax, xlabel="Easting (km)", ylabel="Northing (km)", x_step=5.0, y_step=5.0):
    """Set a world-extent map axes to km ticks with equal aspect and four spines."""
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, EXTENT[1] + 1, x_step * 1000))
    ax.set_yticks(np.arange(0, EXTENT[3] + 1, y_step * 1000))
    ax.set_xticklabels([f"{v / 1000:g}" for v in ax.get_xticks()])
    ax.set_yticklabels([f"{v / 1000:g}" for v in ax.get_yticks()])
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def resid_norm(vmax: float) -> TwoSlopeNorm:
    """Symmetric diverging norm on [-vmax, vmax] with white exactly at zero."""
    vmax = float(abs(vmax))
    if vmax <= 0:
        raise ValueError("vmax must be positive")
    return TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def save(fig, stem: str, out_dir) -> dict:
    """Write ``<stem>.pdf`` and ``<stem>.png`` (300 dpi) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext, dpi in (("pdf", None), ("png", 300)):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi) if dpi else fig.savefig(p)
        paths[ext] = p
    return paths


def _main() -> None:
    print("MM              ", MM)
    print("VE              ", VE)
    print("EXTENT (m)      ", EXTENT)
    print("SECTION_A       ", SECTION_A)
    print("SECTION_B       ", SECTION_B)
    print("COLORS          ", COLORS)
    print("CMAPS           ", CMAPS)
    print("style           ", STYLE_PATH, STYLE_PATH.exists())
    print("toy cache       ", TOY_WORLD_NPZ)


if __name__ == "__main__":
    _main()
