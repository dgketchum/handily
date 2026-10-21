"""Shared paths, palettes and loaders for the split-anatomy figure series.

The series lays out how every well is partitioned, stage by stage, from the
leave-fold-out construction of the regional base R through the deployment-side
state-agency split (NDWR in Nevada, OSE in New Mexico) that the statewide
evaluation scores against.

The state is chosen by the ``NWI_STATE`` environment variable (``NV`` or
``NM``), read once at import so every script in the series resolves the same
bundle, render arm, well tables and output folder::

    NWI_STATE=NM uv run python figures/nwi/fig00_population_histograms.py

Style, panel lettering and file writing are inherited from
``utils.figures.fig_common`` so this series matches the existing figures.

Code lives here; images are written to ``FIG_DIR``
(``/data/ssd2/handily/figures/<project>/``: ``nwi`` for NV, ``nmose`` for NM).
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.figures.fig_common import (  # noqa: E402
    MM,
    WIDTH_2COL,
    panel_letter,
    save,
    use_style,
)

__all__ = [
    "MM",
    "WIDTH_2COL",
    "panel_letter",
    "save",
    "use_style",
    "STATE",
    "CFG",
    "FIG_DIR",
    "BUNDLE_DIR",
    "RUN_DIR",
    "RENDER_ARM",
    "WELL_DIR",
    "SOURCES_PATH",
    "TARGETS_PATH",
    "SITES_PATH",
    "EVAL_DIR",
    "HUC8_POLYS",
    "HUC12_POLYS",
    "BANDS",
    "BAND_LABELS",
    "FOLD_COLORS",
    "SOURCE_ORDER",
    "SOURCE_COLORS",
    "SOURCE_LABELS",
    "PSEUDO_SOURCE",
    "WELL_CLASSES",
    "load_bundle",
    "rendered_basins",
    "huc12_folds",
    "state_outline",
    "outside_letter",
    "scale_bar",
    "round_bar_length_m",
    "emptiest_corner",
    "north_arrow",
    "NO_FOLD_COLOR",
    "NO_FOLD_EDGE",
    "SOURCE_HATCHES",
    "STATE_POLYS",
]

# --------------------------------------------------------------------------
# State selection
# --------------------------------------------------------------------------

_CONUS = Path("/data/ssd2/handily/conus/wte_gnn")

#: Everything that differs between the two state packages. Both renders are
#: parity-recipe arms (lattice R at the appended state labels, 2 m ground cap,
#: FAC-REM recipe scalable_v2 pinned by manifest, 1 km drilled-depth floor)
#: trained on the base monitoring bundle plus the admitted state wells as
#: labels, and inferred with those same admitted wells as extra sources.
_STATES = {
    "NV": {
        "name": "Nevada",
        "project": "nwi",
        "agency": "NDWR",
        "agency_long": "Nevada Division of Water Resources",
        "source_name": "nv_ndwr",
        "well_class": "ndwr",
        "bundle": "graph_conus_monitoring_water_v2_ndwr_ufold_parity",
        "arm": "gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold_parity",
        "well_dir": "/data/ssd2/handily/nv/regional/wells",
        "sources": "ndwr_admitted_sources_frozen.parquet",
        "targets": "ndwr_admission_targets.parquet",
        "sites": "ndwr_sites_a4_input.parquet",
        "manifest": None,
        "basin_list": "/data/ssd2/handily/nv/regional/fix_round/nv_huc8_72.txt",
        "eval_dir": "/data/ssd2/handily/nv/regional/statewide_eval_round3_parity_sm4",
        "block_col": "w25r_basin",
        "block_name": "NDWR hydrographic basin",
        "predecessor": "gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold",
    },
    "NM": {
        "name": "New Mexico",
        "project": "nmose",
        "agency": "OSE",
        "agency_long": "New Mexico Office of the State Engineer",
        "source_name": "nm_ose",
        "well_class": "ose",
        "bundle": "graph_conus_monitoring_water_v2_ndwr_ufold_ose_parity",
        "arm": "gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold_oselbl_parity",
        "well_dir": "/data/ssd2/handily/nm/regional/wells",
        "sources": "ose_admitted_sources_frozen.parquet",
        "targets": "ose_admission_targets.parquet",
        "sites": "ose_sites_input.parquet",
        "manifest": "ose_admission_manifest.json",
        "basin_list": "/data/ssd2/handily/nm/regional/ose_lbl/nm_huc8_97.txt",
        "eval_dir": "/data/ssd2/handily/nm/regional/ose_lbl/statewide_eval_parity_sm4",
        "block_col": "pod_basin",
        "block_name": "OSE point-of-diversion basin",
        "predecessor": "gnn_conus_monitoring_water_src_r1e_m20_ord_ndwrlbl_ufold_oselbl",
    },
}

STATE = os.environ.get("NWI_STATE", "NV").upper()
if STATE not in _STATES:
    raise SystemExit(f"NWI_STATE={STATE!r}; choose one of {sorted(_STATES)}")
CFG = _STATES[STATE]

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

# Per-project output directory: NV = Nevada Water Initiative (nwi),
# NM = New Mexico Office of the State Engineer (nmose).
FIG_DIR = Path("/data/ssd2/handily/figures") / CFG["project"]
STATE_POLYS = Path("/data/ssd2/gwx/boundaries/us_states.parquet")
BUNDLE_DIR = _CONUS / CFG["bundle"]
#: The trained model of record for this state (its ``gnn_run.json`` logs the
#: per-fold row counts the split figures verify against).
RUN_DIR = _CONUS / CFG["arm"]
HUC8_POLYS = Path("/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu8_5070.parquet")
HUC12_POLYS = Path("/nas/hydrography/HUC_Boundaries/wbd_national/wbdhu12_5070.parquet")
#: The rendered model directory name; its presence marks a basin as rendered.
RENDER_ARM = CFG["arm"]
WELL_DIR = Path(CFG["well_dir"])
#: The frozen admitted half (training labels AND deployment sources).
SOURCES_PATH = WELL_DIR / CFG["sources"]
#: The complement of the admitted half in the site panel; the eligible rows
#: are the scoring targets.
TARGETS_PATH = WELL_DIR / CFG["targets"]
#: The full site panel the admission funnel ran over.
SITES_PATH = WELL_DIR / CFG["sites"]
#: The statewide raster evaluation of the render (``sites_sampled.parquet``).
EVAL_DIR = Path(CFG["eval_dir"])

# --------------------------------------------------------------------------
# Depth bands (the reporting doctrine bands, metres below ground)
# --------------------------------------------------------------------------

BANDS = (0.0, 2.0, 5.0, 10.0, 30.0, np.inf)
BAND_LABELS = ("0–2", "2–5", "5–10", "10–30", "≥30")

# --------------------------------------------------------------------------
# Palettes
# --------------------------------------------------------------------------

#: Okabe-Ito: the colour-vision-deficiency-safe qualitative set, exactly eight
#: classes for the eight folds (FIGURE_STYLE_GUIDE §7; tab10 was rejected because
#: its adjacent green/red pair is the banned distinguishing pair).
FOLD_COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#666666",
)
#: HUC12s with no label well are drawn in this fill with a hairline edge,
#: so the full HUC12 tiling of the footprint stays visible.
NO_FOLD_COLOR = "#f2f2f2"
NO_FOLD_EDGE = "#c8c8c8"

#: Water-stage pseudo-rows. They ride the same fold machinery but are labels of
#: a different kind (surface-water stage, weight 0.25), not wells, so the well
#: ledger excludes them.
PSEUDO_SOURCE = "GSW_STAGE"

#: ``well_class`` values that are wells (label rows that are not pseudo-rows):
#: the base monitoring networks plus the appended state-agency labels.
WELL_CLASSES = ("monitoring", "ndwr", "ose")

#: Label classes, stacked in descending magnitude so legend order matches the
#: visual order of the segments (FIGURE_STYLE_GUIDE §8). Colour-vision-
#: deficiency-safe hues, hatch-differentiated so the classes survive grayscale.
SOURCE_ORDER = ("nwis", "nm_ose", "nv_ndwr", "other_state", "doe_gems", "ca_dwr")

SOURCE_COLORS = {
    "nwis": "#0072B2",
    "nm_ose": "#D55E00",
    "nv_ndwr": "#56B4E9",
    "other_state": "#CC79A7",
    "doe_gems": "#009E73",
    "ca_dwr": "#E69F00",
}

#: Grayscale redundancy for the label classes (guide §7/§18.11).
SOURCE_HATCHES = {
    "nwis": "",
    "nm_ose": "\\\\\\",
    "nv_ndwr": "---",
    "other_state": "///",
    "doe_gems": "...",
    "ca_dwr": "xxx",
}

SOURCE_LABELS = {
    "nwis": "NWIS",
    "nm_ose": "NM OSE (appended)",
    "nv_ndwr": "NV NDWR (appended)",
    "ca_dwr": "CA DWR",
    "doe_gems": "DOE GEMS",
    "other_state": "other state networks",
}

#: Networks that keep their own class; the long tail of monitoring networks is
#: folded into ``other_state``.
_NAMED = {"nwis", "ca_dwr", "doe_gems", "nm_ose", "nv_ndwr"}


def load_bundle() -> pd.DataFrame:
    """Bundle query nodes with observed DTW, depth band and collapsed source.

    Returns every label row the model was trained on: the unconfined
    monitoring wells, the appended state-agency wells (``well_class`` ``ndwr``
    / ``ose``) and the water-stage pseudo-rows, each carrying its HUC12 fold
    unit (``cv_unit``) and fold id (``cv_fold``).
    """
    qn = pd.read_parquet(
        BUNDLE_DIR / "query_nodes.parquet",
        columns=[
            "source",
            "well_class",
            "huc8",
            "cv_unit",
            "cv_fold",
            "is_water_pseudo",
            "z_surf_well_m",
            "wte_obs_m",
        ],
    )
    qn["dtw_m"] = qn["z_surf_well_m"] - qn["wte_obs_m"]
    if not np.isfinite(qn["dtw_m"]).all():
        raise ValueError("non-finite observed DTW in the bundle label set")
    qn["band"] = pd.cut(qn["dtw_m"], BANDS, right=False, labels=list(BAND_LABELS))
    qn["huc8"] = qn["huc8"].astype(str).str.zfill(8)
    qn["is_well"] = qn["well_class"].isin(WELL_CLASSES)
    if (qn["is_well"] == qn["is_water_pseudo"].astype(bool)).any():
        raise ValueError(
            "a bundle row is both a well class and a pseudo-row, or neither"
        )
    src = qn["source"].where(qn["source"].isin(_NAMED | {PSEUDO_SOURCE}), "other_state")
    qn["source_class"] = src
    return qn


def basin_list() -> list[str]:
    """The state's HUC8 footprint of record, from the basin list file."""
    ids = sorted(
        line.strip() for line in Path(CFG["basin_list"]).read_text().splitlines()
    )
    ids = [i for i in ids if i]
    if not ids:
        raise ValueError(f"empty basin list {CFG['basin_list']}")
    return ids


def rendered_basins() -> list[str]:
    """HUC8 ids with the state's render on disk, checked against the basin list."""
    pat = f"/data/ssd2/handily/huc8/*/gnn/{RENDER_ARM}"
    ids = sorted(
        os.path.basename(os.path.dirname(os.path.dirname(p))) for p in glob.glob(pat)
    )
    if not ids:
        raise FileNotFoundError(f"no rendered basins matched {pat}")
    want = basin_list()
    if ids != want:
        raise ValueError(
            f"rendered {RENDER_ARM} basins ({len(ids)}) differ from the basin list "
            f"{CFG['basin_list']} ({len(want)}): extra {sorted(set(ids) - set(want))}, "
            f"missing {sorted(set(want) - set(ids))}"
        )
    return ids


def outside_letter(ax, letter: str, x: float = 0.0, y: float = 1.04):
    """Panel letter outside the axes, upper-left (guide §4 default position).

    ``fig_common.panel_letter`` pins ``va="top"`` for inside-the-frame placement,
    so outside placement needs its own call.
    """
    return ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="black",
    )


def scale_bar(
    ax,
    length_m: float = 200_000.0,
    n_seg: int = 2,
    x_frac: float = 0.04,
    y_frac: float = 0.03,
) -> None:
    """Segmented black/white scale bar with numerals and unit (guide §11).

    Call after the map is drawn, so the axes limits are final.
    """
    from matplotlib.patches import Rectangle

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + x_frac * (x1 - x0)
    y = y0 + y_frac * (y1 - y0)
    h = 0.010 * (y1 - y0)
    seg = length_m / n_seg
    for i in range(n_seg):
        ax.add_patch(
            Rectangle(
                (x + i * seg, y),
                seg,
                h,
                facecolor="black" if i % 2 == 0 else "white",
                edgecolor="black",
                linewidth=0.5,
                zorder=6,
            )
        )
    # Label in km when the bar is at least 1 km, else in m; ``:g`` keeps
    # fractional segments honest (a 1 km / 2-segment bar reads 0, 0.5, 1 km).
    unit, div = ("km", 1000.0) if length_m >= 1000 else ("m", 1.0)
    for i in range(n_seg + 1):
        val = i * seg / div
        ax.text(
            x + i * seg,
            y + 1.3 * h,
            f"{val:g} {unit}" if i == n_seg else f"{val:g}",
            ha="center" if i < n_seg else "left",
            va="bottom",
            zorder=6,
        )


def round_bar_length_m(ax, frac: float = 0.22) -> float:
    """Largest round bar length (1, 2, 5 x 10^k m) no wider than ``frac`` of the axes."""
    x0, x1 = ax.get_xlim()
    limit = frac * (x1 - x0)
    cands = [m * 10**k for k in range(2, 7) for m in (1, 2, 5)]
    fit = [c for c in cands if c <= limit]
    if not fit:
        raise ValueError(f"no round scale-bar length fits in {limit:.0f} m")
    return float(fit[-1])


def emptiest_corner(ax, x, y, corners, w: float = 0.44, h: float = 0.30) -> str:
    """The ``loc=`` corner, among ``corners``, holding the fewest data points.

    The legend goes in whichever corner the data vacates (guide §4/§8), decided
    from the data rather than fixed by hand. Ties resolve to the first corner
    listed.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    fx = (np.asarray(x, float) - x0) / (x1 - x0)
    fy = (np.asarray(y, float) - y0) / (y1 - y0)
    side = {"left": fx < w, "right": fx > 1 - w}
    band = {"upper": fy > 1 - h, "lower": fy < h, "center": (fy >= h) & (fy <= 1 - h)}
    n = {c: int((band[c.split()[0]] & side[c.split()[1]]).sum()) for c in corners}
    return min(corners, key=lambda c: n[c])


def north_arrow(ax, x_frac: float = 0.90, y_frac: float = 0.06) -> None:
    """Small plain north arrow (guide §11: regional maps carry one)."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + x_frac * (x1 - x0)
    y = y0 + y_frac * (y1 - y0)
    dy = 0.055 * (y1 - y0)
    ax.annotate(
        "",
        xy=(x, y + dy),
        xytext=(x, y),
        arrowprops={"arrowstyle": "-|>", "color": "black", "linewidth": 0.6},
        zorder=6,
    )
    ax.text(x, y + 1.15 * dy, "N", ha="center", va="bottom", zorder=6)


def state_outline(state: str = STATE):
    """One state's boundary, reprojected to the bundle's EPSG:5070 frame."""
    import geopandas as gpd

    g = gpd.read_parquet(STATE_POLYS)
    sel = g[g["state"].str.upper() == state.upper()]
    if sel.empty:
        raise ValueError(f"state {state!r} not in {STATE_POLYS}")
    return sel.to_crs(5070)


def huc12_folds(basins: list[str]):
    """HUC12 polygons inside ``basins``, joined to their bundle fold assignment.

    Folds are assigned to whole HUC12s, but only HUC12s that contain at least
    one bundle label row appear in the bundle, so a HUC12 with no well has no
    fold (``cv_fold`` is NaN) — drawn in :data:`NO_FOLD_COLOR`.

    The unit -> fold map is read from the WELL rows. A HUC12 that holds only
    water-stage pseudo-rows in the base bundle and gained its first wells when
    the state labels were appended carries the pseudo-rows' original fold and
    the appended wells' fold (the nearest base well's, assigned per unit), so
    the pseudo-rows are left out of the map and the well map must be unique.
    """
    import geopandas as gpd

    h12 = gpd.read_parquet(HUC12_POLYS, columns=["huc12", "geometry"])
    h12["huc8"] = h12["huc12"].astype(str).str.zfill(12).str[:8]
    h12 = h12[h12["huc8"].isin(basins)].copy()

    qn = load_bundle()
    wells = qn[qn["is_well"]]
    fold_of = (
        wells.groupby("cv_unit")["cv_fold"].agg(["first", "nunique"]).reset_index()
    )
    if (fold_of["nunique"] > 1).any():
        raise ValueError("a HUC12 unit carries more than one fold id among its wells")
    n_wells = wells.groupby("cv_unit").size().rename("n_wells")
    h12["huc12"] = h12["huc12"].astype(str).str.zfill(12)
    h12 = h12.merge(
        fold_of.rename(columns={"cv_unit": "huc12", "first": "cv_fold"})[
            ["huc12", "cv_fold"]
        ],
        on="huc12",
        how="left",
    ).merge(n_wells.rename_axis("huc12").reset_index(), on="huc12", how="left")
    h12["n_wells"] = h12["n_wells"].fillna(0).astype(int)
    return h12
