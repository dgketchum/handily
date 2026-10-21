"""Figure 4 — what role each kind of well plays at training and at deployment.

(a) Inside one cross-validation fold: which rows carry a label in the loss,
    which are visible as observation sources in each of the three forwards the
    trainer runs, and which are scored.
(b) At deployment (the state render): which wells are a source for every pixel
    and for every one of the eight fold models, which of them were also
    training labels, and which are scored. The accuracy claim rests on an
    external held-out set that was neither.

Read side by side the two panels show the training-time fold guard (a test well
is never a label and never a source) and the deployment-time absence of one (the
fold-f model reads fold-f wells as sources, by design). The parity arms add a
third fact: the admitted state-agency wells were BOTH training labels and
deployment sources, while the held-out half was neither.

Every cell is read out of the code and the run of record, never asserted:
``utils/train_conus_gnn.py`` (the masked-label source protocol and the three
forwards), ``utils/build_source_edges.py`` (k, self/same-site exclusion),
``utils/infer_conus_gnn.py`` (``deployment_source_ctx``, the fold ensemble, the
by-name source exclusion), the run json of the state's parity arm and the
basin ``infer_run.json`` files of its render. The fold-0 row counts come from
``fig03_fold_split.split_all_folds``, which is itself asserted against the
per-fold counts logged by that run.

Written to the house figure rules in ``~/code/style/FIGURE_STYLE_GUIDE.md``:
no gridlines, no interior text beyond the row/column tick labels, a single
framed legend placed with ``loc=`` only, type sizes left to the shared mplstyle,
and a monochrome three-level glyph (filled / open / half-filled square) so the
encoding survives both grayscale and colour-vision deficiency.

Run::

    NWI_STATE=NM uv run python figures/nwi/fig04_role_matrix.py

Writes ``fig04_role_matrix.{pdf,png}`` plus a ``*_caption.txt`` sidecar to
``/data/ssd2/handily/figures/<project>/`` (``nwi`` for NV, ``nmose`` for NM).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from fig03_fold_split import load_rows, split_all_folds, verify_against_run
from nwi_common import (
    BUNDLE_DIR,
    CFG,
    FIG_DIR,
    MM,
    RENDER_ARM,
    RUN_DIR,
    rendered_basins,
    SOURCES_PATH,
    TARGETS_PATH,
    WELL_CLASSES,
    save,
    use_style,
)
from utils.figures.fig_common import WIDTH_15COL  # noqa: E402 (path set by nwi_common)

#: The fold whose row counts label panel (a).
FOLD = 0

# --------------------------------------------------------------------------
# The matrix. Y = yes, N = no, C = yes but conditioned (caption spells out the
# condition for every C cell). Each cell is checked against code in main().
# --------------------------------------------------------------------------

A_COLS = (
    "label in\nthe loss",
    "source to\ntrain queries",
    "source in the\nearly-stop pass",
    "source in the\nout-of-fold pass",
    "scored",
)
B_COLS = ("source for\nevery pixel", "training\nlabel", "scored")

#: (row key, cell codes). Row labels are built in ``row_labels`` so the counts
#: come from the reproduced split rather than from a literal here.
A_ROWS = (
    ("train", ("Y", "C", "C", "Y", "N")),
    ("val", ("N", "N", "N", "Y", "N")),
    ("test", ("N", "N", "N", "N", "Y")),
    ("pseudo_train", ("C", "N", "N", "N", "N")),
    ("pseudo_test", ("N", "N", "N", "N", "C")),
)
#: Panel (b) rows. Labels appended for another state's package are outside
#: the render footprint and are not drawn; ``deployment_facts`` still checks
#: that every appended source name is excluded from the render's source pool.
B_ROWS = (
    ("monitoring", ("Y", "Y", "C")),
    ("state_src", ("Y", "C", "N")),
    ("state_tgt", ("N", "N", "Y")),
    ("pixel", ("N", "N", "N")),
)

#: Thin rules between row groups (index of the row ABOVE the rule).
A_ROW_RULES = (3,)
#: Thin rules between column groups (index of the column to the LEFT).
A_COL_RULES = (1, 4)
B_COL_RULES = ()

#: Geometry, in millimetres at final printed size.
GUTTER_MM = 52.0  # row-label column
CELL_W_MM = 18.0
CELL_H_MM = 6.0
HEADER_MM = 6.5  # two-line column headers above the matrix
TITLE_MM = 4.0  # the fused panel letter + title line
GAP_MM = 7.0  # between the two panels
MARGIN_MM = 3.0

GLYPH_PT = 10.0
RULE_LW = 0.6
SEP_LW = 0.4


def glyph_style(code: str) -> dict:
    """Marker kwargs for one cell code (monochrome, grayscale/CVD-proof)."""
    base = {
        "marker": "s",
        "markersize": GLYPH_PT,
        "markeredgecolor": "black",
        "markeredgewidth": 0.6,
        "linestyle": "none",
    }
    if code == "Y":
        return base | {"markerfacecolor": "black"}
    if code == "N":
        return base | {"markerfacecolor": "white"}
    if code == "C":
        # half-filled: a qualified yes, distinct from both under grayscale.
        return base | {
            "fillstyle": "left",
            "markerfacecolor": "black",
            "markerfacecoloralt": "white",
        }
    raise ValueError(f"unknown cell code {code!r}")


# --------------------------------------------------------------------------
# Facts, read from code outputs
# --------------------------------------------------------------------------


def fold_counts(fold: int) -> dict:
    """Fold row counts from fig03's reproduction, verified against the run log.

    The quoted role counts are the fold's rows inside the state render
    footprint; the national totals are kept for the pool-size assertions.
    """
    qn = load_rows()
    split = split_all_folds(qn)
    verify_against_run(split)
    if fold not in split:
        raise SystemExit(f"fold {fold} not in {sorted(split)}")
    basins = rendered_basins()
    foot = qn["huc8"].isin(basins).to_numpy(bool)
    water = qn["is_water_pseudo"].to_numpy(bool)
    s = split[fold]
    out = {
        "train": int((s["train"] & ~water & foot).sum()),
        "val": int((s["val"] & ~water & foot).sum()),
        "test": int((s["test"] & ~water & foot).sum()),
        "pseudo_train": int((s["train"] & water & foot).sum()),
        "pseudo_val": int((s["val"] & water).sum()),
        "pseudo_test": int((s["test"] & water & foot).sum()),
        "wells_foot": int((~water & foot).sum()),
        "pseudo_foot": int((water & foot).sum()),
        "wells": int((~water).sum()),
        "n_folds": len(split),
        "n_basins": len(basins),
        "n_monitoring": int((qn["well_class"] == "monitoring").sum()),
        "by_class": {
            c: int(((qn["well_class"] == c).to_numpy() & foot).sum())
            for c in WELL_CLASSES
            if ((qn["well_class"] == c).to_numpy() & foot).any()
        },
    }
    if out["by_class"].get(CFG["well_class"], 0) == 0:
        raise ValueError(f"no appended {CFG['agency']} labels inside the footprint")
    if out["pseudo_val"]:
        raise ValueError(
            "water-stage pseudo-rows landed in val; the trainer returns them to "
            "train (`va &= real`), so the figure's row set would be wrong"
        )
    return out


def run_facts() -> dict:
    """The training-side numbers this figure quotes, from the run of record."""
    run = json.loads((RUN_DIR / "gnn_run.json").read_text())
    hp = run["hyperparams"]
    if run["source_edges"]["enabled"] is not True:
        raise ValueError(f"{RUN_DIR.name} is not a source-edge arm")
    if run["source_obs"]["enabled"] is not False:
        raise ValueError(
            "the run has --source-obs on; drawn sources would then be loss-masked "
            "and the 'label in the loss' cell for train wells is no longer an "
            "unconditional yes"
        )
    se_stats = json.loads((BUNDLE_DIR / "source_edges.json").read_text())
    return {
        "frac_min": float(hp["source_frac_min"]),
        "frac_max": float(hp["source_frac_max"]),
        "val_mix": bool(hp["source_val_mix"]),
        "val_frac": float(hp["val_frac"]),
        "seed": int(hp["seed"]),
        "water_weight": float(hp["water_label_weight"]),
        "n_eligible": int(run["source_obs"]["n_eligible"]),
        "k": int(se_stats["k"]),
        "relief_vw": float(se_stats["relief_vw"]),
        "same_site_dropped": int(se_stats["n_same_site_dropped"]),
        "n_edges": int(se_stats["n_edges"]),
    }


def deployment_facts() -> dict:
    """The state render's source set and the agency admission split."""
    man = json.loads((BUNDLE_DIR / "graph_manifest.json").read_text())
    app = man["appended_wells"]
    if app["source_name"] != CFG["source_name"]:
        raise ValueError(
            f"bundle appended {app['source_name']!r}, not {CFG['source_name']!r}"
        )
    contract = man["parity_rewrite"]["render_contract"]

    pat = f"/data/ssd2/handily/huc8/*/gnn/{RENDER_ARM}/infer_run.json"
    runs = sorted(glob.glob(pat))
    if not runs:
        raise FileNotFoundError(f"no render manifests matched {pat}")
    infer = [json.loads(Path(p).read_text()) for p in runs]
    se = [d["source_edges"] for d in infer]
    for key in ("k", "exclude_m", "n_sources", "n_sources_bundle", "n_sources_extra"):
        vals = {d[key] for d in se}
        if len(vals) != 1:
            raise ValueError(f"basins disagree on source_edges[{key!r}]: {vals}")
    paths = {d["extra_sources_path"] for d in infer}
    if paths != {str(SOURCES_PATH)}:
        raise ValueError(f"render used extra sources {paths}, not {SOURCES_PATH}")
    r_src = {d["r_source"] for d in infer}
    if r_src != {"crossfit"}:
        raise ValueError(f"render R source is {r_src}, not the archived cross-fit")
    wells_mode = {d["wells"] for d in infer}
    if wells_mode != {"monitoring_only"}:
        raise ValueError(f"render well pool is {wells_mode}, not monitoring_only")
    excl = {tuple(d["exclude_source_name"]) for d in infer}
    if excl != {tuple(contract["exclude_source_name"])}:
        raise ValueError(
            f"render excluded {excl} but the bundle contract says "
            f"{contract['exclude_source_name']}"
        )
    exclude_names = list(contract["exclude_source_name"])
    if CFG["source_name"] not in exclude_names:
        raise ValueError(f"{CFG['source_name']} is not excluded from the bundle pool")
    caps = {d["r_ground_cap"]["cap_m"] for d in infer}
    if caps != {float(contract["r_ground_cap_m"])}:
        raise ValueError(f"render ground cap {caps} differs from the contract")

    src = pd.read_parquet(SOURCES_PATH, columns=["sx", "sy"])
    tgt = pd.read_parquet(TARGETS_PATH, columns=["sx", "sy", "is_admission_eligible"])
    elig = tgt[tgt["is_admission_eligible"].to_numpy(bool)]
    overlap = len(
        set(map(tuple, src.to_numpy())) & set(map(tuple, elig[["sx", "sy"]].to_numpy()))
    )
    if overlap:
        raise ValueError(f"{overlap} scored targets are also admitted sources")
    if len(src) != se[0]["n_sources_extra"]:
        raise ValueError(
            f"{len(src)} admitted sources on disk but the render logged "
            f"{se[0]['n_sources_extra']}"
        )
    if len(src) != app["funnel"]["wells_in"]:
        raise ValueError("the appended-label funnel did not start from the frozen file")

    bundle_src = pd.read_parquet(
        BUNDLE_DIR / "query_nodes.parquet", columns=["source", "well_class"]
    )
    n_state = int((bundle_src["source"] == CFG["source_name"]).sum())
    if n_state != int(app["n_appended"]):
        raise ValueError(
            f"{n_state:,} {CFG['source_name']} rows in the bundle, manifest says "
            f"{app['n_appended']:,}"
        )
    is_well = bundle_src["well_class"].isin(WELL_CLASSES).to_numpy()
    appended = set(
        bundle_src.loc[is_well & (bundle_src["well_class"] != "monitoring"), "source"]
    )
    if not appended <= set(exclude_names):
        raise ValueError(
            f"appended label sources {sorted(appended - set(exclude_names))} are "
            "not excluded from the render's bundle source pool"
        )
    mon = bundle_src[bundle_src["well_class"].to_numpy() == "monitoring"]
    if mon["source"].isin(exclude_names).any():
        raise ValueError("a monitoring-class row carries an excluded source name")
    return {
        "n_basins": len(infer),
        "k": int(se[0]["k"]),
        "exclude_m": float(se[0]["exclude_m"]),
        "n_sources": int(se[0]["n_sources"]),
        "n_bundle": int(se[0]["n_sources_bundle"]),
        "n_extra": int(se[0]["n_sources_extra"]),
        "n_state_labels": n_state,
        "append_funnel": app["funnel"],
        "exclude_names": exclude_names,
        "cap_m": float(contract["r_ground_cap_m"]),
        "n_targets": int(len(elig)),
        "n_targets_all": int(len(tgt)),
        # networks that supplied monitoring WELLS (the pseudo-row source is not one)
        "well_sources": sorted(str(s) for s in mon["source"].unique()),
    }


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def row_labels(counts: dict, dep: dict) -> tuple[dict, dict]:
    a = {
        "train": f"train wells ({counts['train']:,})",
        "val": f"val wells ({counts['val']:,})",
        "test": f"test wells ({counts['test']:,})",
        "pseudo_train": f"water-stage pseudo-rows, train ({counts['pseudo_train']:,})",
        "pseudo_test": f"water-stage pseudo-rows, test ({counts['pseudo_test']:,})",
    }
    b = {
        "monitoring": "bundle monitoring wells, every fold "
        f"({counts['by_class']['monitoring']:,} in the footprint)",
        "state_src": f"admitted {CFG['agency']} sources ({dep['n_extra']:,})",
        "state_tgt": f"held-out {CFG['agency']} targets ({dep['n_targets']:,})",
        "pixel": "map pixel (100 m cell)",
    }
    return a, b


def draw_matrix(ax, rows, labels, cols, n_slots, row_rules, col_rules) -> None:
    """One role matrix: glyph per cell, tick labels, group rules, no spines."""
    nrow = len(rows)
    for i, (key, codes) in enumerate(rows):
        y = nrow - i - 0.5
        for j, code in enumerate(codes):
            ax.plot([j + 0.5], [y], **glyph_style(code))
        if key not in labels:
            raise KeyError(f"no row label for {key!r}")

    ax.set_xlim(0.0, n_slots)
    ax.set_ylim(0.0, nrow)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([j + 0.5 for j in range(len(cols))])
    ax.set_xticklabels(cols)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([nrow - i - 0.5 for i in range(nrow)])
    ax.set_yticklabels([labels[k] for k, _ in rows])
    ax.tick_params(axis="both", length=0.0, pad=3.0)
    ax.tick_params(axis="x", labeltop=True, labelbottom=False)

    # Table rules: one above the matrix, one below, thin separators between groups.
    span = len(cols)
    for y, lw in ((0.0, RULE_LW), (float(nrow), RULE_LW)):
        ax.plot([0.0, span], [y, y], color="black", lw=lw, solid_capstyle="butt")
    for i in row_rules:
        ax.plot(
            [0.0, span],
            [nrow - i, nrow - i],
            color="black",
            lw=SEP_LW,
            solid_capstyle="butt",
        )
    for j in col_rules:
        ax.plot([j, j], [0.0, nrow], color="black", lw=SEP_LW, solid_capstyle="butt")


def panel_label(ax, letter: str, title: str, x: float, y: float) -> None:
    """Fused panel letter + short identifier, one text object (guide section 4)."""
    ax.text(
        x,
        y,
        f"({letter}) {title}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        color="black",
    )


def build_figure(a_labels, b_labels, n_slots):
    w_mm = WIDTH_15COL / MM
    a_h = len(A_ROWS) * CELL_H_MM
    b_h = len(B_ROWS) * CELL_H_MM
    h_mm = MARGIN_MM + TITLE_MM + HEADER_MM + a_h + GAP_MM + TITLE_MM + HEADER_MM
    h_mm += b_h + MARGIN_MM
    fig = plt.figure(figsize=(WIDTH_15COL, h_mm * MM))

    mat_w = n_slots * CELL_W_MM
    left = GUTTER_MM / w_mm
    width = mat_w / w_mm
    a_bottom = (MARGIN_MM + b_h + HEADER_MM + TITLE_MM + GAP_MM) / h_mm
    b_bottom = MARGIN_MM / h_mm
    ax_a = fig.add_axes([left, a_bottom, width, a_h / h_mm])
    ax_b = fig.add_axes([left, b_bottom, width, b_h / h_mm])

    # rules: between the scored-target row and the pixel row, and above the
    # held-out row (sources above, non-sources below)
    n_src_rows = sum(1 for k, _ in B_ROWS if k in ("monitoring", "state_src"))
    b_row_rules = (n_src_rows, len(B_ROWS) - 1)
    draw_matrix(ax_a, A_ROWS, a_labels, A_COLS, n_slots, A_ROW_RULES, A_COL_RULES)
    draw_matrix(ax_b, B_ROWS, b_labels, B_COLS, n_slots, b_row_rules, B_COL_RULES)

    # The panel label sits clear of the two-line column headers, left-aligned to
    # the panel bounding box (the row-label gutter), so it collides with nothing.
    lx = -GUTTER_MM / mat_w
    panel_label(ax_a, "a", f"Training, fold {FOLD}", lx, 1.0 + HEADER_MM / a_h)
    panel_label(
        ax_b, "b", f"Deployment, {CFG['name']} render", lx, 1.0 + HEADER_MM / b_h
    )

    # Legend samples are drawn smaller than the cell glyphs (guide section 8:
    # small legend markers). A square sample is taller than a 6 pt text line, so
    # the entry spacing and frame pad are opened up to keep the samples clear of
    # each other and of the frame -- collisions are forbidden (guide section 9).
    handles = [
        Line2D([], [], label=lab, **(glyph_style(code) | {"markersize": 6.0}))
        for code, lab in (("Y", "yes"), ("N", "no"), ("C", "yes, conditionally"))
    ]
    # Panel (b) uses three of the five column slots, so its own right-hand
    # interior is the empty region nearest the data (guide section 8 priority 1).
    ax_b.legend(
        handles=handles,
        loc="center right",
        handlelength=1.4,
        labelspacing=0.6,
        borderpad=0.5,
    )
    return fig


# --------------------------------------------------------------------------
# Caption
# --------------------------------------------------------------------------


def caption(counts: dict, run: dict, dep: dict) -> str:
    fun = dep["append_funnel"]
    cls = "; ".join(
        f"{n:,} {w}"
        for c, n in counts["by_class"].items()
        for w in (
            {
                "monitoring": "unconfined monitoring wells",
                "ndwr": "appended NV NDWR wells",
                "ose": "appended NM OSE wells",
            }[c],
        )
    )
    return (
        "Figure 4. For each kind of well, what role does it play at training "
        "time and at deployment: label, source, or scored target?\n\n"
        f"(a) One cross-validation fold (fold {FOLD}) of the run of record "
        f"({RUN_DIR.name}; val_frac {run['val_frac']}, seed {run['seed']}, "
        f"{counts['n_folds']} folds). Row by row. Train wells "
        f"({counts['train']:,}) carry the loss at full weight; a fresh random "
        f"subset of the eligible train pool becomes visible as observation "
        f"sources every epoch, the subset fraction drawn from a uniform "
        f"distribution on ({run['frac_min']:.2f}, {run['frac_max']:.2f}), which is "
        "why the source cell is conditional. This arm reads sources through "
        "edges only (--source-edges, not --source-obs), so a well drawn as a "
        "source keeps its full label mass that epoch: a source never sees its "
        "own observation because self and same-site edges are dropped when the "
        "edge table is built. Their early-stop cell is conditional for a "
        "different reason: with source_val_mix on, the early-stop metric is the "
        "mean of two numbers, each the median absolute depth-to-water residual "
        "(m) over the val wells, one from a forward with no sources visible at "
        "all and one from a forward with every eligible train well visible, so "
        "a train well is a source in one of the two forwards and not in the "
        f"other. Val wells ({counts['val']:,}) are in no loss term and are not "
        "sources during training; they only steer early stopping, and they "
        "become sources in the final forward, whose test rows are the "
        f"out-of-fold predictions. Test wells ({counts['test']:,}) are the only "
        "scored rows and are in none of the three source pools; on top of that, "
        "every source edge whose source well belongs to the held-out fold is "
        "dropped for the whole of that fold's training, so a fold-f query can "
        "never read a fold-f label. Water-stage pseudo-rows are labels of a "
        "different kind — surface-water stage, not wells — and are never "
        f"eligible sources: the {counts['pseudo_train']:,} in train carry the "
        f"loss at weight {run['water_weight']:g} (the conditional cell), and the "
        f"{counts['pseudo_test']:,} in test are predicted out of fold but "
        "excluded from the headline test metric and reported only in a separate "
        "water-stage diagnostic panel (the other conditional cell). No "
        "pseudo-row is ever a val row: pseudo-rows falling inside a chosen val "
        f"block are returned to train. Row counts are fold {FOLD}'s rows inside "
        f"the {counts['n_basins']}-basin {CFG['name']} render footprint: "
        f"{counts['wells_foot']:,} well labels ({cls}) plus "
        f"{counts['pseudo_foot']:,} pseudo-rows. The split itself is dealt over "
        "the whole national bundle, and the eligible source pool during "
        "training is every well label in that bundle, appended rows "
        "included.\n\n"
        f"(b) The same wells at deployment, in the {dep['n_basins']}-basin "
        f"{CFG['name']} render. There is no fold guard here and none is "
        "intended: deployment_source_ctx exposes every bundle monitoring well "
        f"({counts['by_class']['monitoring']:,} of them inside the footprint) "
        f"plus the {dep['n_extra']:,} admitted {CFG['agency_long']} "
        f"({CFG['agency']}) wells, re-supplied from the frozen admission file, "
        "to every one of "
        f"the {counts['n_folds']} fold models, and the shipped water-table "
        f"elevation is the median over those {counts['n_folds']} per-fold "
        "surfaces. The fold-f model therefore reads fold-f wells as sources "
        "although it never saw them as labels. That is deployment semantics, "
        f"not a leak: source_exclude_m is {dep['exclude_m']:.0f} m, so a cell "
        "at a well reads that well, which is what a map is supposed to do with "
        "an observation in hand. The admitted sources were also training "
        f"labels (the conditional cell): {dep['n_state_labels']:,} of the "
        f"{dep['n_extra']:,} were appended to the bundle at weight 1 "
        f"({fun['dropped_nonfinite_fac_rem']} lacked a finite FAC-REM prior and "
        f"{fun['dropped_collocated_le_1m']} sat within 1 m of a bundle well), "
        "and at render time the bundle copies are excluded from the source "
        "pool by name so that each well enters the pool once, from the frozen "
        "file. The "
        "consequence is that the map cannot be scored against the wells it "
        "reads — the render's leak-gate panel is still computed at bundle "
        "wells but is recorded as informational and never enforced (the "
        "conditional scored cell), and the accuracy claim rests instead on the "
        f"{dep['n_targets']:,} held-out {CFG['agency']} targets, the eligible "
        f"half of the admission split among {dep['n_targets_all']:,} "
        f"label-independent {CFG['agency']} sites (Figure 5). Those targets are "
        "neither labels nor sources anywhere in the pipeline: no scored target "
        "shares a coordinate with any admitted source, and the monitoring "
        f"networks that supplied the bundle wells are {', '.join(dep['well_sources'])}. "
        "A map pixel is a pure consumer — never a source, never a label, never "
        "scored — reading the observations through its own edges.\n\n"
        f"Source edges, both stages: each query reads its k = {run['k']} nearest "
        "eligible source wells in the relief-lifted coordinate the regional base "
        f"R uses — (x, y, {run['relief_vw']:.0f} × land-surface elevation) in "
        "EPSG:5070 metres — with four target-blind edge attributes (log planar "
        "distance, log relief-lifted distance, relative land-surface elevation, "
        "and a same-FAC-basin flag). Self and same-site edges are dropped when "
        "the table is built, where same-site means exact coordinate identity "
        f"(a zero-radius rule, not a disk); {run['same_site_dropped']} same-site "
        f"edges existed to drop in this bundle, out of {run['n_edges']:,} edges. "
        "The deployment edges are rebuilt to the same recipe against the render "
        "lattice.\n\n"
        "Symbols: a filled square is yes, an open square is no, and a "
        "half-filled square is a qualified yes, with the qualification for every "
        "such cell given above. Row counts in panel (a) are fold "
        f"{FOLD}'s, reproduced from utils/train_conus_gnn.py and asserted "
        "against the per-fold counts logged by the run of record (Figure 3 maps "
        "that carving); panel (b) counts are read from the render manifests, "
        "the bundle manifest and the frozen admission tables. The regional base "
        "R is the archived leave-fold-out cross-fit field at both stages — "
        "built without a held-out fold's own wells during training, read as a "
        "lattice at the appended labels, and interpolated from those same "
        f"archived monitoring-well values at deployment, capped {dep['cap_m']:g} "
        "m below ground (Figure 2).\n"
    )


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    use_style()
    counts = fold_counts(FOLD)
    run = run_facts()
    dep = deployment_facts()
    if run["k"] != dep["k"]:
        raise ValueError(f"k differs: training {run['k']}, deployment {dep['k']}")
    if dep["n_sources"] != dep["n_bundle"] + dep["n_extra"]:
        raise ValueError("deployment source counts do not add up")
    if dep["n_bundle"] != counts["n_monitoring"]:
        raise ValueError(
            f"the render exposes {dep['n_bundle']:,} bundle sources but the "
            f"bundle holds {counts['n_monitoring']:,} monitoring wells"
        )
    if run["n_eligible"] != counts["wells"]:
        raise ValueError(
            f"training source pool {run['n_eligible']:,} != {counts['wells']:,} wells"
        )

    a_labels, b_labels = row_labels(counts, dep)
    fig = build_figure(a_labels, b_labels, len(A_COLS))
    paths = save(fig, "fig04_role_matrix", FIG_DIR)
    plt.close(fig)

    cap_path = FIG_DIR / "fig04_role_matrix_caption.txt"
    cap_path.write_text(caption(counts, run, dep))

    hdr = [c.replace("\n", " ") for c in A_COLS]
    print("(a) training, fold", FOLD, "|", " | ".join(hdr))
    for key, codes in A_ROWS:
        print(f"  {a_labels[key]:<48s} {' '.join(codes)}")
    print("(b) deployment |", " | ".join(c.replace("\n", " ") for c in B_COLS))
    for key, codes in B_ROWS:
        print(f"  {b_labels[key]:<48s} {' '.join(codes)}")
    print(
        f"\nsource-edge k={run['k']} vw={run['relief_vw']:.0f} "
        f"same-site dropped={run['same_site_dropped']} "
        f"deployment exclude_m={dep['exclude_m']:.0f} excluded names={dep['exclude_names']}"
    )
    for ext, p in paths.items():
        print(f"{ext}: {p}")
    print(f"caption: {cap_path}")


if __name__ == "__main__":
    main()
