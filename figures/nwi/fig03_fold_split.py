"""Figure 3 — the three roles inside one cross-validation fold.

One map of the state render footprint showing, for a single fold, which well
labels are scored (test), which only steer early stopping (val), and which are
trained on — and the two different spatial units the split is carved by. Test
is whole HUC12 watersheds; val is whole 40 km square grid blocks drawn from
what is left. The two units are nested in the same footprint, so a val block
can straddle test HUC12s: those wells stay test.

The split is reproduced exactly from ``utils/train_conus_gnn.py`` (fold loop) and
``utils/train_wte_gnn.py::val_blocks``, and verified against the per-fold
train/val/test counts logged in the run of record's ``gnn_run.json`` (the
state's parity arm, val_frac 0.15, seed 0). The verification is a hard
assertion: a mismatch raises rather than draws.

Written to the house figure rules in ``~/code/style/FIGURE_STYLE_GUIDE.md``:
no gridlines, no descriptive panel titles, a single framed legend placed with
``loc=`` only, type sizes left to the shared mplstyle, and a
colour-vision-deficiency-safe palette whose role encoding is carried by marker
shape, size, fill and outline weight so it survives grayscale.

Run::

    NWI_STATE=NM uv run python figures/nwi/fig03_fold_split.py --fold 0

Writes ``fig03_fold_split.{pdf,png}`` plus a ``*_caption.txt`` sidecar to
``/data/ssd2/handily/figures/<project>/`` (``nwi`` for NV, ``nmose`` for NM).
"""

from __future__ import annotations

import argparse
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from nwi_common import (
    BUNDLE_DIR,
    CFG,
    FIG_DIR,
    FOLD_COLORS,
    HUC8_POLYS,
    MM,
    NO_FOLD_EDGE,
    PSEUDO_SOURCE,
    RUN_DIR,
    WELL_CLASSES,
    emptiest_corner,
    huc12_folds,
    north_arrow,
    rendered_basins,
    save,
    scale_bar,
    state_outline,
    use_style,
)
from utils.figures.fig_common import WIDTH_1COL  # noqa: E402  (path set by nwi_common)

#: Trainer arguments of the run of record that drive the split.
VAL_FRAC = 0.15
SEED = 0
#: ``build_conus_graph_inputs.py --block-size-m`` (default), the ``block_40km`` scale.
BLOCK_M = 40_000.0

ROLES = ("test", "val", "train")

#: Wells are drawn on top of a pale wash of their own fold colour, so the test
#: HUC12 fill is that colour blended toward white to a FIXED relative luminance
#: (guide §7's two-tier saturation: a uniform pale context layer under saturated
#: marks). A fixed blend fraction would leave the pale members of FOLD_COLORS
#: (yellow, sky blue) invisible and the dark ones competing with the wells.
FILL_LUM = 0.82
#: Val blocks: hatched, unfilled, hairline outline.
VAL_HATCH = "///"
VAL_INK = "#1a1a1a"
TRAIN_INK = "#7a7a7a"


# --------------------------------------------------------------------------
# The trainer's split, reproduced
# --------------------------------------------------------------------------


def val_blocks(
    trainval: np.ndarray, blocks: np.ndarray, frac: float, rng: np.random.RandomState
) -> tuple[np.ndarray, list[str]]:
    """``train_wte_gnn.val_blocks``, verbatim, also returning the chosen block ids.

    Whole 40 km blocks are drawn in a seeded permutation and accumulated by ROW
    count (wells plus water-stage pseudo-rows) until ``frac`` of the train-val
    rows is reached — blocks vary by >100x in density, so a fixed fraction of
    blocks would badly over/under-shoot the row fraction.
    """
    idx = np.where(trainval)[0]
    bl = blocks[idx]
    ub, counts = np.unique(bl, return_counts=True)
    order = rng.permutation(len(ub))
    target = frac * idx.size
    chosen: set = set()
    acc = 0
    for j in order:
        if acc >= target:
            break
        chosen.add(ub[j])
        acc += counts[j]
    return np.isin(blocks, list(chosen)) & trainval, sorted(chosen)


def load_rows() -> pd.DataFrame:
    """Every bundle label row in the trainer's own frame (sorted by node index)."""
    qn = (
        pd.read_parquet(
            BUNDLE_DIR / "query_nodes.parquet",
            columns=[
                "query_node_idx",
                "source",
                "well_class",
                "huc8",
                "cv_unit",
                "cv_fold",
                "block_40km",
                "is_water_pseudo",
                "x5070",
                "y5070",
            ],
        )
        .sort_values("query_node_idx")
        .reset_index(drop=True)
    )
    qn["huc8"] = qn["huc8"].astype(str).str.zfill(8)
    bad = qn["block_40km"].isna() | qn["cv_fold"].isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} bundle rows carry no fold or 40 km block")
    return qn


def split_all_folds(qn: pd.DataFrame) -> dict:
    """Reproduce every fold's (test, val, train) masks and chosen val blocks.

    ``rng`` is one ``RandomState(seed)`` stream created before the fold loop and
    consumed nowhere else in the trainer, so the folds must be walked in sorted
    order together: fold f's val draw depends on every earlier fold's draw.
    """
    # `real` in the trainer is ~water & ~swl & ~shore. This bundle carries only the
    # water-stage pseudo-rows (no SWL aux, no shoreline rows), asserted here rather
    # than silently assumed; the well classes are the monitoring networks plus
    # the appended state-agency rows.
    water = qn["is_water_pseudo"].to_numpy(bool)
    if (qn["well_class"].to_numpy() == "water_pseudo").sum() != water.sum():
        raise ValueError("is_water_pseudo disagrees with well_class == water_pseudo")
    extra = set(qn["well_class"].unique()) - set(WELL_CLASSES) - {"water_pseudo"}
    if extra:
        raise ValueError(f"unexpected well_class values: {extra}")
    real = ~water

    folds = np.array(sorted(qn["cv_fold"].unique()))
    blocks = qn["block_40km"].to_numpy()
    rng = np.random.RandomState(SEED)
    out = {}
    for f in folds:
        test = qn["cv_fold"].to_numpy() == f
        trainval = ~test
        va, chosen = val_blocks(trainval, blocks, VAL_FRAC, rng)
        va &= real  # early stop on wells only; pseudo-rows in val blocks fall to train
        tr = trainval & ~va
        out[int(f)] = {
            "test": test,
            "val": va,
            "train": tr,
            "blocks": chosen,
            "real": real,
        }
    return out


def verify_against_run(split: dict) -> list[dict]:
    """Assert the reproduced row counts equal the run of record's logged counts."""
    run = json.loads((RUN_DIR / "gnn_run.json").read_text())
    if run["graph_dir"].rstrip("/") != str(BUNDLE_DIR):
        raise ValueError(
            f"{RUN_DIR.name} was trained on {run['graph_dir']}, not {BUNDLE_DIR}"
        )
    hp = run["hyperparams"]
    if (float(hp["val_frac"]), int(hp["seed"])) != (VAL_FRAC, SEED):
        raise ValueError(
            f"run val_frac/seed {hp['val_frac']}/{hp['seed']} differ from "
            f"{VAL_FRAC}/{SEED}"
        )
    logged = run["folds"]
    by_fold = {int(d["fold"]): d for d in logged}
    if set(by_fold) != set(split):
        raise ValueError(f"fold sets differ: run {sorted(by_fold)} vs {sorted(split)}")
    for f, s in split.items():
        got = (int(s["train"].sum()), int(s["val"].sum()), int(s["test"].sum()))
        want = (
            int(by_fold[f]["n_train"]),
            int(by_fold[f]["n_val"]),
            int(by_fold[f]["n_test"]),
        )
        if got != want:
            raise ValueError(
                f"fold {f}: reproduced (train, val, test) = {got} but "
                f"{RUN_DIR / 'gnn_run.json'} logs {want}"
            )
    return logged


def block_squares(block_ids, crs=5070):
    """The 40 km EPSG:5070 squares named by ``block_40km`` ids.

    ``build_conus_graph_inputs.py`` writes the id as ``f"{bx}_{by}"`` with
    ``bx = floor(x5070 / 40000)`` and ``by = floor(y5070 / 40000)``, so the square
    is ``[bx, bx+1) x [by, by+1)`` in units of 40 km.
    """
    import geopandas as gpd
    from shapely.geometry import box

    bx, by = zip(*(map(int, b.split("_")) for b in block_ids))
    geom = [
        box(i * BLOCK_M, j * BLOCK_M, (i + 1) * BLOCK_M, (j + 1) * BLOCK_M)
        for i, j in zip(bx, by)
    ]
    return gpd.GeoDataFrame({"block_40km": list(block_ids)}, geometry=geom, crs=crs)


def _pale(hexcolor: str, lum: float = FILL_LUM) -> tuple:
    """Blend a fold colour toward white to a fixed relative luminance.

    Luminance is a weighted sum of the channels, so it is linear in the blend
    fraction ``t`` against white: ``L(t) = t * L(c) + (1 - t)``. Solve for the
    target and clip, so a colour already lighter than the target simply keeps
    the deepest tint it can carry.
    """
    r, g, b = mpl.colors.to_rgb(hexcolor)
    lum_c = 0.2126 * r + 0.7152 * g + 0.0722 * b
    t = float(np.clip((1.0 - lum) / max(1.0 - lum_c, 1e-6), 0.15, 0.85))
    return tuple(t * c + (1.0 - t) for c in (r, g, b))


# --------------------------------------------------------------------------
# The map
# --------------------------------------------------------------------------


def draw_map(ax, fold, h12, basin_polys, footprint, state, vblocks, wells) -> None:
    fold_color = FOLD_COLORS[fold]

    is_test_h12 = (h12["cv_fold"] == fold) & (h12["n_wells"] > 0)
    # The thousands-polygon tiling is rasterized (at savefig.dpi) so the PDF
    # stays under publisher size caps; points, boundaries and text remain vector.
    h12[~is_test_h12].plot(
        ax=ax,
        color="white",
        edgecolor=NO_FOLD_EDGE,
        linewidth=0.15,
        zorder=1,
        rasterized=True,
    )
    h12[is_test_h12].plot(
        ax=ax,
        color=_pale(fold_color),
        edgecolor=NO_FOLD_EDGE,
        linewidth=0.15,
        zorder=2,
        rasterized=True,
    )
    basin_polys.boundary.plot(ax=ax, color="#8c8c8c", linewidth=0.3, zorder=3)
    footprint.boundary.plot(ax=ax, color="#4d4d4d", linewidth=0.4, zorder=4)
    vblocks.plot(
        ax=ax,
        facecolor="none",
        edgecolor=VAL_INK,
        linewidth=0.3,
        hatch=VAL_HATCH,
        zorder=5,
    )
    state.boundary.plot(ax=ax, color="black", linewidth=0.7, zorder=6)

    # train first (the crowd), then val, then test on top: the scored set is the
    # one the reader must find, and it is the smallest.
    styles = {
        "train": {
            "marker": "o",
            "ms": 1.5,
            "mfc": "none",
            "mec": TRAIN_INK,
            "mew": 0.3,
        },
        "val": {"marker": "s", "ms": 1.7, "mfc": VAL_INK, "mec": VAL_INK, "mew": 0.2},
        "test": {
            "marker": "o",
            "ms": 2.1,
            "mfc": fold_color,
            "mec": "black",
            "mew": 0.45,
        },
    }
    for role in ("train", "val", "test"):
        w = wells[wells["role"] == role]
        ax.plot(
            w["x5070"].to_numpy(),
            w["y5070"].to_numpy(),
            ls="none",
            zorder=7 + ROLES[::-1].index(role),
            **styles[role],
        )

    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_xlim(*_pad(footprint.total_bounds[0], footprint.total_bounds[2]))
    ax.set_ylim(*_pad(footprint.total_bounds[1], footprint.total_bounds[3]))
    # Furniture goes where the wells are not: the legend takes the emptiest of
    # the candidate corners (counted on the drawn wells), the scale bar the
    # lower corner the legend left free, and the arrow the NE corner.
    loc = emptiest_corner(
        ax,
        wells["x5070"].to_numpy(),
        wells["y5070"].to_numpy(),
        ("lower left", "lower right", "upper left", "center left"),
        w=0.30,
        h=0.20,
    )
    scale_bar(
        ax,
        length_m=200_000.0,
        x_frac=0.70 if loc == "lower left" else 0.03,
        y_frac=0.02,
    )
    north_arrow(ax, x_frac=0.94, y_frac=0.90)

    handles = [
        Line2D([], [], ls="none", label="test well", **_leg(styles["test"])),
        Line2D([], [], ls="none", label="val well", **_leg(styles["val"])),
        Line2D([], [], ls="none", label="train well", **_leg(styles["train"])),
        Patch(
            facecolor=_pale(fold_color),
            edgecolor=NO_FOLD_EDGE,
            linewidth=0.4,
            label="test HUC12",
        ),
        Patch(
            facecolor="none",
            edgecolor=VAL_INK,
            linewidth=0.3,
            hatch=VAL_HATCH,
            label="val block, 40 km",
        ),
    ]
    ax.legend(handles=handles, loc=loc, title=f"Role, fold {fold}", handlelength=1.0)


def _leg(style: dict) -> dict:
    """Legend-sized copy of a map marker style (guide §8: 2-3 pt samples)."""
    out = dict(style)
    out["ms"] = max(2.4, out["ms"] + 0.6)
    return out


def _pad(lo: float, hi: float, frac: float = 0.02) -> tuple:
    d = (hi - lo) * frac
    return lo - d, hi + d


# --------------------------------------------------------------------------
# Caption
# --------------------------------------------------------------------------


def _ledger(qn: pd.DataFrame, split: dict, keep=None) -> pd.DataFrame:
    """Realised counts per fold: wells by role, pseudo-rows in test and train.

    ``keep`` restricts the counted rows (the footprint ledger); the val-block
    count is the fold's national draw either way.
    """
    water = qn["is_water_pseudo"].to_numpy(bool)
    real = ~water
    keep = np.ones(len(qn), dtype=bool) if keep is None else np.asarray(keep, bool)
    rows = []
    for f in sorted(split):
        s = split[f]
        rows.append(
            {
                "fold": f,
                "test": int((s["test"] & real & keep).sum()),
                "val": int((s["val"] & keep).sum()),
                "train": int((s["train"] & real & keep).sum()),
                "pseudo_test": int((s["test"] & water & keep).sum()),
                "pseudo_train": int((s["train"] & water & keep).sum()),
                "n_val_blocks": len(s["blocks"]),
            }
        )
    d = pd.DataFrame(rows)
    d["val_frac"] = d["val"] / (d["val"] + d["train"])
    return d


CLASS_WORDS = {
    "monitoring": "monitoring-network wells",
    "ndwr": "appended NV NDWR wells",
    "ose": "appended NM OSE wells",
}


def _caption(fold, led, led_foot, st, h12, n_blocks_drawn, basins, classes) -> str:
    """``led`` is the national ledger (val-fraction rule), ``led_foot`` the
    footprint ledger whose counts are quoted."""
    r = led[led["fold"] == fold].iloc[0]
    rf = led_foot[led_foot["fold"] == fold].iloc[0]
    per_fold = "; ".join(
        f"{int(x.fold)}: {int(x.test):,}/{int(x.val):,}/{int(x.train):,}"
        for x in led_foot.itertuples()
    )
    fr = led["val_frac"]
    cls_foot = "; ".join(
        f"{CLASS_WORDS[c]} {n:,}" for c, n in classes["footprint"].items()
    )
    return (
        f"Figure 3. For one cross-validation fold, which wells are scored, which "
        f"steer early stopping, and which are trained on, and by what spatial unit "
        f"is each carved?\n\n"
        f"The {len(basins)}-basin {CFG['name']} render footprint, tiled by its "
        f"{len(h12):,} HUC12s, for fold {fold}. Three rules define the partition. "
        f"Test is every well in a HUC12 assigned to fold {fold}: folds are dealt "
        f"to whole HUC12 watersheds, so the scored set is removed a whole "
        f"watershed at a time. Val is every remaining well inside a set of whole "
        f"40 km square grid blocks, drawn in a seeded permutation and accumulated "
        f"until they hold about 15% of the train-val ROW count. Train is "
        f"everything else. Val is used only to early-stop training and is never "
        f"scored, so it is not a second test set. Wells here are every well label "
        f"of the training bundle inside the footprint ({cls_foot}); the split "
        f"itself is dealt over the whole national bundle, and the appended "
        f"state-agency rows take the fold of their HUC12 where the base bundle "
        f"had one, and otherwise, as a whole unit, the fold of the nearest base "
        f"well, so they are carved by the same three rules.\n\n"
        f"Symbols: filled circles in the fold colour are test wells and the pale "
        f"wash of the same colour is the HUC12s of fold {fold} that hold at least "
        f"one well label; filled squares are val wells and the hatched squares "
        f"are the fold's chosen 40 km blocks ({n_blocks_drawn} of "
        f"{int(r.n_val_blocks)} intersect the footprint); open circles are train "
        f"wells. Hatched blocks overlie fold-colour HUC12s in places: a val block is "
        f"drawn from the train-val wells only, so wells inside it that belong to a "
        f"fold-{fold} HUC12 stay test. A hatched block with no val well drawn "
        f"inside the footprint holds only pseudo-rows, test wells, or wells "
        f"beyond the footprint edge. Hairlines are HUC12 boundaries, mid-grey "
        f"lines HUC8 boundaries, the dark grey line the render footprint, and the "
        f"black line the {CFG['name']} state boundary — the footprint extends "
        f"past it because HUC8 basins straddle the border.\n\n"
        f"Realised counts, fold {fold}, inside the footprint drawn here: "
        f"{st['test']:,} test / {st['val']:,} val / {st['train']:,} train wells "
        f"of {st['total']:,}, plus {int(rf.pseudo_test):,} water-stage "
        f"pseudo-rows in test and {int(rf.pseudo_train):,} in train. Wells are "
        f"the only points drawn; the {st['pseudo']:,} pseudo-rows inside the "
        f"footprint are not. All eight folds, footprint test/val/train wells: "
        f"{per_fold}. The 15% is a share of "
        f"train-val ROWS, and rows include the water-stage pseudo-rows; "
        f"accumulation also stops on the first block that crosses the target, so "
        f"the row target is overshot by up to one block. The pseudo-rows that "
        f"land in a chosen block are then returned to train, so val is a "
        f"well-only set selected by a row-count rule and its realised share of "
        f"train-val WELLS over the whole bundle is not pinned to 15%: it runs "
        f"{fr.min():.1%}-{fr.max():.1%} across the folds "
        f"({fr[led['fold'] == fold].iloc[0]:.1%} for fold {fold}), depending on how "
        f"well-dense the blocks a fold happens to draw are. The block draw "
        f"consumes one seeded random stream shared by the whole fold loop, so each "
        f"fold's blocks are redrawn from where the previous fold left off and the "
        f"val geometry differs fold to fold; folds must be reproduced in sorted "
        f"order together. Split reproduced from utils/train_conus_gnn.py and "
        f"verified against the per-fold row counts logged by the run of record "
        f"({RUN_DIR.name}, val_frac {VAL_FRAC}, seed {SEED}).\n\n"
        f"Albers equal-area projection (EPSG:5070); {len(basins)} HUC8 basins and "
        f"their HUC12s from the WBD, state boundary from the 2022 US Census "
        f"cartographic boundary file. Fold identity is nominal; the role encoding "
        f"is carried by marker shape, size, fill and outline weight, so the three roles "
        f"survive grayscale.\n"
    )


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fold",
        type=int,
        default=0,
        help="cross-validation fold to render as the held-out (test) fold",
    )
    args = ap.parse_args()

    import geopandas as gpd

    use_style()
    mpl.rcParams["hatch.linewidth"] = 0.3

    qn = load_rows()
    split = split_all_folds(qn)
    if args.fold not in split:
        raise SystemExit(f"--fold {args.fold} not in {sorted(split)}")
    verify_against_run(split)
    led = _ledger(qn, split)

    basins = rendered_basins()
    led_foot = _ledger(qn, split, keep=qn["huc8"].isin(basins).to_numpy())
    h12 = huc12_folds(basins)
    h8 = gpd.read_parquet(HUC8_POLYS, columns=["huc8", "geometry"])
    h8["huc8"] = h8["huc8"].astype(str).str.zfill(8)
    basin_polys = h8[h8["huc8"].isin(basins)]
    footprint = basin_polys.dissolve()

    s = split[args.fold]
    role = np.full(len(qn), "", dtype=object)
    for name in ROLES:
        role[s[name]] = name
    qn["role"] = role
    if (qn["role"] == "").any():
        raise ValueError("a bundle row landed in no role")

    is_well = qn["well_class"].isin(WELL_CLASSES)
    wells = qn[is_well & qn["huc8"].isin(basins)].copy()
    if (wells["source"] == PSEUDO_SOURCE).any():
        raise ValueError("a pseudo-source row survived the well filter")
    vblocks = block_squares(s["blocks"])
    vblocks = vblocks[vblocks.intersects(footprint.geometry.iloc[0])]

    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 132 * MM))
    draw_map(
        ax, args.fold, h12, basin_polys, footprint, state_outline(), vblocks, wells
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    stem = f"fig03_fold_split_f{args.fold}" if args.fold else "fig03_fold_split"
    paths = save(fig, stem, FIG_DIR)
    plt.close(fig)

    st = {
        "test": int((wells["role"] == "test").sum()),
        "val": int((wells["role"] == "val").sum()),
        "train": int((wells["role"] == "train").sum()),
        "total": len(wells),
        "pseudo": int(
            ((qn["well_class"] == "water_pseudo") & qn["huc8"].isin(basins)).sum()
        ),
    }
    classes = {
        "footprint": {
            c: int((wells["well_class"] == c).sum())
            for c in WELL_CLASSES
            if (wells["well_class"] == c).any()
        },
    }
    cap_path = FIG_DIR / f"{stem}_caption.txt"
    cap_path.write_text(
        _caption(args.fold, led, led_foot, st, h12, len(vblocks), basins, classes)
    )
    r, rf = led[led["fold"] == args.fold], led_foot[led_foot["fold"] == args.fold]
    if int(rf["test"].iloc[0]) != st["test"] or int(rf["train"].iloc[0]) != st["train"]:
        raise ValueError("footprint ledger disagrees with the drawn well roles")
    if int(r["test"].iloc[0]) < st["test"]:
        raise ValueError("national ledger holds fewer test wells than the footprint")

    print(led.to_string(index=False))
    print("footprint:")
    print(led_foot.to_string(index=False))
    print(f"\nfold {args.fold}  footprint wells: {st}  classes: {classes}")
    print(f"val blocks: {len(s['blocks'])} national, {len(vblocks)} drawn")
    for ext, p in paths.items():
        print(f"{ext}: {p}")
    print(f"caption: {cap_path}")


if __name__ == "__main__":
    main()
