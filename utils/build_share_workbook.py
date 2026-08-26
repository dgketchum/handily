"""Build the formatted accuracy workbook that ships in an external data packet.

Turns the two ``validate_fac_gwx_wells.py`` output dirs (a statewide two-way run
and a FAC-REM-footprint three-way run) into a single .xlsx aimed at a reader who
is not on this project: a plain-language guide tab, the depth-band headline, the
shallow-detection table, the spatial breakdowns, and the raw panel.

xlsx rather than a Google Sheet because the Docs MCP has no Sheets write scope;
the file opens in Sheets with formatting intact.

    uv run python utils/build_share_workbook.py \
        --statewide  <val>/mt_2way_fullfp \
        --facrem-fp  <val>/mt_3way \
        --state MT --out /data/ssd2/handily/share/dnrc_packet/mt_accuracy_panel.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("share_workbook")

NAME = {"FAC_REM": "FAC-REM", "handily": "handily", "Ma": "Ma"}
BANDS = ("0-2m", "2-5m", "5-10m", "10-30m", "30+m")
PRODUCT_ORDER = ("FAC-REM", "handily", "Ma")

INK = "1F3A5F"
PAPER = "FFFFFF"
BAND_TINT = "EEF3F8"
WIN = "D9EAD3"
RULE = Side(style="thin", color="BFC9D4")
THICK = Side(style="medium", color="1F3A5F")

# Columns where a LOWER value is better -- drives the per-band winner highlight.
LOWER_IS_BETTER = ("mad_m", "rmse_m", "p95_abs_err_m", "frac_abs_err_gt_10m")


def _header(ws, row: int, labels: list[str], widths: list[int] | None = None) -> None:
    for i, lab in enumerate(labels, start=1):
        c = ws.cell(row=row, column=i, value=lab)
        c.font = Font(bold=True, color=PAPER, size=10)
        c.fill = PatternFill("solid", fgColor=INK)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = Border(bottom=THICK)
    if widths:
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w


def _title(ws, row: int, text: str, size: int = 13) -> None:
    c = ws.cell(row=row, column=1, value=text)
    c.font = Font(bold=True, size=size, color=INK)


def sheet_guide(wb: Workbook, state: str, n_a: int, n_b: int) -> None:
    ws = wb.create_sheet("How to read this")
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 108
    ws.sheet_view.showGridLines = False

    def block(row: int, heading: str, pairs: list[tuple[str, str]]) -> int:
        _title(ws, row, heading)
        row += 1
        for k, v in pairs:
            a = ws.cell(row=row, column=1, value=k)
            a.font = Font(bold=True, size=10)
            a.alignment = Alignment(vertical="top")
            b = ws.cell(row=row, column=2, value=v)
            b.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[row].height = max(15, 13 * (1 + len(v) // 95))
            row += 1
        return row + 1

    r = 1
    _title(ws, r, f"{state} depth-to-water maps — how to read this workbook", size=15)
    r += 2
    c = ws.cell(
        row=r,
        column=1,
        value=(
            "Three maps of depth to water, each checked against Montana water wells. "
            "Every number in this workbook is in meters. Depth is measured downward "
            "from the ground surface, so a bigger number means deeper water."
        ),
    )
    c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
    ws.row_dimensions[r].height = 30
    r += 2

    r = block(
        r,
        "The three maps",
        [
            (
                "handily",
                "Our machine-learning water-table map. It learns from monitoring "
                "wells together with terrain, climate and streamflow information, "
                "then predicts everywhere in the state. 10 m pixels.",
            ),
            (
                "FAC-REM",
                "Our terrain-only map. It measures how high the ground sits above "
                "the nearest drainage and treats that height as the depth to water. "
                "It uses no well data at all. 10 m pixels, and it only has values "
                "near stream channels.",
            ),
            (
                "Ma",
                "The published national benchmark (Ma and others). The best existing "
                "product we know of and the bar we are trying to clear. 24 m pixels.",
            ),
        ],
    )

    r = block(
        r,
        "Which wells we tested against",
        [
            (
                "The test set",
                "Montana wells that are screened in an unconfined aquifer, are not "
                "from the NWIS/NGWMN networks the Ma product was trained on, and sit "
                "at least 3.5 km from any well handily was trained on. The last rule "
                "is what keeps the comparison honest — a model can look accurate "
                "simply by reproducing wells it was fitted to.",
            ),
            (
                "Why unconfined only",
                "A confined well measures pressure in a sealed aquifer, not the water "
                "table. Its level can sit far above or below the actual water table, "
                "so including confined wells would not test what these maps predict.",
            ),
        ],
    )

    r = block(
        r,
        "The two panels",
        [
            (
                "Panel A",
                f"{n_a:,} wells, handily against Ma. This is the fair statewide "
                "comparison of the two maps that cover the whole state.",
            ),
            (
                "Panel B",
                f"{n_b:,} wells, all three maps. FAC-REM has no value away from "
                "stream channels, so a three-way comparison can only run where all "
                "three maps are defined. That subset sits nearer streams and is "
                "shallower than the state as a whole, so its totals are NOT "
                "comparable with Panel A's — compare within a panel, never across.",
            ),
        ],
    )

    r = block(
        r,
        "What each column means",
        [
            (
                "MAD",
                "Median absolute error, in meters — the typical miss. Half the wells "
                "are closer than this and half are further. Lower is better. It is "
                "unmoved by a handful of wild misses, which is why it needs RMSE "
                "next to it.",
            ),
            (
                "Median residual",
                "The middle signed error, in meters. Positive means the map places "
                "water too deep, negative means too shallow. This tells you which "
                "way a map leans.",
            ),
            (
                "Mean bias",
                "The average signed error, in meters. When mean bias is much larger "
                "than the median residual, the map is not uniformly offset — it has "
                "a tail of large one-sided misses. The difference matters: a uniform "
                "offset can be corrected, a tail cannot.",
            ),
            (
                "RMSE",
                "Root-mean-square error, in meters. It weights large misses heavily, "
                "so it exposes the worst cases MAD hides. A map can win on MAD and "
                "lose badly on RMSE — read the two together, always.",
            ),
            (
                "p95 abs err",
                "95th-percentile absolute error, in meters — the worst 1-in-20 case.",
            ),
            (
                "% miss > 10 m",
                "The share of wells the map missed by more than 10 m. The practical "
                "blunder rate.",
            ),
            ("n", "How many wells are in that row."),
        ],
    )

    r = block(
        r,
        "Why there is no single accuracy number here",
        [
            (
                "",
                "A statewide average hides the finding. These maps behave very "
                "differently depending on how deep the water actually is, and a "
                "single figure just reports whichever depth range happens to hold "
                "the most wells. The depth-band tab is the real result — start there.",
            ),
        ],
    )

    block(
        r,
        "The other tabs",
        [
            (
                "Depth bands",
                "The headline result, split by how deep the water really is.",
            ),
            (
                "Shallow detection",
                "How well each map answers the yes/no question: is water within "
                "2, 5 or 10 m of the surface?",
            ),
            (
                "Where errors live",
                "Errors split by valley against upland, distance to the nearest "
                "stream, and distance to the nearest well handily trained on.",
            ),
            (
                "Full panel",
                "Every number we computed, for anyone who wants to reslice it.",
            ),
        ],
    )


def _write_metric_table(
    ws,
    df: pd.DataFrame,
    group_col: str,
    group_order: list[str],
    start_row: int,
    label_head: str,
) -> int:
    """Grouped metric block: one group per band of rows, winner highlighted."""
    cols = [
        (label_head, group_col, None),
        ("Wells (n)", "n", "#,##0"),
        ("Map", "product", None),
        ("MAD", "mad_m", "0.00"),
        ("Median resid", "median_residual_m", "+0.00;-0.00"),
        ("Mean bias", "bias_m", "+0.00;-0.00"),
        ("RMSE", "rmse_m", "0.00"),
        ("p95 abs err", "p95_abs_err_m", "0.00"),
        ("% miss > 10 m", "frac_abs_err_gt_10m", "0%"),
    ]
    _header(
        ws,
        start_row,
        [c[0] for c in cols],
        widths=[14, 11, 11, 9, 13, 11, 9, 12, 13],
    )
    row = start_row + 1
    shade = False
    for g in group_order:
        sub = df[df[group_col] == g]
        if sub.empty:
            continue
        sub = sub.set_index("product").reindex(
            [p for p in PRODUCT_ORDER if p in set(sub["product"])]
        )
        best = {m: sub[m].min() for m in LOWER_IS_BETTER if sub[m].notna().any()}
        first = row
        for prod, rec in sub.iterrows():
            for i, (_, key, fmt) in enumerate(cols, start=1):
                if key == group_col:
                    val = g if row == first else None
                elif key == "product":
                    val = prod
                elif key == "n":
                    val = int(rec["n"]) if row == first else None
                else:
                    val = rec[key]
                c = ws.cell(row=row, column=i, value=val)
                if fmt and val is not None:
                    c.number_format = fmt
                c.alignment = Alignment(horizontal="center" if i > 1 else "left")
                if shade:
                    c.fill = PatternFill("solid", fgColor=BAND_TINT)
                if key in best and pd.notna(rec[key]) and rec[key] == best[key]:
                    c.font = Font(bold=True, color="1E5C2E")
                    c.fill = PatternFill("solid", fgColor=WIN)
                if key == "product":
                    c.font = Font(bold=True, size=10)
                c.border = Border(bottom=RULE)
            row += 1
        shade = not shade
    return row


def sheet_depth_bands(wb: Workbook, df: pd.DataFrame, n_a: int, n_b: int) -> None:
    ws = wb.create_sheet("Depth bands")
    ws.sheet_view.showGridLines = False
    _title(ws, 1, "Accuracy by how deep the water actually is", size=15)
    ws.cell(
        row=2,
        column=1,
        value=(
            "Green = best of the maps in that row group. Bands are set by the "
            "measured depth at the well, not by the prediction."
        ),
    ).font = Font(italic=True, size=9, color="555555")

    r = 4
    _title(ws, r, f"Panel A — statewide, handily vs Ma  (n = {n_a:,} wells)", size=11)
    d = df[(df.panel == "A_statewide") & (df.group_type == "obs_depth")]
    r = _write_metric_table(ws, d, "group", list(BANDS), r + 1, "Real depth")

    r += 2
    _title(
        ws,
        r,
        f"Panel B — all three maps, FAC-REM footprint only  (n = {n_b:,} wells)",
        size=11,
    )
    ws.cell(
        row=r + 1,
        column=1,
        value="Not comparable with Panel A — a nearer-stream, shallower subset.",
    ).font = Font(italic=True, size=9, color="555555")
    d = df[(df.panel == "B_facrem_footprint") & (df.group_type == "obs_depth")]
    _write_metric_table(ws, d, "group", list(BANDS), r + 2, "Real depth")
    ws.freeze_panes = "A5"


def sheet_where(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.create_sheet("Where errors live")
    ws.sheet_view.showGridLines = False
    _title(ws, 1, "Where the errors are, by setting and distance", size=15)
    blocks = [
        ("setting", "Valley floor vs upland", "Setting"),
        ("fac_dist_stream", "Distance to the nearest mapped stream", "To stream"),
        (
            "train_dist",
            "Distance to the nearest well handily trained on",
            "To training well",
        ),
        ("well_class", "By type of well record", "Well type"),
    ]
    r = 3
    for gt, heading, lab in blocks:
        d = df[(df.panel == "A_statewide") & (df.group_type == gt)]
        if d.empty:
            continue
        _title(ws, r, f"{heading}  —  Panel A (statewide, handily vs Ma)", size=11)
        order = list(dict.fromkeys(d["group"]))
        r = _write_metric_table(ws, d, "group", order, r + 1, lab) + 2
    ws.freeze_panes = "A4"


def sheet_shallow(wb: Workbook, pr: pd.DataFrame) -> None:
    ws = wb.create_sheet("Shallow detection")
    ws.sheet_view.showGridLines = False
    _title(ws, 1, "Finding shallow water: precision and recall", size=15)
    for i, txt in enumerate(
        [
            "Recall — of the wells that really are shallower than the cutoff, what "
            "share did the map flag as shallow? A high-recall map misses few of them.",
            "Precision — of the places the map called shallow, what share really "
            "were? A high-precision map raises few false alarms.",
            "Neither is 'the' right number; which one matters depends on whether a "
            "missed shallow area or a false alarm is the costlier mistake for you.",
        ]
    ):
        c = ws.cell(row=2 + i, column=1, value=txt)
        c.font = Font(italic=True, size=9, color="555555")
        ws.merge_cells(start_row=2 + i, start_column=1, end_row=2 + i, end_column=6)

    r = 7
    for panel, lab in (
        ("A_statewide", "Panel A — statewide (handily vs Ma)"),
        ("B_facrem_footprint", "Panel B — all three maps, FAC-REM footprint"),
    ):
        d = pr[(pr.panel == panel) & (pr.scope == "all")]
        if d.empty:
            continue
        _title(ws, r, lab, size=11)
        _header(
            ws,
            r + 1,
            [
                "Cutoff",
                "Map",
                "Recall",
                "Precision",
                "Wells truly shallow",
                "Places called shallow",
            ],
            widths=[12, 12, 11, 11, 20, 22],
        )
        row = r + 2
        shade = False
        for thr in ("<2m", "<5m", "<10m"):
            sub = d[d.threshold == thr]
            sub = sub.set_index("product").reindex(
                [p for p in PRODUCT_ORDER if p in set(sub["product"])]
            )
            best_r = sub["recall"].max()
            first = row
            for prod, rec in sub.iterrows():
                vals = [
                    thr if row == first else None,
                    prod,
                    rec["recall"],
                    rec["precision"],
                    int(rec["n_obs_shallow"]) if row == first else None,
                    int(rec["n_pred_shallow"]),
                ]
                for i, v in enumerate(vals, start=1):
                    c = ws.cell(row=row, column=i, value=v)
                    if i in (3, 4):
                        c.number_format = "0.00"
                    if i in (5, 6):
                        c.number_format = "#,##0"
                    c.alignment = Alignment(horizontal="center" if i > 1 else "left")
                    if shade:
                        c.fill = PatternFill("solid", fgColor=BAND_TINT)
                    if i == 3 and rec["recall"] == best_r:
                        c.font = Font(bold=True, color="1E5C2E")
                        c.fill = PatternFill("solid", fgColor=WIN)
                    if i == 2:
                        c.font = Font(bold=True, size=10)
                    c.border = Border(bottom=RULE)
                row += 1
            shade = not shade
        r = row + 2


def sheet_full(wb: Workbook, df: pd.DataFrame) -> None:
    ws = wb.create_sheet("Full panel")
    heads = list(df.columns)
    _header(ws, 1, heads, widths=[20, 17, 16, 10, 10, 10, 15, 10, 13, 13, 15, 15])
    for j, rec in enumerate(df.itertuples(index=False), start=2):
        for i, v in enumerate(rec, start=1):
            c = ws.cell(row=j, column=i, value=v)
            col = heads[i - 1]
            if col == "n":
                c.number_format = "#,##0"
            elif col.startswith("frac_"):
                c.number_format = "0.0%"
            elif col.endswith("_m"):
                c.number_format = "0.00"
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(heads))}{len(df) + 1}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statewide", required=True, help="2-way validation out-dir")
    p.add_argument("--facrem-fp", required=True, help="3-way validation out-dir")
    p.add_argument("--state", default="MT")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    frames, prs = [], []
    for d, panel in (
        (args.statewide, "A_statewide"),
        (args.facrem_fp, "B_facrem_footprint"),
    ):
        s = pd.read_csv(Path(d) / "score_summary.csv")
        s.insert(0, "panel", panel)
        frames.append(s)
        run = json.loads((Path(d) / "validation_run.json").read_text())
        for scope, per in run["shallow_pr"].items():
            for prod, v in per.items():
                for thr, st in v.items():
                    prs.append(
                        {
                            "panel": panel,
                            "scope": scope,
                            "product": NAME[prod],
                            "threshold": thr,
                            **st,
                        }
                    )

    df = pd.concat(frames, ignore_index=True).rename(columns={"predictor": "product"})
    df["product"] = df["product"].map(NAME)
    df["group"] = (
        df["group"]
        .str.replace("-infm", "+m", regex=False)
        .str.replace("-infkm", "+km", regex=False)
    )
    for c in (
        "mad_m",
        "bias_m",
        "median_residual_m",
        "rmse_m",
        "p90_abs_err_m",
        "p95_abs_err_m",
    ):
        df[c] = df[c].round(2)
    pr = pd.DataFrame(prs)

    n_a = int(df[(df.panel == "A_statewide") & (df.group_type == "all")]["n"].iloc[0])
    n_b = int(
        df[(df.panel == "B_facrem_footprint") & (df.group_type == "all")]["n"].iloc[0]
    )

    wb = Workbook()
    wb.remove(wb.active)
    sheet_guide(wb, args.state, n_a, n_b)
    sheet_depth_bands(wb, df, n_a, n_b)
    sheet_shallow(wb, pr)
    sheet_where(wb, df)
    sheet_full(wb, df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)

    # Plain-text twins of the same numbers, so the product naming and the panel
    # definitions come from one place rather than an ad-hoc side script.
    csv_panel = out.with_suffix(".csv")
    csv_pr = out.parent / f"{args.state.lower()}_shallow_detection.csv"
    df.to_csv(csv_panel, index=False)
    pr.to_csv(csv_pr, index=False)
    log.info(
        "wrote %s + %s + %s (panel A n=%d, panel B n=%d)",
        out.name,
        csv_panel.name,
        csv_pr.name,
        n_a,
        n_b,
    )


if __name__ == "__main__":
    main()
