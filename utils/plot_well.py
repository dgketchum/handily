#!/usr/bin/env python
"""Plot annual depth-to-water time series for a single well as stacked panels.

Usage:
    uv run python utils/plot_well.py --file /nas/gwx/products/wells/nwis_USGS_453404113272601.parquet
    uv run python utils/plot_well.py --file /nas/gwx/products/wells/nwis_USGS_453404113272601.parquet --out well.pdf
"""

import argparse
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WELLS_META = "/nas/gwx/products/wells.geoparquet"


def _load_metadata(parquet_path):
    """Load station metadata from wells.geoparquet by matching file_path."""
    if not os.path.exists(WELLS_META):
        return {}
    meta = pd.read_parquet(
        WELLS_META,
        columns=[
            "canonical_id",
            "source",
            "h_accuracy_class",
            "mean_dtw",
            "obs_count",
            "file_path",
        ],
    )
    basename = os.path.basename(parquet_path)
    row = meta[meta["file_path"].str.contains(basename, na=False)]
    if len(row) == 0:
        return {}
    r = row.iloc[0]
    return {
        "canonical_id": r["canonical_id"],
        "source": r["source"],
        "h_acc_cls": r["h_accuracy_class"],
        "mean_dtw": r["mean_dtw"],
        "obs_count": int(r["obs_count"]),
    }


def plot_well(parquet_path, out_path=None, note=None):
    df = pd.read_parquet(parquet_path)

    if "dtime" not in df.columns or "dtw" not in df.columns:
        print(f"Missing dtime/dtw columns in {parquet_path}", file=sys.stderr)
        sys.exit(1)

    df = df.dropna(subset=["dtime", "dtw"]).copy()
    df["dtime"] = pd.to_datetime(df["dtime"], utc=True)
    df = df.sort_values("dtime").reset_index(drop=True)

    if len(df) == 0:
        print(f"No valid data in {parquet_path}", file=sys.stderr)
        sys.exit(1)

    # Station name from filename
    station_name = os.path.splitext(os.path.basename(parquet_path))[0]

    # Metadata
    meta = _load_metadata(parquet_path)

    # Group by year
    df["year"] = df["dtime"].dt.year
    years = sorted(df["year"].unique())

    # Filter years with no data
    years = [y for y in years if len(df[df["year"] == y]) > 0]
    if not years:
        print("No data years to plot", file=sys.stderr)
        sys.exit(1)

    n_years = len(years)
    fig_height = max(4, 1.8 * n_years)
    fig, axes = plt.subplots(n_years, 1, figsize=(10, fig_height), squeeze=False)
    axes = axes.ravel()

    # Global y-axis limits
    dtw_min = df["dtw"].min()
    dtw_max = df["dtw"].max()
    y_pad = (dtw_max - dtw_min) * 0.1 if dtw_max > dtw_min else 1.0
    ylim = (
        dtw_max + y_pad,
        dtw_min - y_pad,
    )  # Inverted: deeper water = higher value at bottom

    for i, year in enumerate(years):
        ax = axes[i]
        ydf = df[df["year"] == year].copy()
        ydf = ydf.sort_values("dtime")

        # Day-of-year dates mapped to a common reference year for x-axis
        ref_year = 2000  # arbitrary leap year for alignment
        ydf["plot_date"] = ydf["dtime"].apply(
            lambda dt: dt.replace(year=ref_year, tzinfo=None)
        )

        dates = ydf["plot_date"].values
        dtw_vals = ydf["dtw"].values

        # Plot points
        ax.scatter(
            dates, dtw_vals, s=12, color="steelblue", zorder=3, edgecolors="none"
        )

        # Connect points that are within 31 days of each other
        if len(dates) > 1:
            diffs = np.diff(dates).astype("timedelta64[D]").astype(float)
            seg_x, seg_y = [], []
            for j in range(len(dates)):
                seg_x.append(dates[j])
                seg_y.append(dtw_vals[j])
                if j < len(diffs) and diffs[j] > 31:
                    # Gap > 1 month: draw current segment, start new
                    if len(seg_x) > 1:
                        ax.plot(
                            seg_x, seg_y, color="steelblue", linewidth=0.8, zorder=2
                        )
                    seg_x, seg_y = [], []
            if len(seg_x) > 1:
                ax.plot(seg_x, seg_y, color="steelblue", linewidth=0.8, zorder=2)

        ax.set_ylim(ylim)
        ax.set_xlim(
            pd.Timestamp(f"{ref_year}-01-01"),
            pd.Timestamp(f"{ref_year}-12-31"),
        )
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.set_ylabel(f"{year}", fontsize=9, rotation=0, labelpad=35, va="center")
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.3)

        if i < n_years - 1:
            ax.set_xticklabels([])

    # Title
    fig.suptitle(station_name, fontsize=11, fontweight="bold", y=0.995)

    # Metadata annotation
    if meta:
        ann_lines = []
        if note:
            ann_lines.append(note)
        ann_lines += [
            f"n={meta.get('obs_count', len(df))}",
            f"mean DTW={meta.get('mean_dtw', df['dtw'].mean()):.2f} m",
            f"h_acc: {meta.get('h_acc_cls', 'unknown')}",
            f"source: {meta.get('source', 'unknown')}",
        ]
        fig.text(
            0.98,
            0.98,
            "\n".join(ann_lines),
            transform=fig.transFigure,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

    axes[-1].set_xlabel("Month", fontsize=9)
    fig.supylabel("Depth to Water (m)", fontsize=10, x=0.01)

    fig.tight_layout(rect=[0.05, 0.02, 0.95, 0.97])

    if out_path is None:
        out_path = os.path.splitext(parquet_path)[0] + ".pdf"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot annual well DTW time series")
    parser.add_argument("--file", required=True, help="Path to well parquet file")
    parser.add_argument(
        "--wells-dir",
        default="/nas/gwx/products/wells",
        help="Wells directory (default /nas/gwx/products/wells)",
    )
    parser.add_argument(
        "--plots-dir",
        default="/nas/gwx/products/wells/plots",
        help="Output plots directory (default /nas/gwx/products/wells/plots)",
    )
    parser.add_argument(
        "--out", default=None, help="Output PDF path (overrides --plots-dir)"
    )
    parser.add_argument(
        "--note", default=None, help="Annotation note printed above obs count"
    )
    args = parser.parse_args()
    out = args.out
    if out is None:
        os.makedirs(args.plots_dir, exist_ok=True)
        out = os.path.join(
            args.plots_dir, os.path.splitext(os.path.basename(args.file))[0] + ".pdf"
        )
    plot_well(args.file, out, note=args.note)


if __name__ == "__main__":
    main()
