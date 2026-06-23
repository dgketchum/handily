"""Build water-table ELEVATION drainage anchors from Hydrography90m + the CONUS DEM.

Step 3 of the WTE-elevation-surface pipeline (see notes/WTE_ELEVATION_SURFACE.md).
At a perennial stream/river the water table daylights at the land surface, so the
land-surface elevation along the channel network is a Dirichlet anchor (boundary
condition) for the smooth regional head solve (step 4).

Given two grid-aligned 100 m EPSG:5070 rasters:
  - accumulation_conus_100m_5070.tif  (Hydrography90m flow accumulation, signed
    upstream *cell* count; nodata -9999999; negatives = inter-region inflow)
  - elev48i0100a.tif                  (USGS 100 m land surface, m, nodata 32767)

produce:
  - anchor_head_100m_5070.tif   float32, land-surface elevation on channel cells,
                                out-nodata off-channel.
  - channel_mask_100m_5070.tif  uint8, 1=channel / 0=off-channel / 255=nodata.

Channel = valid & |accumulation| >= threshold. The magnitude (abs) bridges the
Hydrography90m 20-degree tile seams, where a river continuing downstream of a
regional boundary is encoded with a large *negative* accumulation. nodata is
excluded BEFORE thresholding -- |(-9999999)| would otherwise pass any threshold.

A single global drainage threshold over-anchors ephemeral washes in arid basins
(where the regional table is deep, not at the wash) and under-anchors in humid
terrain. That climate-dependence is a step-4 tuning concern; here the threshold is
a parameter and the run reports the network size at a ladder of candidate
thresholds (one pass) so it can be retuned without recomputing.

Run:
  uv run python utils/build_wte_anchors.py --min-drainage-km2 25
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio

log = logging.getLogger("build_wte_anchors")

HYDRO_DIR = "/data/ssd2/handily/conus/hydrography90m"
ACC_PATH = f"{HYDRO_DIR}/accumulation_conus_100m_5070.tif"
ELEV_PATH = "/data/ssd1/streamflow-ml-data/conus-dem/data/elev48i0100a.tif"

# Hydrography90m accumulation is an upstream *cell* count on the native 3 arc-sec
# (~90 m) grid. Nominal cell area = (90 m)^2 = 0.0081 km^2. This is the equatorial
# nominal; true 3 arc-sec cells shrink with cos(lat), so at CONUS latitudes the
# real contributing area per count is ~10-30% smaller -- a km2 threshold thus
# anchors marginally more (smaller true area) toward the north. Acceptable for a
# regional network; documented rather than silently corrected.
CELL_AREA_KM2 = 0.0081

# Candidate thresholds (km2 upstream drainage) reported every run for retuning.
CANDIDATE_KM2 = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0)

OUT_NODATA = -9999.0
MASK_NODATA = 255


def compute_anchor_head(
    acc: np.ndarray,
    elev: np.ndarray,
    *,
    acc_nodata: float,
    elev_nodata: float,
    threshold_cells: float,
    out_nodata: float = OUT_NODATA,
) -> tuple[np.ndarray, np.ndarray]:
    """Channel = valid & |acc| >= threshold_cells; head = land-surface elev there.

    nodata is masked out BEFORE the magnitude test so the -9999999 accumulation
    sentinel (|.| huge) cannot masquerade as a giant river. Returns
    (anchor_head float32 with *out_nodata* off-channel,
     channel_mask uint8: 1 channel / 0 off-channel / 255 input-nodata).
    """
    acc = acc.astype(np.float64, copy=False)
    valid = (acc != acc_nodata) & (elev != elev_nodata)
    channel = valid & (np.abs(acc) >= threshold_cells)

    head = np.full(acc.shape, out_nodata, dtype=np.float32)
    head[channel] = elev[channel].astype(np.float32)

    mask = np.where(valid, channel.astype(np.uint8), np.uint8(MASK_NODATA))
    return head, mask.astype(np.uint8)


def count_channel_cells(
    acc: np.ndarray,
    valid: np.ndarray,
    threshold_cells_list: list[float],
) -> np.ndarray:
    """Per-threshold count of channel cells in this block (for the report ladder)."""
    mag = np.abs(acc.astype(np.float64))
    return np.array([int((valid & (mag >= t)).sum()) for t in threshold_cells_list])


def _assert_aligned(a: rasterio.DatasetReader, b: rasterio.DatasetReader) -> None:
    if (a.width, a.height) != (b.width, b.height):
        raise SystemExit(f"shape mismatch: {a.name} {a.shape} vs {b.name} {b.shape}")
    if not a.transform.almost_equals(b.transform, precision=1e-6):
        raise SystemExit(f"transform mismatch: {a.name} vs {b.name}")
    if a.crs != b.crs:
        raise SystemExit(f"CRS mismatch: {a.crs} vs {b.crs}")


def build(
    acc_path: str,
    elev_path: str,
    out_dir: str,
    min_drainage_km2: float,
    stripe_rows: int = 2048,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head_path = out / "anchor_head_100m_5070.tif"
    mask_path = out / "channel_mask_100m_5070.tif"

    threshold_cells = min_drainage_km2 / CELL_AREA_KM2
    candidate_cells = [k / CELL_AREA_KM2 for k in CANDIDATE_KM2]
    log.info(
        "threshold %.1f km2 -> |accumulation| >= %.0f cells (nominal %.4f km2/cell)",
        min_drainage_km2,
        threshold_cells,
        CELL_AREA_KM2,
    )

    with rasterio.open(acc_path) as acc_src, rasterio.open(elev_path) as elev_src:
        _assert_aligned(acc_src, elev_src)
        acc_nodata = acc_src.nodata
        elev_nodata = elev_src.nodata
        if acc_nodata is None or elev_nodata is None:
            raise SystemExit(
                f"missing nodata: acc={acc_nodata} elev={elev_nodata} "
                "(refusing to guess -- inspect the inputs)"
            )

        height, width = acc_src.height, acc_src.width
        head_profile = acc_src.profile | dict(
            dtype="float32",
            nodata=OUT_NODATA,
            count=1,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            bigtiff="yes",
        )
        mask_profile = acc_src.profile | dict(
            dtype="uint8",
            nodata=MASK_NODATA,
            count=1,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            bigtiff="yes",
        )

        ladder = np.zeros(len(candidate_cells), dtype=np.int64)
        n_channel = 0
        n_valid = 0

        with (
            rasterio.open(head_path, "w", **head_profile) as head_dst,
            rasterio.open(mask_path, "w", **mask_profile) as mask_dst,
        ):
            for r0 in range(0, height, stripe_rows):
                r1 = min(r0 + stripe_rows, height)
                win = ((r0, r1), (0, width))
                acc = acc_src.read(1, window=win)
                elev = elev_src.read(1, window=win)

                head, mask = compute_anchor_head(
                    acc,
                    elev,
                    acc_nodata=acc_nodata,
                    elev_nodata=elev_nodata,
                    threshold_cells=threshold_cells,
                )
                head_dst.write(head, 1, window=win)
                mask_dst.write(mask, 1, window=win)

                valid = (acc.astype(np.float64) != acc_nodata) & (elev != elev_nodata)
                ladder += count_channel_cells(acc, valid, candidate_cells)
                n_channel += int((mask == 1).sum())
                n_valid += int(valid.sum())
                log.info("rows %d-%d / %d", r0, r1, height)

    log.info("=== anchor network ===")
    log.info(
        "valid cells: %d   channel cells: %d (%.3f%%)",
        n_valid,
        n_channel,
        100.0 * n_channel / max(n_valid, 1),
    )
    log.info("approx network length: %.0f km (cells x 100 m)", n_channel * 100 / 1000)
    log.info("--- candidate threshold ladder (one pass) ---")
    for km2, cells, n in zip(CANDIDATE_KM2, candidate_cells, ladder):
        log.info(
            "  >= %6.0f km2 (%8.0f cells): %12d channel cells (~%.0f km)",
            km2,
            cells,
            n,
            n * 100 / 1000,
        )
    log.info("wrote %s", head_path)
    log.info("wrote %s", mask_path)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acc", default=ACC_PATH, help="flow accumulation raster")
    p.add_argument("--elev", default=ELEV_PATH, help="land-surface DEM raster")
    p.add_argument("--out-dir", default=HYDRO_DIR)
    p.add_argument("--min-drainage-km2", type=float, default=25.0)
    p.add_argument("--stripe-rows", type=int, default=2048)
    args = p.parse_args(argv)
    build(args.acc, args.elev, args.out_dir, args.min_drainage_km2, args.stripe_rows)


if __name__ == "__main__":
    main()
