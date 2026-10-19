"""Write a FAC-REM registry pin for a render job.

The registry (``fac_rem_registry``) is auto-discovered from disk, so a render on a
machine with a different basin set staged samples a different raster wherever the
watershed-clipped FAC rasters overlap (first finite sample wins). This writes the
exact ordered set of registry entries that can touch a target basin set -- every
discovered raster whose bounds intersect any target basin's lattice bbox (+ buffer)
-- as a manifest ``basin<TAB>recipe<TAB>bytes<TAB>path``. Set
``HANDILY_FAC_PIN=<manifest>`` on both machines (with the same
``HANDILY_FAC_RECIPE``) and stage exactly those rasters: the registry is then
identical by construction and import fails if it is not.

    HANDILY_FAC_RECIPE=scalable_v2 uv run python utils/pin_fac_registry.py \
        --basins nm_huc8_97.txt --out fac_pin_nm97_scalable_v2.tsv
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import geopandas as gpd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fac_rem_registry import (  # noqa: E402
    _SCALABLE_ROOT,
    FAC_RECIPE,
    FAC_REM_REGISTRY,
    _recipe_of,
)


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--basins", required=True, help="text file, one target HUC8 per line"
    )
    ap.add_argument("--out", required=True, help="manifest path (.tsv)")
    ap.add_argument(
        "--buffer-m",
        type=float,
        default=1000.0,
        help="pad on each target basin bbox before the bounds-intersection test",
    )
    return ap.parse_args()


def main() -> None:
    args = parse()
    targets = [t for t in Path(args.basins).read_text().split() if t]
    boxes = []
    for b in targets:
        bnd = gpd.read_file(f"{_SCALABLE_ROOT}/{b}/basin_boundary.fgb").to_crs(5070)
        x0, y0, x1, y1 = bnd.total_bounds
        boxes.append(
            (
                x0 - args.buffer_m,
                y0 - args.buffer_m,
                x1 + args.buffer_m,
                y1 + args.buffer_m,
            )
        )

    kept = []
    for p in FAC_REM_REGISTRY:
        with rasterio.open(p) as s:
            bb = s.bounds
        hit = any(
            bb.left < x1 and bb.right > x0 and bb.bottom < y1 and bb.top > y0
            for x0, y0, x1, y1 in boxes
        )
        if hit:
            basin, recipe = _recipe_of(p)
            kept.append((basin, recipe, Path(p).stat().st_size, p))

    missing = sorted(set(targets) - {k[0] for k in kept})
    if missing:
        raise SystemExit(
            f"{len(missing)} target basins have no FAC-REM raster in the registry: {missing}"
        )
    extras = [k[0] for k in kept if k[0] not in set(targets)]
    not_on_recipe = [k[0] for k in kept if k[1] != FAC_RECIPE]
    lines = [
        f"# FAC-REM registry pin written {dt.date.today().isoformat()} by utils/pin_fac_registry.py",
        f"# HANDILY_FAC_RECIPE={FAC_RECIPE}  targets={args.basins} ({len(targets)})  buffer_m={args.buffer_m:g}",
        f"# {len(kept)} entries in registry order: {len(targets)} targets + {len(extras)} overlapping neighbours; "
        f"{len(not_on_recipe)} resolve to a non-{FAC_RECIPE} recipe",
        "# basin\trecipe\tbytes\tpath",
    ]
    lines += [f"{b}\t{r}\t{n}\t{p}" for b, r, n, p in kept]
    Path(args.out).write_text("\n".join(lines) + "\n")
    gb = sum(k[2] for k in kept) / 1e9
    print(
        f"pin -> {args.out}: {len(kept)} rasters ({gb:.1f} GB); targets {len(targets)}, neighbours {len(extras)}"
    )
    print(f"  neighbours: {' '.join(extras)}")
    print(
        f"  not on recipe {FAC_RECIPE} ({len(not_on_recipe)}): {' '.join(not_on_recipe)}"
    )


if __name__ == "__main__":
    main()
