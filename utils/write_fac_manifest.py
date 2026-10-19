"""Write (or subset) a FAC-REM raster manifest.

A manifest is the explicit, ordered list of FAC-REM rasters an arm's FAC feature
is sampled from (see ``fac_rem_registry``). Bundle builders write one themselves;
this tool covers the other cases:

  * a manifest for an arm trained before manifests existed (pass to
    ``--fac-manifest`` of infer / render / predict):
        write_fac_manifest.py --recipe scalable --legacy-pilots --out <path>
  * a mixed-recipe manifest for a reach graph or bundle build (variant where it
    was built, baseline elsewhere; every entry records its recipe):
        write_fac_manifest.py --recipe scalable_v2 --fallback-recipe scalable --out <path>
  * the subset of an arm's manifest that a render of some basins needs (what to
    stage on another machine; the subset verifies there with the same tool):
        write_fac_manifest.py --from <arm>/fac_rem_manifest.json --basins nm97.txt --out <subset>
  * verifying a machine holds every raster of a manifest, byte-exact:
        write_fac_manifest.py --from <manifest> --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fac_rem_registry import (  # noqa: E402
    _SCALABLE_ROOT,
    count_by_recipe,
    discover,
    load_manifest,
    select_paths,
    write_manifest,
)


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--recipe", help="discover rasters of this recipe under huc8/")
    src.add_argument("--from", dest="src", help="start from an existing manifest")
    ap.add_argument("--out", help="manifest to write")
    ap.add_argument(
        "--fallback-recipe",
        default=None,
        help="with --recipe: basins lacking the recipe are listed on this one instead",
    )
    ap.add_argument(
        "--legacy-pilots",
        action="store_true",
        help="with --recipe: list the named pilots first (pre-2026-10 arm sourcing)",
    )
    ap.add_argument(
        "--basins",
        help="text file of HUC8s: keep only entries whose bounds touch one of their "
        "lattice bboxes (+ --buffer-m)",
    )
    ap.add_argument("--buffer-m", type=float, default=1000.0)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check every (selected) raster exists with the recorded size; exit 1 if not",
    )
    ap.add_argument("--note", default="")
    return ap.parse_args()


def main() -> None:
    args = parse()
    if args.recipe:
        entries = discover(
            args.recipe,
            fallback=args.fallback_recipe,
            legacy_pilots=args.legacy_pilots,
        )
        recipe, fallback = args.recipe, args.fallback_recipe
    else:
        doc = load_manifest(args.src)
        entries, recipe = doc["entries"], doc["recipe"]
        fallback = doc.get("fallback_recipe")

    if args.basins:
        targets = [t for t in Path(args.basins).read_text().split() if t]
        boxes = []
        for b in targets:
            bnd = gpd.read_file(f"{_SCALABLE_ROOT}/{b}/basin_boundary.fgb").to_crs(5070)
            x0, y0, x1, y1 = bnd.total_bounds
            m = args.buffer_m
            boxes.append((x0 - m, y0 - m, x1 + m, y1 + m))
        keep = []
        for e in entries:
            left, bottom, right, top = e["bounds"]
            if any(
                left < x1 and right > x0 and bottom < y1 and top > y0
                for x0, y0, x1, y1 in boxes
            ):
                keep.append(e)
        have = {e["basin"] for e in keep}
        missing = [t for t in targets if t not in have]
        if missing:
            raise SystemExit(
                f"{len(missing)} target basins have no raster in the source: {missing}"
            )
        extras = [e["basin"] for e in keep if e["basin"] not in set(targets)]
        print(
            f"subset for {len(targets)} targets: {len(keep)} entries "
            f"({len(extras)} overlapping neighbours: {' '.join(extras)})"
        )
        entries = keep

    if args.verify:
        try:
            select_paths(entries, strict=True)
        except SystemExit as e:
            print(e)
            raise SystemExit(1) from None
        print(f"verified {len(entries)} rasters present with recorded sizes")

    if args.out:
        write_manifest(
            args.out, entries, recipe, note=args.note, fallback_recipe=fallback
        )
        gb = sum(e["bytes"] for e in entries) / 1e9
        print(
            f"wrote {args.out}: {len(entries)} rasters {count_by_recipe(entries)}, {gb:.1f} GB"
        )
    elif not args.verify:
        raise SystemExit("pass --out and/or --verify")


if __name__ == "__main__":
    main()
