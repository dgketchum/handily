#!/usr/bin/env python3
"""Write a GIS_DATA.txt manifest of remote raster source paths and local destinations.

Each non-comment line in the output manifest is tab-separated:

    <remote absolute source path>    <local relative destination path>

The manifest is intended to be generated on the remote handily machine, then consumed by
the local sync script in this repo.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


STATE_CHOICES = ("mt", "nm", "nv")
PRODUCT_CHOICES = ("rem", "streams", "ndwi", "points")
DEFAULT_SOURCE_ROOT = Path("/data/ssd2/handily")
DEFAULT_DEST_PREFIX = {
    "mt": "handily/rem/mt",
    "nm": "handily/nm",
    "nv": "handily/nv",
}
NM_AOI_LIST_RE = re.compile(r"\.isin\(\[(?P<body>.*?)\]\)", re.DOTALL)
AOI_DIR_RE = re.compile(r"^aoi_(\d{4})$")
AOI_IDS_SECTION_RE = re.compile(r"--aoi-ids\s+(?P<body>.+?)(?:\s+>\s|$)")


@dataclass(frozen=True)
class ManifestEntry:
    state: str
    aoi: str
    product: str
    source: Path
    destination: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GIS_DATA.txt from repo-known AOI selections and/or discovered "
            "remote AOI directories."
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("GIS_DATA.txt"),
        help="Manifest output path. Default: GIS_DATA.txt",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Remote handily data root. Default: /data/ssd2/handily",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=STATE_CHOICES,
        default=list(STATE_CHOICES),
        help="States to include. Default: mt nm nv",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=PRODUCT_CHOICES,
        default=list(PRODUCT_CHOICES),
        help="Products to include. Default: rem streams ndwi",
    )
    parser.add_argument(
        "--state-aois",
        action="append",
        default=[],
        metavar="STATE:ID,ID,...",
        help="Explicit AOI IDs for a state, e.g. mt:7,8,9. Repeat as needed.",
    )
    parser.add_argument(
        "--dest-prefix",
        action="append",
        default=[],
        metavar="STATE=REL_PATH",
        help="Override destination prefix, e.g. mt=handily/rem/mt",
    )
    parser.add_argument(
        "--commands-file",
        type=Path,
        default=Path("commands.sh"),
        help="Repo script used to infer AOI lists when available. Default: commands.sh",
    )
    parser.add_argument(
        "--nm-selection-file",
        type=Path,
        default=Path("zh_commands.sh"),
        help="Repo script containing the NM pilot AOI list. Default: zh_commands.sh",
    )
    parser.add_argument(
        "--no-auto-repo-selections",
        action="store_true",
        help="Do not infer AOI selections from repo scripts; only use explicit AOIs or discovery.",
    )
    parser.add_argument(
        "--discover-mode",
        choices=("any", "all"),
        default="any",
        help=(
            "For states without explicit/repo AOI lists, discover AOIs with any requested "
            "product or all requested products. Default: any"
        ),
    )
    args = parser.parse_args()
    args.out = args.out.expanduser().resolve()
    args.source_root = args.source_root.expanduser()
    args.commands_file = args.commands_file.expanduser().resolve()
    args.nm_selection_file = args.nm_selection_file.expanduser().resolve()
    return args


def canonical_aoi(value: int | str) -> str:
    return f"{int(value):04d}"


def parse_state_aois(values: list[str]) -> dict[str, tuple[str, ...]]:
    out: dict[str, set[str]] = defaultdict(set)
    for raw in values:
        if ":" not in raw:
            raise ValueError(f"Invalid --state-aois value: {raw}")
        state, body = raw.split(":", 1)
        state = state.strip().lower()
        if state not in STATE_CHOICES:
            raise ValueError(f"Invalid state in --state-aois: {state}")
        for token in re.split(r"[,\s]+", body.strip()):
            if token:
                out[state].add(canonical_aoi(token))
    return {state: tuple(sorted(aois)) for state, aois in out.items()}


def parse_dest_prefix_overrides(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --dest-prefix value: {raw}")
        state, rel_path = raw.split("=", 1)
        state = state.strip().lower()
        if state not in STATE_CHOICES:
            raise ValueError(f"Invalid state in --dest-prefix: {state}")
        rel_path = rel_path.strip().strip("/")
        if not rel_path:
            raise ValueError(f"Empty destination prefix in --dest-prefix: {raw}")
        out[state] = rel_path
    return out


def parse_nm_repo_selection(selection_file: Path) -> tuple[str, ...] | None:
    if not selection_file.exists():
        return None
    text = selection_file.read_text(encoding="utf-8")
    match = NM_AOI_LIST_RE.search(text)
    if not match:
        return None
    values = [canonical_aoi(token) for token in re.findall(r"\d+", match.group("body"))]
    return tuple(sorted(set(values))) if values else None


def parse_commands_repo_selection(
    commands_file: Path, state: str
) -> tuple[str, ...] | None:
    if not commands_file.exists():
        return None
    state_markers = (f"/handily/{state}/", f"configs/{state}_", f"{state}_aois.shp")
    for line in commands_file.read_text(encoding="utf-8").splitlines():
        if "--aoi-ids" not in line:
            continue
        if not any(marker in line for marker in state_markers):
            continue
        match = AOI_IDS_SECTION_RE.search(line)
        if not match:
            continue
        values = [
            canonical_aoi(token) for token in re.findall(r"\d+", match.group("body"))
        ]
        if values:
            return tuple(sorted(set(values)))
    return None


def product_filename(product: str, aoi: str) -> str:
    if product == "rem":
        return "rem_bounds.tif"
    if product == "streams":
        return "streams_bounds.tif"
    if product == "ndwi":
        return f"naip_ndwi_aoi_{aoi}.tif"
    if product == "points":
        return "points/donor_review.gpkg"
    raise ValueError(f"Unsupported product: {product}")


def has_requested_products(
    aoi_dir: Path, aoi: str, products: tuple[str, ...], mode: str
) -> bool:
    present = [
        (aoi_dir / product_filename(product, aoi)).exists() for product in products
    ]
    return all(present) if mode == "all" else any(present)


def discover_aois(
    source_root: Path,
    state: str,
    products: tuple[str, ...],
    mode: str,
) -> tuple[str, ...]:
    state_dir = source_root / state
    if not state_dir.is_dir():
        return ()
    aois: list[str] = []
    for child in sorted(state_dir.iterdir()):
        if not child.is_dir():
            continue
        match = AOI_DIR_RE.match(child.name)
        if not match:
            continue
        aoi = match.group(1)
        if has_requested_products(child, aoi, products, mode):
            aois.append(aoi)
    return tuple(aois)


def resolve_state_selection(
    args: argparse.Namespace,
    explicit: dict[str, tuple[str, ...]],
    products: tuple[str, ...],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    selected: dict[str, tuple[str, ...]] = {}
    sources: dict[str, str] = {}

    for state in args.states:
        if state in explicit:
            selected[state] = explicit[state]
            sources[state] = "cli"
            continue

        discovered = discover_aois(
            args.source_root, state, products, args.discover_mode
        )

        if not args.no_auto_repo_selections:
            repo_aois: tuple[str, ...] | None = None
            if state == "nm":
                repo_aois = parse_nm_repo_selection(args.nm_selection_file)
                repo_source = str(args.nm_selection_file)
            else:
                repo_source = str(args.commands_file)
            if repo_aois is None:
                repo_aois = parse_commands_repo_selection(args.commands_file, state)
                repo_source = str(args.commands_file)

            if repo_aois and discovered:
                selected[state] = tuple(sorted(set(repo_aois).union(discovered)))
                sources[state] = f"{repo_source}+discover:{args.discover_mode}"
                continue
            if repo_aois:
                selected[state] = repo_aois
                sources[state] = repo_source
                continue

        selected[state] = discovered
        sources[state] = f"discover:{args.discover_mode}"

    return selected, sources


def build_entries(
    source_root: Path,
    state_aois: dict[str, tuple[str, ...]],
    dest_prefixes: dict[str, str],
    products: tuple[str, ...],
) -> tuple[list[ManifestEntry], dict[str, dict[str, list[str]]]]:
    entries: list[ManifestEntry] = []
    missing: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for state in sorted(state_aois):
        dest_prefix = Path(dest_prefixes[state])
        for aoi in state_aois[state]:
            source_dir = source_root / state / f"aoi_{aoi}"
            for product in products:
                filename = product_filename(product, aoi)
                source_path = source_dir / filename
                if not source_path.exists():
                    missing[state][aoi].append(product)
                    continue
                destination = dest_prefix / f"aoi_{aoi}" / filename
                entries.append(
                    ManifestEntry(
                        state=state,
                        aoi=aoi,
                        product=product,
                        source=source_path,
                        destination=destination,
                    )
                )
    entries.sort(
        key=lambda item: (item.state, item.aoi, PRODUCT_CHOICES.index(item.product))
    )
    return entries, missing


def write_manifest(
    out_path: Path,
    entries: list[ManifestEntry],
    state_aois: dict[str, tuple[str, ...]],
    selection_sources: dict[str, str],
    products: tuple[str, ...],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# GIS_DATA manifest\n")
        f.write("# columns: source_abs_path<TAB>destination_rel_path\n")
        f.write(f"# products: {', '.join(products)}\n")
        for state in sorted(state_aois):
            aois = ", ".join(state_aois[state]) if state_aois[state] else "(none)"
            f.write(
                f"# state={state} source={selection_sources[state]} "
                f"aoi_count={len(state_aois[state])} aois={aois}\n"
            )
        for entry in entries:
            f.write(f"{entry.source}\t{entry.destination.as_posix()}\n")


def print_summary(
    entries: list[ManifestEntry],
    state_aois: dict[str, tuple[str, ...]],
    selection_sources: dict[str, str],
    missing: dict[str, dict[str, list[str]]],
    out_path: Path,
) -> None:
    print(f"manifest: {out_path}")
    print(f"entries: {len(entries)}")
    print()

    entries_by_state: dict[str, int] = defaultdict(int)
    for entry in entries:
        entries_by_state[entry.state] += 1

    for state in sorted(state_aois):
        selected_count = len(state_aois[state])
        written_count = entries_by_state.get(state, 0)
        missing_count = sum(
            len(products) for products in missing.get(state, {}).values()
        )
        print(
            f"{state}: selected_aois={selected_count} entries={written_count} "
            f"selection_source={selection_sources[state]} missing_products={missing_count}"
        )


def main() -> int:
    args = parse_args()
    products = tuple(args.products)

    try:
        explicit = parse_state_aois(args.state_aois)
        dest_prefixes = dict(DEFAULT_DEST_PREFIX)
        dest_prefixes.update(parse_dest_prefix_overrides(args.dest_prefix))
        state_aois, selection_sources = resolve_state_selection(
            args, explicit, products
        )
        entries, missing = build_entries(
            args.source_root, state_aois, dest_prefixes, products
        )
        write_manifest(args.out, entries, state_aois, selection_sources, products)
        print_summary(entries, state_aois, selection_sources, missing, args.out)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
