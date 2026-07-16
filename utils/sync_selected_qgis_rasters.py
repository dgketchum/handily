#!/usr/bin/env python3
"""Sync GIS rasters from a GIS_DATA.txt manifest.

Each non-comment line in the manifest must be tab-separated:

    <remote absolute source path>    <local relative destination path>
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


STATE_CHOICES = ("mt", "nm", "nv")
PRODUCT_CHOICES = ("rem", "streams", "ndwi", "points")
MANIFEST_LINE_RE = re.compile(r"^(?P<source>/\S+)\t(?P<destination>\S+)$")
DEST_PATH_RE = re.compile(
    r"^(?P<prefix>.+)/aoi_(?P<aoi>\d{4})/"
    r"(?P<filename>rem_bounds\.tif|streams_bounds\.tif|naip_ndwi_aoi_\d{4}\.tif"
    r"|points/donor_review\.gpkg)$"
)


@dataclass(frozen=True)
class ManifestEntry:
    source: Path
    destination: Path
    state: str
    aoi: str
    product: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync REM, streams, and NDWI rasters using a GIS_DATA.txt manifest."
    )
    parser.add_argument(
        "--remote",
        default=os.environ.get("HANDILY_REMOTE"),
        help="SSH target for rsync, e.g. user@host. Can also be set via HANDILY_REMOTE.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("GIS_DATA.txt"),
        help="Local GIS_DATA manifest path. Default: GIS_DATA.txt",
    )
    parser.add_argument(
        "--fetch-manifest",
        action="store_true",
        help="Fetch the remote manifest to --manifest before reading it.",
    )
    parser.add_argument(
        "--remote-manifest",
        default=None,
        help="Remote manifest path used with --fetch-manifest.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("~/data"),
        help="Local root for manifest destination paths. Default: ~/data",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        choices=STATE_CHOICES,
        default=list(STATE_CHOICES),
        help="Filter to these states. Default: mt nm nv",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        choices=PRODUCT_CHOICES,
        default=list(PRODUCT_CHOICES),
        help="Filter to these products. Default: rem streams ndwi",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show rsync actions without copying raster files.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print the grouped rsync commands and stop.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each rsync command before executing it.",
    )
    args = parser.parse_args()

    if not args.remote:
        parser.error("--remote is required unless HANDILY_REMOTE is set")
    if args.fetch_manifest and not args.remote_manifest:
        parser.error("--remote-manifest is required with --fetch-manifest")

    args.manifest = args.manifest.expanduser().resolve()
    args.local_root = args.local_root.expanduser().resolve()
    return args


def fetch_manifest(remote: str, remote_manifest: str, local_manifest: Path) -> None:
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-av", f"{remote}:{remote_manifest}", str(local_manifest)]
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"Failed to fetch manifest from {remote}:{remote_manifest}")


def infer_state(prefix: str) -> str:
    for state in STATE_CHOICES:
        if prefix.endswith(f"/{state}"):
            return state
    raise ValueError(f"Could not infer state from destination prefix: {prefix}")


def infer_product(filename: str) -> str:
    if filename == "rem_bounds.tif":
        return "rem"
    if filename == "streams_bounds.tif":
        return "streams"
    if filename.startswith("naip_ndwi_aoi_") and filename.endswith(".tif"):
        return "ndwi"
    if filename == "points/donor_review.gpkg":
        return "points"
    raise ValueError(f"Unrecognized product filename: {filename}")


def read_manifest(manifest_path: Path) -> list[ManifestEntry]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries: list[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            match = MANIFEST_LINE_RE.match(line)
            if not match:
                raise ValueError(f"Invalid manifest line: {line}")

            source = Path(match.group("source"))
            destination = Path(match.group("destination"))

            dest_match = DEST_PATH_RE.match(destination.as_posix())
            if not dest_match:
                raise ValueError(f"Invalid destination path in manifest: {destination}")

            prefix = dest_match.group("prefix")
            filename = dest_match.group("filename")
            entries.append(
                ManifestEntry(
                    source=source,
                    destination=destination,
                    state=infer_state(prefix),
                    aoi=dest_match.group("aoi"),
                    product=infer_product(filename),
                )
            )

    return entries


def filter_entries(
    entries: list[ManifestEntry], states: set[str], products: set[str]
) -> list[ManifestEntry]:
    return [
        entry
        for entry in entries
        if entry.state in states and entry.product in products
    ]


def summarize_entries(entries: list[ManifestEntry]) -> list[str]:
    state_aois: dict[str, set[str]] = defaultdict(set)
    state_files: dict[str, int] = defaultdict(int)
    for entry in entries:
        state_aois[entry.state].add(entry.aoi)
        state_files[entry.state] += 1

    lines: list[str] = []
    for state in sorted(state_aois):
        aois = ", ".join(sorted(state_aois[state]))
        lines.append(
            f"{state}: aois={len(state_aois[state])} files={state_files[state]} ids={aois}"
        )
    return lines


def build_grouped_rsync_commands(
    entries: list[ManifestEntry],
    remote: str,
    local_root: Path,
    dry_run: bool,
) -> list[list[str]]:
    grouped: dict[tuple[Path, Path], set[str]] = defaultdict(set)
    for entry in entries:
        source_dir = entry.source.parent
        dest_dir = local_root / entry.destination.parent
        grouped[(source_dir, dest_dir)].add(entry.source.name)

    commands: list[list[str]] = []
    for (source_dir, dest_dir), filenames in sorted(grouped.items()):
        cmd = ["rsync", "-av", "--prune-empty-dirs"]
        if dry_run:
            cmd.append("--dry-run")
        for filename in sorted(filenames):
            cmd.append(f"--include=/{filename}")
        cmd.append("--exclude=*")
        cmd.append(f"{remote}:{source_dir.as_posix().rstrip('/')}/")
        cmd.append(f"{dest_dir.as_posix().rstrip('/')}/")
        commands.append(cmd)
    return commands


def run_sync(args: argparse.Namespace, entries: list[ManifestEntry]) -> int:
    print(f"manifest: {args.manifest}")
    print(f"local root: {args.local_root}")
    print(f"remote: {args.remote}")
    print(f"states: {', '.join(args.states)}")
    print(f"products: {', '.join(args.products)}")
    print(f"manifest entries: {len(entries)}")
    print()

    for line in summarize_entries(entries):
        print(line)
    print()

    commands = build_grouped_rsync_commands(
        entries=entries,
        remote=args.remote,
        local_root=args.local_root,
        dry_run=args.dry_run,
    )

    if args.plan_only:
        for cmd in commands:
            print(" ".join(cmd))
        return 0

    for cmd in commands:
        dest_dir = Path(cmd[-1])
        dest_dir.mkdir(parents=True, exist_ok=True)
        if args.verbose or args.dry_run:
            print(" ".join(cmd))
        completed = subprocess.run(cmd)
        if completed.returncode != 0:
            return completed.returncode

    return 0


def main() -> int:
    args = parse_args()

    try:
        if args.fetch_manifest:
            fetch_manifest(args.remote, args.remote_manifest, args.manifest)
        entries = read_manifest(args.manifest)
        entries = filter_entries(entries, set(args.states), set(args.products))
        if not entries:
            raise ValueError("No manifest entries remain after filtering")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return run_sync(args, entries)


if __name__ == "__main__":
    raise SystemExit(main())
