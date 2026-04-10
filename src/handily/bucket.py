"""Google Cloud Storage bucket utilities for syncing EE exports to local filesystem.

The local filesystem mirrors the bucket structure:
    gs://wudr/handily/beaverhead/irrmapper/...
    ->
    /nas/handily/handily/beaverhead/irrmapper/...

This allows consistent paths between bucket exports and local processing.
"""

import fnmatch
import logging
import os
import subprocess

LOGGER = logging.getLogger("handily.bucket")

# Default bucket and local root
DEFAULT_BUCKET = "wudr"
DEFAULT_LOCAL_ROOT = "/nas/handily"
DEFAULT_BUCKET_PREFIX = "handily"


def get_gsutil_path() -> str:
    """Find gsutil command path."""
    # Try common locations
    candidates = [
        "gsutil",
        os.path.expanduser("~/google-cloud-sdk/bin/gsutil"),
    ]
    for cmd in candidates:
        try:
            result = subprocess.run([cmd, "version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                return cmd
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    raise FileNotFoundError("gsutil not found. Install Google Cloud SDK.")


def list_bucket_contents(
    bucket_path: str,
    gsutil: str | None = None,
    recursive: bool = False,
) -> list[str]:
    """List contents of a GCS bucket path.

    Parameters
    ----------
    bucket_path : str
        Full bucket path (e.g., 'gs://wudr/handily/irrmapper/')
    gsutil : str, optional
        Path to gsutil command.
    recursive : bool
        If True, list recursively.

    Returns
    -------
    list[str]
        List of file paths in the bucket.
    """
    if gsutil is None:
        gsutil = get_gsutil_path()

    if not bucket_path.startswith("gs://"):
        bucket_path = f"gs://{bucket_path}"

    list_path = bucket_path
    if recursive:
        if not list_path.endswith("/"):
            list_path += "/"
        list_path = f"{list_path}**"
    cmd = [gsutil, "ls", list_path]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        LOGGER.warning("gsutil ls failed: %s", result.stderr.strip())
        return []

    files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    return files


def _coerce_glob_pattern(pattern: str) -> str:
    """Treat bare strings as substring matches, otherwise honor glob syntax."""
    if not pattern:
        return "*"
    if any(ch in pattern for ch in ("*", "?", "[")):
        return pattern
    return f"*{pattern}*"


def sync_bucket_to_local(
    bucket: str,
    bucket_prefix: str,
    local_root: str,
    local_prefix: str | None = None,
    subdir: str | None = None,
    glob_pattern: str = "*",
    overwrite: bool = False,
    dry_run: bool = False,
    gsutil: str | None = None,
    recursive: bool = True,
) -> dict:
    """Sync files from GCS bucket to local filesystem.

    Downloads from GCS using bucket_prefix, writes locally using local_prefix:
        gs://{bucket}/{bucket_prefix}/{subdir}/*
        ->
        {local_root}/{local_prefix}/{subdir}/*

    Parameters
    ----------
    bucket : str
        Bucket name (e.g., 'wudr').
    bucket_prefix : str
        Prefix within bucket (e.g., 'handily/mt').
    local_root : str
        Local root directory.
    local_prefix : str, optional
        Local subdirectory prefix (e.g., 'mt'). Defaults to bucket_prefix.
    subdir : str, optional
        Subdirectory to sync (e.g., 'ndwi').
    glob_pattern : str
        Glob pattern to filter files (e.g., '*irr_freq*' or '*.csv').
        If no glob characters are present, it is treated as a substring match.
    overwrite : bool
        Overwrite existing local files.
    dry_run : bool
        Print files without copying.
    gsutil : str, optional
        Path to gsutil command.
    recursive : bool
        Sync nested directories (preserve remote folder structure locally).

    Returns
    -------
    dict
        Summary with keys: copied, skipped, errors, files.
    """
    if gsutil is None:
        gsutil = get_gsutil_path()

    local_root = os.path.expanduser(local_root)

    glob_pattern = _coerce_glob_pattern(glob_pattern)

    if local_prefix is None:
        local_prefix = bucket_prefix

    # Build paths
    if subdir:
        bucket_path = f"gs://{bucket}/{bucket_prefix}/{subdir}/"
        local_dir = os.path.join(local_root, local_prefix, subdir)
    else:
        bucket_path = f"gs://{bucket}/{bucket_prefix}/"
        local_dir = os.path.join(local_root, local_prefix)

    LOGGER.info("Syncing: %s -> %s", bucket_path, local_dir)

    # Create local directory
    os.makedirs(local_dir, exist_ok=True)

    # List remote objects (optionally recursively), then copy individually.
    # This keeps glob filtering + overwrite behavior consistent.
    files = list_bucket_contents(bucket_path, gsutil=gsutil, recursive=recursive)
    files = [f for f in files if f and not f.endswith("/")]

    if glob_pattern != "*":
        filtered = []
        for remote_path in files:
            rel = remote_path
            if remote_path.startswith(bucket_path):
                rel = remote_path[len(bucket_path) :]
            if fnmatch.fnmatch(os.path.basename(rel), glob_pattern) or fnmatch.fnmatch(
                rel, glob_pattern
            ):
                filtered.append(remote_path)
        files = filtered

    if dry_run:
        LOGGER.info("Dry run - would copy %d files:", len(files))
        for f in files:
            rel = (
                f[len(bucket_path) :]
                if f.startswith(bucket_path)
                else os.path.basename(f)
            )
            print(f"  {f} -> {os.path.join(local_dir, rel)}")
        return {"copied": 0, "skipped": 0, "errors": 0, "files": files}

    copied, skipped, errors = 0, 0, 0
    local_files: list[str] = []
    for file_path in files:
        rel = (
            file_path[len(bucket_path) :]
            if file_path.startswith(bucket_path)
            else os.path.basename(file_path)
        )
        local_path = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path) and not overwrite:
            skipped += 1
            local_files.append(local_path)
            continue

        cmd = [gsutil, "cp", file_path, local_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            copied += 1
            local_files.append(local_path)
            LOGGER.debug("Copied: %s", local_path)
        else:
            errors += 1
            LOGGER.warning("Failed to copy %s: %s", file_path, result.stderr.strip())

    LOGGER.info(
        "Sync complete: copied=%d, skipped=%d, errors=%d", copied, skipped, errors
    )
    return {
        "copied": copied,
        "skipped": skipped,
        "errors": errors,
        "files": local_files,
    }


def build_bucket_path(
    bucket: str,
    bucket_prefix: str,
    subdir: str,
    filename: str | None = None,
) -> str:
    """Build a GCS bucket path.

    Parameters
    ----------
    bucket : str
        Bucket name.
    bucket_prefix : str
        Prefix within bucket.
    subdir : str
        Subdirectory.
    filename : str, optional
        Filename to append.

    Returns
    -------
    str
        Full bucket path (without gs:// prefix for EE export).
    """
    if filename:
        return f"{bucket_prefix}/{subdir}/{filename}"
    return f"{bucket_prefix}/{subdir}"


def build_local_path(
    local_root: str,
    local_prefix: str,
    subdir: str,
    filename: str | None = None,
) -> str:
    """Build a local filesystem path under the project directory.

    Parameters
    ----------
    local_root : str
        Local root directory.
    local_prefix : str
        Project prefix (e.g., 'mt').
    subdir : str
        Subdirectory.
    filename : str, optional
        Filename to append.

    Returns
    -------
    str
        Full local path.
    """
    local_root = os.path.expanduser(local_root)
    if filename:
        return os.path.join(local_root, local_prefix, subdir, filename)
    return os.path.join(local_root, local_prefix, subdir)


def sync_irrmapper(
    project_name: str,
    local_root: str = DEFAULT_LOCAL_ROOT,
    bucket: str = DEFAULT_BUCKET,
    bucket_prefix: str = DEFAULT_BUCKET_PREFIX,
    overwrite: bool = False,
    dry_run: bool = False,
) -> str | None:
    """Sync IrrMapper exports from bucket to local.

    Parameters
    ----------
    project_name : str
        Project name (e.g., 'beaverhead').
    local_root : str
        Local root directory.
    bucket : str
        Bucket name.
    bucket_prefix : str
        Bucket prefix.
    overwrite : bool
        Overwrite existing files.
    dry_run : bool
        Print without copying.

    Returns
    -------
    str or None
        Path to local IrrMapper CSV if found, None otherwise.
    """
    full_prefix = f"{bucket_prefix}/{project_name}"
    result = sync_bucket_to_local(
        bucket=bucket,
        bucket_prefix=full_prefix,
        local_root=local_root,
        local_prefix=project_name,
        subdir="irrmapper",
        glob_pattern="irr_freq",
        overwrite=overwrite,
        dry_run=dry_run,
    )

    if result["copied"] > 0 or result["skipped"] > 0:
        # Find the CSV file
        local_dir = build_local_path(local_root, project_name, "irrmapper")
        csvs = [f for f in os.listdir(local_dir) if f.endswith(".csv")]
        if csvs:
            return os.path.join(local_dir, csvs[0])

    return None
