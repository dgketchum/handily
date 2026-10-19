"""FAC-REM manifest: explicit per-arm raster list (discover / select / sample)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

import fac_rem_registry as reg  # noqa: E402


def _tif(root: Path, basin: str, recipe: str, x0: float, y0: float, val: float) -> Path:
    """2x2 10 m EPSG:5070 raster covering x0..x0+20, y0-20..y0 filled with val."""
    d = root / basin / "rem" / f"{basin}_{recipe}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / reg.RASTER_NAME
    arr = np.full((2, 2), val, dtype="float32")
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:5070",
        transform=from_origin(x0, y0, 10, 10),
    ) as dst:
        dst.write(arr, 1)
    return p


def test_discover_sorted_fallback_only_when_named(tmp_path):
    _tif(tmp_path, "16060001", "scalable", 0, 0, 1)
    _tif(tmp_path, "16060001", "scalable_v2", 0, 0, 1)
    _tif(tmp_path, "16040108", "scalable", 100, 0, 1)  # baseline only
    _tif(tmp_path, "16050203", "scalable_v2", 200, 0, 1)  # variant only
    _tif(tmp_path, "nm_rio_grande_abq", "scalable", 300, 0, 1)  # legacy pilot

    v2 = reg.discover("scalable_v2", str(tmp_path))
    assert [(e["basin"], e["recipe"]) for e in v2] == [
        ("16050203", "scalable_v2"),
        ("16060001", "scalable_v2"),
    ]

    mixed = reg.discover("scalable_v2", str(tmp_path), fallback="scalable")
    assert [(e["basin"], e["recipe"]) for e in mixed] == [
        ("16040108", "scalable"),
        ("16050203", "scalable_v2"),
        ("16060001", "scalable_v2"),
    ]
    assert reg.count_by_recipe(mixed) == {"scalable": 1, "scalable_v2": 2}

    base = reg.discover("scalable", str(tmp_path))
    assert [e["basin"] for e in base] == ["16040108", "16060001"]  # no pilot
    legacy = reg.discover("scalable", str(tmp_path), legacy_pilots=True)
    assert [e["basin"] for e in legacy] == ["nm_rio_grande_abq", "16040108", "16060001"]
    assert all(e["bounds"][0] < e["bounds"][2] and e["bytes"] > 0 for e in legacy)


def test_manifest_roundtrip_and_strict_select(tmp_path):
    _tif(tmp_path, "16060001", "scalable", 0, 0, 1)
    _tif(tmp_path, "16040108", "scalable", 1000, 0, 1)
    entries = reg.discover("scalable", str(tmp_path))
    man = tmp_path / reg.MANIFEST_NAME
    reg.write_manifest(man, entries, "scalable", note="t", fallback_recipe=None)
    doc = reg.load_manifest(man)
    assert doc["recipe"] == "scalable" and doc["n"] == 2
    assert doc["entries"] == entries

    # bounds filter: a window touching only the second raster
    sel = reg.select_paths(doc["entries"], bounds=(1005, -15, 1010, -5))
    assert [Path(p).parents[2].name for p in sel] == ["16040108"]
    assert len(reg.select_paths(doc["entries"])) == 2

    # strict: size change and missing file both refuse
    p0 = Path(entries[0]["path"])
    with open(p0, "ab") as f:
        f.write(b"\0")
    with pytest.raises(SystemExit, match="bytes on disk"):
        reg.select_paths(doc["entries"], strict=True)
    p0.unlink()
    with pytest.raises(SystemExit, match="missing"):
        reg.select_paths(doc["entries"], strict=True)
    assert len(reg.select_paths(doc["entries"], strict=False)) == 2

    with pytest.raises(SystemExit, match="manifest missing"):
        reg.load_manifest(tmp_path / "nope.json")
    with pytest.raises(SystemExit, match="carries no"):
        reg.manifest_for_model(tmp_path / "arm_without_manifest")
    assert reg.manifest_for_model(tmp_path, override=str(man)) == man


def test_sample_first_finite_wins_in_list_order(tmp_path):
    a = _tif(tmp_path, "a", "scalable", 0, 0, 1.0)
    b = _tif(tmp_path, "b", "scalable", 0, 0, 2.0)
    with rasterio.open(a, "r+") as dst:
        arr = dst.read(1)
        arr[0, 0] = np.nan
        dst.write(arr, 1)
    x = np.array([5.0, 15.0, 500.0])
    y = np.array([-5.0, -5.0, -5.0])
    got = reg.sample_fac_rem(x, y, [str(a), str(b)])
    assert got[0] == 2.0  # a is NaN there -> b fills
    assert got[1] == 1.0  # a wins where finite
    assert np.isnan(got[2])  # outside every raster
    rev = reg.sample_fac_rem(x, y, [str(b), str(a)])
    assert rev[1] == 2.0
