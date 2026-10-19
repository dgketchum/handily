"""HANDILY_FAC_RECIPE pointer: prefer the variant recipe, fall back to _scalable."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))


def _touch(root: Path, basin: str, recipe: str) -> None:
    d = root / basin / "rem" / f"{basin}_{recipe}"
    d.mkdir(parents=True)
    (d / "fac_head_depth_rem_10m.tif").write_bytes(b"")


def _reload(monkeypatch, recipe: str | None):
    if recipe is None:
        monkeypatch.delenv("HANDILY_FAC_RECIPE", raising=False)
    else:
        monkeypatch.setenv("HANDILY_FAC_RECIPE", recipe)
    import fac_rem_registry

    return importlib.reload(fac_rem_registry)


def test_recipe_pointer_prefers_variant_and_falls_back(tmp_path, monkeypatch):
    _touch(tmp_path, "16060001", "scalable")
    _touch(tmp_path, "16060001", "scalable_v2")
    _touch(tmp_path, "16040108", "scalable")  # no variant -> falls back
    _touch(tmp_path, "16050203", "scalable_v2")  # variant only

    reg = _reload(monkeypatch, "scalable_v2")
    monkeypatch.setattr(reg, "_SCALABLE_ROOT", str(tmp_path))
    paths = reg._discover(reg._fac_path)
    recipes = dict(reg._recipe_of(p) for p in paths)
    assert recipes == {
        "16040108": "scalable",
        "16050203": "scalable_v2",
        "16060001": "scalable_v2",
    }

    base = _reload(monkeypatch, None)
    monkeypatch.setattr(base, "_SCALABLE_ROOT", str(tmp_path))
    paths = base._discover(base._fac_path)
    recipes = dict(base._recipe_of(p) for p in paths)
    # default pointer = baseline only; the variant-only basin has no baseline raster
    assert recipes == {"16040108": "scalable", "16060001": "scalable"}
    assert base.FAC_RECIPE == "scalable"
    _reload(monkeypatch, None)


def test_pin_restricts_registry_and_checks_recipes(tmp_path, monkeypatch):
    _touch(tmp_path, "16060001", "scalable_v2")
    _touch(tmp_path, "16040108", "scalable")
    _touch(tmp_path, "16050203", "scalable_v2")
    reg = _reload(monkeypatch, "scalable_v2")
    monkeypatch.setattr(reg, "_SCALABLE_ROOT", str(tmp_path))
    full = reg._discover(reg._fac_path)
    assert len(full) == 3

    pin = tmp_path / "pin.tsv"
    pin.write_text("# recipe scalable_v2\n16050203\tscalable_v2\n16040108\tscalable\n")
    kept = reg._apply_pin(full, str(pin))
    assert [reg._recipe_of(p) for p in kept] == [
        ("16040108", "scalable"),
        ("16050203", "scalable_v2"),
    ]  # discovery order kept, unpinned basin dropped

    pin.write_text("16050203\tscalable_v2\n16060012\tscalable_v2\n")
    try:
        reg._apply_pin(full, str(pin))
    except SystemExit as e:
        assert "missing 1" in str(e)
    else:
        raise AssertionError("missing pinned basin must fail")

    pin.write_text("16040108\tscalable_v2\n")  # resolves to baseline on this disk
    try:
        reg._apply_pin(full, str(pin))
    except SystemExit as e:
        assert "recipe mismatch 1" in str(e)
    else:
        raise AssertionError("recipe mismatch must fail")
    _reload(monkeypatch, None)
