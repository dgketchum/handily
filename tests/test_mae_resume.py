"""Tests for train_neighborhood_mae --resume: fresh start, exact continuation from
.latest.pt (model/opt/sched/rng state), and fail-loud on config or arm mismatch.

Uses a tiny synthetic patch dir (memmap patches/mask + meta.json) so train() runs
end-to-end on CPU in a few seconds.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tnm = _load("train_neighborhood_mae")

N, NS, NC, WIN = 24, 4, 2, 16


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    # keep tests off the (shared) GPU and bit-deterministic for the exact-resume check
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture()
def patch_dir(tmp_path):
    d = tmp_path / "patches"
    d.mkdir()
    rng = np.random.default_rng(7)
    patches = np.memmap(d / "patches.f16", "float16", "w+", shape=(N, NS, NC, WIN, WIN))
    patches[:] = rng.standard_normal((N, NS, NC, WIN, WIN)).astype("float16")
    patches.flush()
    mask = np.memmap(d / "mask.u8", "uint8", "w+", shape=(N, NS, WIN, WIN))
    mask[:] = 1
    mask.flush()
    (d / "meta.json").write_text(
        json.dumps(
            {
                "n_patches": N,
                "n_channels": NC,
                "win": WIN,
                "scales": [100, 500, 1000, 4000],
                "channels": [f"c{i}" for i in range(NC)],
            }
        )
    )
    (d / "norm_stats.json").write_text(json.dumps({}))
    return d


def _args(patch_dir, out, epochs, resume=False, **over):
    a = dict(
        patch_dir=str(patch_dir),
        arm="s100",
        out=str(out),
        epochs=epochs,
        batch=8,
        lr=1e-3,
        dim=16,
        depth=1,
        heads=2,
        dec_dim=8,
        dec_depth=1,
        patch=8,
        mask_ratio=0.70,
        ckpt_every=100,
        seed=0,
        no_ram=False,
        resume=resume,
    )
    a.update(over)
    return argparse.Namespace(**a)


def test_fresh_run_writes_resumable_latest(patch_dir, tmp_path):
    out = tmp_path / "m.pt"
    tnm.train(_args(patch_dir, out, epochs=2))
    latest = out.with_suffix(".latest.pt")
    assert latest.exists()
    ck = torch.load(latest, map_location="cpu", weights_only=False)
    assert ck["epoch"] == 2
    for k in ("opt", "sched", "best", "torch_rng", "np_rng"):
        assert k in ck


def test_resume_without_latest_starts_fresh(patch_dir, tmp_path):
    out = tmp_path / "m.pt"
    tnm.train(_args(patch_dir, out, epochs=1, resume=True))
    ck = torch.load(
        out.with_suffix(".latest.pt"), map_location="cpu", weights_only=False
    )
    assert ck["epoch"] == 1


def test_resume_matches_uninterrupted_run(patch_dir, tmp_path, monkeypatch):
    # A: 4 epochs straight through
    out_a = tmp_path / "a.pt"
    tnm.train(_args(patch_dir, out_a, epochs=4))
    # B: same 4-epoch job preempted right after epoch 2's .latest.pt save, then
    # requeued with identical args — restored model/opt/sched/rng state must
    # reproduce A's continuation exactly
    out_b = tmp_path / "b.pt"
    real_save = torch.save

    def kill_after_ep2(obj, path, *a, **kw):
        real_save(obj, path, *a, **kw)
        if str(path).endswith(".latest.pt") and obj["epoch"] == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(tnm.torch, "save", kill_after_ep2)
    with pytest.raises(KeyboardInterrupt):
        tnm.train(_args(patch_dir, out_b, epochs=4))
    monkeypatch.setattr(tnm.torch, "save", real_save)
    tnm.train(_args(patch_dir, out_b, epochs=4, resume=True))
    ck_a = torch.load(
        out_a.with_suffix(".latest.pt"), map_location="cpu", weights_only=False
    )
    ck_b = torch.load(
        out_b.with_suffix(".latest.pt"), map_location="cpu", weights_only=False
    )
    assert ck_b["epoch"] == 4
    assert ck_a["val_loss"] == pytest.approx(ck_b["val_loss"], abs=1e-6)
    for k in ck_a["model"]:
        assert torch.allclose(ck_a["model"][k], ck_b["model"][k], atol=1e-6), k


def test_resume_config_mismatch_fails_loud(patch_dir, tmp_path):
    out = tmp_path / "m.pt"
    tnm.train(_args(patch_dir, out, epochs=1))
    with pytest.raises(SystemExit, match="mismatch"):
        tnm.train(_args(patch_dir, out, epochs=2, resume=True, dim=32))


def test_resume_arm_mismatch_fails_loud(patch_dir, tmp_path):
    out = tmp_path / "m.pt"
    tnm.train(_args(patch_dir, out, epochs=1))
    with pytest.raises(SystemExit, match="mismatch"):
        tnm.train(_args(patch_dir, out, epochs=1, resume=True, arm="s500"))


def test_resume_epochs_mismatch_fails_loud(patch_dir, tmp_path):
    # OneCycleLR total_steps is baked into the schedule state: a resume must
    # rerun the identical --epochs, never extend it
    out = tmp_path / "m.pt"
    tnm.train(_args(patch_dir, out, epochs=1))
    with pytest.raises(SystemExit, match="mismatch"):
        tnm.train(_args(patch_dir, out, epochs=2, resume=True))
