"""Tests for the MAE harder-pretext levers: block masking and cross-channel drop.

Model-level only (no patch dir): mask-count invariance across modes, spatial
coherence of block masks, variable-tied channel drop with >=1 variable kept,
eval-mode passthrough, and constructor validation.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from scipy import ndimage


def _load(name):
    p = Path(__file__).resolve().parents[1] / "utils" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tnm = _load("train_neighborhood_mae")


def _model(**over):
    kw = dict(
        in_ch=8,
        img=64,
        patch=8,
        dim=16,
        depth=1,
        heads=2,
        dec_dim=8,
        dec_depth=1,
        mask_ratio=0.70,
        n_vars=4,
    )
    kw.update(over)
    return tnm.MAE(**kw)


def test_mask_count_identical_token_vs_block():
    # the argsort keep/scatter machinery fixes the masked count at
    # n_patch - int(n_patch * (1 - mask_ratio)) regardless of mode
    torch.manual_seed(0)
    x = torch.randn(4, 8, 64, 64)
    for mode in ("token", "block"):
        m = _model(mask_mode=mode)
        m.train()
        _, mask = m(x)
        n = m.n_patch
        expect = n - int(n * (1 - m.mask_ratio))
        assert mask.shape == (4, n)
        assert (mask.sum(1) == expect).all(), mode


def test_block_mask_is_spatially_coherent():
    # block-mode masked tokens form few connected components on the token grid;
    # iid token masking at the same ratio fragments into many
    torch.manual_seed(0)
    g = 64 // 8

    def n_components(model):
        noise = model._mask_noise(32, "cpu")
        keep = int(model.n_patch * (1 - model.mask_ratio))
        masked = torch.zeros_like(noise, dtype=torch.bool)
        masked.scatter_(1, noise.argsort(1)[:, keep:], True)
        counts = [ndimage.label(masked[i].reshape(g, g).numpy())[1] for i in range(32)]
        return sum(counts) / len(counts)

    blk = n_components(_model(mask_mode="block", mask_ratio=0.30))
    tok = n_components(_model(mask_mode="token", mask_ratio=0.30))
    assert blk < tok
    assert blk <= 4.0


def test_block_noise_covers_target():
    torch.manual_seed(0)
    m = _model(mask_mode="block")
    noise = m._mask_noise(16, "cpu")
    covered = noise > 0.5  # base noise < 0.01, covered tokens > 1.0
    target = int(m.n_patch * m.mask_ratio)
    assert (covered.sum(1) >= target).all()


def test_channel_keep_tied_across_scales_and_never_empty():
    torch.manual_seed(0)
    m = _model(channel_drop_p=0.9)  # high p exercises the all-dropped rescue path
    keep = m._channel_keep(256, "cpu").squeeze(-1).squeeze(-1)  # [B, in_ch]
    assert keep.shape == (256, 8)
    # variable v's mask is identical in both scale groups (scale-major layout)
    assert torch.equal(keep[:, :4], keep[:, 4:])
    # at least one variable survives in every sample
    assert (keep[:, :4].sum(1) >= 1).all()
    # with p=0.9 drops actually happen
    assert keep.mean() < 0.5


def test_channel_drop_train_forward_finite_loss():
    torch.manual_seed(0)
    m = _model(mask_mode="block", channel_drop_p=0.5)
    m.train()
    x = torch.randn(4, 8, 64, 64)
    pred, mask = m(x)
    wmap = torch.ones(4, 64, 64)
    loss = tnm.masked_recon_loss(pred, x, mask, wmap, m)
    assert torch.isfinite(loss)
    loss.backward()


def test_channel_drop_inactive_in_eval():
    # eval-mode forward must not consume drop RNG or corrupt the input: with the
    # same seed, drop_p=0.9 and drop_p=0.0 give identical outputs
    x = torch.randn(2, 8, 64, 64)
    m = _model(channel_drop_p=0.9)
    m.eval()
    torch.manual_seed(1)
    with torch.no_grad():
        pred_a, mask_a = m(x)
    m.channel_drop_p = 0.0
    torch.manual_seed(1)
    with torch.no_grad():
        pred_b, mask_b = m(x)
    assert torch.equal(pred_a, pred_b)
    assert torch.equal(mask_a, mask_b)


def test_constructor_validation():
    with pytest.raises(ValueError, match="mask_mode"):
        _model(mask_mode="grid")
    with pytest.raises(ValueError, match="multiple"):
        _model(n_vars=3)  # in_ch=8 not a multiple of 3
