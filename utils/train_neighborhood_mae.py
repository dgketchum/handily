"""Compact ViT-MAE pretraining on the neighborhood covariate stack.

Self-supervised masked-autoencoder over the multi-scale 64x64 patches built by
build_mae_patches.py. One arm per resolution (single scale) or the pyramid (all scales
stacked as channel groups). Reconstruction loss is masked-token MSE, weighted per pixel
by the DEM valid mask so nodata never enters the loss (nodata is imputed 0 in the input
but zero-weighted in the target).

The embedding = mean-pooled encoder tokens with mask_ratio=0; extract_mae_embeddings.py
loads a checkpoint and calls MAE.encode().

    uv run python utils/train_neighborhood_mae.py \
        --patch-dir /data/ssd2/handily/conus/mae/patches/ladder \
        --arm s100 --out /data/ssd2/handily/conus/mae/checkpoints/s100.pt \
        --epochs 40 --batch 384
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_neighborhood_mae")

ARMS = {
    "s100": (0,),
    "s500": (1,),
    "s1000": (2,),
    "s4000": (3,),
    "pyr": (0, 1, 2),
    "pyr_wide": (0, 1, 2, 3),
}


# --------------------------------------------------------------------------- model
class MAE(nn.Module):
    """Minimal ViT-MAE for CxHxW inputs; square image, square patch."""

    def __init__(
        self,
        in_ch: int,
        img: int = 64,
        patch: int = 8,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        dec_dim: int = 64,
        dec_depth: int = 2,
        mask_ratio: float = 0.70,
    ):
        super().__init__()
        self.in_ch, self.img, self.patch = in_ch, img, patch
        self.n_patch = (img // patch) ** 2
        self.pdim = patch * patch * in_ch
        self.mask_ratio = mask_ratio
        self.patch_embed = nn.Conv2d(in_ch, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, dim))
        enc = nn.TransformerEncoderLayer(
            dim, heads, dim * 4, batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.enc_norm = nn.LayerNorm(dim)
        self.dec_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, self.n_patch, dec_dim))
        dec = nn.TransformerEncoderLayer(
            dec_dim,
            heads,
            dec_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec, dec_depth)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_head = nn.Linear(dec_dim, self.pdim)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B,C,H,W] -> [B,n_patch, patch*patch*C]."""
        b, c, h, w = x.shape
        p = self.patch
        x = x.reshape(b, c, h // p, p, w // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # b, gh, gw, p, p, c
        return x.reshape(b, (h // p) * (w // p), p * p * c)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B,N,dim]
        return t + self.pos

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Full-image embedding (no masking): mean-pooled encoder tokens -> [B,dim]."""
        t = self.enc_norm(self.encoder(self._tokens(x)))
        return t.mean(dim=1)

    def forward(self, x: torch.Tensor):
        b = x.shape[0]
        tok = self._tokens(x)
        n, keep = self.n_patch, int(self.n_patch * (1 - self.mask_ratio))
        noise = torch.rand(b, n, device=x.device)
        ids_shuf = noise.argsort(1)
        ids_rest = ids_shuf.argsort(1)
        ids_keep = ids_shuf[:, :keep]
        vis = torch.gather(tok, 1, ids_keep.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
        lat = self.enc_norm(self.encoder(vis))
        # decoder: scatter visible + mask tokens
        d = self.dec_embed(lat)
        mt = self.mask_token.expand(b, n - keep, -1)
        d_full = torch.cat([d, mt], 1)
        d_full = torch.gather(
            d_full, 1, ids_rest.unsqueeze(-1).expand(-1, -1, d.shape[-1])
        )
        d_full = d_full + self.dec_pos
        pred = self.dec_head(self.dec_norm(self.decoder(d_full)))  # [B,N,pdim]
        mask = torch.ones(b, n, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_rest)  # 1 = masked token (in loss)
        return pred, mask


def masked_recon_loss(pred, target_img, mask, wmap, model):
    """Weighted MSE over MASKED tokens; wmap is a per-pixel [B,H,W] validity weight."""
    tgt = model.patchify(target_img)  # [B,N,pdim]
    # per-token, per-pixel weight (broadcast over channels): patchify weight map
    b, h, w = wmap.shape
    p = model.patch
    wm = wmap.reshape(b, 1, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1)
    wm = wm.reshape(b, (h // p) * (w // p), p * p)  # [B,N,patch*patch], pixel-ordered
    # patchify orders a token as (pixel, channel) with channel fastest -> interleave the
    # per-pixel weight across channels so weight[pixel*C+c] == w[pixel].
    wm = wm.repeat_interleave(model.in_ch, dim=2)  # [B,N,pdim]
    tok_mask = mask.unsqueeze(-1)  # [B,N,1] 1=masked
    w = wm * tok_mask
    err = (pred - tgt) ** 2 * w
    denom = w.sum().clamp_min(1.0)
    return err.sum() / denom


# --------------------------------------------------------------------------- data
def load_arm(patch_dir: Path, arm: str, in_ram: bool = True):
    meta = json.loads((patch_dir / "meta.json").read_text())
    n, ns, nc, win = (
        meta["n_patches"],
        len(meta["scales"]),
        meta["n_channels"],
        meta["win"],
    )
    patches = np.memmap(
        patch_dir / "patches.f16", "float16", "r", shape=(n, ns, nc, win, win)
    )
    mask = np.memmap(patch_dir / "mask.u8", "uint8", "r", shape=(n, ns, win, win))
    if in_ram:  # materialize once (fits: ~0.29 MB/patch); avoids per-batch disk reads
        log.info("loading %d patches into RAM (%.1f GB)", n, patches.nbytes / 1e9)
        patches = np.asarray(patches)
        mask = np.asarray(mask)
    sidx = ARMS[arm]
    return meta, patches, mask, sidx


def make_batch(patches, mask, sidx, idx, device):
    """Assemble [B,C,H,W] input and [B,H,W] pixel-weight map for arm scales sidx."""
    p = np.asarray(patches[idx])  # [B,ns,nc,H,W]
    m = np.asarray(mask[idx])  # [B,ns,H,W]
    x = np.concatenate([p[:, s] for s in sidx], axis=1)  # [B, C*len(sidx), H, W]
    wmap = np.mean([m[:, s] for s in sidx], axis=0).astype("float32")  # [B,H,W]
    return (
        torch.from_numpy(x.astype("float32")).to(device),
        torch.from_numpy(wmap).to(device),
    )


# --------------------------------------------------------------------------- train
def train(args) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    patch_dir = Path(args.patch_dir)
    meta, patches, mask, sidx = load_arm(patch_dir, args.arm, in_ram=not args.no_ram)
    n = meta["n_patches"]
    in_ch = meta["n_channels"] * len(sidx)
    n_val = max(1, int(n * 0.05))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = MAE(
        in_ch,
        img=meta["win"],
        patch=args.patch,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dec_dim=args.dec_dim,
        dec_depth=args.dec_depth,
        mask_ratio=args.mask_ratio,
    ).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    log.info(
        "arm=%s in_ch=%d params=%.2fM n_train=%d",
        args.arm,
        in_ch,
        nparam / 1e6,
        len(tr_idx),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(tr_idx) // args.batch),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    start_ep = 0
    if args.resume:
        latest = out.with_suffix(".latest.pt")
        if latest.exists():
            ck = torch.load(latest, map_location=device, weights_only=False)
            same = all(
                ck["config"][k] == v
                for k, v in {
                    "in_ch": in_ch,
                    "patch": args.patch,
                    "dim": args.dim,
                    "depth": args.depth,
                    "heads": args.heads,
                    "dec_dim": args.dec_dim,
                    "dec_depth": args.dec_depth,
                    "mask_ratio": args.mask_ratio,
                }.items()
            )
            # OneCycleLR state carries total_steps: resuming under a different
            # schedule length silently corrupts the LR curve, so require identity
            same = same and ck["epochs_total"] == args.epochs
            if not same or ck["arm"] != args.arm:
                raise SystemExit(
                    f"--resume: {latest} config/arm mismatch with current args"
                )
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            best = ck.get("best", float("inf"))
            start_ep = ck["epoch"]
            torch.set_rng_state(ck["torch_rng"].cpu())
            rng.bit_generator.state = ck["np_rng"]
            log.info(
                "resumed from %s at epoch %d (best val %.4f)", latest, start_ep, best
            )
        else:
            log.info("--resume: no %s, starting fresh", latest)
    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        ep_idx = rng.permutation(tr_idx)
        losses = []
        for b0 in range(0, len(ep_idx) - args.batch + 1, args.batch):
            idx = np.sort(ep_idx[b0 : b0 + args.batch])
            x, wmap = make_batch(patches, mask, sidx, idx, device)
            pred, m = model(x)
            loss = masked_recon_loss(pred, x, m, wmap, model)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        # validation
        model.eval()
        vl = []
        with torch.no_grad():
            for b0 in range(0, len(val_idx), args.batch):
                idx = np.sort(val_idx[b0 : b0 + args.batch])
                x, wmap = make_batch(patches, mask, sidx, idx, device)
                pred, m = model(x)
                vl.append(masked_recon_loss(pred, x, m, wmap, model).item())
        vloss = float(np.mean(vl))
        log.info(
            "ep %d/%d train %.4f val %.4f (%.0fs)",
            ep + 1,
            args.epochs,
            float(np.mean(losses)),
            vloss,
            time.time() - t0,
        )
        ckpt = {
            "model": model.state_dict(),
            "config": {
                "in_ch": in_ch,
                "img": meta["win"],
                "patch": args.patch,
                "dim": args.dim,
                "depth": args.depth,
                "heads": args.heads,
                "dec_dim": args.dec_dim,
                "dec_depth": args.dec_depth,
                "mask_ratio": args.mask_ratio,
            },
            "arm": args.arm,
            "scales": [meta["scales"][s] for s in sidx],
            "channels": meta["channels"],
            "patch_dir": str(patch_dir),
            "norm_stats": str(patch_dir / "norm_stats.json"),
            "epoch": ep + 1,
            "epochs_total": args.epochs,
            "val_loss": vloss,
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "best": min(best, vloss),
            "torch_rng": torch.get_rng_state(),
            "np_rng": rng.bit_generator.state,
        }
        torch.save(ckpt, out.with_suffix(".latest.pt"))
        if (ep + 1) % max(1, args.ckpt_every) == 0:
            torch.save(ckpt, out.with_suffix(f".ep{ep + 1}.pt"))
        if vloss < best:
            best = vloss
            torch.save(ckpt, out)
    log.info("arm=%s best val %.4f -> %s", args.arm, best, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch-dir", required=True)
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dec-dim", type=int, default=64)
    ap.add_argument("--dec-depth", type=int, default=2)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--mask-ratio", type=float, default=0.70)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-ram",
        action="store_true",
        help="stream patches from the memmap instead of materializing in RAM "
        "(required at v2 scale: patch sets larger than node memory)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="resume from <out>.latest.pt if present (model/opt/sched/rng state); "
        "fails loud on config mismatch",
    )
    train(ap.parse_args())


if __name__ == "__main__":
    main()
