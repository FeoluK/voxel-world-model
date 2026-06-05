"""v14 — voxel-native MaskGIT world model.

A single-frame 3D conv U-Net over voxel block-IDs, trained BERT-style (random
mask -> predict the masked block classes via cross-entropy). At inference it is
driven by the FRONTIER mask (carry fills the known region, model fills the rest)
with confidence-based iterative un-masking. Temporal consistency comes from the
persistent-map carry + write-back, NOT from the network — so this net is purely
spatial. No VAE, no temporal attention, no continuous latent.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _gn(c):
    return nn.GroupNorm(num_groups=min(32, c), num_channels=c)


class ResBlock3D(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1 = _gn(cin); self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.n2 = _gn(cout); self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class VoxelMaskGIT3D(nn.Module):
    """3D U-Net mapping a (partly-masked) voxel-ID grid -> per-voxel class logits.

    num_classes real block classes; index `num_classes` is the [MASK] token.
    """
    def __init__(self, num_classes=2021, dim=96, ch_mult=(1, 2, 4), embed_dim=64, cond_classes=0):
        super().__init__()
        self.num_classes = num_classes
        self.mask_id = num_classes
        self.grad_ckpt = False
        self.use_cond = cond_classes > 0
        self.embed = nn.Embedding(num_classes + 1, embed_dim)
        if self.use_cond:
            # coarse-layout conditioning (super-cat per voxel, upsampled to fine res)
            self.cond_embed = nn.Embedding(cond_classes, embed_dim)
            self.stem = nn.Conv3d(embed_dim * 2, dim, 3, padding=1)
        else:
            self.stem = nn.Conv3d(embed_dim, dim, 3, padding=1)

        chs = [dim * m for m in ch_mult]
        # encoder
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        cin = dim
        for c in chs:
            self.down.append(ResBlock3D(cin, c))
            self.downsample.append(nn.Conv3d(c, c, 4, stride=2, padding=1))
            cin = c
        self.mid = ResBlock3D(cin, cin)
        # decoder
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for c in reversed(chs):
            self.upsample.append(nn.ConvTranspose3d(cin, c, 4, stride=2, padding=1))
            self.up.append(ResBlock3D(c + c, c))  # skip concat
            cin = c
        self.out_norm = _gn(cin)
        self.head = nn.Conv3d(cin, num_classes, 1)

    def forward(self, ids, cond=None):
        # ids: (B, D, H, W) long, with mask_id at masked positions
        # cond: (B, D, H, W) long super-cat layout (upsampled), if use_cond
        x = self.embed(ids).permute(0, 4, 1, 2, 3).contiguous()  # (B,embed,D,H,W)
        if self.use_cond:
            ce = self.cond_embed(cond).permute(0, 4, 1, 2, 3).contiguous()
            x = torch.cat([x, ce], dim=1)
        x = self.stem(x)
        ckpt = self.grad_ckpt and self.training
        skips = []
        for blk, ds in zip(self.down, self.downsample):
            x = checkpoint(blk, x, use_reentrant=False) if ckpt else blk(x)
            skips.append(x); x = ds(x)
        x = checkpoint(self.mid, x, use_reentrant=False) if ckpt else self.mid(x)
        for ups, blk, skip in zip(self.upsample, self.up, reversed(skips)):
            x = ups(x)
            cat = torch.cat([x, skip], dim=1)
            x = checkpoint(blk, cat, use_reentrant=False) if ckpt else blk(cat)
        x = F.silu(self.out_norm(x))
        return self.head(x)  # (B, num_classes, D, H, W)


def cosine_mask_ratio(u):
    # u in [0,1) -> mask ratio in (0,1]; cosine schedule (MaskGIT)
    return torch.cos(0.5 * math.pi * (1.0 - u)).clamp(1e-3, 1.0)


def build_slab_mask(ids, generator=None, frac_lo=0.15, frac_hi=0.85):
    """Contiguous-region mask: mask a slab covering a random fraction along a
    random axis/side -> the model must EXTRAPOLATE a coherent contiguous region
    from the adjacent known half (matches the inference frontier / big-chunk gen),
    unlike scattered random masks which only teach interpolation."""
    B = ids.shape[0]; dev = ids.device
    mask = torch.zeros_like(ids, dtype=torch.bool)
    dims = ids.shape[1:]  # (D,H,W)
    for b in range(B):
        a = int(torch.randint(0, 3, (1,), device=dev, generator=generator))
        n = dims[a]
        frac = float(torch.empty(1, device=dev).uniform_(frac_lo, frac_hi, generator=generator))
        k = max(1, int(round(frac * n)))
        side = int(torch.randint(0, 2, (1,), device=dev, generator=generator))
        idx = [slice(None)] * 3
        idx[a] = slice(n - k, n) if side == 0 else slice(0, k)
        mask[b][tuple(idx)] = True
    return mask


def build_random_mask(ids, generator=None):
    """Per-sample random mask at a sampled ratio. Returns (masked_ids, mask_bool)."""
    B = ids.shape[0]
    dev = ids.device
    u = torch.rand(B, device=dev, generator=generator)
    ratio = cosine_mask_ratio(u).view(B, 1, 1, 1)
    rnd = torch.rand(ids.shape, device=dev, generator=generator)
    mask = rnd < ratio                          # True = masked
    # guarantee at least one masked voxel per sample
    flat = mask.view(B, -1)
    none = ~flat.any(dim=1)
    if none.any():
        flat[none, 0] = True
    return mask


@torch.no_grad()
def maskgit_generate(model, known_ids, fill_mask, n_steps=12, temp=1.0, top_p=1.0, cond=None, logit_bias=None):
    """Iterative confidence-based un-masking over fill_mask (True=generate).
    known_ids holds the carried/known voxels; masked positions are sampled.
    cond (B,D,H,W) optional super-cat layout for the cond-conditioned fine model.
    top_p<1.0 => nucleus sampling. logit_bias (C,) added to per-class logits each step
    — boost rare structure classes (trees) so confidence-greedy unmask commits them
    instead of always collapsing to the air/common mode."""
    B = known_ids.shape[0]
    cur = known_ids.clone()
    cur[fill_mask] = model.mask_id
    rem = fill_mask.clone()
    total = fill_mask.reshape(B, -1).sum(1).float()
    for step in range(n_steps):
        logits = model(cur, cond=cond).float()                       # (B,C,D,H,W)
        if logit_bias is not None:
            logits = logits + logit_bias.to(logits.device).view(1, -1, 1, 1, 1)
        C = logits.shape[1]
        flatp = F.softmax(logits / max(temp, 1e-3), dim=1).permute(0, 2, 3, 4, 1).reshape(B, -1, C)
        if top_p < 1.0:
            sp, si = torch.sort(flatp, dim=-1, descending=True)
            cum = sp.cumsum(-1)
            remove = cum > top_p
            remove[..., 1:] = remove[..., :-1].clone()              # keep first token crossing top_p
            remove[..., 0] = False
            sp = sp.masked_fill(remove, 0.0)
            sp = sp / sp.sum(-1, keepdim=True).clamp_min(1e-8)
            ssort = torch.multinomial(sp.reshape(-1, C), 1).reshape(B, -1, 1)
            samp = torch.gather(si, 2, ssort).squeeze(-1)           # (B,V) original class id
        else:
            samp = torch.multinomial(flatp.reshape(-1, C), 1).reshape(B, -1)     # (B,V)
        conf = torch.gather(flatp, 2, samp.unsqueeze(-1)).squeeze(-1)            # (B,V)
        remflat = rem.reshape(B, -1); curflat = cur.reshape(B, -1)
        conf = torch.where(remflat, conf, torch.full_like(conf, -1.0))
        ratio = math.cos(0.5 * math.pi * (step + 1) / n_steps)                   # 1 -> 0
        for b in range(B):
            rem_n = int(remflat[b].sum())
            if rem_n == 0:
                continue
            k = rem_n if step == n_steps - 1 else max(1, int(rem_n - total[b].item() * ratio))
            k = min(k, rem_n)
            idx = torch.topk(conf[b], k).indices
            curflat[b, idx] = samp[b, idx].to(curflat.dtype)
            remflat[b, idx] = False
        cur = curflat.reshape(known_ids.shape); rem = remflat.reshape(fill_mask.shape)
        if not rem.any():
            break
    return cur
