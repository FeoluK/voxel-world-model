"""Flow-matching (rectified-flow) DiT/UNet SHADER for the v14 explicit renderer.

The prior renderer_explicit.py (MSE) and renderer_sharp.py (MSE + latent GAN)
both produced BLURRY output: an MSE shader regresses to the conditional mean of
all plausible RGB-latents, which is low-frequency mush. The latent-GAN added
high-frequency noise but no correct structure.

PERSIST-style fix: a GENERATIVE flow-matching shader that SAMPLES one sharp
rgb-latent instead of averaging. We train a velocity field v_theta(x_tau, tau, cond)
under rectified flow:

    x0  = clean Wan rgb_lat (16, 9, 27, 48)            (target)
    x1  ~ N(0, I)                                       (noise)
    tau ~ U(0,1)
    x_tau = (1 - tau) * x0 + tau * x1                   (linear interpolant)
    v_target = x1 - x0   (== d x_tau / d tau)
    loss = || v_theta(x_tau, tau, cond) - v_target ||^2

cond = the EXPLICIT rasterized buffer (voxel_raycaster -> id/depth/hit/ray_dir),
encoded to a feature map aligned to the 27x48 latent grid. The backbone is a
per-frame conditional U-Net with AdaGN timestep+cond modulation and a temporal
conv across the 9 latent frames. cond features are concatenated to x_tau at the
input AND injected via AdaGN at every block (cond is summarized to a vector).

Inference: integrate the ODE from tau=1 (pure noise) -> tau=0 with N Euler
steps to obtain a SHARP rgb_lat sample, then Wan-decode -> RGB.

Reuses unchanged: voxel_raycaster coord math, VAE3D, VoxelDiTDataset,
batched_raycast, voxel_to_compact_grids, the screen-buffer feature layout.
"""
from __future__ import annotations
import argparse, os, sys, time, math
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/scratch/users/flukol/mg_pt_voxel/v14code")
from voxel_dataset import VoxelDiTDataset
from vae_3d import VAE3D
from renderer_explicit import (batched_raycast, voxel_to_compact_grids,
                               LAT_TO_SRC, EYE_CENTER, EYE_HEIGHT,
                               RGB_T, RGB_C, RGB_H, RGB_W, N_COMPACT)
from raster_torch import rasterize_batch, MAX_DEPTH  # render_v2: exact-eye z-buffer rasterizer
USE_RASTER = os.environ.get('USE_RASTER', '1') == '1'  # 1=rasterize, 0=legacy raycast


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# Timestep (tau) sinusoidal embedding
# ---------------------------------------------------------------------------
def timestep_embedding(t, dim, max_period=10000.0):
    # t: (N,) in [0,1]; scale to [0,1000] for resolution
    t = t.float() * 1000.0
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ---------------------------------------------------------------------------
# Conditioning encoder: screen buffer -> (per-frame) feature map at latent res
# Same input channels as the regression shader's screen buffer.
# ---------------------------------------------------------------------------
class CondEncoder(nn.Module):
    def __init__(self, emb_dim=32, base=96, cond_ch=64):
        super().__init__()
        self.emb = nn.Embedding(N_COMPACT, emb_dim)
        in_ch = emb_dim + 1 + 1 + 3   # id_emb + depth + hit + ray_dir(3)
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
        )
        self.proj = nn.Conv2d(base * 2, cond_ch, 1)

    def build_buffer(self, id_map, depth, hit, dirs):
        emb = self.emb(id_map)                                  # (B,T,H,W,E)
        d = (depth / MAX_DEPTH).clamp(0, 1).unsqueeze(-1)
        h = hit.float().unsqueeze(-1)
        buf = torch.cat([emb, d, h, dirs], dim=-1)              # (B,T,H,W,Cin)
        return buf.permute(0, 1, 4, 2, 3).contiguous()         # (B,T,Cin,H,W)

    def forward(self, id_map, depth, hit, dirs):
        B, T, H, W = id_map.shape
        buf = self.build_buffer(id_map, depth, hit, dirs)
        x = buf.reshape(B * T, *buf.shape[2:])
        x = self.enc(x)                                         # (B*T, 2*base, H/2, W/2)
        x = self.proj(x)                                        # (B*T, cond_ch, H/2, W/2)
        # align to latent grid (27x48)
        x = F.adaptive_avg_pool2d(x, (RGB_H, RGB_W))           # (B*T, cond_ch, 27,48)
        return x.view(B, T, -1, RGB_H, RGB_W)                  # (B,T,cond_ch,27,48)


# ---------------------------------------------------------------------------
# AdaGN residual block (modulated by tau+cond summary vector)
# ---------------------------------------------------------------------------
class AdaGNBlock(nn.Module):
    def __init__(self, ci, co, cmod):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ci)
        self.conv1 = nn.Conv2d(ci, co, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, co)
        self.conv2 = nn.Conv2d(co, co, 3, padding=1)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        # produce scale,shift for both norms
        self.mod = nn.Linear(cmod, co * 4)

    def forward(self, x, m):
        s1, b1, s2, b2 = self.mod(m).chunk(4, dim=-1)
        N = x.shape[0]
        s1 = s1.view(N, -1, 1, 1); b1 = b1.view(N, -1, 1, 1)
        s2 = s2.view(N, -1, 1, 1); b2 = b2.view(N, -1, 1, 1)
        h = self.conv1(self.norm1(x))
        # AdaGN on h (post-conv channels = co): re-norm then modulate
        h = self.norm2(h) * (1 + s1) + b1
        h = F.silu(h)
        h = self.conv2(h)
        h = h * (1 + s2) + b2
        h = F.silu(h)
        return self.skip(x) + h


# ---------------------------------------------------------------------------
# Flow-matching velocity net.
#  input:  x_tau (B,16,9,27,48) + cond feat (B,cond_ch,9,27,48)
#  modln:  tau embedding + global cond summary
#  output: v (B,16,9,27,48)
# Per-frame 2D U-Net (small, since 27x48 latent) + temporal conv mix.
# ---------------------------------------------------------------------------
class FlowVNet(nn.Module):
    def __init__(self, cond_ch=64, base=128, tdim=256):
        super().__init__()
        self.cond_ch = cond_ch
        self.tdim = tdim
        self.t_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        # global cond summary: pool cond map -> vector, add to tau embed
        self.cond_summary = nn.Sequential(nn.Linear(cond_ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        in_ch = RGB_C + cond_ch
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        self.b0 = AdaGNBlock(base, base, tdim)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)   # 27x48 -> 14x24
        self.b1 = AdaGNBlock(base * 2, base * 2, tdim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)  # 14x24 -> 7x12
        self.b2 = AdaGNBlock(base * 2, base * 2, tdim)
        self.mid = AdaGNBlock(base * 2, base * 2, tdim)
        # temporal mix at the bottleneck (channels = base*2)
        self.tconv = nn.Conv1d(base * 2, base * 2, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)   # 7x12 -> 14x24
        self.c2 = AdaGNBlock(base * 4, base * 2, tdim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)       # 14x24 -> 28x48
        self.c1 = AdaGNBlock(base * 2, base, tdim)
        self.outnorm = nn.GroupNorm(8, base)
        self.outc = nn.Conv2d(base, RGB_C, 3, padding=1)
        nn.init.zeros_(self.outc.weight); nn.init.zeros_(self.outc.bias)

    def forward(self, x_tau, tau, cond):
        # x_tau (B,16,T,H,W) ; cond (B,cond_ch,T,H,W) ; tau (B,)
        B, _, T, H, W = x_tau.shape
        temb = self.t_mlp(timestep_embedding(tau, self.tdim))      # (B,tdim)
        # global cond summary (per clip)
        csum = cond.mean(dim=(2, 3, 4))                            # (B,cond_ch)
        m = temb + self.cond_summary(csum)                        # (B,tdim)
        m = m.unsqueeze(1).expand(B, T, self.tdim).reshape(B * T, self.tdim)

        x = torch.cat([x_tau, cond], dim=1)                       # (B,16+cond,T,H,W)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, RGB_C + self.cond_ch, H, W)
        x = self.inc(x)
        s0 = self.b0(x, m)
        s1 = self.b1(self.down1(s0), m)
        s2 = self.b2(self.down2(s1), m)
        h = self.mid(s2, m)
        # temporal mix at bottleneck
        Cb, hh, ww = h.shape[1], h.shape[2], h.shape[3]
        ht = h.view(B, T, Cb, hh, ww).permute(0, 3, 4, 2, 1).reshape(B * hh * ww, Cb, T)
        ht = self.tconv(ht) + ht
        h = ht.reshape(B, hh, ww, Cb, T).permute(0, 4, 3, 1, 2).reshape(B * T, Cb, hh, ww)
        u2 = self.up2(h)
        # match size with s1 (14x24)
        u2 = F.interpolate(u2, size=s1.shape[-2:], mode="nearest") if u2.shape[-2:] != s1.shape[-2:] else u2
        u2 = self.c2(torch.cat([u2, s1], 1), m)
        u1 = self.up1(u2)
        u1 = F.interpolate(u1, size=s0.shape[-2:], mode="nearest") if u1.shape[-2:] != s0.shape[-2:] else u1
        u1 = self.c1(torch.cat([u1, s0], 1), m)
        out = self.outc(F.silu(self.outnorm(u1)))                 # (B*T,16,H,W)
        return out.view(B, T, RGB_C, H, W).permute(0, 2, 1, 3, 4)  # (B,16,T,H,W)


# ---------------------------------------------------------------------------
# Build cond once (no grad through raycaster); helper used by train + sample
# ---------------------------------------------------------------------------
def _exact_eye(raw_cam):
    """raw_cam (B,9,5) UNNORMALIZED (xpos,ypos,zpos,yaw,pitch).
    Per-frame voxel cube re-centering -> grid-local eye.
      bx0=floor(xp)-24 ; by0=clip(floor(yp)-24,0,256-48) ; bz0=floor(zp)-24
      eye=(xp-bx0, (yp-by0)+1.62, zp-bz0)
    Returns origins (B,9,3), yaw (B,9), pit (B,9)."""
    xp = raw_cam[..., 0]; yp = raw_cam[..., 1]; zp = raw_cam[..., 2]
    yaw = raw_cam[..., 3]; pit = raw_cam[..., 4]
    bx0 = torch.floor(xp) - 24.0
    by0 = torch.clamp(torch.floor(yp) - 24.0, 0.0, 256.0 - 48.0)
    bz0 = torch.floor(zp) - 24.0
    origins = torch.stack([xp - bx0, (yp - by0) + 1.62, zp - bz0], dim=-1)
    return origins, yaw, pit


def make_cond(cond_enc, vae, id2compact, vox, cam, dev, rast_h, rast_w, raw_cam=None):
    grids = voxel_to_compact_grids(vae, id2compact, vox, dev)
    if raw_cam is not None:
        # FIX 1: exact camera eye from raw recorded position (per-frame re-centered cube)
        rc = raw_cam[:, LAT_TO_SRC]                              # (B,9,5)
        origins, yaw, pit = _exact_eye(rc)
    else:
        # legacy fallback: grid-center + normalized cumulative delta (WRONG, kept for compat)
        yaw = cam[:, LAT_TO_SRC, 3] * 180.0
        pit = cam[:, LAT_TO_SRC, 4] * 180.0
        dxyz = cam[:, LAT_TO_SRC, 0:3]
        origins = torch.stack([EYE_CENTER + dxyz[..., 0],
                               EYE_CENTER + dxyz[..., 1] + EYE_HEIGHT,
                               EYE_CENTER + dxyz[..., 2]], dim=-1)
    with torch.no_grad():
        if USE_RASTER:
            # FIX 2: z-buffer rasterize every solid voxel (no fixed-step skipping)
            id_map, depth, hit, dirs = rasterize_batch(grids, origins, yaw, pit, rast_h, rast_w)
        else:
            id_map, depth, hit, dirs = batched_raycast(grids, origins, yaw, pit, rast_h, rast_w)
    cond = cond_enc(id_map, depth, hit, dirs)                    # (B,T,cond_ch,27,48)
    cond = cond.permute(0, 2, 1, 3, 4).contiguous()             # (B,cond_ch,T,27,48)
    return cond, hit


@torch.no_grad()
def sample_ode(vnet, cond_enc, vae, id2compact, vox, cam, dev, rast_h, rast_w,
               n_steps=32, seed=0, raw_cam=None):
    """Integrate rectified-flow ODE from tau=1 (noise) -> tau=0. Returns rgb_lat."""
    cond, _ = make_cond(cond_enc, vae, id2compact, vox, cam, dev, rast_h, rast_w, raw_cam=raw_cam)
    B = vox.shape[0]
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(B, RGB_C, RGB_T, RGB_H, RGB_W, device=dev, generator=g)  # x at tau=1
    taus = torch.linspace(1.0, 0.0, n_steps + 1, device=dev)
    for i in range(n_steps):
        tau = taus[i]
        dt = (taus[i] - taus[i + 1])   # positive
        tvec = torch.full((B,), float(tau), device=dev)
        v = vnet(x, tvec, cond)        # dx/dtau = x1 - x0
        x = x - dt * v                 # move toward tau=0
    return x  # predicted clean rgb_lat (B,16,9,27,48)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--rast", type=int, default=72)
    ap.add_argument("--base", type=int, default=128)
    ap.add_argument("--cond_base", type=int, default=96)
    ap.add_argument("--cond_ch", type=int, default=64)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    RAST_H = (args.rast // 4) * 4
    RAST_W = (int(round(RAST_H * 16 / 9)) // 4) * 4
    log(f"RAST {RAST_H}x{RAST_W}")

    dev = torch.device("cuda")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    VOX_VAE = os.environ.get("VOX_VAE", "/scratch/users/flukol/mg_pt_voxel/v14code/vae_final.pt")
    VOX_VOCAB = os.environ.get("VOX_VOCAB", "/scratch/users/flukol/mg_pt_voxel/v14code/voxel_vocab.npz")
    vae = VAE3D(num_classes=2021, latent_channels=48, middle_channels=(32, 128, 512))
    sd = torch.load(VOX_VAE, map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd))
    vae.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    vae = vae.to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    vv = np.load(VOX_VOCAB)
    id2compact = torch.from_numpy(vv["id2compact"]).long().to(dev)

    cond_enc = CondEncoder(base=args.cond_base, cond_ch=args.cond_ch).to(dev)
    vnet = FlowVNet(cond_ch=args.cond_ch, base=args.base).to(dev)
    nparams = sum(p.numel() for p in vnet.parameters()) + sum(p.numel() for p in cond_enc.parameters())
    log(f"flow params: {nparams/1e6:.2f}M")

    params = list(vnet.parameters()) + list(cond_enc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        vnet.load_state_dict(ck["vnet"]); cond_enc.load_state_dict(ck["cond_enc"])
        if "opt" in ck: opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        log(f"RESUMED {args.resume} @ {start_step}")

    ds = VoxelDiTDataset(clip_len=33, require_rgb=True, require_clip=False,
                         require_clip_pf=False, clip_dir=None, clip_pf_dir=None,
                         samples_per_epoch=400000, samples_per_episode=1)
    log(f"valid episodes: {len(ds.index)}")
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers,
                        drop_last=True, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    it = iter(loader)
    t0 = time.time(); ema = None
    n_steps = 8 if args.smoke else args.steps
    scaler = torch.cuda.amp.GradScaler()

    for step in range(start_step, start_step + n_steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        vox = b["voxel_lat"].to(dev, non_blocking=True)
        cam = b["camera"].to(dev, non_blocking=True)
        raw_cam = b["raw_camera"].to(dev, non_blocking=True) if "raw_camera" in b else None
        x0 = b["rgb_lat"].to(dev, non_blocking=True).float()      # (B,16,9,27,48)
        B = vox.shape[0]

        cond, hit = make_cond(cond_enc, vae, id2compact, vox, cam, dev, RAST_H, RAST_W, raw_cam=raw_cam)

        lr = args.lr * min(1.0, (step + 1 - start_step) / max(1, args.warmup))
        for pg in opt.param_groups: pg["lr"] = lr

        # rectified flow
        x1 = torch.randn_like(x0)
        tau = torch.rand(B, device=dev)
        tv = tau.view(B, 1, 1, 1, 1)
        x_tau = (1 - tv) * x0 + tv * x1
        v_target = x1 - x0

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            v_pred = vnet(x_tau, tau, cond)
            loss = F.mse_loss(v_pred.float(), v_target)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update()

        l = loss.item(); ema = l if ema is None else 0.98 * ema + 0.02 * l
        if step % args.log_every == 0 or args.smoke:
            sps = (step + 1 - start_step) / (time.time() - t0)
            log(f"step {step} vloss {l:.4f} ema {ema:.4f} hit% {hit.float().mean().item():.2f} "
                f"{sps:.2f} it/s x0std {x0.std():.3f} vpred_std {v_pred.float().std():.3f}")
        if (step + 1) % args.ckpt_every == 0 and not args.smoke:
            torch.save({"step": step + 1, "vnet": vnet.state_dict(),
                        "cond_enc": cond_enc.state_dict(), "opt": opt.state_dict(),
                        "rast_h": RAST_H, "rast_w": RAST_W,
                        "cond_ch": args.cond_ch, "base": args.base, "cond_base": args.cond_base},
                       os.path.join(args.ckpt_dir, "flow_latest.pt"))
            log(f"  ckpt @ {step+1}")

    if not args.smoke:
        torch.save({"step": start_step + n_steps, "vnet": vnet.state_dict(),
                    "cond_enc": cond_enc.state_dict(), "opt": opt.state_dict(),
                    "rast_h": RAST_H, "rast_w": RAST_W,
                    "cond_ch": args.cond_ch, "base": args.base, "cond_base": args.cond_base},
                   os.path.join(args.ckpt_dir, "flow_final.pt"))
    log("TRAINING DONE")


if __name__ == "__main__":
    main()
