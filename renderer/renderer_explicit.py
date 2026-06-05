"""v14 EXPLICIT-rasterizer renderer (PERSIST-style).

Pipeline per clip:
  voxel_lat (B,48,33,12,12,12) --VAE.decode(argmax)--> block ids (2021 vocab)
      -> map to compact 0..128 ids
  for each of 9 representative frames:
      raycast(grid, eye, yaw, pitch) -> (H,W) compact-id map + depth + hitmask
      build screen buffer: [id_embed | depth | hitmask | ray_dir(3)]
  -> per-frame 2D U-Net -> (16,27,48) latent ; light temporal conv across 9 frames
Target: GT rgb_lat (B,16,9,27,48). Loss: MSE.

The VAE decode of voxel latents is done ONCE per clip (no grad). The shader net
is the only trainable part. This is the explicit-projection design that replaces
the failed implicit renderer_v0.
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
from voxel_dataset import VoxelDiTDataset
from vae_3d import VAE3D
from voxel_raycaster import camera_basis

RGB_T, RGB_C, RGB_H, RGB_W = 9, 16, 27, 48
# 9 Wan latent frames correspond to source frames ~ [0,4,8,...,32]
LAT_TO_SRC = [0, 4, 8, 12, 16, 20, 24, 28, 32]
N_COMPACT = 129          # compact vocab size (0..128)
RAST_H, RAST_W = 56, 96  # rasterization buffer (divisible by 4 for clean U-Net up/down)
FOV_DEG = 70.0
GRID = 48
EYE_CENTER = 24.0
EYE_HEIGHT = 1.6
MAX_STEPS = 120
STEP = 0.5
MAX_DEPTH = MAX_STEPS * STEP


def log(m, r=0):
    if r == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------
# Batched raycaster: B clips, T frames, one grid per (clip,frame).
# grids: (B,T,X,Y,Z) long compact-ids ; origins (B,T,3) ; yaw,pitch (B,T) deg
# Returns id_map (B,T,H,W) long, depth (B,T,H,W), dirs (B,T,H,W,3)
# ---------------------------------------------------------------------------
def batched_raycast(grids, origins, yaw_deg, pitch_deg, H, W, air_id=0):
    dev = grids.device
    B, T, X, Y, Z = grids.shape
    BT = B * T
    g = grids.reshape(BT, X, Y, Z)
    org = origins.reshape(BT, 3)
    yaw = yaw_deg.reshape(BT)
    pit = pitch_deg.reshape(BT)
    fwd, right, up = camera_basis(yaw, pit)          # (BT,3) each

    aspect = W / H
    tan_v = math.tan(math.radians(FOV_DEG) / 2.0)
    tan_h = tan_v * aspect
    ys = torch.linspace(1 - 1.0 / H, -1 + 1.0 / H, H, device=dev)
    xs = torch.linspace(-1 + 1.0 / W, 1 - 1.0 / W, W, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")   # (H,W)
    gx = gx.reshape(1, H, W, 1); gy = gy.reshape(1, H, W, 1)
    dirs = (fwd.view(BT, 1, 1, 3)
            + (gx * tan_h) * right.view(BT, 1, 1, 3)
            + (gy * tan_v) * up.view(BT, 1, 1, 3))    # (BT,H,W,3)
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)

    org_e = org.view(BT, 1, 1, 3)
    id_map = torch.zeros((BT, H, W), dtype=torch.long, device=dev)
    depth = torch.full((BT, H, W), MAX_DEPTH, device=dev)
    hit = torch.zeros((BT, H, W), dtype=torch.bool, device=dev)
    bt_idx = torch.arange(BT, device=dev).view(BT, 1, 1).expand(BT, H, W)

    t = torch.full((BT, H, W), STEP * 0.5, device=dev)
    for _ in range(MAX_STEPS):
        pos = org_e + dirs * t.unsqueeze(-1)
        ix = torch.floor(pos[..., 0]).long()
        iy = torch.floor(pos[..., 1]).long()
        iz = torch.floor(pos[..., 2]).long()
        inside = (ix >= 0) & (ix < X) & (iy >= 0) & (iy < Y) & (iz >= 0) & (iz < Z)
        active = inside & (~hit)
        cix = ix.clamp(0, X - 1); ciy = iy.clamp(0, Y - 1); ciz = iz.clamp(0, Z - 1)
        ids = g[bt_idx, cix, ciy, ciz]
        new_hit = active & (ids != air_id)
        if new_hit.any():
            id_map = torch.where(new_hit, ids, id_map)
            depth = torch.where(new_hit, t, depth)
            hit = hit | new_hit
        t = t + STEP
        if hit.all():
            break
    return (id_map.view(B, T, H, W), depth.view(B, T, H, W),
            hit.view(B, T, H, W), dirs.view(B, T, H, W, 3))


# ---------------------------------------------------------------------------
# Shader net: screen buffer -> rgb latent
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)


class ShaderUNet(nn.Module):
    def __init__(self, emb_dim=32, base=96):
        super().__init__()
        self.emb = nn.Embedding(N_COMPACT, emb_dim)
        in_ch = emb_dim + 1 + 1 + 3   # id_emb + depth + hitmask + ray_dir(3)
        self.inc = DoubleConv(in_ch, base)
        self.d1 = DoubleConv(base, base * 2)
        self.d2 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.c1 = DoubleConv(base * 4, base * 2)
        self.up2 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c2 = DoubleConv(base * 2, base)
        # temporal conv across the 9 frames (channels = base)
        self.tconv = nn.Conv1d(base, base, 3, padding=1)
        self.head = nn.Conv2d(base, RGB_C, 1)

    def forward(self, buf):  # buf: (B,T,Cin,H,W)
        B, T = buf.shape[:2]
        x = buf.reshape(B * T, *buf.shape[2:])
        x0 = self.inc(x)
        x1 = self.d1(self.pool(x0))
        x2 = self.d2(self.pool(x1))
        u1 = self.up1(x2)
        u1 = self.c1(torch.cat([u1, x1], 1))
        u2 = self.up2(u1)
        u2 = self.c2(torch.cat([u2, x0], 1))           # (B*T, base, H, W)
        # temporal mix: pool to latent res first, then mix over T
        feat = F.adaptive_avg_pool2d(u2, (RGB_H, RGB_W))  # (B*T, base, 27,48)
        C = feat.shape[1]
        feat = feat.view(B, T, C, RGB_H, RGB_W).permute(0, 3, 4, 2, 1).reshape(B * RGB_H * RGB_W, C, T)
        feat = self.tconv(feat) + feat
        feat = feat.reshape(B, RGB_H, RGB_W, C, T).permute(0, 4, 3, 1, 2).reshape(B * T, C, RGB_H, RGB_W)
        out = self.head(feat).view(B, T, RGB_C, RGB_H, RGB_W).permute(0, 2, 1, 3, 4)
        return out  # (B,16,9,27,48)


def build_screen_buffer(shader, id_map, depth, hit, dirs):
    # id_map (B,T,H,W) long ; depth (B,T,H,W) ; hit bool ; dirs (B,T,H,W,3)
    B, T, H, W = id_map.shape
    emb = shader.emb(id_map)                                   # (B,T,H,W,E)
    d = (depth / MAX_DEPTH).clamp(0, 1).unsqueeze(-1)          # (B,T,H,W,1)
    h = hit.float().unsqueeze(-1)
    buf = torch.cat([emb, d, h, dirs], dim=-1)                 # (B,T,H,W,Cin)
    return buf.permute(0, 1, 4, 2, 3).contiguous()            # (B,T,Cin,H,W)


@torch.no_grad()
def voxel_to_compact_grids(vae, id2compact, vox, dev):
    """vox (B,48,33,12,12,12) -> compact-id grids for the 9 LAT_TO_SRC frames.
    Returns (B,9,48,48,48) long compact ids."""
    B = vox.shape[0]
    sel = vox[:, :, LAT_TO_SRC]                                # (B,48,9,12,12,12)
    sel = sel.permute(0, 2, 1, 3, 4, 5).reshape(B * RGB_T, 48, 12, 12, 12).float()
    out = []
    for i in range(0, sel.shape[0], 8):
        logit = vae.decode(sel[i:i + 8])                       # (n,2021,48,48,48)
        out.append(logit.argmax(1))
    ids2021 = torch.cat(out, 0)                                # (B*9,48,48,48)
    compact = id2compact[ids2021.clamp(0, 2020)]              # (B*9,48,48,48)
    return compact.view(B, RGB_T, 48, 48, 48).long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--ckpt_every", type=int, default=500)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    # VAE (frozen) + compact map
    vae = VAE3D(num_classes=2021, latent_channels=48, middle_channels=(32, 128, 512))
    sd = torch.load("/scratch/users/flukol/mg_pt_voxel/v14code/vae_final.pt", map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd))
    vae.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    vae = vae.to(dev).eval()
    for p in vae.parameters(): p.requires_grad_(False)
    vv = np.load("/scratch/users/flukol/mg_pt_voxel/v14code/voxel_vocab.npz")
    id2compact = torch.from_numpy(vv["id2compact"]).long().to(dev)

    shader = ShaderUNet().to(dev)
    log(f"shader params: {sum(p.numel() for p in shader.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(shader.parameters(), lr=args.lr, betas=(0.9, 0.95))

    ds = VoxelDiTDataset(clip_len=33, require_rgb=True, require_clip=False,
                         require_clip_pf=False, clip_dir=None, clip_pf_dir=None,
                         samples_per_epoch=200000, samples_per_episode=1)
    log(f"valid episodes: {len(ds.index)}")
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers,
                        drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0)

    it = iter(loader)
    t0 = time.time(); ema = None
    n_steps = 5 if args.smoke else args.steps
    for step in range(n_steps):
        try:
            b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        vox = b["voxel_lat"].to(dev, non_blocking=True)          # (B,48,33,12,12,12)
        cam = b["camera"].to(dev, non_blocking=True)             # (B,33,5) normalized
        target = b["rgb_lat"].to(dev, non_blocking=True).float() # (B,16,9,27,48)
        B = vox.shape[0]

        grids = voxel_to_compact_grids(vae, id2compact, vox, dev)  # (B,9,48,48,48)
        yaw = cam[:, LAT_TO_SRC, 3] * 180.0                       # (B,9) deg
        pit = cam[:, LAT_TO_SRC, 4] * 180.0
        dxyz = cam[:, LAT_TO_SRC, 0:3]                            # (B,9,3) block deltas
        origins = torch.stack([
            EYE_CENTER + dxyz[..., 0],
            EYE_CENTER + dxyz[..., 1] + EYE_HEIGHT,
            EYE_CENTER + dxyz[..., 2],
        ], dim=-1)                                                # (B,9,3)

        with torch.no_grad():
            id_map, depth, hit, dirs = batched_raycast(
                grids, origins, yaw, pit, RAST_H, RAST_W)
        buf = build_screen_buffer(shader, id_map, depth, hit, dirs)

        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for pg in opt.param_groups: pg["lr"] = lr

        pred = shader(buf)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shader.parameters(), 1.0)
        opt.step()

        l = loss.item(); ema = l if ema is None else 0.98 * ema + 0.02 * l
        if step % args.log_every == 0 or args.smoke:
            sps = (step + 1) / (time.time() - t0)
            log(f"step {step} loss {l:.4f} ema {ema:.4f} "
                f"hit% {hit.float().mean().item():.2f} {sps:.2f} it/s "
                f"pred[min{pred.min():.2f} max{pred.max():.2f} std{pred.std():.3f}] "
                f"tgt[std{target.std():.3f}]")
        if (step + 1) % args.ckpt_every == 0 and not args.smoke:
            torch.save({"step": step + 1, "state_dict": shader.state_dict()},
                       os.path.join(args.ckpt_dir, "shader_latest.pt"))
    if not args.smoke:
        torch.save({"step": n_steps, "state_dict": shader.state_dict()},
                   os.path.join(args.ckpt_dir, "shader_final.pt"))
    log("TRAINING DONE")


if __name__ == "__main__":
    main()
