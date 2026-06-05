"""render_v3 trainer: temporal-coherent + sharper neural voxel->RGB renderer.

Builds on renderer_flow.py (rectified-flow DiT shader) WITHOUT modifying it.
Adds the four upgrades requested for v3:

  (1) TEMPORAL CONDITIONING (motion fix)
      - FlowVNetV3 gets a prev-frame branch: each clip frame t is conditioned on
        the CLEAN rgb-latent of frame t-1 (teacher-forced, autoregressive in time).
        Frame 0 sees a zero prev + a learned first-frame embedding.
        The prev branch + its input projection are ZERO-INIT so a warm-started
        model is numerically identical to the v2 model at step 0.
      - TEMPORAL-CONSISTENCY LOSS: on STATIC screen regions (same id & ~same depth
        in consecutive rasterized frames), penalise L1 between the model's predicted
        clean frames x0_pred[t] and x0_pred[t-1]. Camera motion (changed pixels) is
        exempt, so real motion is not over-smoothed -> kills flicker, keeps motion.

  (2) RICHER CONDITIONING (sharpness)
      - raster_peel.rasterize_peel: depth-peel (first K voxel hits per ray, not 1)
        + per-hit surface normals.  CondEncoderV3 ingests the extra layers/normals;
        the extra input channels are zero-init in the first conv so warm-start ==
        v2 (which used K=1, no normals).

  (3) PERCEPTUAL LOSS (sharpness)
      - small-weight VGG16 feature loss on the Wan-decoded RGB of a subset of frames
        (pushes HF detail past the flow/MSE blur floor). Falls back to a deterministic
        multi-scale high-frequency (Laplacian) loss if pretrained VGG is unavailable,
        so the run never hard-fails on a missing download.

  (4) plumbing: bf16, grad-ckpt, torchrun --standalone, ckpt every ~20 min,
      PERIODIC (kept) checkpoints + a rolling latest, self-resume.

Warm-start: --resume points at the v2 BEST ckpt (keys "vnet","cond_enc"); we load
those weights into the v3 modules' shared submodules by NAME (strict=False) and
leave the new modules at their zero-init.
"""
from __future__ import annotations
import argparse, os, sys, time, math
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voxel_dataset import VoxelDiTDataset
from vae_3d import VAE3D
from renderer_explicit import (voxel_to_compact_grids, LAT_TO_SRC,
                               EYE_CENTER, EYE_HEIGHT,
                               RGB_T, RGB_C, RGB_H, RGB_W, N_COMPACT)
from renderer_flow import (timestep_embedding, AdaGNBlock, _exact_eye)
from raster_torch import MAX_DEPTH
from raster_peel import rasterize_peel

USE_RASTER = os.environ.get('USE_RASTER', '1') == '1'


def is_dist():
    return dist.is_available() and dist.is_initialized()


def rank0():
    return (not is_dist()) or dist.get_rank() == 0


def log(m):
    if rank0():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


# ===========================================================================
# (2) Conditioning encoder with depth-peel layers + normals.
#     Backward-compatible weight layout: the FIRST conv's first slice of input
#     channels matches the v2 CondEncoder (id_emb + depth + hit + dirs); the
#     extra channels (layers 1..K-1 + normals) get a SEPARATE zero-init conv that
#     is added in, so a warm-started model == v2 at step 0.
# ===========================================================================
class CondEncoderV3(nn.Module):
    def __init__(self, emb_dim=32, base=96, cond_ch=64, K=2):
        super().__init__()
        self.K = K
        self.emb = nn.Embedding(N_COMPACT, emb_dim)            # shared w/ v2 (name "emb")
        # v2 input layout for layer-0: id_emb + depth(1) + hit(1) + dirs(3)
        self.in_base = emb_dim + 1 + 1 + 3
        # extra per-layer channels for layers 1..K-1: id_emb + depth + hit  (no dirs; shared)
        # + normals for ALL K layers: 3*K
        self.in_extra = (K - 1) * (emb_dim + 1 + 1) + 3 * K
        # v2-compatible trunk (names match renderer_flow.CondEncoder.enc / .proj)
        self.enc = nn.Sequential(
            nn.Conv2d(self.in_base, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, padding=1), nn.GroupNorm(8, base * 2), nn.SiLU(),
        )
        self.proj = nn.Conv2d(base * 2, cond_ch, 1)
        # NEW: zero-init injection of the extra (peel+normal) channels into the trunk's first activation
        self.extra_in = nn.Conv2d(self.in_extra, base, 3, padding=1)
        nn.init.zeros_(self.extra_in.weight); nn.init.zeros_(self.extra_in.bias)

    def _layer_buf(self, id_map_k, depth_k, hit_k):
        emb = self.emb(id_map_k)
        d = (depth_k / MAX_DEPTH).clamp(0, 1).unsqueeze(-1)
        h = hit_k.float().unsqueeze(-1)
        return torch.cat([emb, d, h], dim=-1)                  # (B,T,H,W, emb+2)

    def forward(self, id_map, depth, hit, normal, dirs):
        # id_map/depth/hit (B,T,K,H,W) ; normal (B,T,K,H,W,3) ; dirs (B,T,H,W,3)
        B, T, K, H, W = id_map.shape
        # layer 0 == v2 buffer
        buf0 = torch.cat([self._layer_buf(id_map[:, :, 0], depth[:, :, 0], hit[:, :, 0]),
                          dirs], dim=-1)                       # (B,T,H,W, in_base)
        x0 = buf0.permute(0, 1, 4, 2, 3).reshape(B * T, self.in_base, H, W)
        # extra channels: layers 1..K-1 buffers + all normals
        extra = []
        for k in range(1, K):
            extra.append(self._layer_buf(id_map[:, :, k], depth[:, :, k], hit[:, :, k]))
        extra.append(normal.reshape(B, T, K, H, W, 3).permute(0, 1, 3, 4, 2, 5).reshape(B, T, H, W, 3 * K))
        ex = torch.cat(extra, dim=-1).permute(0, 1, 4, 2, 3).reshape(B * T, self.in_extra, H, W)

        # first conv of trunk + zero-init extra injection (== v2 first-conv act at step0)
        h0 = self.enc[0](x0) + self.extra_in(ex)
        h = self.enc[1](h0); h = self.enc[2](h)               # GN, SiLU
        for layer in self.enc[3:]:
            h = layer(h)
        h = self.proj(h)
        h = F.adaptive_avg_pool2d(h, (RGB_H, RGB_W))
        return h.view(B, T, -1, RGB_H, RGB_W)


# ===========================================================================
# (1) Flow velocity net with a prev-frame (autoregressive) branch.
#     Same trunk/names as renderer_flow.FlowVNet so v2 weights load by name.
#     New: prev_proj (zero-init) maps prev clean frame -> extra input channels
#     that are ADDED to the post-inc activation; first-frame learned token.
# ===========================================================================
class FlowVNetV3(nn.Module):
    def __init__(self, cond_ch=64, base=128, tdim=256, grad_ckpt=False):
        super().__init__()
        self.cond_ch = cond_ch; self.tdim = tdim; self.grad_ckpt = grad_ckpt
        self.t_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.cond_summary = nn.Sequential(nn.Linear(cond_ch, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        in_ch = RGB_C + cond_ch
        self.inc = nn.Conv2d(in_ch, base, 3, padding=1)
        self.b0 = AdaGNBlock(base, base, tdim)
        self.down1 = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.b1 = AdaGNBlock(base * 2, base * 2, tdim)
        self.down2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)
        self.b2 = AdaGNBlock(base * 2, base * 2, tdim)
        self.mid = AdaGNBlock(base * 2, base * 2, tdim)
        self.tconv = nn.Conv1d(base * 2, base * 2, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(base * 2, base * 2, 2, stride=2)
        self.c2 = AdaGNBlock(base * 4, base * 2, tdim)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.c1 = AdaGNBlock(base * 2, base, tdim)
        self.outnorm = nn.GroupNorm(8, base)
        self.outc = nn.Conv2d(base, RGB_C, 3, padding=1)
        nn.init.zeros_(self.outc.weight); nn.init.zeros_(self.outc.bias)

        # NEW (temporal): prev-frame branch, zero-init -> warm-start == v2
        self.prev_proj = nn.Conv2d(RGB_C, base, 3, padding=1)
        nn.init.zeros_(self.prev_proj.weight); nn.init.zeros_(self.prev_proj.bias)
        # learned "first frame" token added to the prev branch when t==0
        self.first_frame = nn.Parameter(torch.zeros(1, RGB_C, 1, 1))

    def _core(self, x, m):
        s0 = self.b0(x, m)
        s1 = self.b1(self.down1(s0), m)
        s2 = self.b2(self.down2(s1), m)
        h = self.mid(s2, m)
        return s0, s1, s2, h

    def forward(self, x_tau, tau, cond, prev_frame, first_mask):
        # x_tau,cond (B,*,T,H,W) ; tau (B,) ; prev_frame (B,RGB_C,T,H,W) ; first_mask (B,T) bool
        B, _, T, H, W = x_tau.shape
        temb = self.t_mlp(timestep_embedding(tau, self.tdim))
        csum = cond.mean(dim=(2, 3, 4))
        m = temb + self.cond_summary(csum)
        m = m.unsqueeze(1).expand(B, T, self.tdim).reshape(B * T, self.tdim)

        x = torch.cat([x_tau, cond], dim=1)
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, RGB_C + self.cond_ch, H, W)
        x = self.inc(x)

        # prev-frame conditioning (zero-init branch): add to post-inc activation
        pf = prev_frame.permute(0, 2, 1, 3, 4).reshape(B * T, RGB_C, H, W)
        ff = self.first_frame.expand(B * T, RGB_C, H, W)
        fmask = first_mask.reshape(B * T, 1, 1, 1).float()
        pf = pf * (1 - fmask) + ff * fmask
        x = x + self.prev_proj(pf)

        if self.grad_ckpt and self.training:
            s0, s1, s2, h = checkpoint(self._core, x, m, use_reentrant=False)
        else:
            s0, s1, s2, h = self._core(x, m)

        Cb, hh, ww = h.shape[1], h.shape[2], h.shape[3]
        ht = h.view(B, T, Cb, hh, ww).permute(0, 3, 4, 2, 1).reshape(B * hh * ww, Cb, T)
        ht = self.tconv(ht) + ht
        h = ht.reshape(B, hh, ww, Cb, T).permute(0, 4, 3, 1, 2).reshape(B * T, Cb, hh, ww)
        u2 = self.up2(h)
        u2 = F.interpolate(u2, size=s1.shape[-2:], mode="nearest") if u2.shape[-2:] != s1.shape[-2:] else u2
        u2 = self.c2(torch.cat([u2, s1], 1), m)
        u1 = self.up1(u2)
        u1 = F.interpolate(u1, size=s0.shape[-2:], mode="nearest") if u1.shape[-2:] != s0.shape[-2:] else u1
        u1 = self.c1(torch.cat([u1, s0], 1), m)
        out = self.outc(F.silu(self.outnorm(u1)))
        return out.view(B, T, RGB_C, H, W).permute(0, 2, 1, 3, 4)


# ===========================================================================
# (2/cond) build peeled cond
# ===========================================================================
def make_cond_v3(cond_enc, vae, id2compact, vox, cam, dev, rast_h, rast_w, K, raw_cam=None):
    grids = voxel_to_compact_grids(vae, id2compact, vox, dev)
    if raw_cam is not None:
        rc = raw_cam[:, LAT_TO_SRC]
        origins, yaw, pit = _exact_eye(rc)
    else:
        yaw = cam[:, LAT_TO_SRC, 3] * 180.0
        pit = cam[:, LAT_TO_SRC, 4] * 180.0
        dxyz = cam[:, LAT_TO_SRC, 0:3]
        origins = torch.stack([EYE_CENTER + dxyz[..., 0],
                               EYE_CENTER + dxyz[..., 1] + EYE_HEIGHT,
                               EYE_CENTER + dxyz[..., 2]], dim=-1)
    with torch.no_grad():
        id_map, depth, hit, normal, dirs = rasterize_peel(grids, origins, yaw, pit, rast_h, rast_w, K=K)
    cond = cond_enc(id_map, depth, hit, normal, dirs)          # (B,T,cond_ch,27,48)
    cond = cond.permute(0, 2, 1, 3, 4).contiguous()           # (B,cond_ch,T,27,48)
    # static mask for temporal-consistency loss: nearest-layer id unchanged AND depth ~unchanged
    # downsample to latent grid (27x48), per consecutive frame pair.
    id0 = id_map[:, :, 0].float()                             # (B,T,H,W)
    d0 = (depth[:, :, 0] / MAX_DEPTH).clamp(0, 1)
    id0s = F.adaptive_avg_pool2d(id0, (RGB_H, RGB_W))          # (B,T,27,48)
    d0s = F.adaptive_avg_pool2d(d0, (RGB_H, RGB_W))
    same_id = (id0s[:, 1:] - id0s[:, :-1]).abs() < 0.5
    same_d = (d0s[:, 1:] - d0s[:, :-1]).abs() < 0.02
    static = (same_id & same_d).float().unsqueeze(1)          # (B,1,T-1,27,48)
    return cond, hit, static


# ===========================================================================
# (3) Perceptual loss: VGG16 features w/ deterministic HF fallback.
# ===========================================================================
class PerceptualLoss(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.kind = "hf"
        self.vgg = None
        try:
            import torchvision
            try:
                from torchvision.models import VGG16_Weights
                net = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            except Exception:
                net = torchvision.models.vgg16(pretrained=True)
            # use features up to relu3_3 (index 16)
            self.vgg = net.features[:16].eval().to(dev)
            for p in self.vgg.parameters():
                p.requires_grad_(False)
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.kind = "vgg"
        except Exception as e:
            log(f"[perceptual] VGG unavailable ({type(e).__name__}: {e}); using HF-Laplacian fallback")

    def _lap(self, x):
        k = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                         device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
        k = k.expand(x.shape[1], 1, 3, 3)
        return F.conv2d(x, k, padding=1, groups=x.shape[1])

    def forward(self, pred_rgb, gt_rgb):
        # pred/gt: (N,3,H,W) in [-1,1]
        if self.kind == "vgg":
            p = (pred_rgb.clamp(-1, 1) + 1) / 2
            g = (gt_rgb.clamp(-1, 1) + 1) / 2
            p = (p - self.mean) / self.std
            g = (g - self.mean) / self.std
            fp = self.vgg(p); fg = self.vgg(g)
            return F.l1_loss(fp, fg)
        # fallback: multi-scale Laplacian (HF) L1
        loss = 0.0
        for _ in range(3):
            loss = loss + F.l1_loss(self._lap(pred_rgb), self._lap(gt_rgb))
            pred_rgb = F.avg_pool2d(pred_rgb, 2)
            gt_rgb = F.avg_pool2d(gt_rgb, 2)
        return loss / 3.0


def load_wan(dev):
    """Optional Wan VAE for perceptual decode. Returns module or None."""
    import importlib.util as u
    WAN_MODULE = os.environ.get("WAN_MODULE", "/home/flukol/v12/wan/modules/vae.py")
    WAN_PTH = os.environ.get("WAN_PTH", "/home/flukol/v12/mg2_weights/Wan2.1_VAE.pth")
    if not (os.path.exists(WAN_MODULE) and os.path.exists(WAN_PTH)):
        log(f"[perceptual] Wan VAE not found ({WAN_MODULE} / {WAN_PTH}); perceptual loss disabled")
        return None
    spec = u.spec_from_file_location("wm", WAN_MODULE)
    m = u.module_from_spec(spec); spec.loader.exec_module(m)
    return m.WanVAE(vae_pth=WAN_PTH, device=dev)


# ===========================================================================
# warm-start v2 -> v3 (load shared submodules by name)
# ===========================================================================
def _filtered_load(module, sd, name):
    """Load only keys whose name AND shape match (skip dim-mismatched tensors)."""
    cur = module.state_dict()
    use = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
    skipped = [k for k in sd if k in cur and cur[k].shape != sd[k].shape]
    res = module.load_state_dict(use, strict=False)
    new_keys = [k for k in cur if k not in sd]   # genuinely new (zero-init) modules
    log(f"  {name}: loaded {len(use)}/{len(cur)} | new(zero-init)={new_keys} | "
        f"shape-skipped={skipped}")


def warm_start(cond_enc, vnet, ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    log(f"warm-start <- {ckpt_path} (step {ck.get('step')})")
    _filtered_load(cond_enc, ck["cond_enc"], "cond_enc")
    _filtered_load(vnet, ck["vnet"], "vnet")
    return ck.get("step", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--rast", type=int, default=96)
    ap.add_argument("--base", type=int, default=160)
    ap.add_argument("--cond_base", type=int, default=96)
    ap.add_argument("--cond_ch", type=int, default=64)
    ap.add_argument("--peel_k", type=int, default=2)
    ap.add_argument("--clip_frames", type=int, default=9, help="latent frames per clip (<=9)")
    ap.add_argument("--w_temporal", type=float, default=0.1)
    ap.add_argument("--w_percep", type=float, default=0.05)
    ap.add_argument("--percep_frames", type=int, default=2, help="frames decoded for perceptual loss")
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--ckpt_min", type=float, default=20.0, help="rolling ckpt cadence (min)")
    ap.add_argument("--keep_every", type=int, default=2000, help="keep a PERMANENT ckpt every N steps")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--resume", type=str, default="", help="v2 BEST ckpt (warm-start) or v3 latest")
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--no_percep", action="store_true")
    ap.add_argument("--overfit", action="store_true", help="smoke: lock to one batch to prove loss decreases")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # ---- distributed (torchrun --standalone) ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    RAST_H = (args.rast // 4) * 4
    RAST_W = (int(round(RAST_H * 16 / 9)) // 4) * 4
    K = args.peel_k
    log(f"RAST {RAST_H}x{RAST_W}  peel_K={K}  clip_frames={args.clip_frames}  world={world}")
    if rank0():
        os.makedirs(args.ckpt_dir, exist_ok=True)

    VOX_VAE = os.environ.get("VOX_VAE", "/home/flukol/v12/code/vae_final.pt")
    VOX_VOCAB = os.environ.get("VOX_VOCAB", "/home/flukol/v12/code/voxel_vocab.npz")
    vae = VAE3D(num_classes=2021, latent_channels=48, middle_channels=(32, 128, 512))
    sd = torch.load(VOX_VAE, map_location="cpu")
    sd = sd.get("model", sd.get("state_dict", sd))
    vae.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    vae = vae.to(dev).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vv = np.load(VOX_VOCAB)
    id2compact = torch.from_numpy(vv["id2compact"]).long().to(dev)

    cond_enc = CondEncoderV3(base=args.cond_base, cond_ch=args.cond_ch, K=K).to(dev)
    vnet = FlowVNetV3(cond_ch=args.cond_ch, base=args.base, grad_ckpt=args.grad_ckpt).to(dev)
    nparams = sum(p.numel() for p in vnet.parameters()) + sum(p.numel() for p in cond_enc.parameters())
    log(f"flow params: {nparams/1e6:.2f}M")

    # ---- resume / warm-start ----
    start_step = 0
    params = list(vnet.parameters()) + list(cond_enc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    latest_v3 = os.path.join(args.ckpt_dir, "v3_latest.pt")
    if os.path.exists(latest_v3):
        ck = torch.load(latest_v3, map_location="cpu")
        cond_enc.load_state_dict(ck["cond_enc"]); vnet.load_state_dict(ck["vnet"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        log(f"RESUMED v3 {latest_v3} @ {start_step}")
    elif args.resume and os.path.exists(args.resume):
        start_step = 0  # warm-start = fresh schedule
        warm_start(cond_enc, vnet, args.resume)

    # ---- DDP wrap ----
    if world > 1:
        cond_enc = nn.parallel.DistributedDataParallel(cond_enc, device_ids=[local_rank])
        vnet = nn.parallel.DistributedDataParallel(vnet, device_ids=[local_rank],
                                                   find_unused_parameters=False)

    # ---- perceptual ----
    percep = None; wan = None
    if not args.no_percep and args.w_percep > 0:
        percep = PerceptualLoss(dev).to(dev)
        wan = load_wan(dev)
        if wan is None:
            percep = None  # cannot decode -> disable
            log("[perceptual] disabled (no Wan VAE)")

    # ---- data ----
    ds = VoxelDiTDataset(clip_len=33, require_rgb=True, require_clip=False,
                         require_clip_pf=False, clip_dir=None, clip_pf_dir=None,
                         samples_per_epoch=400000, samples_per_episode=1)
    log(f"valid episodes: {len(ds.index)}")
    sampler = None
    if world > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True, drop_last=True)
    loader = DataLoader(ds, batch_size=args.batch, num_workers=args.num_workers,
                        drop_last=True, pin_memory=True, sampler=sampler,
                        shuffle=(sampler is None),
                        persistent_workers=args.num_workers > 0)

    it = iter(loader)
    t0 = time.time(); ema = None; last_ckpt = time.time()
    n_steps = (120 if args.overfit else 40) if args.smoke else args.steps
    Tf = min(args.clip_frames, RGB_T)
    fixed_b = next(it) if args.overfit else None   # smoke: lock to one batch to prove loss drops

    for step in range(start_step, start_step + n_steps):
        if args.overfit:
            b = fixed_b
        else:
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader); b = next(it)
        vox = b["voxel_lat"].to(dev, non_blocking=True)
        cam = b["camera"].to(dev, non_blocking=True)
        raw_cam = b["raw_camera"].to(dev, non_blocking=True) if "raw_camera" in b else None
        x0_full = b["rgb_lat"].to(dev, non_blocking=True).float()   # (B,16,9,27,48)
        B = vox.shape[0]

        cond, hit, static = make_cond_v3(cond_enc, vae, id2compact, vox, cam, dev,
                                         RAST_H, RAST_W, K, raw_cam=raw_cam)
        # short-clip window: first Tf frames
        x0 = x0_full[:, :, :Tf]
        cond = cond[:, :, :Tf]
        static = static[:, :, :Tf - 1]

        # (1) prev-frame teacher forcing: prev[t] = clean x0[t-1] ; frame0 -> first token
        prev = torch.zeros_like(x0)
        prev[:, :, 1:] = x0[:, :, :-1]
        first_mask = torch.zeros(B, Tf, dtype=torch.bool, device=dev)
        first_mask[:, 0] = True

        lr = args.lr * min(1.0, (step + 1 - start_step) / max(1, args.warmup))
        for pg in opt.param_groups:
            pg["lr"] = lr

        x1 = torch.randn_like(x0)
        tau = torch.rand(B, device=dev)
        tv = tau.view(B, 1, 1, 1, 1)
        x_tau = (1 - tv) * x0 + tv * x1
        v_target = x1 - x0

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            v_pred = vnet(x_tau, tau, cond, prev, first_mask)
            flow_loss = F.mse_loss(v_pred.float(), v_target)
            # predicted clean frame: x0 = x_tau - tau * v
            x0_pred = (x_tau - tv * v_pred).float()
            # (1) temporal-consistency on static regions
            dpred = (x0_pred[:, :, 1:] - x0_pred[:, :, :-1]).abs()    # (B,16,Tf-1,27,48)
            tc_loss = (dpred * static).sum() / (static.sum() * RGB_C + 1e-6)
            loss = flow_loss + args.w_temporal * tc_loss

        # (3) perceptual on a couple decoded frames.
        #   Use the INNER VAE decoder (wan.model.decode) directly so the graph is
        #   kept (the outer WanVAE.decode returns a detached python list). VAE
        #   params are frozen but grad still flows to x0_pred through activations.
        pc_loss_val = 0.0
        if percep is not None:
            nf = min(args.percep_frames, Tf)
            idxs = torch.linspace(0, Tf - 1, nf).long()
            pl = x0_pred[:, :, idxs]                                  # (B,16,nf,27,48)
            gl = x0[:, :, idxs]
            with torch.no_grad():
                gt_rgb = wan.model.decode(gl.detach().float(), wan.scale).float()   # (B,3,nf,H,W)
            pred_rgb = wan.model.decode(pl.float(), wan.scale).float()
            # collapse temporal -> batch of images
            pr = pred_rgb.permute(0, 2, 1, 3, 4).reshape(-1, 3, *pred_rgb.shape[-2:])
            gr = gt_rgb.permute(0, 2, 1, 3, 4).reshape(-1, 3, *gt_rgb.shape[-2:])
            pc_loss = percep(pr, gr.detach())
            loss = loss + args.w_percep * pc_loss
            pc_loss_val = pc_loss.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        l = loss.item(); ema = l if ema is None else 0.98 * ema + 0.02 * l
        if step % args.log_every == 0 or args.smoke:
            sps = (step + 1 - start_step) / (time.time() - t0)
            log(f"step {step} loss {l:.4f} ema {ema:.4f} | flow {flow_loss.item():.4f} "
                f"tc {tc_loss.item():.4f} pc {pc_loss_val:.4f} | static% {static.mean().item():.2f} "
                f"hit% {hit[:, :, 0].float().mean().item():.2f} {sps:.2f} it/s "
                f"vstd {v_pred.float().std():.3f}")

        # ---- checkpointing: rolling latest (time) + PERMANENT every keep_every ----
        if rank0() and not args.smoke:
            now = time.time()
            ce_sd = (cond_enc.module if world > 1 else cond_enc).state_dict()
            vn_sd = (vnet.module if world > 1 else vnet).state_dict()
            meta = {"step": step + 1, "vnet": vn_sd, "cond_enc": ce_sd, "opt": opt.state_dict(),
                    "rast_h": RAST_H, "rast_w": RAST_W, "cond_ch": args.cond_ch,
                    "base": args.base, "cond_base": args.cond_base, "peel_k": K,
                    "clip_frames": args.clip_frames}
            if (now - last_ckpt) > args.ckpt_min * 60:
                torch.save(meta, latest_v3); last_ckpt = now
                log(f"  rolling ckpt @ {step+1}")
            if (step + 1) % args.keep_every == 0:
                kp = os.path.join(args.ckpt_dir, f"v3_step{step+1}.pt")
                torch.save(meta, kp); last_ckpt = now
                log(f"  PERMANENT ckpt -> {kp}")

    if rank0() and not args.smoke:
        ce_sd = (cond_enc.module if world > 1 else cond_enc).state_dict()
        vn_sd = (vnet.module if world > 1 else vnet).state_dict()
        torch.save({"step": start_step + n_steps, "vnet": vn_sd, "cond_enc": ce_sd,
                    "opt": opt.state_dict(), "rast_h": RAST_H, "rast_w": RAST_W,
                    "cond_ch": args.cond_ch, "base": args.base, "cond_base": args.cond_base,
                    "peel_k": K, "clip_frames": args.clip_frames},
                   os.path.join(args.ckpt_dir, "v3_final.pt"))
    log("TRAINING DONE")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
