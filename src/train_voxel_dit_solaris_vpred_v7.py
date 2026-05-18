"""V7 trainer for voxel_dit_solaris (v-prediction — Solaris-faithful loss).

V7 changes vs v6:
  - Feeds PER-FRAME CLIP CLS context (33 tokens per chunk) into cross-attn
    instead of v6's single pooled token. Targets the "new voxels at cube edges"
    failure mode in v6 renders: with per-frame visual signal, model can see new
    terrain entering view before predicting it. Cross-attn handles var-len
    context natively so model code is unchanged.
  - Drops the v6 `.mean(dim=1, keepdim=True)` averaging step.
  - New default --clip_pf_dir points to clip_features_h_pf (built by
    extract_clip_per_frame.py).
  - require_clip_pf=True so only episodes with per-frame CLIP enter training.

IDENTICAL to v6 otherwise: position_cond=camera[..., 0:3], cond_concat anchor,
v-prediction loss, bf16 autocast, fp32 master, Solaris hparams.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import nullcontext as _nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

# add inf_v8 path so we can import voxel_dataset
INF8 = "/scratch/users/flukol/mg_pt_voxel/inf_v8/mg_pt_voxel_local"
# NOTE: top-level path inserted LAST so it wins over inf_v8/ legacy copies.
# inf_v8/ has older voxel_dataset.py (no FIX 31 rank/world_size kwargs) so we
# must prefer the top-level updated copy.
sys.path.insert(0, INF8)
sys.path.insert(0, "/scratch/users/flukol/mg_pt_voxel")

from voxel_dataset import VoxelDiTDataset
from voxel_dit_solaris_v7 import build_voxel_dit_solaris_270m  # v7 = v6 arch + per-frame CLIP


CLIP_DIR_DEFAULT = "/scratch/users/flukol/mg_pt_voxel/extract/clip_features_h"
CLIP_PF_DIR_DEFAULT = "/scratch/users/flukol/mg_pt_voxel/extract/clip_features_h_pf"
NUM_TIMESTEPS = 1000  # bucket count for sinusoidal time embedding
LATENT_STATS_PATH = "/scratch/users/flukol/mg_pt_voxel/voxel_latent_per_channel_stats.npz"


def log(msg, rank):
    if rank == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_ddp():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local)
    return rank, world, local


_ZERO_SENTINEL = None
def _zero_sentinel(device, dtype):
    """FIX 30 (2026-05-08): VAE-encoded all-air sentinel for t>=1 cond_concat,
    matching Solaris's vae.cacheless_encode([first, zeros_pixels]) pattern."""
    global _ZERO_SENTINEL
    if _ZERO_SENTINEL is None:
        path = "/scratch/users/flukol/mg_pt_voxel/zero_sentinel_normalized.pt"
        _ZERO_SENTINEL = torch.load(path, map_location="cpu")
    return _ZERO_SENTINEL.to(device=device, dtype=dtype)

def first_frame_cond_concat_voxel(voxel_lat):
    """Build (B, 52, T, X, Y, Z) cond_concat tensor.

    Concatenates first-frame voxel latent at t=0 + VAE-encoded-air sentinel at
    t>=1 + 4-channel mask (1 at t=0, 0 elsewhere). FIX 30.
    """
    B, C, T, X, Y, Z = voxel_lat.shape
    sentinel = _zero_sentinel(voxel_lat.device, voxel_lat.dtype).expand(B, C, T - 1, X, Y, Z)
    first_only = torch.cat([voxel_lat[:, :, :1], sentinel], dim=2)
    mask = torch.zeros(B, 4, T, X, Y, Z,
                       dtype=voxel_lat.dtype, device=voxel_lat.device)
    mask[:, :, :1] = 1.0
    return torch.cat([mask, first_only], dim=1)  # (B, 4+48=52, T, X, Y, Z)


def load_or_compute_latent_stats(rank, dataset_n_samples=200, device="cuda"):
    """Load per-channel mean+std (48-dim each) from cache, or compute on first run.

    Uses a small subsample to estimate the corpus-wide stats. Computed once on rank 0,
    saved to disk, and broadcast to other ranks.
    """
    if os.path.exists(LATENT_STATS_PATH):
        d = np.load(LATENT_STATS_PATH)
        mean = torch.from_numpy(d["mean"]).float()  # (48,)
        std = torch.from_numpy(d["std"]).float()    # (48,)
        return mean.to(device), std.to(device)
    if rank != 0:
        # other ranks wait until rank 0 computes
        if dist.is_initialized():
            dist.barrier()
        d = np.load(LATENT_STATS_PATH)
        mean = torch.from_numpy(d["mean"]).float()
        std = torch.from_numpy(d["std"]).float()
        return mean.to(device), std.to(device)
    log(f"computing per-channel latent stats from {dataset_n_samples} samples...", rank)
    # IMPORTANT: require_rgb=False — must match the main training dataset's filter set,
    # otherwise stats are computed from a different episode subset (or zero episodes if
    # RGB latents not built).
    ds = VoxelDiTDataset(clip_len=33, samples_per_epoch=dataset_n_samples,
                         samples_per_episode=4, require_rgb=False, require_clip=False,
                         clip_dir=None, verbose=False)
    loader = DataLoader(ds, batch_size=1, num_workers=4)
    sums = torch.zeros(48, dtype=torch.float64)
    sq_sums = torch.zeros(48, dtype=torch.float64)
    n = 0
    for i, b in enumerate(loader):
        v = b["voxel_lat"].float()  # (1, 48, T, X, Y, Z)
        flat = v.flatten(2).flatten(0, 1).reshape(48, -1).double()  # (48, samples)
        sums += flat.sum(dim=1)
        sq_sums += flat.pow(2).sum(dim=1)
        n += flat.size(1)
        if i + 1 >= dataset_n_samples:
            break
    mean = sums / n
    var = (sq_sums / n) - mean.pow(2)
    std = var.clamp_min(1e-6).sqrt()
    np.savez(LATENT_STATS_PATH,
             mean=mean.float().numpy(), std=std.float().numpy())
    log(f"latent stats: mean range [{mean.min():.4f}, {mean.max():.4f}], "
        f"std range [{std.min():.4f}, {std.max():.4f}]", rank)
    if dist.is_initialized():
        dist.barrier()
    return mean.float().to(device), std.float().to(device)


def normalize_latent(voxel_lat, mean, std):
    """(B, 48, T, X, Y, Z) -> normalized."""
    m = mean.view(1, 48, 1, 1, 1, 1)
    s = std.view(1, 48, 1, 1, 1, 1)
    return (voxel_lat - m) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=2, help="per-GPU micro-batch")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    # FIX 23 (2026-05-08): Solaris uses optax.constant_schedule (no warmup) for LR;
    # see solaris_repo/config/lr_scheduler/constant.yaml. Default to 0 to match.
    # Pass --warmup N if you need a warmup; the lr_scale code below handles it.
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ckpt_minutes", type=int, default=30)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--samples_per_epoch", type=int, default=200000)
    ap.add_argument("--samples_per_episode", type=int, default=1)  # FIX 26 (2026-05-08): Solaris draws fresh episode per batch element. 50 caused intra-episode action correlation across consecutive grad steps.
    ap.add_argument("--clip_dir", type=str, default=CLIP_DIR_DEFAULT)
    ap.add_argument("--clip_pf_dir", type=str, default=CLIP_PF_DIR_DEFAULT,
                    help="v7: per-frame CLIP CLS dir; if set, takes precedence over clip_dir")
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--reset_opt", action="store_true", default=False)
    ap.add_argument("--reset_step", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, world, local_rank = setup_ddp()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    if rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    log("building voxel-DiT-Solaris (v-pred — Solaris-faithful)", rank)
    model = build_voxel_dit_solaris_270m().to(device)
    # FIX 12 (2026-05-08): Solaris uses fp32 master weights + bf16 compute via JAX
    # `param_dtype=jnp.float32, dtype=jnp.bfloat16`. Our previous `model.to(bfloat16)`
    # kept the master weights in bf16 (only 7 mantissa bits), known cause of plateau /
    # regression at 1e-4 LR. Now keeping fp32 master; bf16 compute via `torch.autocast`
    # in the forward pass (see `with torch.autocast(...)` below).
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params: {n_params / 1e6:.2f}M  grad_checkpointing=ON", rank)
    model.gradient_checkpointing = True
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=0.0, fused=True)
    log("optimizer: AdamW (fused, bf16 master)", rank)

    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        log(f"resuming from {args.resume}  reset_opt={args.reset_opt}  reset_step={args.reset_step}", rank)
        ckpt = torch.load(args.resume, map_location="cpu")
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        if not args.reset_opt and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if not args.reset_step and "step" in ckpt:
            start_step = ckpt["step"]
        log(f"resumed at step {start_step}", rank)

    if world > 1:
        model = DDP(model, device_ids=[local_rank],
                    gradient_as_bucket_view=True, find_unused_parameters=False)

    # Per-channel latent stats
    mean, std = load_or_compute_latent_stats(rank, device=device)

    # Dataset (v7: prefer per-frame CLIP)
    log("building dataset", rank)
    clip_pf_dir = args.clip_pf_dir if args.clip_pf_dir else None
    clip_dir = None if clip_pf_dir else (args.clip_dir if args.clip_dir else None)
    require_clip = clip_dir is not None
    require_clip_pf = clip_pf_dir is not None
    ds = VoxelDiTDataset(
        clip_len=33,
        samples_per_epoch=args.samples_per_epoch,
        samples_per_episode=args.samples_per_episode,
        require_rgb=False,
        require_clip=require_clip,
        require_clip_pf=require_clip_pf,
        clip_dir=clip_dir,
        clip_pf_dir=clip_pf_dir,
        rank=rank, world_size=world,
    )
    log(f"valid episodes: {len(ds.index)}  "
        f"clip_pf_dir={clip_pf_dir or 'NONE'}  clip_dir={clip_dir or 'NONE'}", rank)

    # FIX 31: dataset is already rank-partitioned via VoxelDiTDataset(rank,world_size)
    # so use a plain RandomSampler (DistributedSampler would double-partition).
    g = torch.Generator(); g.manual_seed(args.seed + rank)
    sampler = torch.utils.data.RandomSampler(ds, generator=g)
    loader = DataLoader(
        ds, batch_size=args.batch, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    loader_iter = iter(loader)

    log(f"training {args.steps} opt-steps, "
        f"micro batch {args.batch}/gpu × {world} GPUs × grad_accum {args.grad_accum} = "
        f"{args.batch * world * args.grad_accum} effective", rank)
    log(f"ckpt every {args.ckpt_minutes} wallclock minutes", rank)

    last_ckpt_time = time.time()
    t_start = time.time()
    opt_steps_done = 0

    for step in range(start_step, args.steps):
        # LR warmup
        lr_scale = min(1.0, (step + 1) / max(1, args.warmup))
        for g in optimizer.param_groups:
            g["lr"] = args.lr * lr_scale

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for accum_idx in range(args.grad_accum):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)  # RandomSampler reshuffles automatically
                batch = next(loader_iter)

            voxel_lat = batch["voxel_lat"].to(device, dtype=torch.float32)  # (B, 48, T, X, Y, Z)
            voxel_lat = normalize_latent(voxel_lat, mean, std)  # per-channel norm
            # v6: position from camera xyz deltas (frame-0-relative; already normalized
            # by voxel_dataset.normalize_camera). Mouse/keyboard streams dropped:
            # voxels are world-axis-aligned so rotation is a no-op; all per-frame voxel
            # change is driven by agent xyz translation.
            position = batch["camera"][..., 0:3].to(device, dtype=torch.float32)  # (B, 33, 3)
            clip_ctx = batch.get("clip_context")
            if clip_ctx is None:
                B = voxel_lat.size(0)
                clip_ctx = torch.zeros(B, 33, 1280, dtype=dtype, device=device)
            else:
                clip_ctx = clip_ctx.to(device, dtype=dtype)
            # v7: per-frame CLIP CLS (B, 33, 1280) — DO NOT mean-pool. Each frame
            # token carries its own visual signal; cross-attn matches voxel frame
            # tokens to their corresponding CLIP frame.

            # cond_concat: (B, 52, T, X, Y, Z)
            cond_concat = first_frame_cond_concat_voxel(voxel_lat).to(dtype)

            # Sample t ~ U(0, 1), then scale to integer index for sinusoidal embedding
            B = voxel_lat.size(0)
            t_cont = torch.rand(B, device=device).clamp_(1e-3, 1.0 - 1e-3)
            t_idx = (t_cont * NUM_TIMESTEPS).long()
            noise = torch.randn_like(voxel_lat)
            tt = t_cont.view(B, 1, 1, 1, 1, 1)
            x_t = (1 - tt) * voxel_lat + tt * noise

            # Sync gradient only on last accum step.
            # FIX (v5): one-shot context manager — wrap forward+loss+backward in a
            # SINGLE `with sync_ctx:` block. Previously this was two separate `with`
            # statements, which crashes because model.no_sync()/nullcontext() instances
            # cannot be re-entered.
            sync_ctx = (model.no_sync() if (world > 1 and accum_idx < args.grad_accum - 1)
                        else _nullcontext())
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    # FIX 12 (cont.): autocast yields bf16 compute while master weights stay
                    # fp32 (Solaris parity). Inputs that were already cast to dtype upstream
                    # are fine; autocast only affects ops inside the `with` block.
                    pred_v = model(
                        x_t.to(dtype), t_idx,
                        visual_context=clip_ctx,
                        cond_concat=cond_concat,
                        position_cond=position.to(dtype),
                    )
                # v-prediction MSE loss: target = noise - x0 (in fp32, outside autocast)
                v_target = (noise - voxel_lat).float()
                loss = F.mse_loss(pred_v.float(), v_target)
                loss = loss / args.grad_accum
                loss.backward()
            accum_loss += loss.item() * args.grad_accum

        # FIX 13 (2026-05-08): Solaris does NOT clip gradients (no `optax.clip_by_global_norm`
        # anywhere in the runner / optimizer config). Our previous default of 1.0 may have
        # been masking real gradient instability. Default raised to 1e6 (effectively off);
        # set --grad_clip to a finite value if you want to keep clipping.
        if args.grad_clip < 1e5:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        else:
            grad_norm = torch.tensor(0.0, device=device)
        optimizer.step()
        opt_steps_done += 1

        # Logging
        if (step + 1) % args.log_every == 0 or step == start_step:
            avg_loss = accum_loss / args.grad_accum
            elapsed = time.time() - t_start
            sps = opt_steps_done / max(elapsed, 1e-6)
            eta_h = (args.steps - step - 1) / max(sps, 1e-6) / 3600
            log(f"step {step + 1}/{args.steps}  loss={avg_loss:.4f}  "
                f"grad={grad_norm.item():.3f}  lr={args.lr * lr_scale:.2e}  "
                f"opt_sps={sps:.3f}  eta={eta_h:.1f}h", rank)

        # Checkpoint
        if rank == 0 and (time.time() - last_ckpt_time) >= args.ckpt_minutes * 60:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_path = os.path.join(args.ckpt_dir,
                                     f"voxel_dit_solaris_v_step{step+1:06d}_{ts}.pt")
            sd = (model.module if world > 1 else model).state_dict()
            torch.save({
                "state_dict": sd,
                "optimizer": optimizer.state_dict(),
                "step": step + 1,
            }, ckpt_path)
            log(f"saved {ckpt_path}", rank)
            last_ckpt_time = time.time()

    # Final ckpt
    if rank == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(args.ckpt_dir,
                                 f"voxel_dit_solaris_v_FINAL_{ts}.pt")
        sd = (model.module if world > 1 else model).state_dict()
        torch.save({"state_dict": sd, "step": args.steps}, ckpt_path)
        log(f"saved final {ckpt_path}", rank)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
