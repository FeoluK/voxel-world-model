"""v14 MaskGIT trainer — voxel-native, discrete, single-frame.

Targets are voxel block-IDs (decoded from the existing VAE latents, no grad — the
VAE is used ONLY to produce training targets; it is not part of the model). Each
step: sample frames, random-mask voxels (cosine ratio), predict masked classes,
cross-entropy on masked positions. Temporal/world behavior is added at inference
by the carry + write-back wrapper — this trainer is pure spatial inpainting.
"""
from __future__ import annotations
import argparse, os, sys, time
from contextlib import nullcontext as _nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voxel_dataset import VoxelDiTDataset
from vae_3d import VAE3D
from voxel_maskgit import VoxelMaskGIT3D, build_random_mask, build_slab_mask, maskgit_generate

_CODE = os.environ.get("V12_CODE_DIR", os.path.dirname(os.path.abspath(__file__)))
VAE_PT = os.environ.get("V12_VAE_PT", os.path.join(_CODE, "vae_final.pt"))
VAE_NUM_CLASSES = 2021   # VAE output vocab (used only to decode targets)
VOCAB_PT = os.environ.get("V14_VOCAB", os.path.join(_CODE, "voxel_vocab.npz"))


def log(m, r):
    if r == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def setup_ddp():
    r = int(os.environ.get("RANK", "0")); w = int(os.environ.get("WORLD_SIZE", "1"))
    l = int(os.environ.get("LOCAL_RANK", "0"))
    if w > 1:
        dist.init_process_group(backend="nccl"); torch.cuda.set_device(l)
    return r, w, l


_VAE = None
def get_vae(dev):
    global _VAE
    if _VAE is None:
        vae = VAE3D(num_classes=VAE_NUM_CLASSES, latent_channels=48, middle_channels=(32, 128, 512))
        ck = torch.load(VAE_PT, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        vae.load_state_dict(sd, strict=False)
        _VAE = vae.to(dev).eval()
        for p in _VAE.parameters():
            p.requires_grad_(False)
    return _VAE


@torch.no_grad()
def decode_ids(vae, lat, chunk=8):
    # lat (N,48,12,12,12) -> ids (N,48,48,48); bf16 decode (no grad) for speed
    out = []
    for i in range(0, lat.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = vae.decode(lat[i:i + chunk].float())
        out.append(logit.argmax(1).to(torch.int16))
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=4, help="clips per GPU")
    ap.add_argument("--frames_per_clip", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ckpt_minutes", type=int, default=30)
    ap.add_argument("--max_minutes", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--ckpt_dir", type=str, required=True)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps_override", type=int, default=0)
    ap.add_argument("--block_prob", type=float, default=0.75, help="prob of contiguous-slab mask vs random mask")
    ap.add_argument("--block_frac_lo", type=float, default=0.20)
    ap.add_argument("--block_frac_hi", type=float, default=0.30)
    ap.add_argument("--curriculum", type=int, default=1, help="15%% random /50%% 1/4 /25%% 1/2 /10%% 3/4")
    ap.add_argument("--self_force_prob", type=float, default=0.4, help="prob a slab step corrupts KNOWN ctx with model's own gen")
    ap.add_argument("--sf_gen_steps", type=int, default=5)
    ap.add_argument("--sf_frac_lo", type=float, default=0.20)
    ap.add_argument("--sf_frac_hi", type=float, default=0.40)
    args = ap.parse_args()

    rank, world, lr_ = setup_ddp()
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank)
    device = torch.device(f"cuda:{lr_}"); dtype = torch.bfloat16
    if rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    _v = np.load(VOCAB_PT)
    MODEL_CLASSES = int(_v["num_classes"])
    id2c = torch.from_numpy(_v["id2compact"]).long().to(device)   # (2021,) -> compact
    log(f"=== v14 MaskGIT  dim={args.dim} batch={args.batch}x{args.frames_per_clip}fpc  "
        f"vocab={MODEL_CLASSES} ===", rank)
    model = VoxelMaskGIT3D(num_classes=MODEL_CLASSES, dim=args.dim).to(device)
    model.grad_ckpt = bool(args.grad_ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params: {n_params/1e6:.1f}M  grad_ckpt={model.grad_ckpt}", rank)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)

    start = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in (ck.get("state_dict", ck)).items()}
        model.load_state_dict(sd, strict=True)
        if "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
        if "step" in ck: start = ck["step"]
        log(f"resumed at {start}", rank)

    if world > 1:
        model = DDP(model, device_ids=[lr_], gradient_as_bucket_view=True, static_graph=True)

    vae = get_vae(device)
    log("VAE loaded (targets only)", rank)
    ds = VoxelDiTDataset(clip_len=33, samples_per_epoch=200000, samples_per_episode=1,
                         require_rgb=False, require_jsonl=False, require_clip=False, require_clip_pf=False,
                         clip_dir=None, clip_pf_dir=None, rank=rank, world_size=world)
    log(f"valid episodes: {len(ds.index)}", rank)
    g = torch.Generator(); g.manual_seed(args.seed + rank)
    loader = DataLoader(ds, batch_size=args.batch,
                        sampler=torch.utils.data.RandomSampler(ds, generator=g),
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=(args.num_workers > 0),
                        prefetch_factor=(4 if args.num_workers > 0 else None))
    it = iter(loader)
    mgen = torch.Generator(device=device); mgen.manual_seed(args.seed + 99 + rank)

    last_ck = time.time(); t0 = time.time(); step = start - 1
    def save(stp, tag):
        p = os.path.join(args.ckpt_dir, f"v14_maskgit_dim{args.dim}_{tag}{stp:06d}.pt")
        sd = (model.module if world > 1 else model).state_dict()
        torch.save({"state_dict": sd, "optimizer": opt.state_dict(), "step": stp,
                    "dim": args.dim, "num_classes": MODEL_CLASSES}, p)
        log(f"saved {p}", rank)

    for step in range(start, args.steps):
        for pg in opt.param_groups:
            pg["lr"] = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0; acc_accm = 0.0
        for ai in range(args.grad_accum):
            try: batch = next(it)
            except StopIteration: it = iter(loader); batch = next(it)
            lat = batch["voxel_lat"].to(device).float()      # (B,48,T,12,12,12)
            B, C, T = lat.shape[:3]
            # sample frames_per_clip random frames per clip
            fpc = min(args.frames_per_clip, T)
            fidx = torch.randint(0, T, (B, fpc), device=device)
            sel = torch.stack([lat[b, :, fidx[b]] for b in range(B)])   # (B,48,fpc,12,12,12)
            sel = sel.permute(0, 2, 1, 3, 4, 5).reshape(B * fpc, C, 12, 12, 12)
            with torch.no_grad():
                ids = decode_ids(vae, sel).long()             # (N,48,48,48) 2021-vocab
                ids = id2c[ids]                               # -> compact vocab (0..MODEL_CLASSES-1)
            if args.curriculum:
                _r = float(torch.rand(1, generator=mgen, device=device))
                if _r < 0.15:
                    _use_block = False; _kind = "random"
                    mask = build_random_mask(ids, generator=mgen)        # interpolation, retained
                else:
                    _use_block = True
                    if _r < 0.65:   _lo, _hi, _kind = 0.22, 0.28, "quarter"        # 50%
                    elif _r < 0.90: _lo, _hi, _kind = 0.45, 0.55, "half"           # 25%
                    else:           _lo, _hi, _kind = 0.72, 0.78, "three_quarter"  # 10%
                    mask = build_slab_mask(ids, generator=mgen, frac_lo=_lo, frac_hi=_hi)
            else:
                _use_block = bool(torch.rand(1, generator=mgen, device=device) < args.block_prob)
                if _use_block:
                    _kind = "slab"
                    mask = build_slab_mask(ids, generator=mgen, frac_lo=args.block_frac_lo, frac_hi=args.block_frac_hi)
                else:
                    _kind = "random"
                    mask = build_random_mask(ids, generator=mgen)
            # ---- self-forcing: corrupt KNOWN context with the model's OWN generation ----
            # so the model learns to fill the frontier (target = real GT) even when its
            # neighbouring context is its own (imperfect, maybe-airy) output -> breaks the
            # air-attractor that teacher-forcing can't reach (exposure-bias / recovery).
            ctx = ids.clone()
            do_sf = _use_block and bool(torch.rand(1, generator=mgen, device=device) < args.self_force_prob)
            if do_sf:
                with torch.no_grad():
                    sf_mask = build_slab_mask(ids, generator=mgen, frac_lo=args.sf_frac_lo, frac_hi=args.sf_frac_hi)
                    sf_mask = sf_mask & (~mask)                 # corrupt only KNOWN (non-target) voxels
                    if sf_mask.any():
                        base_m = model.module if world > 1 else model
                        filled_sf = maskgit_generate(base_m, ids.clone(), sf_mask,
                                                     n_steps=args.sf_gen_steps, temp=1.0)
                        ctx[sf_mask] = filled_sf[sf_mask]       # context now partly self-generated
            if step < start + 8:  # VERIFY curriculum mix + slab geometry + self-forcing
                _f = mask.float().mean().item()
                _b0 = mask[0]
                _ax = [int((_b0.any(1).any(1)).sum()), int((_b0.any(0).any(1)).sum()), int((_b0.any(0).any(0)).sum())]
                log(f"  [maskchk] kind={_kind} frac={_f:.3f} per-axis-extent(of48)={_ax} self_force={do_sf}", rank)
            inp = ctx.clone(); inp[mask] = MODEL_CLASSES      # mask token; target stays GT ids[mask]
            sync = (model.no_sync() if (world > 1 and ai < args.grad_accum - 1) else _nullcontext())
            with sync:
                with torch.autocast("cuda", dtype=dtype):
                    logits = model(inp)                       # (N,C,48,48,48)
                # memory-efficient: CE only on masked voxels (gather first)
                lm = logits.permute(0, 2, 3, 4, 1)[mask]      # (n_masked, C)
                tm = ids[mask]                                # (n_masked,)
                loss = F.cross_entropy(lm.float(), tm)
                (loss / args.grad_accum).backward()
            with torch.no_grad():
                accm = (lm.argmax(-1) == tm).float().mean()
            acc_loss += loss.item() / args.grad_accum
            acc_accm += accm.item() / args.grad_accum
        opt.step()

        if (step + 1) % args.log_every == 0 or step == start:
            el = time.time() - t0
            log(f"step {step+1}/{args.steps}  loss={acc_loss:.4f}  masked_acc={acc_accm:.3f}  "
                f"sps={(step+1-start)/max(el,1e-6):.3f}  mins={el/60:.1f}", rank)
        if rank == 0 and (time.time() - last_ck) >= args.ckpt_minutes * 60:
            save(step + 1, "step"); last_ck = time.time()
        if args.max_minutes > 0 and (time.time() - t0) >= args.max_minutes * 60:
            log(f"hit max_minutes at {step+1}", rank); break
        if args.max_steps_override and (step + 1 - start) >= args.max_steps_override:
            log(f"hit max_steps_override at {step+1}", rank); break

    if rank == 0: save(min(step + 1, args.steps), "FINAL")
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
