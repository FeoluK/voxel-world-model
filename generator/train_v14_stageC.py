"""v14 MaskGIT — Stage C TEMPORAL multi-frame self-forcing trainer (voxel-native).

Combines what works from the two prior self-forcing attempts:

  * train_v14_sf.py (GOOD, spatial):  full-frame fat-slab GT loss — robust, non-trivial
    targets, fixed the stone/air collapse.  But NO temporal horizon: it never rolls the
    model on a sequence of its OWN frames.
  * train_v14_sf3.py (FAILED, temporal):  did a genuine multi-frame rollout but supervised
    ONLY the thin world-shift frontier.  On low-motion treechop the frontier is near-empty
    -> trivial targets -> collapse to stone.

Stage C does a genuine MULTI-FRAME autoregressive rollout (model generates frame t from its
OWN frame t-1, over a horizon H that ramps lo->hi using the SAME translate()+maskgit_generate()
world-shift machinery as eval_v14), but supervises with a FULL-FRAME fat-slab GT loss at the
rolled frame (like train_v14_sf.py spatially), NOT frontier-only.  This trains long-horizon
temporal consistency without the frontier-triviality collapse.

Each grad step:
  * with prob --self_force_prob:  TEMPORAL SF step.  Roll the model forward H steps on its
    own output, then mask a fat slab (slab_frac 0.20-0.40) of the FINAL rolled frame and
    train (CE) to predict the GT voxels there.  GT = real data frame at that horizon offset,
    decoded from the GT voxel latent at that frame index (same target source as train_v14_sf).
    Anti-triviality guard: if the rolled frame's GT slab is degenerate (too little non-air
    content), fall back to a TF step — never emit a trivial loss.
  * else (1-prob):  ordinary TEACHER-FORCED curriculum slab-mask step, exactly like
    train_v14_sf.py (always a valid non-trivial gradient).

VAE is used ONLY (no grad) to decode GT block-id targets; it is not part of the model.
Checkpoint format matches train_v14_sf.py so eval_v14.py loads it unchanged.
bf16, gradient checkpointing on, torchrun --standalone single-node multi-GPU, self-resume.
"""
from __future__ import annotations
import argparse, glob, os, re, sys, time
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
from voxel_maskgit import (VoxelMaskGIT3D, build_random_mask, build_slab_mask,
                           maskgit_generate)
from eval_v14 import translate   # REUSE the canvas-shift used at eval/inference time

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


def base_of(model, world):
    return model.module if world > 1 else model


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
    """lat (N,48,12,12,12) -> ids (N,48,48,48) 2021-vocab; bf16 decode (no grad)."""
    out = []
    for i in range(0, lat.shape[0], chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = vae.decode(lat[i:i + chunk].float())
        out.append(logit.argmax(1).to(torch.int16))
    return torch.cat(out, 0)


def cum_shifts(cam_xyz, K):
    """cam_xyz (T,3) normalized (frame-0-subtracted) -> cumulative int shifts, one per
    rolled frame.  shift[k] carries the frame-0 canvas to align with GT frame k+1.
    Matches eval_v14 EXACTLY: shift[t] = round(-(pos[t]-pos[0])).  We always pass the
    cumulative shift; per-step deltas are computed in the rollout (carry running world)."""
    p = cam_xyz.detach().cpu().numpy()
    return [tuple((-np.round(p[t] - p[0])).astype(int)) for t in range(1, K + 1)]


@torch.no_grad()
def selfforce_rollout(bmodel, world0, shifts, K, n_unmask=5, temp=1.0, top_p=1.0):
    """Roll a single clip forward K steps on the model's OWN output, mirroring
    eval_v14.rollout (compound=True): carry the RUNNING world by each per-step shift,
    then maskgit_generate the revealed frontier.

    world0  : (48,48,48) compact GT seed (real frame 0).
    shifts  : list of cumulative int shifts from frame 0 (len >= K).
    Returns the drifted world (48,48,48) compact at horizon K.
    """
    world = world0.clone()
    prev_sh = (0, 0, 0)
    for k in range(K):
        sh = shifts[k]
        step_sh = tuple(int(sh[a]) - int(prev_sh[a]) for a in range(3))   # per-step delta
        prev_sh = sh
        carried, valid = translate(world, step_sh)
        fill = ~valid
        if fill.any():
            world = maskgit_generate(bmodel, carried.unsqueeze(0), fill.unsqueeze(0),
                                     n_steps=n_unmask, temp=temp, top_p=top_p)[0]
        else:
            world = carried
    return world


def latest_self_ckpt(ckpt_dir, dim):
    """Find newest v14_stageC_dim{dim}_step{N}.pt in ckpt_dir (self-resume)."""
    if not os.path.isdir(ckpt_dir):
        return None, -1
    best, best_n = None, -1
    pat = re.compile(rf"v14_stageC_dim{dim}_step(\d+)\.pt$")
    for f in glob.glob(os.path.join(ckpt_dir, f"v14_stageC_dim{dim}_step*.pt")):
        m = pat.search(os.path.basename(f))
        if m and int(m.group(1)) > best_n:
            best, best_n = f, int(m.group(1))
    return best, best_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=8, help="clips per GPU")
    ap.add_argument("--frames_per_clip", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ckpt_minutes", type=int, default=20)
    ap.add_argument("--max_minutes", type=float, default=0.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--ckpt_dir", type=str, default="/home/flukol/v14_tc_stageC_ckpts")
    ap.add_argument("--resume", type=str, required=True,
                    help="warm-start ckpt (loaded strict, fresh optimizer) if no self-ckpt present")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps_override", type=int, default=0)
    # ---- temporal self-forcing schedule ----
    ap.add_argument("--horizon_lo", type=int, default=2)
    ap.add_argument("--horizon_hi", type=int, default=64)
    ap.add_argument("--horizon_ramp_steps", type=int, default=4000,
                    help="steps over which rollout horizon H ramps lo->hi")
    ap.add_argument("--self_force_prob", type=float, default=0.8)
    ap.add_argument("--sf_gen_steps", type=int, default=5, help="maskgit unmask steps per rolled frame")
    ap.add_argument("--sf_temp", type=float, default=1.0)
    ap.add_argument("--sf_top_p", type=float, default=1.0)
    # ---- full-frame slab loss (both TF and the supervised SF frame) ----
    ap.add_argument("--slab_frac_lo", type=float, default=0.20)
    ap.add_argument("--slab_frac_hi", type=float, default=0.40)
    ap.add_argument("--curriculum", type=int, default=1, help="TF-step mask curriculum (as train_v14_sf)")
    # ---- anti-triviality guard ----
    ap.add_argument("--min_nonair_frac", type=float, default=0.02,
                    help="GT slab must contain >= this fraction non-air voxels, else fall back to TF")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    rank, world, lr_ = setup_ddp()
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank)
    device = torch.device(f"cuda:{lr_}"); dtype = torch.bfloat16
    if rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.smoke:
        args.num_workers = min(args.num_workers, 2)
        args.log_every = 1
        args.warmup = min(args.warmup, 5)
        # tiny checkpoint cadence so the smoke proves mid-run checkpointing (not just FINAL)
        args.ckpt_minutes = 0.0

    _v = np.load(VOCAB_PT)
    MODEL_CLASSES = int(_v["num_classes"])
    id2c = torch.from_numpy(_v["id2compact"]).long().to(device)   # (2021,) -> compact
    AIR = int(_v["id2compact"][0])                                # compact id of air
    log(f"=== v14 Stage C (temporal SF)  dim={args.dim} batch={args.batch}x{args.frames_per_clip}fpc "
        f"vocab={MODEL_CLASSES} air={AIR} H={args.horizon_lo}->{args.horizon_hi} "
        f"sf_prob={args.self_force_prob} smoke={args.smoke} ===", rank)

    model = VoxelMaskGIT3D(num_classes=MODEL_CLASSES, dim=args.dim).to(device)
    model.grad_ckpt = bool(args.grad_ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params: {n_params/1e6:.1f}M  grad_ckpt={model.grad_ckpt}", rank)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0, fused=True)

    # -------- resume: self-ckpt (full restore) > --resume (weights only, fresh opt) --------
    start = 0
    self_ck, self_n = latest_self_ckpt(args.ckpt_dir, args.dim)
    if self_ck is not None:
        ck = torch.load(self_ck, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in (ck.get("state_dict", ck)).items()}
        model.load_state_dict(sd, strict=True)
        if "optimizer" in ck: opt.load_state_dict(ck["optimizer"])
        start = int(ck.get("step", self_n))
        log(f"SELF-RESUME from {self_ck} at step {start} (optimizer restored)", rank)
    elif args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in (ck.get("state_dict", ck)).items()}
        rck = int(ck.get("num_classes", MODEL_CLASSES))
        if rck != MODEL_CLASSES:
            log(f"WARN warm-start num_classes={rck} != vocab {MODEL_CLASSES}", rank)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # we want a STRICT warm-start; report and hard-fail if anything is off
        if missing or unexpected:
            raise RuntimeError(f"warm-start key mismatch: missing={list(missing)} unexpected={list(unexpected)}")
        log(f"WARM-START (strict, fresh optimizer) from {args.resume} "
            f"(was step {ck.get('step')}, dim {ck.get('dim')}, K {rck}) — no missing/unexpected keys", rank)
    else:
        raise FileNotFoundError(f"no self-ckpt in {args.ckpt_dir} and --resume '{args.resume}' missing")

    if world > 1:
        model = DDP(model, device_ids=[lr_], gradient_as_bucket_view=True,
                    find_unused_parameters=False, static_graph=True)
    bmodel = base_of(model, world)

    vae = get_vae(device)
    log("VAE loaded (targets only)", rank)
    ds = VoxelDiTDataset(clip_len=66, samples_per_epoch=200000, samples_per_episode=1,
                         require_rgb=False, require_jsonl=True, require_clip=False,
                         require_clip_pf=False, clip_dir=None, clip_pf_dir=None,
                         rank=rank, world_size=world)
    log(f"valid episodes: {len(ds.index)}  (require_jsonl=True for per-frame positions)", rank)
    g = torch.Generator(); g.manual_seed(args.seed + rank)
    loader = DataLoader(ds, batch_size=args.batch,
                        sampler=torch.utils.data.RandomSampler(ds, generator=g),
                        num_workers=args.num_workers, pin_memory=False, drop_last=True,
                        persistent_workers=(args.num_workers > 0),
                        prefetch_factor=(4 if args.num_workers > 0 else None))
    it = iter(loader)
    mgen = torch.Generator(device=device); mgen.manual_seed(args.seed + 99 + rank)

    def next_batch():
        nonlocal it
        try: return next(it)
        except StopIteration:
            it = iter(loader); return next(it)

    last_ck = time.time(); t0 = time.time(); step = start - 1

    def save(stp):
        if rank != 0:
            return
        p = os.path.join(args.ckpt_dir, f"v14_stageC_dim{args.dim}_step{stp:06d}.pt")
        sd = base_of(model, world).state_dict()
        torch.save({"state_dict": sd, "optimizer": opt.state_dict(), "step": stp,
                    "dim": args.dim, "num_classes": MODEL_CLASSES}, p)
        log(f"saved {p}", rank)

    # ---- masked CE on a fat slab of GT ids; returns (loss, masked_acc) ----
    def slab_ce(inp_ids, gt_ids, mask):
        with torch.autocast("cuda", dtype=dtype):
            logits = model(inp_ids)                              # (N,C,48,48,48)
            lm = logits.permute(0, 2, 3, 4, 1)[mask].float()    # (n_masked,C)
            tm = gt_ids[mask]
            loss = F.cross_entropy(lm, tm)
        with torch.no_grad():
            acc = (lm.argmax(-1) == tm).float().mean()
        return loss, acc

    # ---- TEACHER-FORCED curriculum slab step (identical recipe to train_v14_sf) ----
    def tf_step(ai):
        batch = next_batch()
        lat = batch["voxel_lat"].to(device).float()              # (B,48,T,12,12,12)
        B, C, T = lat.shape[:3]
        fpc = min(args.frames_per_clip, T)
        fidx = torch.randint(0, T, (B, fpc), device=device)
        sel = torch.stack([lat[b, :, fidx[b]] for b in range(B)])
        sel = sel.permute(0, 2, 1, 3, 4, 5).reshape(B * fpc, C, 12, 12, 12)
        with torch.no_grad():
            ids = id2c[decode_ids(vae, sel).long()]             # (N,48,48,48) compact
        if args.curriculum:
            _r = float(torch.rand(1, generator=mgen, device=device))
            if _r < 0.15:
                mask = build_random_mask(ids, generator=mgen)
            elif _r < 0.65:
                mask = build_slab_mask(ids, generator=mgen, frac_lo=0.22, frac_hi=0.28)
            elif _r < 0.90:
                mask = build_slab_mask(ids, generator=mgen, frac_lo=0.45, frac_hi=0.55)
            else:
                mask = build_slab_mask(ids, generator=mgen, frac_lo=0.72, frac_hi=0.78)
        else:
            mask = build_slab_mask(ids, generator=mgen,
                                   frac_lo=args.slab_frac_lo, frac_hi=args.slab_frac_hi)
        inp = ids.clone(); inp[mask] = MODEL_CLASSES            # mask token
        sync = (model.no_sync() if (world > 1 and ai < args.grad_accum - 1) else _nullcontext())
        with sync:
            loss, acc = slab_ce(inp, ids, mask)
            (loss / args.grad_accum).backward()
        return float(loss), float(acc)

    # ---- TEMPORAL SELF-FORCED step: roll H frames on own output, supervise final
    #      frame with a full-frame fat slab vs GT.  Returns (loss, acc, did_sf). ----
    def sf_step(ai, H):
        batch = next_batch()
        lat = batch["voxel_lat"].to(device).float()             # (B,48,T,12,12,12)
        cam = batch["camera"].to(device)                        # (B,T,5) normalized xyz in 0:3
        B, C, T = lat.shape[:3]
        Hc = max(1, min(H, T - 1))                              # cap horizon to clip length
        # choose a batch element with a valid horizon; build drifted contexts (no grad)
        drifted_list, gt_list = [], []
        for b in range(B):
            shifts = cum_shifts(cam[b, :, :3], Hc)
            # need GT seed (frame 0) and GT target (frame Hc), both decoded compact
            seed_lat = lat[b, :, 0:1].permute(1, 0, 2, 3, 4)               # (1,48,12,12,12)
            tgt_lat  = lat[b, :, Hc:Hc + 1].permute(1, 0, 2, 3, 4)         # (1,48,12,12,12)
            with torch.no_grad():
                seed = id2c[decode_ids(vae, seed_lat).long()][0]          # (48,48,48) compact
                gt_tgt = id2c[decode_ids(vae, tgt_lat).long()][0]         # GT at horizon
                drifted = selfforce_rollout(bmodel, seed, shifts, Hc,
                                            n_unmask=args.sf_gen_steps,
                                            temp=args.sf_temp, top_p=args.sf_top_p)
            drifted_list.append(drifted)
            gt_list.append(gt_tgt)
        drifted = torch.stack(drifted_list)                                # (B,48,48,48)
        gt_tgt = torch.stack(gt_list)                                      # (B,48,48,48)

        # full-frame fat slab on the rolled frame; train to predict GT there.
        mask = build_slab_mask(gt_tgt, generator=mgen,
                               frac_lo=args.slab_frac_lo, frac_hi=args.slab_frac_hi)
        # anti-triviality guard: slab must cover substantial non-air GT content.
        nonair = ((gt_tgt != AIR) & mask).float().sum() / mask.float().sum().clamp_min(1.0)
        if float(nonair) < args.min_nonair_frac:
            return None, None, False                          # caller falls back to TF

        # KEY: input is the model's OWN drifted frame, but the masked slab is replaced by
        # the mask token; supervision target is the REAL GT frame at this horizon (full
        # slab, NOT frontier-only).  This is train_v14_sf's spatial loss applied on top of
        # a genuine multi-frame temporal rollout.
        inp = drifted.clone(); inp[mask] = MODEL_CLASSES
        sync = (model.no_sync() if (world > 1 and ai < args.grad_accum - 1) else _nullcontext())
        with sync:
            loss, acc = slab_ce(inp, gt_tgt, mask)
            (loss / args.grad_accum).backward()
        return float(loss), float(acc), True

    # ==================================================================
    #  TRAINING LOOP
    # ==================================================================
    for step in range(start, args.steps):
        for pg in opt.param_groups:
            pg["lr"] = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        opt.zero_grad(set_to_none=True)

        # horizon ramp lo -> hi
        prog = min(1.0, (step - start) / max(1, args.horizon_ramp_steps))
        H = int(round(args.horizon_lo + prog * (args.horizon_hi - args.horizon_lo)))

        acc_loss = 0.0; acc_accm = 0.0; n_sf = 0
        for ai in range(args.grad_accum):
            do_sf = bool(torch.rand(1, generator=mgen, device=device) < args.self_force_prob)
            ls = ac = None; did_sf = False
            if do_sf:
                ls, ac, did_sf = sf_step(ai, H)
            if not did_sf:
                ls, ac = tf_step(ai)                            # SF declined / guard tripped
            acc_loss += ls / args.grad_accum
            acc_accm += ac / args.grad_accum
            n_sf += int(did_sf)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.log_every == 0 or step == start:
            el = time.time() - t0
            log(f"step {step+1}/{args.steps}  loss={acc_loss:.4f}  masked_acc={acc_accm:.3f}  "
                f"H={H}  sf={n_sf}/{args.grad_accum}  "
                f"sps={(step+1-start)/max(el,1e-6):.3f}  mins={el/60:.1f}", rank)
        if rank == 0 and (time.time() - last_ck) >= args.ckpt_minutes * 60:
            save(step + 1); last_ck = time.time()
        if args.max_minutes > 0 and (time.time() - t0) >= args.max_minutes * 60:
            log(f"hit max_minutes at {step+1}", rank); break
        if args.max_steps_override and (step + 1 - start) >= args.max_steps_override:
            log(f"hit max_steps_override at {step+1}", rank); break

    save(min(step + 1, args.steps))
    if dist.is_initialized(): dist.destroy_process_group()


if __name__ == "__main__":
    main()
