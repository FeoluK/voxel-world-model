"""V7 ablation: per-frame CLIP variant, position-only action stream (v6/v7 spec).

Patched 2026-05-13: prior version called model() with mouse_cond/keyboard_cond,
but v6/v7/v7.5 use position_cond only (mouse/keyboard removed in v6_minimal).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

INF8 = "/scratch/users/flukol/mg_pt_voxel/inf_v8/mg_pt_voxel_local"
sys.path.insert(0, "/scratch/users/flukol/mg_pt_voxel")
sys.path.insert(0, INF8)
sys.path.insert(0, "/scratch/users/flukol/mg_pt_voxel/inf_v8")

from voxel_dataset import VoxelDiTDataset
from voxel_dit_solaris_v7 import build_voxel_dit_solaris_270m
import vae_3d as vae_module

CLIP_PF_DIR = "/scratch/users/flukol/mg_pt_voxel/extract/clip_features_h_pf"
VAE_PT = "/scratch/users/flukol/mg_pt_voxel/vae_ckpts/vae_final.pt"
LATENT_STATS = "/scratch/users/flukol/mg_pt_voxel/voxel_latent_per_channel_stats.npz"
NUM_TIMESTEPS = 1000
NUM_CLASSES = 2021


_ZERO_SENTINEL = None
def _zero_sentinel(device, dtype):
    global _ZERO_SENTINEL
    if _ZERO_SENTINEL is None:
        _ZERO_SENTINEL = torch.load(
            "/scratch/users/flukol/mg_pt_voxel/zero_sentinel_normalized.pt",
            map_location="cpu")
    return _ZERO_SENTINEL.to(device=device, dtype=dtype)


def first_frame_cond_concat(voxel_lat):
    B, C, T, X, Y, Z = voxel_lat.shape
    sentinel = _zero_sentinel(voxel_lat.device, voxel_lat.dtype).expand(B, C, T-1, X, Y, Z)
    first = torch.cat([voxel_lat[:, :, :1], sentinel], dim=2)
    mask = torch.zeros(B, 4, T, X, Y, Z, dtype=voxel_lat.dtype, device=voxel_lat.device)
    mask[:, :, :1] = 1.0
    return torch.cat([mask, first], dim=1)


def normalize(x, mean, std): return (x - mean.view(1, 48, 1, 1, 1, 1)) / std.view(1, 48, 1, 1, 1, 1)
def denormalize(x, mean, std): return x * std.view(1, 48, 1, 1, 1, 1) + mean.view(1, 48, 1, 1, 1, 1)


def flow_match_inference_timesteps(n_steps, timestep_shift=5.0):
    sigmas = torch.linspace(1.0, 0.0, n_steps + 1, dtype=torch.float32)[:-1]
    sigmas = timestep_shift * sigmas / (1.0 + (timestep_shift - 1.0) * sigmas)
    return torch.cat([sigmas, torch.zeros(1, dtype=torch.float32)], dim=0)


@torch.no_grad()
def sample_rectified_flow(model, target, x0_gt_norm, vc, cc, pc, n_steps=50,
                          dtype=torch.bfloat16, timestep_shift=5.0):
    """v6/v7/v7.5: position_cond only (camera xyz). No mouse/keyboard."""
    B, C, T, X, Y, Z = x0_gt_norm.shape
    device = x0_gt_norm.device
    x = torch.randn(B, C, T, X, Y, Z, device=device, dtype=torch.float32)
    sigmas = flow_match_inference_timesteps(n_steps, timestep_shift=timestep_shift).to(device)
    for i in range(n_steps):
        s_cur = sigmas[i].item(); s_next = sigmas[i + 1].item()
        t_idx = torch.tensor([min(int(s_cur * NUM_TIMESTEPS), NUM_TIMESTEPS - 1)],
                             device=device).long()
        with torch.amp.autocast('cuda', dtype=dtype):
            pred = model(x, t_idx, visual_context=vc, cond_concat=cc,
                         position_cond=pc).float()
        v_pred = (x - pred) / max(s_cur, 1e-3) if target == "x0" else pred
        x_start = x - s_cur * v_pred
        if i == n_steps - 1:
            x = x_start
        else:
            x = (1.0 - s_next) * x_start + s_next * torch.randn_like(x_start)
    return x


def vae_decode_to_ids(vae, z):
    T = z.shape[2]
    frames = []
    for t in range(T):
        zf = z[:, :, t]
        logits = vae.decode(zf)
        ids = logits.argmax(dim=1)
        frames.append(ids.cpu())
        del logits
        torch.cuda.empty_cache()
    return torch.stack(frames, dim=1).numpy()[0]


def get_or_build_pool(pool_path, n_chunks=6, seed=42):
    if os.path.exists(pool_path):
        return torch.load(pool_path, map_location="cpu")
    print(f"[pool] building {n_chunks}-chunk v7 pool at {pool_path}", flush=True)
    torch.manual_seed(seed); np.random.seed(seed)
    ds = VoxelDiTDataset(
        clip_len=33, samples_per_epoch=n_chunks, samples_per_episode=1,
        require_rgb=False, require_clip=False, require_clip_pf=True,
        clip_dir=None, clip_pf_dir=CLIP_PF_DIR, verbose=False,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0)
    pool = []
    for i, batch in enumerate(loader):
        if i >= n_chunks: break
        pool.append({k: v for k, v in batch.items() if isinstance(v, torch.Tensor)})
    torch.save(pool, pool_path)
    print(f"[pool] saved {len(pool)} chunks", flush=True)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["x0", "vpred"], default="vpred")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--chunk_idx", type=int, required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_rf_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda"); dtype = torch.bfloat16
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] v7 model {args.ckpt}", flush=True)
    model = build_voxel_dit_solaris_270m().to(device)
    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck["state_dict"] if "state_dict" in ck else ck
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True); model.eval()
    print(f"[load] target={args.target} step={ck.get('step', '?')}", flush=True)

    vae = vae_module.VAE3D(num_classes=NUM_CLASSES, latent_channels=48,
                           middle_channels=(32, 128, 512)).to(device).eval()
    vck = torch.load(VAE_PT, map_location="cpu", weights_only=False)
    if isinstance(vck, dict):
        for k in ("state_dict", "model", "ema"):
            if k in vck: vck = vck[k]; break
    vae.load_state_dict(vck, strict=False)

    d = np.load(LATENT_STATS)
    mean = torch.from_numpy(d["mean"]).float().to(device)
    std = torch.from_numpy(d["std"]).float().to(device)

    pool = get_or_build_pool(args.pool)
    print(f"[load] pool idx={args.chunk_idx}/{len(pool)}", flush=True)
    batch = pool[args.chunk_idx]
    voxel_lat0 = batch["voxel_lat"].to(device, dtype=torch.float32)
    voxel_lat0_norm = normalize(voxel_lat0, mean, std)
    # v6/v7/v7.5: position-only action stream (xyz from camera)
    position = batch["camera"][..., 0:3].to(device).float()  # (B, 33, 3)
    clip_ctx = batch["clip_context"].to(device).float()  # (B, 33, 1280) per-frame CLS (v7+)
    cond_concat = first_frame_cond_concat(voxel_lat0_norm).float()
    print(f"[shapes] vox={tuple(voxel_lat0.shape)} clip={tuple(clip_ctx.shape)} "
          f"pos={tuple(position.shape)}", flush=True)

    with torch.no_grad():
        gt_ids = vae_decode_to_ids(vae, voxel_lat0)
    np.savez(os.path.join(args.output_dir, "voxels_GT.npz"), voxels=gt_ids.astype(np.int64))
    nonair = (gt_ids != 0)

    # Ablation tensors (position-only spec)
    zeros_clip = torch.zeros_like(clip_ctx)
    clip_first_only = clip_ctx.clone()
    clip_first_only[:, 1:, :] = 0.0
    zeros_cc = torch.zeros_like(cond_concat)
    zeros_pos = torch.zeros_like(position)
    reverse_pos = position.flip(dims=[1])

    abls = [
        ("normal",                clip_ctx,        cond_concat, position),
        ("zero_clip",             zeros_clip,      cond_concat, position),
        ("zero_clip_post_first",  clip_first_only, cond_concat, position),
        ("zero_cond_concat",      clip_ctx,        zeros_cc,    position),
        ("zero_position",         clip_ctx,        cond_concat, zeros_pos),
        ("reverse_position",      clip_ctx,        cond_concat, reverse_pos),
    ]
    metrics = {}
    for name, vc, cc, pc in abls:
        t0 = time.time()
        x_pred = sample_rectified_flow(model, args.target, voxel_lat0_norm, vc, cc, pc,
                                        n_steps=args.n_rf_steps, dtype=dtype)
        x_pred = denormalize(x_pred, mean, std)
        latent_mse = F.mse_loss(x_pred, voxel_lat0).item()
        with torch.no_grad():
            pred_ids = vae_decode_to_ids(vae, x_pred)
        acc_all = float((pred_ids == gt_ids).mean())
        acc_nonair = float((pred_ids[nonair] == gt_ids[nonair]).mean()) if nonair.sum() > 0 else 0.0
        metrics[name] = {"latent_mse": latent_mse, "acc_all": acc_all,
                          "acc_nonair": acc_nonair, "elapsed_s": time.time() - t0}
        np.savez(os.path.join(args.output_dir, f"voxels_{name}.npz"),
                 voxels=pred_ids.astype(np.int64))
        print(f"[abl] {name:24s} latent_mse={latent_mse:.4f} acc_all={acc_all:.4f} "
              f"acc_nonair={acc_nonair:.4f}", flush=True)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"chunk_idx": args.chunk_idx, "target": args.target, "ckpt": args.ckpt,
                   "ckpt_step": ck.get('step', None), "metrics": metrics}, f, indent=2)
    print(f"[done] saved {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
