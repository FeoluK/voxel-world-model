"""v14 MaskGIT eval/rollout harness.

Modes (AR rollout, carry previous world + maskgit-fill the frontier + write-back):
  normal          : carry by true per-frame position delta
  zero_pos        : no carry shift (no motion) -> frontier empty -> world should stay
  reverse_pos     : carry the opposite direction
  zero_context    : mask the WHOLE grid each step -> pure unconditional generation
  long_roll       : synthetic walk in one direction for --long steps (generative)

Outputs: per-mode voxel-id sequences (mapped back to original 2021-vocab for the
renderer) + numbers (vs-GT overlap acc where GT exists; block-distribution / air
fraction / 'other' fraction to detect collapse or drift).
"""
import argparse, json, os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voxel_dataset import VoxelDiTDataset
from vae_3d import VAE3D
from voxel_maskgit import VoxelMaskGIT3D, maskgit_generate

_CODE = os.environ.get("V12_CODE_DIR", os.path.dirname(os.path.abspath(__file__)))
VAE_PT = os.environ.get("V12_VAE_PT", os.path.join(_CODE, "vae_final.pt"))
VOCAB_PT = os.environ.get("V14_VOCAB", os.path.join(_CODE, "voxel_vocab.npz"))
dev = "cuda"


def translate(world, shift):
    D, H, W = world.shape
    out = torch.zeros_like(world); valid = torch.zeros_like(world, dtype=torch.bool)
    sds, sss = [], []
    for s, n in zip(shift, (D, H, W)):
        s = int(s)
        if s >= 0: ds0, ds1, ss0, ss1 = s, n, 0, n - s
        else:      ds0, ds1, ss0, ss1 = 0, n + s, -s, n
        ds0 = max(0, min(n, ds0)); ds1 = max(ds0, min(n, ds1))
        ss0 = max(0, min(n, ss0)); ss1 = max(ss0, min(n, ss1))
        if ds1 <= ds0: return out, valid
        sds.append(slice(ds0, ds1)); sss.append(slice(ss0, ss1))
    out[sds[0], sds[1], sds[2]] = world[sss[0], sss[1], sss[2]]
    valid[sds[0], sds[1], sds[2]] = True
    return out, valid


def fill_frontier(model, known, fill, n_unmask, temp=1.0, top_p=1.0):
    if not fill.any():
        return known
    return maskgit_generate(model, known.unsqueeze(0), fill.unsqueeze(0), n_steps=n_unmask, temp=temp, top_p=top_p)[0]


def rollout(model, world0, shifts, mode, n_unmask=12, compound=False, temp=1.0, top_p=1.0):
    """world0 (D,H,W) compact seed. compound=False: carry FRAME-0 by each (cumulative)
    shift [ablations — preserves full motion]. compound=True: carry the running world
    by each per-step shift [long AR rollout]."""
    frames = [world0.clone()]; world = world0.clone()
    for sh in shifts:
        base = world if compound else world0
        if mode == "zero_context":
            known = base.clone(); fill = torch.ones_like(base, dtype=torch.bool)
        else:
            s = (0, 0, 0) if mode == "zero_pos" else (tuple(-np.array(sh)) if mode == "reverse_pos" else sh)
            carried, valid = translate(base, s)
            known, fill = carried, ~valid
        world = fill_frontier(model, known, fill, n_unmask, temp, top_p)
        frames.append(world.clone())
    return torch.stack(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_chunks", type=int, default=30)
    ap.add_argument("--n_unmask", type=int, default=12)
    ap.add_argument("--long", type=int, default=64)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    v = np.load(VOCAB_PT)
    id2c = torch.from_numpy(v["id2compact"]).long().to(dev)
    c2id = torch.from_numpy(v["compact2id"]).long().to(dev)
    K = int(v["num_classes"])

    ck = torch.load(args.ckpt, map_location="cpu")
    model = VoxelMaskGIT3D(num_classes=ck.get("num_classes", K), dim=ck.get("dim", 192)).to(dev).eval()
    model.load_state_dict({k.replace("module.", ""): v_ for k, v_ in ck["state_dict"].items()}, strict=True)
    print(f"[load] step={ck.get('step')} dim={ck.get('dim')} K={ck.get('num_classes')}", flush=True)

    vae = VAE3D(num_classes=2021, latent_channels=48, middle_channels=(32, 128, 512)).to(dev).eval()
    vck = torch.load(VAE_PT, map_location="cpu", weights_only=False)
    vae.load_state_dict(vck.get("state_dict", vck) if isinstance(vck, dict) else vck, strict=False)

    # high-movement chunk
    ds = VoxelDiTDataset(clip_len=33, samples_per_epoch=args.n_chunks, samples_per_episode=1,
                         require_rgb=False, require_clip=False, require_clip_pf=False,
                         clip_dir=None, clip_pf_dir=None, verbose=False)
    items = [ds[i] for i in range(args.n_chunks)]
    def disp(it): return (it["camera"][:, :3] - it["camera"][:1, :3]).norm(dim=1).max().item()
    band = [it for it in items if 6.0 <= disp(it) <= 30.0] or items
    it = sorted(band, key=lambda x: -disp(x))[0]
    lat = it["voxel_lat"].unsqueeze(0).to(dev).float()                    # (1,48,T,12,12,12)
    pos = it["camera"][:, :3].numpy()
    T = lat.shape[2]
    print(f"[chunk] disp={disp(it):.1f} T={T}", flush=True)

    @torch.no_grad()
    def gt_compact(t):
        ids = vae.decode(lat[:, :, t]).argmax(1)[0]    # (48,48,48) 2021-ids
        return id2c[ids]
    gt = torch.stack([gt_compact(t) for t in range(T)])  # (T,48,48,48) compact

    # CUMULATIVE shift from frame 0 (frame-to-frame deltas are sub-block and round to 0,
    # which silently discards all motion — must accumulate THEN round). axis map identity.
    shifts = [tuple((-np.round(pos[t] - pos[0])).astype(int)) for t in range(1, T)]

    # compact-id buckets for diversity (from voxel_vocab name mapping)
    TREE = torch.tensor([11, 20, 21, 28, 35, 46, 92, 98, 111], device=dev)   # logs/leaves/vine
    STRUCT = torch.tensor([14, 15, 16, 26, 55, 83, 105, 121], device=dev)    # planks/fence/glass/brick/cobble
    FOLIAGE = torch.tensor([17, 56, 57, 102, 104], device=dev)               # grass/tallgrass/flowers
    def stats(seq):  # seq (F,48,48,48) compact
        air = (seq == id2c[0].item()).float().mean().item()
        other = (seq == (K - 1)).float().mean().item()
        n_distinct = float(np.mean([int(torch.unique(seq[f]).numel()) for f in range(seq.shape[0])]))
        return {"air_frac": round(air, 3), "other_frac": round(other, 4),
                "tree_frac": round(float(torch.isin(seq, TREE).float().mean()), 4),
                "struct_frac": round(float(torch.isin(seq, STRUCT).float().mean()), 4),
                "foliage_frac": round(float(torch.isin(seq, FOLIAGE).float().mean()), 4),
                "distinct_blocks": round(n_distinct, 1)}

    def to_orig(seq):  # compact -> original ids for renderer
        return c2id[seq.long()].cpu().numpy().astype(np.int16)

    results = {}
    np.savez_compressed(f"{args.out}/voxels_GT.npz", voxels=to_orig(gt))
    for mode in ["normal", "zero_pos", "reverse_pos", "zero_context"]:
        seq = rollout(model, gt[0].clone(), shifts, mode, args.n_unmask, temp=args.temp, top_p=args.top_p)   # (T,...)
        # vs-GT overlap acc (only meaningful for normal/zero_pos/reverse)
        accs = []
        for t in [1, 5, 10, 20, T - 1]:
            g = gt[t]; m = (g != id2c[0].item())
            if m.sum() > 200: accs.append(((seq[t] == g)[m]).float().mean().item())
        results[mode] = {"acc_nonair_vs_gt": round(float(np.mean(accs)), 3) if accs else None, **stats(seq)}
        np.savez_compressed(f"{args.out}/voxels_{mode}.npz", voxels=to_orig(seq))
        print(f"[{mode}] {results[mode]}", flush=True)

    # long generative rollout: walk one direction, +1 voxel/frame on axis 0
    long_shifts = [(-1, 0, 0)] * args.long
    seq = rollout(model, gt[0].clone(), long_shifts, "normal", args.n_unmask, compound=True, temp=args.temp, top_p=args.top_p)
    # drift detector: stats over time (early vs late)
    early = stats(seq[:8]); late = stats(seq[-8:])
    results["long_roll"] = {"frames": seq.shape[0], "early": early, "late": late}
    np.savez_compressed(f"{args.out}/voxels_long_roll.npz", voxels=to_orig(seq))
    print(f"[long_roll] {results['long_roll']}", flush=True)

    json.dump(results, open(f"{args.out}/metrics.json", "w"), indent=2)
    print("EVAL_V14_DONE", flush=True)


if __name__ == "__main__":
    main()
