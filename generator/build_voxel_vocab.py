"""Build a CORRECTED compact voxel vocab from the TREECHOP latents.

Why this exists
---------------
The original voxel_vocab.npz was built on the wrong (non-treechop) data, so
common treechop blocks fell into the catch-all bucket. Worse, the catch-all's
*representative* original-id was clay (vae_id 6), so every catch-all voxel
renders as grey clay. Generated birch (birch_leaves=vae_id 4, birch_log=vae_id
83) was merged into the catch-all -> rendered as clay.

This script:
  1) Decodes every treechop latent through the (frozen) VAE -> 2021-class ids.
  2) Histograms ALL voxels across ALL frames of ALL clips (true frequency).
  3) Assigns its OWN compact class to every block whose voxel-fraction is above
     a coverage threshold (default: covers >= 99.5% of all non-air voxels, with
     an explicit ALWAYS-KEEP allow-list of treechop-critical blocks so birch /
     oak / spruce logs+leaves, grass, dirt, the stone family, water and common
     foliage can NEVER be dropped even if a single clip is under-sampled).
  4) Puts everything else in ONE catch-all class whose representative original-id
     is AIR (vae_id 0) -- a NEUTRAL block. A rare/unknown generated voxel now
     renders as empty space, never as a wrong solid grey block.

Output voxel_vocab.npz schema (unchanged, drop-in for train_v14_sf.py / eval):
    id2compact  (2021,)  int16   vae_id   -> compact class  (0..K-1)
    compact2id  (K,)     int16   compact  -> representative vae_id (for render)
    num_classes ()       int64   K (= number of kept classes + 1 catch-all)
Plus extra diagnostic arrays (ignored by trainer, handy for sanity):
    compact_names (K,) <U…  human-readable name of each compact class
    kept_fraction ()        fraction of non-air voxels covered by own-class blocks

Run on a GPU node via sbatch (decode needs the VAE on CUDA). Example:
    python build_voxel_vocab.py \
        --latents /home/flukol/treechop_gen/latents \
        --vae /home/flukol/v12/code/vae_final.pt \
        --palette /home/flukol/v12/code/global_palette.json \
        --out /home/flukol/v12/code/voxel_vocab_treechop.npz \
        --frames_per_clip 24 --coverage 0.995
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vae_3d import VAE3D

VAE_NUM_CLASSES = 2021
AIR_ID = 0   # vae_id 0 == "" / air in global_palette.json -> NEUTRAL catch-all rep

# Treechop-critical blocks that must ALWAYS get their own compact class.
# Matched as a substring against the palette name (state-stripped), case-insensitive.
ALWAYS_KEEP_SUBSTR = [
    "air",
    "birch_log", "birch_leaves",
    "oak_log", "oak_leaves",
    "spruce_log", "spruce_leaves",
    "jungle_log", "jungle_leaves",
    "acacia_log", "acacia_leaves",
    "dark_oak_log", "dark_oak_leaves",
    "grass_block", "grass", "tall_grass", "fern", "large_fern",
    "dirt", "coarse_dirt", "podzol", "mycelium",
    "stone", "cobblestone", "granite", "diorite", "andesite", "gravel",
    "sand", "sandstone", "clay",
    "water", "flowing_water",
    "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet",
    "oxeye_daisy", "cornflower", "lily_of_the_valley",
    "vine", "leaves", "log", "snow", "ice",
    "coal_ore", "iron_ore", "copper_ore",
    "mushroom", "sapling", "deadbush", "sugar_cane",
]


def log(*a):
    print(*a, flush=True)


def load_palette(path):
    p = json.load(open(path))
    id2names = {}
    for k, v in p.items():
        v = int(v)
        # prefer the shortest / first name for an id; strip block-state "[...]"
        base = k.split("[")[0] if k else ""
        id2names.setdefault(v, base)
    return id2names


def get_lat(npz):
    d = np.load(npz)
    arr = None
    for k in d.files:
        a = d[k]
        if a.ndim >= 4 and a.dtype.kind == "f":
            arr = a; break
    a = np.asarray(arr)
    if a.ndim == 5 and a.shape[0] == 48:        # (48,T,12,12,12) -> (T,48,...)
        a = np.transpose(a, (1, 0, 2, 3, 4))
    return a                                     # (T,48,12,12,12)


@torch.no_grad()
def histogram(latents_dir, vae, dev, frames_per_clip, chunk=8):
    counts = torch.zeros(VAE_NUM_CLASSES, dtype=torch.float64, device=dev)
    files = sorted(glob.glob(os.path.join(latents_dir, "*.npz")))
    log(f"histogramming {len(files)} clips, up to {frames_per_clip} frames each")
    for fi, f in enumerate(files):
        lat = get_lat(f)
        T = lat.shape[0]
        idxs = np.linspace(0, T - 1, min(frames_per_clip, T)).astype(int)
        x = torch.from_numpy(lat[idxs]).float().to(dev)      # (n,48,12,12,12)
        for i in range(0, x.shape[0], chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = vae.decode(x[i:i + chunk])
            ids = logit.argmax(1).reshape(-1)                # (n*48^3,)
            counts += torch.bincount(ids, minlength=VAE_NUM_CLASSES).double()
        if (fi + 1) % 10 == 0:
            log(f"  {fi+1}/{len(files)} clips")
    return counts.cpu().numpy()


def build_vocab(counts, id2names, coverage):
    total = counts.sum()
    nonair = total - counts[AIR_ID]
    order = np.argsort(-counts)                  # most frequent first

    keep = set()
    # 1) always-keep allow-list (only ids that actually appear in the palette / data)
    for vid in range(VAE_NUM_CLASSES):
        nm = id2names.get(vid, "").lower()
        if not nm:
            continue
        if any(s in nm for s in ALWAYS_KEEP_SUBSTR):
            keep.add(vid)
    keep.add(AIR_ID)
    # 2) frequency coverage: add most-frequent blocks until we cover `coverage`
    #    of all NON-AIR voxels (air handled separately as its own class).
    cum = 0.0
    for vid in order:
        if vid == AIR_ID:
            continue
        if cum >= coverage * nonair:
            break
        keep.add(int(vid))
        cum += counts[vid]

    # drop allow-list ids that never appear AT ALL (no point giving them a class
    # the model can never see a target for) -- but keep air always.
    keep = {v for v in keep if counts[v] > 0 or v == AIR_ID}

    # deterministic ordering: air=0 first, then by descending frequency
    kept_sorted = [AIR_ID] + sorted([v for v in keep if v != AIR_ID],
                                    key=lambda v: -counts[v])
    K_keep = len(kept_sorted)
    CATCHALL = K_keep                            # last compact class = catch-all
    num_classes = K_keep + 1

    id2compact = np.full(VAE_NUM_CLASSES, CATCHALL, dtype=np.int16)
    for comp, vid in enumerate(kept_sorted):
        id2compact[vid] = comp

    compact2id = np.zeros(num_classes, dtype=np.int16)
    for comp, vid in enumerate(kept_sorted):
        compact2id[comp] = vid
    compact2id[CATCHALL] = AIR_ID               # NEUTRAL catch-all representative

    names = [id2names.get(int(v), f"id{v}") for v in kept_sorted] + ["<catchall=air>"]
    covered = counts[[v for v in kept_sorted if v != AIR_ID]].sum()
    kept_fraction = float(covered / max(nonair, 1))
    return id2compact, compact2id, num_classes, names, kept_fraction, kept_sorted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--palette", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames_per_clip", type=int, default=24)
    ap.add_argument("--coverage", type=float, default=0.995,
                    help="fraction of NON-AIR voxels that own-classes must cover")
    ap.add_argument("--hist_cache", default="",
                    help="optional .npy to save/load the raw 2021-id counts")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    id2names = load_palette(args.palette)

    if args.hist_cache and os.path.exists(args.hist_cache):
        log(f"loading cached histogram {args.hist_cache}")
        counts = np.load(args.hist_cache)
    else:
        vae = VAE3D(num_classes=VAE_NUM_CLASSES, latent_channels=48,
                    middle_channels=(32, 128, 512)).to(dev).eval()
        ck = torch.load(args.vae, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
        vae.load_state_dict(sd, strict=False)
        for p in vae.parameters():
            p.requires_grad_(False)
        counts = histogram(args.latents, vae, dev, args.frames_per_clip)
        if args.hist_cache:
            np.save(args.hist_cache, counts)
            log(f"saved histogram cache {args.hist_cache}")

    id2compact, compact2id, K, names, kept_frac, kept_sorted = build_vocab(
        counts, id2names, args.coverage)

    log("=" * 64)
    log(f"num_classes = {K}  (={K-1} own-classes + 1 catch-all)")
    log(f"non-air coverage by own-classes = {kept_frac*100:.3f}%")
    log(f"catch-all representative vae_id = {compact2id[K-1]} "
        f"({id2names.get(int(compact2id[K-1]),'?')})  <-- must be air/neutral")
    log("top-30 own-classes (compact -> vae_id : name : voxel_count):")
    for comp, vid in enumerate(kept_sorted[:30]):
        log(f"  {comp:3d} -> {vid:4d} : {names[comp]:32s} : {int(counts[vid]):>12,}")
    # sanity: are the birch blocks present and own-classed?
    name2id = {}
    for vid, nm in id2names.items():
        name2id.setdefault(nm, vid)
    for nm in ["minecraft:birch_log", "minecraft:birch_leaves",
               "minecraft:oak_log", "minecraft:oak_leaves",
               "minecraft:spruce_log", "minecraft:spruce_leaves",
               "minecraft:grass_block", "minecraft:dirt", "minecraft:water"]:
        vid = name2id.get(nm)
        if vid is None:
            log(f"  CHECK {nm}: NOT IN PALETTE")
            continue
        comp = int(id2compact[vid])
        own = comp != (K - 1)
        log(f"  CHECK {nm}: vae_id={vid} count={int(counts[vid]):>10,} "
            f"compact={comp} {'OWN-CLASS' if own else '!!! CATCH-ALL !!!'}")

    np.savez(args.out,
             id2compact=id2compact,
             compact2id=compact2id,
             num_classes=np.int64(K),
             compact_names=np.array(names),
             kept_fraction=np.float64(kept_frac))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
