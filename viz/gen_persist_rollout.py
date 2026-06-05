"""Pod-side: build a GENUINELY MODEL-GENERATED, PERSISTENT voxel world for the
CS153 persistence demo, then traverse it along a scripted out-and-back path.

Approach (persistent canvas, lossless integer translation):
  1. Seed a 48^3 compact-id world from a high-movement treechop chunk (frame 0),
     same selection as eval_v14.
  2. Allocate an 80^3 canvas, place the seed at the centre (offset 16..63).
  3. ROLL THE GENERATOR OUTWARD once in +X,-X,+Z,-Z to FILL the canvas frontier:
     slide a 48^3 window outward in small steps, translate() the known region,
     maskgit_generate() the freshly-exposed strip, write the generated voxels back
     into the canvas. After this the 80^3 canvas is fully model-generated and FIXED.
  4. Traverse the now-fixed canvas with a 48^3 window along the path
     (centre -> 15 E -> back -> 15 W -> back -> 15 N -> back -> 15 S -> back).
     Each frame is a pure integer slice of the fixed canvas => return-to-centre is
     BIT-IDENTICAL by construction (this is the canvas_shift_lossless property).

Outputs (into $OUT):
  voxels_path.npz   : voxels (F,48,48,48) original 2021-ids for the iso renderer
  camera_path.json  : per-frame eye/yaw/pitch/key/label for iso + RGB panels
  canvas.npz        : the full 80^3 generated canvas (compact + original ids)
  persistence.json  : measured %% identical at each centre-return vs the start frame

Run inside nvcr pytorch container via srun (see run_persist_rollout.sbatch).
"""
import argparse, json, os, sys
import numpy as np, torch

CODE = "/home/flukol/v12/code"
sys.path.insert(0, CODE)
from voxel_maskgit import VoxelMaskGIT3D, maskgit_generate

dev = "cuda"
W = 48                      # window edge (model native)
CANVAS = 80                 # persistent canvas edge (holds centre +-15 in X,Z)
OFF = (CANVAS - W) // 2     # =16: where the seed window sits inside the canvas
DIST = 15                   # blocks out per leg
AXIS = {"X": 0, "Y": 1, "Z": 2}   # grid axes; Y is up


# ---------- translate / fill (mirror eval_v14) ----------
def translate(world, shift):
    D, H, Wd = world.shape
    out = torch.zeros_like(world); valid = torch.zeros_like(world, dtype=torch.bool)
    sds, sss = [], []
    for s, n in zip(shift, (D, H, Wd)):
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


def fill_frontier(model, known, fill, n_unmask, temp, top_p):
    if not fill.any():
        return known
    return maskgit_generate(model, known.unsqueeze(0), fill.unsqueeze(0),
                            n_steps=n_unmask, temp=temp, top_p=top_p)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed_npz", default="/home/flukol/persist_rollout/f0.npz",
                    help="real decoded world (48,48,48 'block_ids' 2021-ids) used as canvas seed")
    ap.add_argument("--n_unmask", type=int, default=12)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--fill_step", type=int, default=8, help="outward window slide per fill step")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    v = np.load(args.vocab)
    id2c = torch.from_numpy(v["id2compact"]).long().to(dev)
    c2id = torch.from_numpy(v["compact2id"]).long().to(dev)
    K = int(v["num_classes"])
    AIR_C = int(id2c[0].item())

    ck = torch.load(args.ckpt, map_location="cpu")
    model = VoxelMaskGIT3D(num_classes=ck.get("num_classes", K), dim=ck.get("dim", 192)).to(dev).eval()
    model.load_state_dict({k.replace("module.", ""): t for k, t in ck["state_dict"].items()}, strict=True)
    print(f"[load] step={ck.get('step')} dim={ck.get('dim')} K={ck.get('num_classes')}", flush=True)

    # ---- seed: load a REAL decoded world directly from npz (block_ids = 2021-ids).
    # Bypasses VoxelDiTDataset entirely (the seed only needs one voxel grid), which
    # avoids the "0 valid episodes -> high<=0" failure when the dataset env is empty.
    sd = np.load(args.seed_npz)
    ids = torch.from_numpy(sd["block_ids"].astype(np.int64)).to(dev)   # (48,48,48) 2021-ids
    ids = ids.clamp(0, 2020)
    seed = id2c[ids].long()                              # compact (48,48,48)
    print(f"[seed] from {args.seed_npz} shape={tuple(ids.shape)} "
          f"air_frac={(seed==AIR_C).float().mean():.3f} n_unique={len(torch.unique(seed))}", flush=True)

    # ---- canvas: place seed at centre; fill outward in each horizontal direction ----
    canvas = torch.full((CANVAS, CANVAS, CANVAS), AIR_C, dtype=torch.long, device=dev)
    known_mask = torch.zeros((CANVAS, CANVAS, CANVAS), dtype=torch.bool, device=dev)
    sl = slice(OFF, OFF + W)
    canvas[sl, sl, sl] = seed
    known_mask[sl, sl, sl] = True

    @torch.no_grad()
    def fill_direction(axis, sgn):
        """Slide a 48^3 window from centre outward along (axis, sign), filling the
        newly exposed strip with the generator and writing it back to the canvas."""
        steps = (OFF + args.fill_step - 1) // args.fill_step   # cover the OFF=16 margin
        # window lower corner, starts at the seed position
        corner = [OFF, OFF, OFF]
        for st in range(steps):
            corner[axis] += sgn * args.fill_step
            # clamp so window stays inside canvas
            corner[axis] = max(0, min(CANVAS - W, corner[axis]))
            cs = [slice(corner[d], corner[d] + W) for d in range(3)]
            win = canvas[cs[0], cs[1], cs[2]].clone()
            winknown = known_mask[cs[0], cs[1], cs[2]].clone()
            fill = ~winknown
            if not fill.any():
                continue
            gen = fill_frontier(model, win, fill, args.n_unmask, args.temp, args.top_p)
            # write back ONLY the newly generated (previously unknown) voxels
            newvox = torch.where(winknown, win, gen)
            canvas[cs[0], cs[1], cs[2]] = newvox
            known_mask[cs[0], cs[1], cs[2]] = True
        print(f"[fill axis={axis} sgn={sgn}] known_frac={known_mask.float().mean():.3f}", flush=True)

    for axis, sgn in [(0, +1), (0, -1), (2, +1), (2, -1)]:
        fill_direction(axis, sgn)
    # any leftover unknown corners: fill once with a centred-ish pass per Z-slab not needed;
    # report coverage
    print(f"[canvas] final known_frac={known_mask.float().mean():.3f} "
          f"air_frac={(canvas==AIR_C).float().mean():.3f}", flush=True)

    # ---- build the scripted path of WINDOW OFFSETS into the fixed canvas ----
    # camera stays centred; the window into the canvas shifts. Offset (dx,dz) in blocks.
    # base window corner = OFF (so dx=dz=0 -> the seed window).
    legs = [("X", +1, "E", "W"), ("X", -1, "W", "S"),
            ("Z", +1, "N", "D"), ("Z", -1, "S", "A")]
    # yaw per travel direction (matches camera_basis_np in panel_a):
    #   yaw=0 -> +Z, yaw=180 -> -Z, yaw=90 -> -X, yaw=270 -> +X
    YAW_DIR = {("X", +1): 270.0, ("X", -1): 90.0, ("Z", +1): 0.0, ("Z", -1): 180.0}
    KEY_DIR = {("X", +1): "W", ("X", -1): "S", ("Z", +1): "D", ("Z", -1): "A"}
    N_LEG = 15           # frames per half-leg (out / back)
    TURN = 18            # frames for a smooth 90deg turn (~1.5s @ 12fps)
    HOLD = 3
    CX_RGB, CZ_RGB = 24.0, 24.0
    EYE_Y, PITCH = 28.0, 12.0

    def ease(t): return t * t * (3 - 2 * t)

    def ang_lerp(a0, a1, t):
        d = ((a1 - a0 + 180) % 360) - 180
        return (a0 + d * t) % 360

    frames = []   # each: dict(offx, offz, yaw, key, label, at_center, phase)

    def add(offx, offz, yaw, key, label, at_center, phase):
        frames.append(dict(offx=int(round(offx)), offz=int(round(offz)),
                           yaw=float(yaw), key=key, label=label,
                           at_center=at_center, phase=phase))

    cur_yaw = YAW_DIR[("X", +1)]
    for _ in range(HOLD):
        add(0, 0, cur_yaw, None, "start", True, "hold")

    for li, (ax, sgn, name, _k) in enumerate(legs):
        out_yaw = YAW_DIR[(ax, sgn)]
        out_key = KEY_DIR[(ax, sgn)]
        back_yaw = YAW_DIR[(ax, -sgn)]
        back_key = KEY_DIR[(ax, -sgn)]
        # smooth turn from cur_yaw -> out_yaw before moving out
        for i in range(1, TURN + 1):
            y = ang_lerp(cur_yaw, out_yaw, ease(i / TURN))
            add(0, 0, y, None, "turning", True, "turn")
        cur_yaw = out_yaw
        # OUT
        for i in range(1, N_LEG + 1):
            s = ease(i / N_LEG) * DIST
            ox = sgn * s if ax == "X" else 0.0
            oz = sgn * s if ax == "Z" else 0.0
            add(ox, oz, out_yaw, out_key, f"{out_key}  out {name} +{int(round(s))}", False, "out")
        # smooth turn to face back direction (180 flip done over a turn too)
        for i in range(1, TURN + 1):
            y = ang_lerp(out_yaw, back_yaw, ease(i / TURN))
            ox = sgn * DIST if ax == "X" else 0.0
            oz = sgn * DIST if ax == "Z" else 0.0
            add(ox, oz, y, None, "turning", False, "turn")
        cur_yaw = back_yaw
        # BACK
        for i in range(1, N_LEG + 1):
            s = (1.0 - ease(i / N_LEG)) * DIST
            ox = sgn * s if ax == "X" else 0.0
            oz = sgn * s if ax == "Z" else 0.0
            add(ox, oz, back_yaw, back_key, f"{back_key}  return {name}", False, "back")
        # HOLD at centre (persistence moment)
        for _ in range(HOLD):
            add(0, 0, back_yaw, None, "BACK AT CENTER", True, "hold")

    # ---- materialise each frame's 48^3 window from the FIXED canvas ----
    F = len(frames)
    out_vox = np.zeros((F, W, W, W), dtype=np.int16)
    start_window = None
    persistence = []
    for fi, fr in enumerate(frames):
        dx, dz = fr["offx"], fr["offz"]
        x0, z0 = OFF + dx, OFF + dz
        x0 = max(0, min(CANVAS - W, x0)); z0 = max(0, min(CANVAS - W, z0))
        win = canvas[x0:x0 + W, OFF:OFF + W, z0:z0 + W]    # (48,48,48) compact
        orig = c2id[win.long()].cpu().numpy().astype(np.int16)
        out_vox[fi] = orig
        if fr["at_center"] and dx == 0 and dz == 0:
            if start_window is None:
                start_window = win.clone()
            else:
                same = (win == start_window).float().mean().item()
                persistence.append({"frame": fi, "label": fr["label"], "pct_identical": round(100 * same, 3)})

    np.savez_compressed(os.path.join(args.out, "voxels_path.npz"), voxels=out_vox)
    np.savez_compressed(os.path.join(args.out, "canvas.npz"),
                        canvas_compact=canvas.cpu().numpy().astype(np.int16),
                        canvas_orig=c2id[canvas.long()].cpu().numpy().astype(np.int16))

    # camera_path.json for iso + RGB panels (eye is grid-local; camera stays centred)
    cam = [{"eye": [CX_RGB, EYE_Y, CZ_RGB], "yaw": fr["yaw"], "pitch": PITCH,
            "key": fr["key"], "label": fr["label"], "at_center": fr["at_center"],
            "phase": fr["phase"], "offx": fr["offx"], "offz": fr["offz"]} for fr in frames]
    json.dump(cam, open(os.path.join(args.out, "camera_path.json"), "w"), indent=1)
    json.dump({"n_frames": F, "canvas": CANVAS, "dist": DIST,
               "center_returns": persistence,
               "mean_pct_identical": round(float(np.mean([p["pct_identical"] for p in persistence])), 3) if persistence else None},
              open(os.path.join(args.out, "persistence.json"), "w"), indent=2)
    print(f"[done] frames={F} persistence={persistence}", flush=True)
    print("GEN_PERSIST_DONE", flush=True)


if __name__ == "__main__":
    main()
