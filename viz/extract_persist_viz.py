"""Pod-side extractor for the PERSIST camera-visibility figure.

For a chosen episode + chunk-aligned 33-frame window, this:
  - decodes the voxel latent of each requested frame -> 2021 block-ids AND
    compact-id grid (48^3),
  - computes the exact grid-local eye/yaw/pitch the renderer uses (_exact_eye),
  - raycasts (batched_raycast: depth = distance ALONG ray, dir = unit) to get
    the FIRST-HIT voxel coordinate per pixel -> the VISIBLE voxel SET,
  - Wan-decodes the GT rgb_lat for that chunk (panel B),
  - runs the flow renderer (panel C) via sample_ode.

Saves one .npz per requested source-frame index with everything panel A/B/C need.
The isometric panel-A render happens locally on the Mac (palette lives there).

Run inside the nvcr pytorch container via srun (see sbatch).
"""
import os, sys, json
import numpy as np
import torch
import importlib.util as u

RV2 = "/home/flukol/render_v2"
sys.path.insert(0, RV2)
os.chdir(RV2)

from voxel_dataset import normalize_camera, _extract_actions_camera, _read_jsonl
from vae_3d import VAE3D
from renderer_explicit import (batched_raycast, voxel_to_compact_grids,
                               LAT_TO_SRC)
from renderer_flow import (CondEncoder, FlowVNet, sample_ode, _exact_eye)

dev = torch.device("cuda")

DATA = "/home/flukol/treechop_gen"
EP = os.environ["EP"]
T0 = int(os.environ["T0"])                       # chunk-aligned window start (source frame)
OUT = os.environ.get("OUT", "/home/flukol/persist_viz")
os.makedirs(OUT, exist_ok=True)
# which of the 9 LAT_TO_SRC frames to fully process (still=one, turn=all)
WHICH = os.environ.get("WHICH", "all")           # "all" or comma list of indices into LAT_TO_SRC

VOX_VAE = "/home/flukol/v12/code/vae_final.pt"
VOX_VOCAB = "/home/flukol/v12/code/voxel_vocab.npz"
WAN_MODULE = "/home/flukol/v12/wan/modules/vae.py"
WAN_PTH = "/home/flukol/v12/mg2_weights/Wan2.1_VAE.pth"
FLOW_CKPT = "/home/flukol/render_v2_treechop/ckpt/BEST_treechop_renderer_step29950.pt"

# ----- load voxel VAE + vocab -----
vae = VAE3D(num_classes=2021, latent_channels=48, middle_channels=(32, 128, 512))
sd = torch.load(VOX_VAE, map_location="cpu"); sd = sd.get("model", sd.get("state_dict", sd))
vae.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
vae = vae.to(dev).eval()
vv = np.load(VOX_VOCAB)
id2compact = torch.from_numpy(vv["id2compact"]).long().to(dev)
print("vae+vocab loaded", flush=True)

# ----- Wan VAE -----
spec = u.spec_from_file_location("wm", WAN_MODULE); m = u.module_from_spec(spec); spec.loader.exec_module(m)
wan = m.WanVAE(vae_pth=WAN_PTH, device=dev)
print("wan loaded", flush=True)

# ----- flow renderer -----
fck = torch.load(FLOW_CKPT, map_location="cpu")
cond_ch = fck.get("cond_ch", 64); fbase = fck.get("base", 128); cbase = fck.get("cond_base", 96)
FH = fck.get("rast_h", 72); FW = fck.get("rast_w", 128)
cond_enc = CondEncoder(base=cbase, cond_ch=cond_ch).to(dev).eval()
vnet = FlowVNet(cond_ch=cond_ch, base=fbase).to(dev).eval()
cond_enc.load_state_dict(fck["cond_enc"]); vnet.load_state_dict(fck["vnet"])
print(f"flow loaded @ {FH}x{FW} step {fck.get('step')}", flush=True)

# ----- load episode -----
with np.load(f"{DATA}/latents/{EP}.npz") as z:
    vox_all = np.asarray(z["latents"])            # (N,48,12,12,12) fp16
with np.load(f"{DATA}/rgb_latents/{EP}.npz") as z:
    rgb_all = np.asarray(z["latents"])            # (16, F_lat, 27, 48)
recs = _read_jsonl(f"{DATA}/jsonl/{EP}.jsonl")
n_use = min(vox_all.shape[0], len(recs))
_, _, cam_full = _extract_actions_camera(recs, n_use)   # (N,5) raw

CLIP = 33
t0, t1 = T0, T0 + CLIP
# Match dataset FIX 20: voxel slice shifted +1
vox_clip = vox_all[t0 + 1:t1 + 1]                       # (33,48,12,12,12)
vox_clip = np.transpose(vox_clip, (1, 0, 2, 3, 4))      # (48,33,12,12,12)
vox = torch.from_numpy(vox_clip.copy()).unsqueeze(0).to(dev).float()  # (1,48,33,...)

raw_cam = torch.from_numpy(cam_full[t0:t1].copy().astype("float32")).unsqueeze(0).to(dev)  # (1,33,5)
cam_norm = torch.from_numpy(normalize_camera(cam_full[t0:t1])).unsqueeze(0).to(dev)        # (1,33,5)

c = t0 // CLIP
rgb_clip = rgb_all[:, c * 9:c * 9 + 9]                  # (16,9,27,48)
rgb_lat = torch.from_numpy(rgb_clip.copy()).float().to(dev)

# ----- decode compact grids for the 9 LAT_TO_SRC frames (48^3) -----
grids = voxel_to_compact_grids(vae, id2compact, vox, dev)     # (1,9,48,48,48) compact
# also decode 2021 ids for the isometric palette
sel = vox[:, :, LAT_TO_SRC].permute(0, 2, 1, 3, 4, 5).reshape(9, 48, 12, 12, 12).float()
ids2021 = []
with torch.no_grad():
    for i in range(0, 9, 3):
        logit = vae.decode(sel[i:i + 3])
        ids2021.append(logit.argmax(1))
ids2021 = torch.cat(ids2021, 0).cpu().numpy().astype(np.int32)  # (9,48,48,48) world block ids

# ----- exact eye / yaw / pitch (renderer convention) -----
rc = raw_cam[:, LAT_TO_SRC]                                    # (1,9,5)
origins, yaw, pit = _exact_eye(rc)                             # (1,9,3),(1,9),(1,9)

# ----- raycast for VISIBLE set (depth = distance along ray) -----
RH, RW = 144, 256
with torch.no_grad():
    id_map, depth, hit, dirs = batched_raycast(grids, origins, yaw, pit, RH, RW)
# id_map (1,9,RH,RW); depth (1,9,RH,RW); hit bool; dirs (1,9,RH,RW,3)

# ----- Wan-decode GT (panel B) -----
with torch.no_grad():
    gt = wan.decode(rgb_lat.unsqueeze(0))[0]
gt = ((gt.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()  # (9,216,384,3)

# ----- flow render (panel C) -----
with torch.no_grad():
    pred = sample_ode(vnet, cond_enc, vae, id2compact, vox, cam_norm, dev, FH, FW,
                      n_steps=32, seed=0, raw_cam=raw_cam)
    pr = wan.decode(pred)[0]
pr = ((pr.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()   # (9,216,384,3)

# ----- DENSE per-frame Panel-A data (all 33 frames in the window) -----
# Decode every voxel frame -> compact + 2021 ids, raycast at its own camera.
DENSE = os.environ.get("DENSE", "0") == "1"
if DENSE:
    RHd, RWd = 120, 213
    voxf = vox[0].permute(1, 0, 2, 3, 4)               # (33,48,12,12,12)
    raw33 = raw_cam[0]                                  # (33,5)
    bx0 = torch.floor(raw33[:, 0]) - 24.0
    by0 = torch.clamp(torch.floor(raw33[:, 1]) - 24.0, 0.0, 256.0 - 48.0)
    bz0 = torch.floor(raw33[:, 2]) - 24.0
    eye33 = torch.stack([raw33[:, 0] - bx0, (raw33[:, 1] - by0) + 1.62, raw33[:, 2] - bz0], dim=-1)
    yaw33 = raw33[:, 3]; pit33 = raw33[:, 4]
    for fi in range(0, 33, 2):                          # every other frame -> 17 frames
        with torch.no_grad():
            logit = vae.decode(voxf[fi:fi+1].float())
            ids21 = logit.argmax(1)[0]                  # (48,48,48)
            cg = id2compact[ids21.clamp(0, 2020)].unsqueeze(0).unsqueeze(0)  # (1,1,48,48,48)
            o = eye33[fi:fi+1].view(1, 1, 3)
            im, dp, ht, dr = batched_raycast(cg, o, yaw33[fi:fi+1].view(1, 1),
                                             pit33[fi:fi+1].view(1, 1), RHd, RWd)
            pts = o.view(1, 1, 3) + dr[0, 0] * dp[0, 0].unsqueeze(-1)   # (RHd,RWd,3)
            vix = torch.floor(pts).long()[ht[0, 0]].clamp(0, 47)
            vis = torch.unique(vix, dim=0).cpu().numpy().astype(np.int32)
        np.savez_compressed(
            f"{OUT}/{EP}_t{T0}_dense{fi:02d}.npz",
            block_ids=ids21.cpu().numpy().astype(np.int32), visible=vis,
            eye=eye33[fi].cpu().numpy(), yaw=float(yaw33[fi]), pitch=float(pit33[fi]),
            src_frame=T0 + fi, hit_frac=float(ht.float().mean().item()))
        print(f"dense{fi:02d} yaw={float(yaw33[fi]):.1f} vis={len(vis)} hit%={ht.float().mean().item():.2f}", flush=True)

# ----- per-frame visible-set + save (the 9 render-aligned frames) -----
which = list(range(9)) if WHICH == "all" else [int(x) for x in WHICH.split(",")]
origins_np = origins[0].cpu().numpy()
yaw_np = yaw[0].cpu().numpy(); pit_np = pit[0].cpu().numpy()
for f in which:
    o = origins[0, f].view(1, 1, 3)
    d = dirs[0, f]                                  # (RH,RW,3)
    dep = depth[0, f]                               # (RH,RW)
    h = hit[0, f]                                   # (RH,RW)
    pts = o + d * dep.unsqueeze(-1)                 # (RH,RW,3) hit points
    vox_ix = torch.floor(pts).long()
    vox_ix = vox_ix[h]                              # (Nhit,3)
    vox_ix = vox_ix.clamp(0, 47)
    vis = torch.unique(vox_ix, dim=0).cpu().numpy().astype(np.int32)  # (M,3) visible voxel coords (grid X,Y,Z)
    src_frame = T0 + LAT_TO_SRC[f]
    np.savez_compressed(
        f"{OUT}/{EP}_t{T0}_f{f}.npz",
        block_ids=ids2021[f],                        # (48,48,48) world ids, grid (X,Y,Z)
        visible=vis,                                 # (M,3) X,Y,Z grid coords visible from camera
        eye=origins_np[f], yaw=float(yaw_np[f]), pitch=float(pit_np[f]),
        gt_rgb=gt[f], render_rgb=pr[f],              # (216,384,3) uint8
        hit_frac=float(h.float().mean().item()),
        src_frame=src_frame,
    )
    print(f"f{f} src{src_frame} eye={origins_np[f].round(2)} yaw={yaw_np[f]:.1f} pit={pit_np[f]:.1f} "
          f"hit%={h.float().mean().item():.2f} visible_vox={len(vis)}", flush=True)

print("DONE", flush=True)
