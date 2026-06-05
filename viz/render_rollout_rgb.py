"""Pod-side: render the first-person RGB panel for the GENERATIVE persistence demo.

Unlike v1 (single fixed chunk + strafing eye), here the WORLD is the per-frame
model-generated 48^3 window from gen_persist_rollout.py and the camera stays
CENTRED (eye ~grid centre, only yaw rotates). Each frame's generated voxels are
rasterised at the centred pose and run through the v2 flow renderer.

Inputs (from gen_persist_rollout output, on the pod):
  $VOX_NPZ   voxels_path.npz : voxels (F,48,48,48) ORIGINAL 2021-ids
  $PATH_JSON camera_path.json : yaw/pitch/label per frame (centred eye)
Outputs: one rgb_XXXX.png (216x384) per frame into $OUT.
"""
import os, sys, json
import numpy as np
import torch

RV2 = "/home/flukol/render_v2"
sys.path.insert(0, RV2)
os.chdir(RV2)

from renderer_explicit import RGB_C, RGB_H, RGB_W, RGB_T
from raster_torch import rasterize_batch
from renderer_flow import CondEncoder, FlowVNet
import importlib.util as u

dev = torch.device("cuda")

VOX_NPZ = os.environ["VOX_NPZ"]
PATH_JSON = os.environ["PATH_JSON"]
OUT = os.environ.get("OUT", "/home/flukol/persist_rollout/rgb")
os.makedirs(OUT, exist_ok=True)

RENDER_VOCAB = "/home/flukol/v12/code/voxel_vocab.npz"     # 129-class render space
WAN_MODULE = "/home/flukol/v12/wan/modules/vae.py"
WAN_PTH = "/home/flukol/v12/mg2_weights/Wan2.1_VAE.pth"
FLOW_CKPT = "/home/flukol/render_v2_treechop/ckpt/BEST_treechop_renderer_step29950.pt"
EYE_Y_RGB = float(os.environ.get("EYE_Y_RGB", "25.6"))

# ---- render-space vocab (2021-id -> render compact) ----
rv = np.load(RENDER_VOCAB)
r_id2c = torch.from_numpy(rv["id2compact"]).long().to(dev)   # (2021,)
print("render vocab loaded", flush=True)

# ---- Wan VAE ----
spec = u.spec_from_file_location("wm", WAN_MODULE); m = u.module_from_spec(spec); spec.loader.exec_module(m)
wan = m.WanVAE(vae_pth=WAN_PTH, device=dev)
print("wan loaded", flush=True)

# ---- flow renderer ----
fck = torch.load(FLOW_CKPT, map_location="cpu")
cond_ch = fck.get("cond_ch", 64); fbase = fck.get("base", 128); cbase = fck.get("cond_base", 96)
FH = fck.get("rast_h", 72); FW = fck.get("rast_w", 128)
cond_enc = CondEncoder(base=cbase, cond_ch=cond_ch).to(dev).eval()
vnet = FlowVNet(cond_ch=cond_ch, base=fbase).to(dev).eval()
cond_enc.load_state_dict(fck["cond_enc"]); vnet.load_state_dict(fck["vnet"])
print(f"flow loaded @ {FH}x{FW} step {fck.get('step')}", flush=True)

# ---- generated voxels + path ----
vox_seq = np.load(VOX_NPZ)["voxels"].astype(np.int64)        # (F,48,48,48) 2021-ids
path = json.load(open(PATH_JSON))
N = min(len(vox_seq), len(path))
yaws = np.array([path[i]["yaw"] for i in range(N)], dtype=np.float32)
pits = np.array([path[i]["pitch"] for i in range(N)], dtype=np.float32)
print(f"frames: {N} grid {vox_seq.shape[1:]}", flush=True)

# centred eye for every frame (world shifts under the camera)
eye = np.array([24.0, EYE_Y_RGB, 24.0], dtype=np.float32)


def compact_grid(frame_2021):
    g = torch.from_numpy(frame_2021).long().to(dev)           # (48,48,48) 2021-id
    return r_id2c[g.clamp(0, 2020)]                            # (48,48,48) render-compact


def sample(cond, B, seed=0, n_steps=32):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(B, RGB_C, RGB_T, RGB_H, RGB_W, device=dev, generator=g)
    taus = torch.linspace(1.0, 0.0, n_steps + 1, device=dev)
    for i in range(n_steps):
        dt = (taus[i] - taus[i + 1])
        tvec = torch.full((B,), float(taus[i]), device=dev)
        with torch.no_grad():
            v = vnet(x, tvec, cond)
        x = x - dt * v
    return x


# process in windows of RGB_T(=9) frames; each frame uses ITS OWN generated grid
W = RGB_T
rgbs = []
for s in range(0, N, W):
    chunk = list(range(s, min(s + W, N)))
    pad = W - len(chunk)
    idx = chunk + [chunk[-1]] * pad
    grids = torch.stack([compact_grid(vox_seq[i]) for i in idx], 0).unsqueeze(0)   # (1,9,48,48,48)
    o = torch.from_numpy(np.tile(eye, (W, 1))).view(1, W, 3).to(dev)
    y = torch.from_numpy(yaws[idx]).view(1, W).to(dev)
    p = torch.from_numpy(pits[idx]).view(1, W).to(dev)
    with torch.no_grad():
        id_map, depth, hit, dirs = rasterize_batch(grids, o, y, p, FH, FW)
        cond = cond_enc(id_map, depth, hit, dirs)                 # (1,9,cond_ch,27,48)
        cond = cond.permute(0, 2, 1, 3, 4).contiguous()
    pred = sample(cond, 1, seed=0)
    with torch.no_grad():
        pr = wan.decode(pred)[0]
    pr = ((pr.clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()  # (9,216,384,3)
    for j, fi in enumerate(chunk):
        rgbs.append((fi, pr[j], float(hit[0, j].float().mean().item())))
    print(f"window {s}-{chunk[-1]} hit%={[round(float(hit[0,j].float().mean()),2) for j in range(len(chunk))]}", flush=True)

from PIL import Image
hits = []
for fi, im, hf in rgbs:
    Image.fromarray(im).save(f"{OUT}/rgb_{fi:04d}.png")
    hits.append(hf)
print(f"SAVED {len(rgbs)} rgb frames to {OUT}; mean hit%={np.mean(hits):.3f}", flush=True)
print("DONE", flush=True)
