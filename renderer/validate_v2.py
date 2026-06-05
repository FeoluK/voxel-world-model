"""Validate the FLOW shader vs the blurry MSE regression baseline.

For the SAME held-out samples decode 3-up RGB:
    GT | REGRESSION(blurry MSE) | FLOW(ODE-sampled)
Saves stills + mp4 + an HF-energy metric (higher = sharper; GT is the ceiling).

Picks SURFACE/forest samples (bright frames). Wan-decode for all three.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import imageio.v3 as iio
import importlib.util as u

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # render_v2 wins over PYTHONPATH code dir
from voxel_dataset import VoxelDiTDataset
from vae_3d import VAE3D
from renderer_explicit import (ShaderUNet, batched_raycast, build_screen_buffer,
                               voxel_to_compact_grids, LAT_TO_SRC,
                               EYE_CENTER, EYE_HEIGHT)
from renderer_flow import CondEncoder, FlowVNet, sample_ode

OUT = os.environ.get("OUT", "/home/flukol/render_v2/val")
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda")
FLOW_CKPT = os.environ.get("FLOW_CKPT", f"{OUT}/ckpt/flow_final.pt")
BASE_CKPT = os.environ.get("BASE_CKPT", "/scratch/users/flukol/render_real/ckpt/shader_final.pt")
USE_BASE = os.environ.get("USE_BASE", "1") == "1"   # set 0 on pod (no regression ckpt)
BASE_RAST_H, BASE_RAST_W = 56, 96
TAG = os.environ.get("TAG", "flow")
N = int(os.environ.get("N", "4"))
N_ODE = int(os.environ.get("N_ODE", "32"))
BRIGHT = os.environ.get("BRIGHT", "1") == "1"
WAN_MODULE = os.environ.get("WAN_MODULE", "/home/flukol/v12/wan/modules/vae.py")
WAN_PTH = os.environ.get("WAN_PTH", "/scratch/users/flukol/mg2_weights/Wan2.1_VAE.pth")
VOX_VAE = os.environ.get("VOX_VAE", "/home/flukol/v12/code/vae_final.pt")
VOX_VOCAB = os.environ.get("VOX_VOCAB", "/home/flukol/v12/code/voxel_vocab.npz")

# Wan VAE (direct-load to avoid flash_attn)
spec = u.spec_from_file_location("wm", WAN_MODULE)
m = u.module_from_spec(spec); spec.loader.exec_module(m)
wan = m.WanVAE(vae_pth=WAN_PTH, device=dev)
print("Wan loaded", flush=True)

vae = VAE3D(num_classes=2021, latent_channels=48, middle_channels=(32, 128, 512))
sd = torch.load(VOX_VAE, map_location="cpu")
sd = sd.get("model", sd.get("state_dict", sd))
vae.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
vae = vae.to(dev).eval()
vv = np.load(VOX_VOCAB)
id2compact = torch.from_numpy(vv["id2compact"]).long().to(dev)

# regression baseline (optional; absent on H100 pod)
base = None
if USE_BASE and os.path.exists(BASE_CKPT):
    base = ShaderUNet().to(dev).eval()
    base.load_state_dict(torch.load(BASE_CKPT, map_location="cpu")["state_dict"])

# flow
fck = torch.load(FLOW_CKPT, map_location="cpu")
cond_ch = fck.get("cond_ch", 64); fbase = fck.get("base", 128); cbase = fck.get("cond_base", 96)
FH = fck.get("rast_h", 72); FW = fck.get("rast_w", 128)
cond_enc = CondEncoder(base=cbase, cond_ch=cond_ch).to(dev).eval()
vnet = FlowVNet(cond_ch=cond_ch, base=fbase).to(dev).eval()
cond_enc.load_state_dict(fck["cond_enc"]); vnet.load_state_dict(fck["vnet"])
print(f"base @ {BASE_RAST_H}x{BASE_RAST_W} ; flow @ {FH}x{FW} step {fck.get('step')} | {N_ODE} ODE steps", flush=True)

def wan_decode(lat):
    with torch.no_grad():
        v = wan.decode(lat.unsqueeze(0).float().to(dev))[0]
    v = ((v.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return v.permute(1, 2, 3, 0).cpu().numpy()   # (9,216,384,3)

def run_base(vox, cam):
    grids = voxel_to_compact_grids(vae, id2compact, vox, dev)
    yaw = cam[:, LAT_TO_SRC, 3] * 180.0; pit = cam[:, LAT_TO_SRC, 4] * 180.0
    dxyz = cam[:, LAT_TO_SRC, 0:3]
    origins = torch.stack([EYE_CENTER + dxyz[..., 0], EYE_CENTER + dxyz[..., 1] + EYE_HEIGHT,
                           EYE_CENTER + dxyz[..., 2]], dim=-1)
    with torch.no_grad():
        id_map, depth, hit, dirs = batched_raycast(grids, origins, yaw, pit, BASE_RAST_H, BASE_RAST_W)
        buf = build_screen_buffer(base, id_map, depth, hit, dirs)
        return base(buf)[0].cpu()

def hf_energy(rgb):
    g = rgb.astype(np.float32).mean(-1)
    return np.abs(np.diff(g, axis=1)).mean() + np.abs(np.diff(g, axis=2)).mean()

ds = VoxelDiTDataset(clip_len=33, require_rgb=True, require_clip=False,
                     require_clip_pf=False, clip_dir=None, clip_pf_dir=None,
                     samples_per_epoch=200, samples_per_episode=1)

all_frames = []; si = 0; tries = 0
hf_gt = hf_base = hf_flow = 0.0
mse_b_tot = mse_f_tot = 0.0
while si < N and tries < 120:
    tries += 1
    s = ds[tries]
    vox = s["voxel_lat"].unsqueeze(0).to(dev)
    cam = s["camera"].unsqueeze(0).to(dev)
    raw_cam = s["raw_camera"].unsqueeze(0).to(dev)
    target = s["rgb_lat"].float()
    gt_rgb = wan_decode(target)
    if BRIGHT and gt_rgb.mean() < 95:
        continue
    pred_f = sample_ode(vnet, cond_enc, vae, id2compact, vox, cam, dev, FH, FW,
                        n_steps=N_ODE, seed=si, raw_cam=raw_cam)[0].cpu()
    mse_f = F.mse_loss(pred_f, target).item(); mse_f_tot += mse_f
    f_rgb = wan_decode(pred_f)
    e_gt, e_f = hf_energy(gt_rgb), hf_energy(f_rgb)
    hf_gt += e_gt; hf_flow += e_f
    panels = [gt_rgb]
    if base is not None:
        pred_b = run_base(vox, cam)
        mse_b = F.mse_loss(pred_b, target).item(); mse_b_tot += mse_b
        b_rgb = wan_decode(pred_b); e_b = hf_energy(b_rgb); hf_base += e_b
        panels.append(b_rgb)
        print(f"s{si}: mse base {mse_b:.4f} flow {mse_f:.4f} | std base {pred_b.std():.3f} "
              f"flow {pred_f.std():.3f} tgt {target.std():.3f} | HF gt {e_gt:.2f} base {e_b:.2f} flow {e_f:.2f}",
              flush=True)
    else:
        print(f"s{si}: mse flow {mse_f:.4f} | std flow {pred_f.std():.3f} tgt {target.std():.3f} "
              f"| HF gt {e_gt:.2f} flow {e_f:.2f}", flush=True)
    panels.append(f_rgb)
    sb = np.concatenate(panels, axis=2)
    all_frames.append(sb)
    Image.fromarray(sb[4]).save(f"{OUT}/{TAG}_sample{si}_midframe.png")
    si += 1

n = max(si, 1)
print(f"\nMEAN over {n}: mse base {mse_b_tot/n:.4f} flow {mse_f_tot/n:.4f}", flush=True)
print(f"MEAN HF energy: GT {hf_gt/n:.2f} | REGRESSION {hf_base/n:.2f} | FLOW {hf_flow/n:.2f}", flush=True)
print(f"(higher HF = sharper; GT is ceiling; FLOW should >> REGRESSION)", flush=True)
big = np.concatenate(all_frames, axis=1)
iio.imwrite(f"{OUT}/{TAG}_compare.mp4", big, fps=3, codec="libx264")
print(f"DONE -> {OUT}/{TAG}_compare.mp4  (layout: GT | REGRESSION | FLOW)", flush=True)
