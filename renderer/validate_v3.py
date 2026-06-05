"""Render v3 (temporal + depth-peel) vs v2, on the SAME held-out clips.
4-up per frame:  GT | v2(flow) | v3-TF(prev=GT) | v3-AR(prev=own pred, autoregressive)
v3-TF = teacher-forced (matches training, directly comparable to v2).
v3-AR = real autoregressive rollout (reveals drift; the honest motion test).
"""
import os, sys
import numpy as np, torch, torch.nn.functional as F
import imageio.v3 as iio
import importlib.util as u
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voxel_dataset import VoxelDiTDataset
from vae_3d import VAE3D
from renderer_flow import CondEncoder, FlowVNet, sample_ode, RGB_C, RGB_T, RGB_H, RGB_W
import train_render_v3 as T3
from train_render_v3 import CondEncoderV3, FlowVNetV3, make_cond_v3

dev = "cuda"
OUT = os.environ.get("OUT", "/home/flukol/val_fix/v3show")
N = int(os.environ.get("N", "6")); N_ODE = int(os.environ.get("N_ODE", "32"))
START_TRY = int(os.environ.get("START_TRY", "40"))
V3_CKPT = os.environ.get("V3_CKPT", "/home/flukol/render_v2_treechop/ckpt_v3/v3_latest.pt")
V2_CKPT = os.environ.get("V2_CKPT", "/home/flukol/render_v2_treechop/ckpt/flow_latest.pt")
WAN_MODULE = os.environ["WAN_MODULE"]; WAN_PTH = os.environ["WAN_PTH"]
VOX_VAE = os.environ["VOX_VAE"]; VOX_VOCAB = os.environ["VOX_VOCAB"]
os.makedirs(OUT, exist_ok=True)

spec = u.spec_from_file_location("wm", WAN_MODULE); m = u.module_from_spec(spec); spec.loader.exec_module(m)
wan = m.WanVAE(vae_pth=WAN_PTH, device=dev); print("Wan loaded", flush=True)

vae = VAE3D(num_classes=2021, latent_channels=48, middle_channels=(32, 128, 512))
sd = torch.load(VOX_VAE, map_location="cpu"); sd = sd.get("model", sd.get("state_dict", sd))
vae.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
vae = vae.to(dev).eval()
id2compact = torch.from_numpy(np.load(VOX_VOCAB)["id2compact"]).long().to(dev)

# v3 model
c3 = torch.load(V3_CKPT, map_location="cpu")
RH, RW = c3["rast_h"], c3["rast_w"]; K = c3["peel_k"]
cond3 = CondEncoderV3(base=c3["cond_base"], cond_ch=c3["cond_ch"], K=K).to(dev).eval()
vnet3 = FlowVNetV3(cond_ch=c3["cond_ch"], base=c3["base"]).to(dev).eval()
cond3.load_state_dict(c3["cond_enc"]); vnet3.load_state_dict(c3["vnet"])
print(f"v3 step {c3.get('step')}  rast {RH}x{RW}  K={K}", flush=True)

# v2 model
c2 = torch.load(V2_CKPT, map_location="cpu")
FH2, FW2 = c2.get("rast_h", 72), c2.get("rast_w", 128)
cond2 = CondEncoder(base=c2.get("cond_base", 96), cond_ch=c2.get("cond_ch", 64)).to(dev).eval()
vnet2 = FlowVNet(cond_ch=c2.get("cond_ch", 64), base=c2.get("base", 128)).to(dev).eval()
cond2.load_state_dict(c2["cond_enc"]); vnet2.load_state_dict(c2["vnet"])
print(f"v2 step {c2.get('step')}  rast {FH2}x{FW2}", flush=True)

def wan_decode(lat):
    with torch.no_grad():
        v = wan.decode(lat.unsqueeze(0).float().to(dev))[0]
    v = ((v.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
    return v.permute(1, 2, 3, 0).cpu().numpy()  # (frames,H,W,3)

@torch.no_grad()
def v3_render(vox, cam, x0_gt, raw_cam, ar, seed=0):
    cond, _, _ = make_cond_v3(cond3, vae, id2compact, vox, cam, dev, RH, RW, K, raw_cam=raw_cam)
    B, _, T, H, W = x0_gt.shape
    cond = cond[:, :, :T]
    taus = torch.linspace(1.0, 0.0, N_ODE + 1, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    if not ar:
        prev = torch.zeros_like(x0_gt); prev[:, :, 1:] = x0_gt[:, :, :-1]
        fm = torch.zeros(B, T, dtype=torch.bool, device=dev); fm[:, 0] = True
        x = torch.randn(B, RGB_C, T, H, W, device=dev, generator=g)
        for i in range(N_ODE):
            tv = torch.full((B,), float(taus[i]), device=dev)
            x = x - (taus[i] - taus[i + 1]) * vnet3(x, tv, cond, prev, fm)
        return x
    # autoregressive: frame-by-frame, prev = own previous clean prediction
    preds = []
    for t in range(T):
        ct = cond[:, :, t:t + 1]
        if t == 0:
            prev = torch.zeros(B, RGB_C, 1, H, W, device=dev)
            fm = torch.ones(B, 1, dtype=torch.bool, device=dev)
        else:
            prev = preds[-1]; fm = torch.zeros(B, 1, dtype=torch.bool, device=dev)
        x = torch.randn(B, RGB_C, 1, H, W, device=dev, generator=g)
        for i in range(N_ODE):
            tv = torch.full((B,), float(taus[i]), device=dev)
            x = x - (taus[i] - taus[i + 1]) * vnet3(x, tv, ct, prev, fm)
        preds.append(x)
    return torch.cat(preds, dim=2)

ds = VoxelDiTDataset(clip_len=33, require_rgb=True, require_clip=False,
                     require_clip_pf=False, clip_dir=None, clip_pf_dir=None,
                     samples_per_epoch=200, samples_per_episode=1)
all_frames = []; si = 0; tries = START_TRY
while si < N and tries < START_TRY + 160:
    tries += 1
    s = ds[tries]
    vox = s["voxel_lat"].unsqueeze(0).to(dev)
    cam = s["camera"].unsqueeze(0).to(dev)
    raw_cam = s["raw_camera"].unsqueeze(0).to(dev)
    x0_gt = s["rgb_lat"].unsqueeze(0).float().to(dev)
    gt_rgb = wan_decode(x0_gt[0])
    if gt_rgb.mean() < 95:
        continue
    v2 = wan_decode(sample_ode(vnet2, cond2, vae, id2compact, vox, cam, dev, FH2, FW2, n_steps=N_ODE, seed=si, raw_cam=raw_cam)[0])
    v3tf = wan_decode(v3_render(vox, cam, x0_gt, raw_cam, ar=False, seed=si)[0])
    v3ar = wan_decode(v3_render(vox, cam, x0_gt, raw_cam, ar=True, seed=si)[0])
    Tn = min(len(gt_rgb), len(v2), len(v3tf), len(v3ar))
    panel = np.concatenate([gt_rgb[:Tn], v2[:Tn], v3tf[:Tn], v3ar[:Tn]], axis=2)  # hstack
    all_frames.append(panel)
    mid = Tn // 2
    iio.imwrite(f"{OUT}/v3show_sample{si}_midframe.png", panel[mid])
    print(f"sample {si} (try {tries}) done  frames={Tn}", flush=True)
    si += 1

stacked = np.concatenate(all_frames, axis=1) if all_frames else None  # vstack samples
if stacked is not None:
    iio.imwrite(f"{OUT}/v3show_compare.mp4", stacked, fps=8, codec="libx264")
    print(f"DONE -> {OUT}/v3show_compare.mp4  (layout per row: GT | v2 | v3-TF | v3-AR)", flush=True)
