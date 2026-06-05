"""Build the CS153 "persistence demo" stitched video.

Concept: ONE FIXED 3D voxel world. A camera moves out-and-back along each
cardinal direction. Because the world is a fixed 3D object, when the camera
returns to center it sees the EXACT same thing -> persistence proof.

Panels (left -> right):
  [LEFT]   WASD overlay (keys lit by current motion direction)
  [MIDDLE] isometric voxel world (FIXED) + semi-transparent gray camera marker
  [RIGHT]  first-person RGB (added later by the cluster job; optional placeholder)

LEFT + MIDDLE are pure-CPU/PIL and deterministic. Run:
  python build_persist_demo.py --out figs/persist_demo/attemptN [--rgb DIR]
"""
import os, sys, math, argparse, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import panel_a as PA

NAVY = (14, 18, 48)            # #0E1230
WORLD_NPZ = os.path.join(HERE, "npz/f0.npz")   # FIXED world: trees+lake+terrain

# ----- camera path -----
# World center in grid-local coords. Renderer eye is grid (X, Y, Z), Y up.
CX, CZ = 24.0, 24.0
EYE_Y = 28.0                    # ~3 above the center surface (~Y25)
PITCH = 12.0                    # slightly down
DIST = 15.0                    # blocks out per leg
N_LEG = 15                     # interpolated frames per (out / back) half-leg
HOLD = 3                       # frames to hold at center on each return

# yaw convention (camera_basis_np): fwd = [-sin(yaw)*cp, -sin(pit), cos(yaw)*cp]
#   yaw=0   -> +Z      yaw=180 -> -Z
#   yaw=90  -> -X      yaw=270 -> +X
# Axis->key mapping requested: +X=W, -X=S, +Z=D, -Z=A
YAW = {"+Z": 0.0, "-Z": 180.0, "-X": 90.0, "+X": 270.0}
KEY = {"+X": "W", "-X": "S", "+Z": "D", "-Z": "A"}


def ease(t):
    """smoothstep ease in/out on [0,1]."""
    return t * t * (3 - 2 * t)


def build_path():
    """Return list of frames: dict(eye, yaw, dir, key, leg_label, at_center)."""
    legs = ["+X", "-X", "+Z", "-Z"]   # East, West, North, South (out & back each)
    frames = []

    def add(eye, yaw, d, key, label, at_center):
        frames.append(dict(eye=np.array(eye, float), yaw=yaw, dir=d,
                           key=key, label=label, at_center=at_center))

    # initial hold at center, facing +Z (no key lit)
    for _ in range(HOLD):
        add([CX, EYE_Y, CZ], YAW["+Z"], None, None, "start", True)

    for d in legs:
        yaw = YAW[d]
        key = KEY[d]
        # unit step vector in world XZ for this direction
        dx = (1.0 if d == "+X" else -1.0 if d == "-X" else 0.0)
        dz = (1.0 if d == "+Z" else -1.0 if d == "-Z" else 0.0)
        # OUT
        for i in range(1, N_LEG + 1):
            s = ease(i / N_LEG) * DIST
            add([CX + dx * s, EYE_Y, CZ + dz * s], yaw, d, key,
                f"{key} +{int(round(s))}", False)
        # BACK (still facing direction of motion -> motion is now opposite dir,
        # but we keep yaw facing the way we travel: traveling back = opposite key)
        back_d = {"+X": "-X", "-X": "+X", "+Z": "-Z", "-Z": "+Z"}[d]
        back_yaw = YAW[back_d]
        back_key = KEY[back_d]
        for i in range(1, N_LEG + 1):
            s = (1.0 - ease(i / N_LEG)) * DIST
            add([CX + dx * s, EYE_Y, CZ + dz * s], back_yaw, back_d, back_key,
                f"{back_key} returning", False)
        # HOLD at center (persistence moment)
        for _ in range(HOLD):
            add([CX, EYE_Y, CZ], back_yaw, None, None, "BACK AT CENTER", True)
    return frames


# ----- WASD panel -----
def render_wasd(active_key, label, size=(230, 300), supersample=2):
    """COMPACT WASD cluster — much smaller keys + tight spacing than v1.
    The canvas is sized snugly around the key cluster so it stays legible after the
    stitcher scales it into its (narrow) cell."""
    W, H = size[0] * supersample, size[1] * supersample
    img = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(img, "RGBA")

    def font(sz):
        for p in ["/System/Library/Fonts/SFNSRounded.ttf",
                  "/System/Library/Fonts/Helvetica.ttc",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf"]:
            try:
                return ImageFont.truetype(p, sz)
            except Exception:
                continue
        return ImageFont.load_default()

    # ---- smaller keys + tighter gap (was 120/18) ----
    key_sz = int(58 * supersample)
    gap = int(8 * supersample)
    cluster_h = 2 * key_sz + gap
    cx = W // 2
    top_y = (H - cluster_h) // 2 - int(20 * supersample)   # vertically centred-ish
    # positions: W on top centred; A S D in a row below
    pos = {
        "W": (cx - key_sz // 2, top_y),
        "A": (cx - key_sz - gap - key_sz // 2, top_y + key_sz + gap),
        "S": (cx - key_sz // 2, top_y + key_sz + gap),
        "D": (cx + gap + key_sz // 2, top_y + key_sz + gap),
    }
    kf = font(int(28 * supersample))

    for k, (x, y) in pos.items():
        on = (k == active_key)
        rad = int(10 * supersample)
        box = [x, y, x + key_sz, y + key_sz]
        if on:
            d.rounded_rectangle(box, radius=rad, fill=(90, 200, 255, 255),
                                outline=(180, 235, 255, 255),
                                width=max(2, 2 * supersample))
            tcol = (8, 14, 36, 255)
        else:
            d.rounded_rectangle(box, radius=rad, fill=(30, 38, 78, 180),
                                outline=(90, 100, 150, 220),
                                width=max(1, supersample))
            tcol = (120, 132, 175, 255)
        tb = d.textbbox((0, 0), k, font=kf)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.text((x + key_sz // 2 - tw // 2 - tb[0],
                y + key_sz // 2 - th // 2 - tb[1]), k, font=kf, fill=tcol)

    # readout label (compact)
    lf = font(int(17 * supersample))
    lab = label if label else "idle"
    d.text((cx, top_y + 2 * key_sz + gap + int(28 * supersample)), lab,
           font=lf, fill=(140, 220, 255, 255) if active_key else (110, 122, 165, 255),
           anchor="mm")
    # axis hint
    hf = font(int(13 * supersample))
    d.text((cx, H - int(34 * supersample)),
           "+X=W  -X=S  +Z=D  -Z=A", font=hf,
           fill=(95, 108, 150, 255), anchor="mm")

    if supersample > 1:
        img = img.resize(size, Image.LANCZOS)
    return img


def pad_to_height(img, target_h, bg=NAVY):
    if img.height == target_h:
        return img
    scale = target_h / img.height
    return img.resize((max(1, int(img.width * scale)), target_h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rgb", default=None, help="dir with rgb_XXXX.png frames")
    # GENERATIVE rollout inputs (from gen_persist_rollout.py, pulled off the pod):
    ap.add_argument("--vox_npz", required=True,
                    help="voxels_path.npz: voxels (F,48,48,48) per-frame generated world")
    ap.add_argument("--cam_json", required=True,
                    help="camera_path.json from the pod (has turns + yaw per frame)")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    frames_dir = os.path.join(args.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # per-frame GENERATED voxel windows + matching camera path (with smooth turns)
    vox_seq = np.load(args.vox_npz)["voxels"].astype(np.int64)   # (F,48,48,48)
    cam = json.load(open(args.cam_json))
    F = min(len(vox_seq), len(cam))
    print(f"generative rollout: {F} frames, vox grid {vox_seq.shape[1:]}")

    # FIXED canvas bounds: union over a few frames so the iso panel never resizes
    # even as the world content shifts under the (centred) camera.
    fb = PA.compute_voxel_bounds(vox_seq[0], supersample=2)
    for idx in [F // 4, F // 2, 3 * F // 4, F - 1]:
        b = PA.compute_voxel_bounds(vox_seq[idx], supersample=2)
        fb = (min(fb[0], b[0]), max(fb[1], b[1]), min(fb[2], b[2]), max(fb[3], b[3]))

    PANEL_H = 700        # final stitched height
    # fixed cell widths so proportions are stable & balanced (panels centered).
    # WASD cell is narrower now that the key cluster is compact.
    CELL_W = {"wasd": 300, "iso": 680, "rgb": 560}
    TITLE_H = 40

    def cell(img, w, title):
        """Fit img into a (w x PANEL_H) navy cell, centered, with a small title."""
        avail_h = PANEL_H - TITLE_H
        scale = min(w / img.width, avail_h / img.height)
        nw, nh = max(1, int(img.width * scale)), max(1, int(img.height * scale))
        im = img.resize((nw, nh), Image.LANCZOS)
        c = Image.new("RGB", (w, PANEL_H), NAVY)
        c.paste(im, ((w - nw) // 2, TITLE_H + (avail_h - nh) // 2))
        d = ImageDraw.Draw(c)
        try:
            tf = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        except Exception:
            tf = ImageFont.load_default()
        d.text((w // 2, TITLE_H // 2 + 4), title, font=tf,
               fill=(150, 162, 205), anchor="mm")
        return c

    # camera marker is always centred in the 48^3 window (the WORLD shifts, not it)
    EYE = np.array([24.0, EYE_Y, 24.0])

    stitched = []
    center_returns = []
    for i in range(F):
        fr = cam[i]
        block_ids = vox_seq[i]
        wasd = render_wasd(fr["key"], fr["label"])
        iso = PA.render_panel_a(block_ids, np.zeros((0, 3), int),
                                eye=EYE, yaw=fr["yaw"], pitch=PITCH,
                                cam_mode="gray", fixed_bounds=fb,
                                world_dim_factor=0.62, supersample=2)
        cells = [cell(wasd, CELL_W["wasd"], "controls"),
                 cell(iso, CELL_W["iso"], "generated 3D world + camera")]
        if args.rgb:
            rp = os.path.join(args.rgb, f"rgb_{i:04d}.png")
            if os.path.exists(rp):
                cells.append(cell(Image.open(rp).convert("RGB"),
                                  CELL_W["rgb"], "first-person render"))
        gap = 14
        total_w = sum(c.width for c in cells) + gap * (len(cells) - 1)
        canvas = Image.new("RGB", (total_w, PANEL_H), NAVY)
        x = 0
        for c in cells:
            canvas.paste(c, (x, 0))
            x += c.width + gap
        canvas.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
        stitched.append(canvas)
        if fr.get("label") == "BACK AT CENTER":
            center_returns.append(i)
        if i % 10 == 0:
            print(f"frame {i}/{F} key={fr['key']} yaw={fr['yaw']} label={fr['label']}")

    # write mp4 with ffmpeg
    mp4 = os.path.join(args.out, "rollout_persistence.mp4")
    os.system(
        f"ffmpeg -y -framerate {args.fps} -i '{frames_dir}/frame_%04d.png' "
        f"-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -pix_fmt yuv420p "
        f"-crf 18 '{mp4}' >/dev/null 2>&1")
    print("WROTE", mp4, "frames:", F)
    # representative still: a BACK-AT-CENTER moment (or last frame)
    cidx = center_returns[0] if center_returns else F - 1
    stitched[cidx].save(os.path.join(args.out, "still.png"))
    print("STILL", os.path.join(args.out, "still.png"), "frame", cidx)


if __name__ == "__main__":
    main()
