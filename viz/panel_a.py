"""Panel A: isometric voxel world + camera marker + frustum + GOLD visible voxels.

Reuses render_isometric3's projection math EXACTLY so the camera marker and the
highlighted voxels land in the right place. Renders on a dark navy bg to match
the slide deck.

World/grid convention (from the pod extractor):
  block_ids: (48,48,48) indexed (X, Y, Z), Y = Minecraft UP, air = 0.
  visible:   (M,3) grid coords (X, Y, Z).
  eye:       (3,) grid-local (X, Y, Z).
  yaw,pitch: Minecraft degrees.

render_isometric3 applies np.swapaxes(grid,1,2) so its internal render grid is
indexed (X, Z, Y). Its projection for render-index (rx,ry,rz):
    sx = (rx - ry)*TH + off_x
    sy = (rx + ry)*H2 - rz*TD + off_y
We map a continuous WORLD point (wx,wy,wz) -> render coords (wx, wz, wy) and use
the same formula, so points and voxels share one projection.
"""
import sys, math
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, "/Users/frozone/Documents/IWorld/testing/persist_small_v5")
import render_isometric3 as R3

NAVY = (14, 18, 48)        # #0E1230
GOLD = (255, 210, 60)
GOLD_TOP = (255, 224, 110)
GOLD_LEFT = (236, 184, 40)
GOLD_RIGHT = (208, 156, 24)
CAM_COLOR = (255, 70, 220)  # magenta
FRUST_COLOR = (255, 120, 230)


def camera_basis_np(yaw_deg, pitch_deg):
    yaw = math.radians(yaw_deg); pit = math.radians(pitch_deg)
    cp = math.cos(pit)
    fwd = np.array([-math.sin(yaw) * cp, -math.sin(pit), math.cos(yaw) * cp])
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, world_up); right /= (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, fwd); up /= (np.linalg.norm(up) + 1e-8)
    return fwd, right, up


def render_panel_a(block_ids, visible, eye, yaw, pitch,
                   tile_h=12, tile_depth=11, supersample=2,
                   dim_factor=0.55, fov_deg=70.0,
                   cam_mode="magenta", fixed_bounds=None,
                   world_dim_factor=None):
    """Render the isometric world with gold visible voxels + camera marker.

    cam_mode: "magenta" (original) or "gray" (semi-transparent gray camera
              object/cone for the persistence demo).
    fixed_bounds: optional (min_sx, max_sx, min_sy, max_sy) in supersampled
              screen units (computed from voxels only, i.e. WITHOUT the camera)
              so the canvas is identical every frame and the world never shifts.
              When given, no autocrop is applied.
    world_dim_factor: if set, overrides dim_factor for the world voxels (lets the
              persistence demo show the full world a touch brighter).

    Returns a PIL.Image (RGB) on navy background.
    """
    pal = R3.build_palette()
    if world_dim_factor is not None:
        dim_factor = world_dim_factor
    # swap to render grid (X, Z, Y) — Y up
    vox = np.swapaxes(block_ids.astype(np.int64), 1, 2)  # (X, Zr, Yr)
    Sx, Sy, Sz = vox.shape

    # build a visible mask in RENDER index space (swap Y<->Z of the coords)
    vis_set = set()
    if visible is not None and len(visible):
        for (wx, wy, wz) in visible:
            vis_set.add((int(wx), int(wz), int(wy)))   # (rx, ry, rz)

    kind = pal["kind"]; id_top = pal["top"]; id_side = pal["side"]; id_glow = pal["glow"]
    leaf_ids = pal["leaf_ids"]
    air_ids = {cid for cid, k in kind.items() if k == "air"}

    is_cube = np.zeros(vox.max() + 2, dtype=bool)
    is_air = np.zeros(vox.max() + 2, dtype=bool)
    for cid in range(vox.max() + 1):
        k = kind.get(cid, "cube" if cid not in air_ids else "air")
        if k == "air":
            is_air[cid] = True
        elif k == "cube":
            is_cube[cid] = True
    solid = is_cube[vox]
    occupied = ~is_air[vox]

    top_exp = np.zeros_like(solid)
    top_exp[:, :, :-1] = solid[:, :, :-1] & ~solid[:, :, 1:]; top_exp[:, :, -1] = solid[:, :, -1]
    right_exp = np.zeros_like(solid)
    right_exp[:-1] = solid[:-1] & ~solid[1:]; right_exp[-1] = solid[-1]
    left_exp = np.zeros_like(solid)
    left_exp[:, :-1] = solid[:, :-1] & ~solid[:, 1:]; left_exp[:, -1] = solid[:, -1]
    cube_vis = (top_exp | right_exp | left_exp) & solid
    plant_torch = occupied & ~solid
    draw_mask = cube_vis | plant_torch
    xs, ys, zs = np.where(draw_mask)

    th = tile_h * supersample; td = tile_depth * supersample; h2 = th // 2

    def proj(rx, ry, rz):
        sx = (rx - ry) * th
        sy = (rx + ry) * h2 - rz * td
        return sx, sy

    # camera point in render space (world (ex,ey,ez) -> render (ex, ez, ey))
    ex, ey, ez = float(eye[0]), float(eye[1]), float(eye[2])
    cam_r = (ex, ez, ey)
    cam_sx, cam_sy = proj(*cam_r)

    # bounds over voxels (+ camera only in the legacy/autocrop path)
    sxs = (xs - ys) * th
    sys_ = (xs + ys) * h2 - zs * td
    margin = th * 3
    if fixed_bounds is not None:
        # FIXED canvas computed from voxels alone -> world pixels never move.
        min_sx, max_sx, min_sy, max_sy = fixed_bounds
    else:
        allx = np.append(sxs, cam_sx); ally = np.append(sys_, cam_sy)
        min_sx = int(allx.min()) - th - margin; max_sx = int(allx.max()) + th + margin
        min_sy = int(ally.min()) - h2 - margin; max_sy = int(ally.max()) + td + h2 + margin
    img_w = max_sx - min_sx; img_h = max_sy - min_sy
    off_x, off_y = -min_sx, -min_sy

    img = Image.new("RGB", (img_w, img_h), NAVY)
    draw = ImageDraw.Draw(img, "RGBA")

    # AO
    sp = np.pad(solid.astype(np.float32), 1)
    nbr = np.zeros_like(solid, dtype=np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                nbr += sp[1+dx:1+dx+Sx, 1+dy:1+dy+Sy, 1+dz:1+dz+Sz]
    ao = 1.0 - 0.16 * (nbr / 26.0)

    order = np.lexsort((-ys, xs, xs + ys, zs))
    xs, ys, zs = xs[order], ys[order], zs[order]

    def shade(rgb, f):
        return tuple(max(0, min(255, int(c * f))) for c in rgb)

    for i in range(len(xs)):
        x, y, z = int(xs[i]), int(ys[i]), int(zs[i])
        bid = int(vox[x, y, z]); k = kind.get(bid, "cube")
        cx = (x - y) * th + off_x; cy = (x + y) * h2 - z * td + off_y
        is_vis = (x, y, z) in vis_set

        if k == "plant":
            # render plants as a small subtle tuft, not the figure-shaped billboard
            col = id_top.get(bid, (90, 160, 60))
            if is_vis:
                col = GOLD
                a_tuft = 235
            else:
                col = tuple(int(c * dim_factor + NAVY[j] * (1 - dim_factor)) for j, c in enumerate(col))
                a_tuft = 150
            base_y = cy + th // 2
            r = max(1, int(th * 0.28))
            draw.ellipse([cx - r, base_y - 2 * r, cx + r, base_y], fill=col + (a_tuft,))
            continue
        if k == "torch":
            R3._draw_torch(draw, cx, cy, th, td, id_glow.get(bid, (255, 214, 120)))
            continue

        a = float(ao[x, y, z])
        if is_vis:
            top_rgb, side_rgb = GOLD_TOP, GOLD
            sh_top, sh_left, sh_right = 1.0, 0.92, 0.82  # keep gold bright
            alpha = 255
        else:
            top_rgb = id_top.get(bid, (150, 138, 120))
            side_rgb = id_side.get(bid, top_rgb)
            # dim non-visible toward navy
            top_rgb = tuple(int(c * dim_factor + NAVY[j] * (1 - dim_factor)) for j, c in enumerate(top_rgb))
            side_rgb = tuple(int(c * dim_factor + NAVY[j] * (1 - dim_factor)) for j, c in enumerate(side_rgb))
            sh_top, sh_left, sh_right = R3.SHADE_TOP, R3.SHADE_LEFT, R3.SHADE_RIGHT
            alpha = int(255 * 0.9) if bid in leaf_ids else 255

        if left_exp[x, y, z]:
            fc = shade(side_rgb, sh_left * a) + (alpha,)
            draw.polygon([(cx - th, cy), (cx, cy + h2), (cx, cy + h2 + td), (cx - th, cy + td)], fill=fc)
        if right_exp[x, y, z]:
            fc = shade(side_rgb, sh_right * a) + (alpha,)
            draw.polygon([(cx + th, cy), (cx, cy + h2), (cx, cy + h2 + td), (cx + th, cy + td)], fill=fc)
        if top_exp[x, y, z]:
            fc = shade(top_rgb, sh_top * a) + (alpha,)
            draw.polygon([(cx, cy - h2), (cx + th, cy), (cx, cy + h2), (cx - th, cy)], fill=fc)
        # gold rim glow on top edge for pop
        if is_vis and top_exp[x, y, z]:
            draw.line([(cx - th, cy), (cx, cy - h2), (cx + th, cy)], fill=(255, 240, 170, 180), width=max(1, supersample))

    # ---- camera marker + frustum ----
    fwd, right, up = camera_basis_np(yaw, pitch)
    tan_v = math.tan(math.radians(fov_deg) / 2.0)
    tan_h = tan_v * (256 / 144)
    L = 11.0  # frustum line length (world units)
    # gray (persistence-demo) marker: SMALL + SLIM camera.
    #   overall length/height ~0.5x of the magenta marker, width ~0.25x.
    GRAY_L = L * 0.5            # half the length
    GRAY_W_SCALE = 0.25        # quarter the cone half-width (slim, not a fat cone)

    def world_to_screen(p):
        rx, ry, rz = p[0], p[2], p[1]  # world (x,y,z) -> render (x, z, y)
        sx = (rx - ry) * th + off_x
        sy = (rx + ry) * h2 - rz * td + off_y
        return (sx, sy)

    eye_w = np.array([ex, ey, ez])
    cam_pt = world_to_screen(eye_w)
    # 4 frustum corners (use the HORIZONTAL fov edges; flatten vertical a touch for clarity)
    corners = []
    for sgnx in (-1, 1):
        for sgny in (-1, 1):
            d = fwd + sgnx * tan_h * right + sgny * (tan_v * 0.7) * up
            d = d / (np.linalg.norm(d) + 1e-8)
            corners.append(eye_w + d * L)
    center_far = eye_w + fwd * L

    if cam_mode == "gray":
        # Semi-transparent GRAY camera object: SMALL + SLIM. A short, narrow view
        # cone pointing along the motion/yaw direction + a tiny camera body.
        GRAY = (200, 205, 215)
        GRAY_DK = (150, 156, 168)
        a_cone = 105
        a_body = 165
        # recompute SLIM, SHORT cone corners (own scale, not the magenta L/fov).
        gl, gr = [], []
        for sgny in (-1, 1):
            d = fwd + GRAY_W_SCALE * tan_h * right + sgny * (tan_v * 0.6) * up
            d = d / (np.linalg.norm(d) + 1e-8)
            gr.append(eye_w + d * GRAY_L)
        for sgny in (-1, 1):
            d = fwd - GRAY_W_SCALE * tan_h * right + sgny * (tan_v * 0.6) * up
            d = d / (np.linalg.norm(d) + 1e-8)
            gl.append(eye_w + d * GRAY_L)
        tl = world_to_screen(gl[1]); bl = world_to_screen(gl[0])
        tr = world_to_screen(gr[1]); br = world_to_screen(gr[0])
        fl = ((tl[0] + bl[0]) / 2, (tl[1] + bl[1]) / 2)
        fr = ((tr[0] + br[0]) / 2, (tr[1] + br[1]) / 2)
        gcf = world_to_screen(eye_w + fwd * GRAY_L)
        # slim translucent gray cone
        draw.polygon([cam_pt, fl, fr], fill=GRAY + (a_cone,))
        draw.line([cam_pt, fl], fill=GRAY + (175,), width=max(1, supersample))
        draw.line([cam_pt, fr], fill=GRAY + (175,), width=max(1, supersample))
        draw.line([fl, fr], fill=GRAY + (140,), width=max(1, supersample))
        # short forward direction arrow (gray-white)
        draw.line([cam_pt, gcf], fill=(235, 238, 245, 205), width=max(1, supersample))
        # tiny camera body diamond at the eye (half the old size)
        r = max(3, int(2.3 * supersample))
        draw.polygon([(cam_pt[0], cam_pt[1]-r), (cam_pt[0]+r, cam_pt[1]),
                      (cam_pt[0], cam_pt[1]+r), (cam_pt[0]-r, cam_pt[1])],
                     fill=GRAY + (a_body,), outline=GRAY_DK + (230,))
        # little lens dot toward forward
        lp = world_to_screen(eye_w + fwd * 1.3)
        lr = max(1, int(1.0 * supersample))
        draw.ellipse([lp[0]-lr, lp[1]-lr, lp[0]+lr, lp[1]+lr], fill=(245, 248, 255, 215))
    else:
        cs = [world_to_screen(corners[i]) for i in (0, 1, 3, 2)]  # ordered ring
        # translucent filled view cone (eye -> two horizontal far edges)
        top_l = world_to_screen(corners[1]); top_r = world_to_screen(corners[3])
        bot_l = world_to_screen(corners[0]); bot_r = world_to_screen(corners[2])
        fl = ((top_l[0] + bot_l[0]) / 2, (top_l[1] + bot_l[1]) / 2)
        fr = ((top_r[0] + bot_r[0]) / 2, (top_r[1] + bot_r[1]) / 2)
        draw.polygon([cam_pt, fl, fr], fill=(255, 120, 230, 55))
        draw.line([cam_pt, fl], fill=FRUST_COLOR + (230,), width=max(2, supersample))
        draw.line([cam_pt, fr], fill=FRUST_COLOR + (230,), width=max(2, supersample))
        draw.line([fl, fr], fill=FRUST_COLOR + (170,), width=max(1, supersample))
        # forward arrow (eye -> center_far) bright white
        cf = world_to_screen(center_far)
        draw.line([cam_pt, cf], fill=(255, 255, 255, 235), width=max(2, supersample))
        # camera body sphere
        r = max(6, int(4.5 * supersample))
        draw.ellipse([cam_pt[0]-r-3, cam_pt[1]-r-3, cam_pt[0]+r+3, cam_pt[1]+r+3], fill=(255, 255, 255, 80))
        draw.ellipse([cam_pt[0]-r, cam_pt[1]-r, cam_pt[0]+r, cam_pt[1]+r], fill=CAM_COLOR + (255,))
        draw.ellipse([cam_pt[0]-r//2, cam_pt[1]-r//2, cam_pt[0]+r//2, cam_pt[1]+r//2], fill=(255, 255, 255, 255))

    # crop to content but keep navy bg (skip when a fixed canvas is requested)
    if fixed_bounds is None:
        img = _autocrop_navy(img, NAVY, margin=10 * supersample)
    if supersample > 1:
        img = img.resize((max(1, img.width // supersample), max(1, img.height // supersample)), Image.LANCZOS)
    return img


def compute_voxel_bounds(block_ids, tile_h=12, tile_depth=11, supersample=2,
                         margin_tiles=4.0):
    """Screen-space bounds (min_sx,max_sx,min_sy,max_sy) over the WORLD voxels
    only (no camera), in supersampled units — so a fixed canvas can be reused
    every frame and the world never shifts. Mirrors render_panel_a's projection.
    """
    pal = R3.build_palette()
    vox = np.swapaxes(block_ids.astype(np.int64), 1, 2)
    kind = pal["kind"]
    air_ids = {cid for cid, k in kind.items() if k == "air"}
    is_air = np.zeros(vox.max() + 2, dtype=bool)
    for cid in range(vox.max() + 1):
        if kind.get(cid, "cube") == "air" or cid in air_ids:
            is_air[cid] = True
    occ = ~is_air[vox]
    xs, ys, zs = np.where(occ)
    th = tile_h * supersample; td = tile_depth * supersample; h2 = th // 2
    sxs = (xs - ys) * th
    sys_ = (xs + ys) * h2 - zs * td
    m = int(th * margin_tiles)
    min_sx = int(sxs.min()) - th - m; max_sx = int(sxs.max()) + th + m
    min_sy = int(sys_.min()) - h2 - m; max_sy = int(sys_.max()) + td + h2 + m
    return (min_sx, max_sx, min_sy, max_sy)


def _autocrop_navy(img, bg, margin=16):
    arr = np.array(img)
    mask = np.any(np.abs(arr.astype(int)[:, :, :3] - np.array(bg)) > 6, axis=2)
    rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
    if not rows.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]; cmin, cmax = np.where(cols)[0][[0, -1]]
    rmin = max(0, rmin - margin); rmax = min(arr.shape[0], rmax + margin + 1)
    cmin = max(0, cmin - margin); cmax = min(arr.shape[1], cmax + margin + 1)
    return img.crop((cmin, rmin, cmax, rmax))
