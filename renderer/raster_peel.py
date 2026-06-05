"""Depth-peeling + surface-normal extension of raster_torch.rasterize_batch (render_v3).

The base render_v2 rasterizer (raster_torch.rasterize_batch) keeps only the FIRST
(nearest) voxel hit per ray.  That throws away parallax / see-through-foliage cues
and gives the shader no surface-orientation signal, which caps how sharp the
conditioning can be (the v2 vloss plateaued ~0.246 -> conditioning-bound).

This module adds, without touching the original file:

  rasterize_peel(grids, origins, yaw, pitch, H, W, K=2)
      -> id_map  (B,T,K,H,W)   long  compact-ids of the first K hits (near->far)
         depth   (B,T,K,H,W)   float (MAX_DEPTH where miss)
         hit     (B,T,K,H,W)   bool
         normal  (B,T,K,H,W,3) float  per-hit surface normal (grid axes), 0 where miss
         dirs    (B,T,H,W,3)   float  per-pixel ray dir (shared across layers)

  K=1 reproduces the single-hit behaviour of raster_torch.rasterize_batch (the
  layer-0 id/depth/hit match within painter-order tolerance).

Normals are estimated from the per-layer depth buffer via screen-space gradients
of the 3D hit position (cross product of d(pos)/dx and d(pos)/dy), which is robust
and cheap and needs no neighbour voxel lookups.

Reuses camera_basis + the exact coordinate/eye convention from raster_torch.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F

from voxel_raycaster import camera_basis
from raster_torch import FOV_DEG, MAX_DEPTH, RAD_MAX


def _normals_from_depth(pos, hit):
    """pos (BT,H,W,3) world/grid hit positions ; hit (BT,H,W) bool.
    Returns normal (BT,H,W,3) unit vectors (0 where miss)."""
    # central-ish differences via shifted slices (replicate pad)
    BT, H, W, _ = pos.shape
    p = pos.permute(0, 3, 1, 2)                                  # (BT,3,H,W)
    p = F.pad(p, (1, 1, 1, 1), mode="replicate")
    dpdx = (p[:, :, 1:-1, 2:] - p[:, :, 1:-1, :-2])             # (BT,3,H,W)
    dpdy = (p[:, :, 2:, 1:-1] - p[:, :, :-2, 1:-1])
    n = torch.cross(dpdx, dpdy, dim=1)                          # (BT,3,H,W)
    n = n.permute(0, 2, 3, 1)                                   # (BT,H,W,3)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
    n = torch.where(hit.unsqueeze(-1), n, torch.zeros_like(n))
    return n


@torch.no_grad()
def rasterize_peel(grids, origins, yaw_deg, pitch_deg, H, W, K=2, air_id=0,
                   fov_deg=FOV_DEG, rad_max=RAD_MAX):
    """Depth-peeled rasterizer. See module docstring for shapes."""
    dev = grids.device
    B, T, X, Y, Z = grids.shape
    BT = B * T
    g = grids.reshape(BT, X, Y, Z)
    org = origins.reshape(BT, 3).float()
    yaw = yaw_deg.reshape(BT).float()
    pit = pitch_deg.reshape(BT).float()
    fwd, right, up = camera_basis(yaw, pit)

    tanv = math.tan(math.radians(fov_deg) / 2.0)
    tanh = tanv * (W / H)
    focal = (H / 2.0) / tanv

    ys = torch.linspace(1 - 1.0 / H, -1 + 1.0 / H, H, device=dev)
    xs = torch.linspace(-1 + 1.0 / W, 1 - 1.0 / W, W, device=dev)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    dirs = (fwd.view(BT, 1, 1, 3)
            + (gx.view(1, H, W, 1) * tanh) * right.view(BT, 1, 1, 3)
            + (gy.view(1, H, W, 1) * tanv) * up.view(BT, 1, 1, 3))
    dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-8)       # (BT,H,W,3)

    NPIX = H * W
    # K depth/id layers (near->far)
    depth_layers = torch.full((K, BT, NPIX), float(MAX_DEPTH), device=dev)
    id_layers = torch.zeros((K, BT, NPIX), dtype=torch.long, device=dev)

    xx = torch.arange(X, device=dev); yy = torch.arange(Y, device=dev); zz = torch.arange(Z, device=dev)
    gx_i, gy_i, gz_i = torch.meshgrid(xx, yy, zz, indexing="ij")
    centers = torch.stack([gx_i, gy_i, gz_i], dim=-1).reshape(-1, 3).float() + 0.5

    for bt in range(BT):
        grid = g[bt]
        solid = grid.reshape(-1) != air_id
        if not solid.any():
            continue
        ids = grid.reshape(-1)[solid]
        c = centers[solid]
        rel = c - org[bt].view(1, 3)
        depth = rel @ fwd[bt]
        front = depth > 0.1
        if not front.any():
            continue
        rel = rel[front]; depth = depth[front]; ids = ids[front]
        ndx = (rel @ right[bt]) / (depth * tanh)
        ndy = (rel @ up[bt]) / (depth * tanv)
        inf = (ndx.abs() < 1.15) & (ndy.abs() < 1.15)
        if not inf.any():
            continue
        ndx = ndx[inf]; ndy = ndy[inf]; depth = depth[inf]; ids = ids[inf]
        px = (((ndx + 1) / 2) * W).long()
        py = (((1 - ndy) / 2) * H).long()
        rad = torch.clamp(torch.round(0.6 * focal / depth).long(), 0, rad_max)

        flat_pix = []; flat_dep = []; flat_id = []
        rmax = int(rad.max().item())
        for r in range(0, rmax + 1):
            m = rad == r
            if not m.any():
                continue
            pxr = px[m]; pyr = py[m]; depr = depth[m]; idr = ids[m]
            if r == 0:
                ok = (pxr >= 0) & (pxr < W) & (pyr >= 0) & (pyr < H)
                if ok.any():
                    flat_pix.append(pyr[ok] * W + pxr[ok]); flat_dep.append(depr[ok]); flat_id.append(idr[ok])
                continue
            offs = torch.arange(-r, r + 1, device=dev)
            oy, ox = torch.meshgrid(offs, offs, indexing="ij")
            oy = oy.reshape(-1); ox = ox.reshape(-1); S = oy.shape[0]
            yy2 = pyr.view(-1, 1) + oy.view(1, -1)
            xx2 = pxr.view(-1, 1) + ox.view(1, -1)
            dd2 = depr.view(-1, 1).expand(-1, S)
            ii2 = idr.view(-1, 1).expand(-1, S)
            ok = (xx2 >= 0) & (xx2 < W) & (yy2 >= 0) & (yy2 < H)
            if ok.any():
                flat_pix.append((yy2 * W + xx2)[ok]); flat_dep.append(dd2[ok]); flat_id.append(ii2[ok])
        if not flat_pix:
            continue
        P = torch.cat(flat_pix); D = torch.cat(flat_dep); I = torch.cat(flat_id)

        # --- depth-peel: resolve K nearest distinct-depth samples per pixel ---
        # sort all (pixel, depth) samples by (pixel, depth ascending)
        order = torch.argsort(D)            # depth ascending
        Ps = P[order]; Ds = D[order]; Is = I[order]
        order2 = torch.argsort(Ps, stable=True)
        Ps = Ps[order2]; Ds = Ds[order2]; Is = Is[order2]
        # rank within each pixel group: first occurrence -> 0, etc.
        # boundaries where pixel id changes
        new_grp = torch.ones_like(Ps, dtype=torch.bool)
        new_grp[1:] = Ps[1:] != Ps[:-1]
        grp_id = torch.cumsum(new_grp.long(), 0) - 1
        grp_start = torch.zeros(int(grp_id[-1].item()) + 1, dtype=torch.long, device=dev)
        grp_start.scatter_(0, grp_id, torch.arange(Ps.shape[0], device=dev))
        # rank = position - start_of_group
        rank = torch.arange(Ps.shape[0], device=dev) - grp_start[grp_id]
        for k in range(K):
            sel = rank == k
            if sel.any():
                pk = Ps[sel]; dk = Ds[sel]; ik = Is[sel]
                depth_layers[k, bt].scatter_(0, pk, dk)
                id_layers[k, bt].scatter_(0, pk, ik)

    id_map = id_layers.view(K, B, T, H, W).permute(1, 2, 0, 3, 4).contiguous()
    depth = depth_layers.view(K, B, T, H, W).permute(1, 2, 0, 3, 4).contiguous()
    hit = depth < (MAX_DEPTH - 1e-3)
    id_map = torch.where(hit, id_map, torch.zeros_like(id_map))

    # --- per-layer normals from screen-space depth gradients ---
    dirs_bt = dirs                                              # (BT,H,W,3)
    org_bt = org.view(BT, 1, 1, 3)
    normals = torch.zeros((K, BT, H, W, 3), device=dev)
    for k in range(K):
        dk = depth_layers[k].view(BT, H, W)
        hk = dk < (MAX_DEPTH - 1e-3)
        pos = org_bt + dirs_bt * dk.unsqueeze(-1)              # (BT,H,W,3)
        normals[k] = _normals_from_depth(pos, hk)
    normal = normals.view(K, B, T, H, W, 3).permute(1, 2, 0, 3, 4, 5).contiguous()

    return id_map, depth, hit, normal, dirs.view(B, T, H, W, 3)
