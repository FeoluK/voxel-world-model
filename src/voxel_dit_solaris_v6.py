"""Voxel-DiT V6_minimal — v5 minus keyboard, mouse replaced with raw camera xyz position deltas (3-d).

Why: voxels are world-axis-aligned, so mouse rotation is a no-op for voxel content.
ALL voxel changes between frames are driven by agent POSITION shifting the cube.
v5/v4 included a 2/4-d mouse stream and a 7/23-d keyboard stream that are redundant
for the voxel domain — the only signal that maps to per-frame voxel translation is
the camera xyz.

V6 changes vs v5:
- Drop keyboard branch entirely (no keyboard_embed, keyboard_attn_kv, proj_keyboard,
  enable_keyboard, key_attn_*_norm, kbd_rope_freqs).
- Rename mouse_* -> position_* throughout (position_mlp, position_attn_q, proj_position,
  position_rope_freqs, enable_position).
- Mouse channel count 4 -> position channel count 3 (raw xyz).
- ActionModule.forward: forward(x, tt, ts, position_condition=None).
- Block.forward: forward(x, e, grid_sizes, freqs, context, position_cond=None).
- Top-level VoxelDiT.forward: drops mouse_cond/keyboard_cond, adds position_cond.
- Builder: position_dim_in=3 (no keyboard_dim_in, no enable_keyboard).
v6 ckpts are NOT cross-loadable with v5 (different action-module weights and shapes).
Architecturally otherwise identical to v5.

==========================================================================
Voxel-DiT — direct Solaris/Wan2.1 adaptation for 4D voxel latents.

This is a minimal port of /scratch/users/flukol/mg_pt/Matrix-Game-2/wan/modules/model.py
to the voxel domain. Only the changes strictly required for 4D voxel input
were made; every other ratio, init, norm scheme, and block structure matches Solaris exactly.

ADAPTATIONS FROM SOLARIS:
1. Patch embedding: 3D Conv (T,H,W) -> 4D-style patches (T,X,Y,Z) via flatten+Linear
   (PyTorch has no native Conv4d; we patchify in einops then project)
2. RoPE: 3-axis (T,H,W) split [22,21,21] complex -> 4-axis (T,X,Y,Z) equal split [16,16,16,16]
   complex (preserves Solaris's near-equal split character in the 4D extension)
3. in_dim 36 -> 100 (48 voxel x_t + 48 first-frame voxel cond_concat + 4 mask channels)
4. out_dim 16 -> 48 (predict voxel latent)
5. patch_size (1,2,2) -> (1,2,2,2)
6. ActionModule.vae_time_compression_ratio: 4 -> 1 (our voxel VAE has no temporal compression)
7. action_module sizing scaled proportionally to dim (hidden 128->64, mouse_hidden 1024->384, etc.)

KNOWN MINOR DEVIATIONS (verified by audit, severity MEDIUM, may revisit):
- ActionRMSNorm omits the unused `self.weight` parameter (Solaris declares it but doesn't
  use it in forward — see action_module.py:23-35). Cosmetic; only matters if loading
  Solaris pretrained weights.

OUTSTANDING TODO (HIGH severity, needs careful port):
- Action-module mouse Q/K and keyboard Q/K should have RoPE applied (Solaris
  action_module.py:337,347,449,459 calls apply_rotary_emb). Solaris's RoPE for
  action attn is a 3D rotary over (T, patch_h, patch_w) with mouse_qk_dim_list=[8,28,28]
  (sum=64=head_dim). Our action attn operates per-spatial-position over T=33 with
  head_dim=128, so the equivalent port is 1D-along-T RoPE rotating all 128 head_dims
  (or matching solaris's split). Not yet ported — the first 4 fixes above (window
  off-by-one, kbd Q-shape, proj init, plus this comment header) are the priority.
  Apply this RoPE port BEFORE retraining if possible; otherwise keep without RoPE
  and re-evaluate.

FIXED 2026-05-08 (verified vs solaris_repo at /scratch/users/flukol/solaris_repo/):
1. Action window off-by-one: pad_t now = frames_window_size - 1 (was frames_window_size).
   Window for latent i now covers real frames [i*vae_tc - W + 1, i*vae_tc + 1) = past +
   current (was [i*vae_tc - W, i*vae_tc) = pure past, missing current).
2. Keyboard cross-attn Q-shape: now reshape Q to (B*S, T, H, D) so softmax runs over
   T=33 keys per spatial position (was over T*S=7128 with K/V tiled spatially, diluting
   the time-correct attention by 216×). Matches solaris_repo:437-441.
3. proj_mouse / proj_keyboard zero-init removed: now use xavier_uniform from the global
   init loop. Solaris uses default flax LeCun normal (no zero override).
4. Header comment now reflects the actual remaining deviations (just RoPE).

NOT ADAPTED (intentional, kept Solaris-exact):
- RMSNorm + LayerNorm classes (eps, no learnable RMS bias)
- Pre-norm structure for self-attn and FFN
- AdaLN modulation init: randn(1,6,dim)/sqrt(dim) (NOT zero-init)
- Output head linear init: zeros (NOT std=0.02)
- proj_mouse/proj_keyboard init: zeros (NOT std=0.02)
- xavier_uniform on all other Linears
- ffn_dim/dim = 5.833 ratio (3744/640)
- num_layers = 30
- num_heads = 5 (preserves head_dim = 128 = Solaris's exact value)
- bf16 weights throughout (caller is expected to do model.to(bfloat16))
- CLIP MLPProj: in_dim=1280 -> dim, no other text path
- Cross-attention type 'i2v_cross_attn' (single CLIP visual_context stream)
- Time embedding: sinusoidal_1d with freq_dim=256 -> dim via SiLU MLP
- Gradient checkpointing: identical wrapper as Solaris model.py:683-688

DROPPED (vs our previous voxel_dit.py):
- Camera module (privileged-info leak — game state)
- Plucker rays (depended on camera)
- Separate RGB cross-attention (will add back later if needed)
- Register tokens (Solaris always has CLIP)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


__all__ = ['VoxelDiTSolaris', 'build_voxel_dit_solaris_270m', 'build_voxel_dit_solaris_1_5b']

# FIX 24 (2026-05-08): Apply 1D RoPE on T to mouse + keyboard action attention.
# Solaris action_module.py:347-353 (mouse) and :459-465 (keyboard) both call
# apply_rotary_emb(q, k, freqs_cis) on action attention. We were missing this.
# Without temporal RoPE the action attention over T=33 frames is permutation-
# invariant in time, so the per-frame mouse delta gets averaged across all
# frames — explaining the observed mouse-blindness (drop_zero_mouse=0 across
# all 7+ tested chunks with mouse_mag 0.07-0.26). Solaris uses 3D RoPE
# [8,28,28] over (t,ph,pw); since K side has no spatial info in our reshape
# pattern, we apply 1D RoPE on T (the axis that distinguishes per-frame
# action signal — the one whose absence causes mouse blindness).


def _apply_rope_1d(x, freqs):
    """Apply 1D rotary embedding over T axis to FIRST 2*freqs.shape[-1] dims of
    head_dim, leaving the rest as identity.

    x: (B*S, T, H, D_total)
    freqs: complex (max_T, n_pairs) — from rope_params(max_T, 2*n_pairs)

    FIX 32 (2026-05-08): Solaris's mouse_qk_dim_list=[8,28,28] splits head_dim
    across (T, ph, pw). Only the first 8 dims (4 complex pairs) get T-rotation;
    spatial axes collapse to identity for our K-side (no spatial info on K).
    Was rotating all head_dim/2 pairs — over-rotation with low-k aliasing over
    T=33. Now matches Solaris.
    """
    T = x.shape[-3]
    n_pairs = freqs.shape[-1]
    n = 2 * n_pairs
    D = x.shape[-1]
    if n >= D:
        f = freqs[:T].view(T, 1, n_pairs)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], n_pairs, 2))
        return torch.view_as_real(x_complex * f).reshape(*x.shape).type_as(x)
    f = freqs[:T].view(T, 1, n_pairs)
    x_rot = x[..., :n]
    x_rest = x[..., n:]
    x_complex = torch.view_as_complex(x_rot.float().reshape(*x_rot.shape[:-1], n_pairs, 2))
    x_rot_out = torch.view_as_real(x_complex * f).reshape(*x_rot.shape).type_as(x)
    return torch.cat([x_rot_out, x_rest], dim=-1)


# =============================================================================
# Norms (identical to Solaris)
# =============================================================================

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):  # FIX 33: default 1e-5 was dead-code mismatch with Solaris 1e-6 (every callsite already passes 1e-6 explicitly; this aligns the default).
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


# FIX 11 (2026-05-08): Solaris explicitly upcasts to fp32 inside norm1/2/3 forward
# (world_model.py:370, 386, 446: `x.astype(jnp.float32)` before LN, then `.astype(x.dtype)`
# after). Our previous impl let LN run in input dtype (potentially bf16). Override forward
# to do the upcast.
# FIX 11b (2026-05-08): also force weight/bias to fp32 for the LN compute. If a caller
# does `model.to(bf16)` (legacy path), nn.LayerNorm's params end up in bf16 and the call
# `F.layer_norm(x_fp32, weight_bf16)` raises a dtype mismatch. We use F.layer_norm
# directly with fp32-cast params instead of super().forward.
class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        w = self.weight.float() if self.weight is not None else None
        b = self.bias.float() if self.bias is not None else None
        return torch.nn.functional.layer_norm(
            x.float(), self.normalized_shape, w, b, self.eps
        ).type_as(x)


# =============================================================================
# Time embedding (identical to Solaris)
# =============================================================================

def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# =============================================================================
# 4D RoPE (extended from Solaris's 3D)
# =============================================================================

def rope_params(max_seq_len, dim, theta=10000):
    """Identical to Solaris."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply_4d(x, grid_sizes, freqs):
    """4D RoPE for voxel tokens at positions (t, x, y, z).

    Args:
        x: (B, L, num_heads, head_dim) where L = T*X*Y*Z
        grid_sizes: tuple of ints (T, X, Y, Z) — same for all batch (TIER1 fix style).
        freqs: complex tensor (max_seq_len, head_dim/2) — concatenated 4-axis freqs.
            Split sizes: [c_T, c_X, c_Y, c_Z] complex dims, summing to head_dim/2.
    """
    n, c = x.size(2), x.size(3) // 2
    # 4-way equal split (preserves Solaris's near-equal 3D split character).
    chunk = c // 4
    freqs_split = freqs.split([chunk, chunk, chunk, c - 3 * chunk], dim=1)

    T, X, Y, Z = grid_sizes
    seq_len = T * X * Y * Z

    output = []
    for i in range(len(x)):
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        # Build 4D positional indexing as outer products in complex space.
        f_t = freqs_split[0][:T].view(T, 1, 1, 1, -1).expand(T, X, Y, Z, -1)
        f_x = freqs_split[1][:X].view(1, X, 1, 1, -1).expand(T, X, Y, Z, -1)
        f_y = freqs_split[2][:Y].view(1, 1, Y, 1, -1).expand(T, X, Y, Z, -1)
        f_z = freqs_split[3][:Z].view(1, 1, 1, Z, -1).expand(T, X, Y, Z, -1)
        freqs_i = torch.cat([f_t, f_x, f_y, f_z], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


# =============================================================================
# Self-attention (identical to Solaris)
# =============================================================================

def _attn(q, k, v, causal=False):
    """Fallback SDPA — Solaris uses flash_attn.flash_attention; we use SDPA for portability.
    Behavior is mathematically identical (no varlen needed for our fixed-size training).
    """
    q = q.transpose(1, 2)  # (B, H, L, D)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.transpose(1, 2)  # (B, L, H, D)


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q = rope_apply_4d(q, grid_sizes, freqs)
        k = rope_apply_4d(k, grid_sizes, freqs)
        x = _attn(q, k, v, causal=False)
        x = x.flatten(2)
        x = self.o(x)
        return x


# =============================================================================
# I2V cross-attention (identical to Solaris)
# =============================================================================

class WanI2VCrossAttention(WanSelfAttention):
    def forward(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = _attn(q, k, v, causal=False)
        x = x.flatten(2)
        x = self.o(x)
        return x


# =============================================================================
# Action module (Solaris exact, with vae_time_compression_ratio=1 for voxel)
# Adapted minimally from /scratch/users/flukol/mg_pt/Matrix-Game-2/wan/modules/action_module.py
# Drops the flex_attention / kv_cache code paths (not needed for Stage 1 bidirectional).
# =============================================================================


class ActionRMSNorm(nn.Module):
    """Solaris's action_module uses the same WanRMSNorm class as the rest of the model
    (action_module.py:10 imports from transformer.py). transformer.py:25-38 defines and
    USES self.weight in forward (`* self.weight.value.astype(input_dtype)`). Our previous
    "weightless" assumption was wrong (FIX 7, 2026-05-08). Adding learnable weight to match.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class VoxelActionModule(nn.Module):
    """V6 ActionModule — position-only (mouse/keyboard removed).

    Direct port of Solaris ActionModule mouse-self-attn path, but the input
    is the agent's xyz position (3 channels) instead of mouse/orientation.
    Structure (windowed MLP -> per-spatial-position self-attn over T -> proj add)
    is unchanged from v5's mouse path; only the channel dim and naming differ.

    - vae_time_compression_ratio = 1 (our voxel VAE is 1x temporal)
    - Spatial dim S = X*Y*Z (4D voxel) instead of H*W (2D video)
    - Bidirectional only (drops flex_attention/kv_cache)
    """

    def __init__(self,
                 position_dim_in: int = 3,   # v6: raw camera xyz (was mouse_dim_in=4)
                 hidden_size: int = 64,
                 img_hidden_size: int = 640,
                 position_hidden_dim: int = 384,
                 vae_time_compression_ratio: int = 1,
                 windows_size: int = 3,
                 heads_num: int = 8,
                 qk_norm: bool = True,
                 qkv_bias: bool = False,
                 enable_position: bool = True):
        super().__init__()
        self.enable_position = enable_position
        self.heads_num = heads_num
        self.vae_time_compression_ratio = vae_time_compression_ratio
        self.windows_size = windows_size

        if self.enable_position:
            c = position_hidden_dim
            self.position_mlp = nn.Sequential(
                nn.Linear(position_dim_in * vae_time_compression_ratio * windows_size + img_hidden_size, c, bias=True),
                nn.GELU(approximate='tanh'),
                nn.Linear(c, c),
                nn.LayerNorm(c),
            )
            head_dim = c // heads_num
            self.t_qkv = nn.Linear(c, c * 3, bias=qkv_bias)
            self.img_attn_q_norm = ActionRMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
            self.img_attn_k_norm = ActionRMSNorm(head_dim, eps=1e-6) if qk_norm else nn.Identity()
            self.proj_position = nn.Linear(c, img_hidden_size, bias=qkv_bias)
            # 1D RoPE on T for position self-attn (max 256 frames). theta=256 matches
            # single_player.yaml:54. Only 8 dims rotated (4 freq pairs), matching
            # Solaris's mouse_qk_dim_list=[8,28,28] T-axis allocation in v5.
            self.register_buffer('position_rope_freqs', rope_params(256, 8, theta=256), persistent=False)

    def forward(self, x, tt, ts, position_condition=None):
        """
        Args:
            x: (B, tt*ts, C) image tokens, where ts = X*Y*Z (flattened spatial volume)
            tt: int, number of latent frames (33 for us)
            ts: int, total spatial tokens per frame
            position_condition: (B, N_frames, 3) — raw camera xyz (frame-0-relative,
                already normalized by voxel_dataset.normalize_camera).
        """
        if position_condition is None:
            return x
        B, N_frames, _ = position_condition.shape
        # vae_tc=1 so N_feats = N_frames
        assert ((N_frames - 1) + self.vae_time_compression_ratio) % self.vae_time_compression_ratio == 0
        N_feats = int((N_frames - 1) / self.vae_time_compression_ratio) + 1
        assert N_feats == tt, f"N_feats {N_feats} != tt {tt}"
        assert tt * ts == x.shape[1], f"tt*ts {tt*ts} != x.shape[1] {x.shape[1]}"

        # Action window: pad with `frames_window_size - 1` so that
        # window i covers real frames `[i*vae_tc - (W-1), i*vae_tc + 1)` —
        # the previous (W-1) actions PLUS the current frame's action.
        frames_window_size = self.vae_time_compression_ratio * self.windows_size
        pad_t = frames_window_size - 1
        win = frames_window_size  # length of each window slice

        if self.enable_position and position_condition is not None:
            hidden_states = rearrange(x, "B (T S) C -> (B S) T C", T=tt, S=ts)
            B_orig, N_frames_p, C_p = position_condition.shape

            pad = position_condition[:, 0:1, :].expand(-1, pad_t, -1)
            position_padded = torch.cat([pad, position_condition], dim=1)
            group_position = [
                position_padded[:,
                                i * self.vae_time_compression_ratio:
                                i * self.vae_time_compression_ratio + win, :]
                for i in range(N_feats)
            ]
            group_position = torch.stack(group_position, dim=1)  # (B, T, win, C_p)
            S = ts
            group_position = group_position.unsqueeze(-1).expand(B_orig, N_feats, win, C_p, S)
            group_position = group_position.permute(0, 4, 1, 2, 3).reshape(B_orig * S, N_feats, win * C_p)

            group_position = torch.cat([hidden_states, group_position], dim=-1)
            group_position = self.position_mlp(group_position)
            position_qkv = self.t_qkv(group_position)
            q, k, v = rearrange(position_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
            q = self.img_attn_q_norm(q).to(v)
            k = self.img_attn_k_norm(k).to(v)
            # 1D RoPE on T axis (q,k shape (B*S, T, H, D))
            q = _apply_rope_1d(q, self.position_rope_freqs)
            k = _apply_rope_1d(k, self.position_rope_freqs)

            attn = _attn(q, k, v, causal=False)
            attn = rearrange(attn, '(b S) T h d -> b (T S) (h d)', b=B_orig)
            hidden_states = rearrange(x, "(B S) T C -> B (T S) C", B=B_orig)
            attn = self.proj_position(attn)
            hidden_states = hidden_states + attn
        else:
            hidden_states = x

        return hidden_states


# =============================================================================
# Attention Block (Solaris exact structure)
# =============================================================================

WAN_CROSSATTENTION_CLASSES = {
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads,
                 qk_norm=True, cross_attn_norm=True,
                 action_config=None, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.action_model = VoxelActionModule(**action_config) if action_config else None

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        # FIX 8 (2026-05-08): Solaris FFN uses exact GELU (jax.nn.gelu default
        # approximate=False), not tanh-approximation. world_model.py:316-324.
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),  # exact GELU to match Solaris
            nn.Linear(ffn_dim, dim),
        )
        # AdaLN modulation — Solaris uses randn/sqrt(dim), NOT zero-init.
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim ** 0.5)

    def forward(self, x, e, grid_sizes, freqs, context, position_cond=None):
        if e.dim() == 3:
            e = (self.modulation + e).chunk(6, dim=1)
        elif e.dim() == 4:
            modulation = self.modulation.unsqueeze(2)
            e = (modulation + e).chunk(6, dim=1)
            e = [ei.squeeze(1) for ei in e]

        # self-attention
        y = self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], grid_sizes, freqs)
        x = x + y * e[2]

        # cross-attention + action + FFN
        x = x + self.cross_attn(self.norm3(x.to(context.dtype)), context)
        if self.action_model is not None:
            T, X, Y, Z = grid_sizes
            ts = X * Y * Z
            x = self.action_model(x.to(context.dtype), T, ts, position_cond)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x


# =============================================================================
# Output head (Solaris exact, AdaLN before final Linear)
# =============================================================================

class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim ** 0.5)

    def forward(self, x, e):
        if e.dim() == 2:
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        elif e.dim() == 3:
            modulation = self.modulation.unsqueeze(2)
            e = (modulation + e.unsqueeze(1)).chunk(2, dim=1)
            e = [ei.squeeze(1) for ei in e]
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # FIX 9 (2026-05-08): Solaris MLPProj uses tanh-approx GELU (transformer.py:312:
        # `jax.nn.gelu(x, approximate=True)`). Direction was flipped in our impl.
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(approximate='tanh'),  # tanh-approx to match Solaris MLPProj
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, image_embeds):
        # FIX 10 (2026-05-08): Solaris explicitly upcasts to fp32 for the LN
        # (transformer.py:308-314: `self.proj[0](image_embeds.astype(jnp.float32))`).
        # Cast input to fp32 for LN, back to original dtype for downstream.
        return self.proj(image_embeds.float()).to(image_embeds.dtype)


# =============================================================================
# 4D Patch Embedding (PyTorch has no Conv4d; use einops + Linear)
# =============================================================================

class VoxelPatchEmbed4D(nn.Module):
    """Patchify (B, C_in, T, X, Y, Z) -> (B, dim, T_p, X_p, Y_p, Z_p) tokens.

    patch_size = (pT, pX, pY, pZ). T must be divisible by pT, X/Y/Z by pX/pY/pZ.
    Implementation: einops rearrange to extract patches, then Linear project.
    Mathematically identical to a (nonexistent) nn.Conv4d with stride=kernel=patch_size.
    """

    def __init__(self, in_dim, dim, patch_size):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.patch_size = patch_size
        flat = in_dim * math.prod(patch_size)
        self.proj = nn.Linear(flat, dim)

    def forward(self, x):
        # x: (B, C, T, X, Y, Z)
        pT, pX, pY, pZ = self.patch_size
        x = rearrange(
            x, "B C (T pt) (X px) (Y py) (Z pz) -> B (T X Y Z) (C pt px py pz)",
            pt=pT, px=pX, py=pY, pz=pZ,
        )
        x = self.proj(x)  # (B, L, dim)
        return x


# =============================================================================
# VoxelDiT main model — direct Solaris WanModel adaptation
# =============================================================================

class VoxelDiTSolaris(nn.Module):
    def __init__(self,
                 patch_size=(1, 2, 2, 2),
                 in_dim=100,           # 48 voxel + 48 cond_concat + 4 mask
                 dim=640,
                 ffn_dim=3744,
                 freq_dim=256,
                 out_dim=48,
                 num_heads=5,           # head_dim = 128 (Solaris-exact)
                 num_layers=30,
                 qk_norm=True,
                 cross_attn_norm=True,
                 action_config=None,
                 clip_in_dim=1280,
                 eps=1e-6):
        super().__init__()
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.eps = eps

        # 4D patch embedding (replaces Solaris's Conv3d patch_embedding)
        self.patch_embedding = VoxelPatchEmbed4D(in_dim, dim, patch_size)

        # Time embedding (identical to Solaris)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # Blocks. FIX #16 (2026-05-08): match Solaris's action_module_zipper.
        # Solaris world_model.py:818 with default matrix_game_forward=True does
        # `action_module_zipper = jnp.array([True]*15 + [False]*15)`; only the
        # first half of blocks inject action features into the residual stream.
        # Previously we ran action_module on every block, doubling the action-
        # path gradient and (per the symptom we observed) causing keyboard
        # injection to interfere with itself.
        cross_attn_type = 'i2v_cross_attn'
        n_action = num_layers // 2  # 15 of 30 by default
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              qk_norm, cross_attn_norm,
                              action_config if i < n_action else None,
                              eps=eps)
            for i in range(num_layers)
        ])

        # Output head
        self.head = Head(dim, out_dim, patch_size, eps)

        # 4D RoPE freqs — equal split [c/4, c/4, c/4, c-3*(c/4)] complex per axis.
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        head_dim = dim // num_heads
        c = head_dim // 2
        chunk = c // 4
        self.freqs = torch.cat([
            rope_params(1024, chunk * 2),
            rope_params(1024, chunk * 2),
            rope_params(1024, chunk * 2),
            rope_params(1024, (c - 3 * chunk) * 2),
        ], dim=1)

        # CLIP MLPProj (image embedding -> dim)
        self.img_emb = MLPProj(clip_in_dim, dim)

        # Solaris-style init
        self.init_weights()
        self.gradient_checkpointing = False

    def init_weights(self):
        # Xavier on all Linears, zero biases
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # FIX 14 (2026-05-08): Patch embed init scale. Solaris's transformer.py:108-118
        # initializes Conv3d with `random_normal * 2.0/sqrt(in_dim*kernel_volume)`. Our
        # patch_embedding is nn.Linear after einops, but the equivalent init is the same
        # std. With in_dim=100 (48 noise + 52 cond_concat), patch_volume=8: std = 2/sqrt(800)
        # ≈ 0.0707. Our prior xavier_uniform gave std ≈ sqrt(2/(800+640)) ≈ 0.037 (~2× smaller).
        in_dim_patch = self.patch_embedding.proj.in_features  # = in_dim * patch_volume
        nn.init.normal_(self.patch_embedding.proj.weight, std=2.0 / (in_dim_patch ** 0.5))
        nn.init.zeros_(self.patch_embedding.proj.bias)
        # FIX 15 (2026-05-08): Time embedding init asymmetry. Previously we overrode only
        # `time_embedding` (with std=0.02) but NOT `time_projection`. Solaris's flax default
        # for both is LeCun normal (≈std=1/sqrt(fan_in)). Removing the std=0.02 override so
        # both go through xavier_uniform from the global Linear-init loop above. Same
        # treatment for both halves of the time-embedding pathway.
        # (Was: for m in self.time_embedding.modules(): nn.init.normal_(m.weight, std=0.02))
        # FIX 6 (2026-05-08): Output head was incorrectly zero-init. Solaris uses default
        # LeCun normal (flax `nnx.Linear`); see solaris_repo:world_model.py:472-478. Zero-init
        # of the final Linear in a DiT head is a known anti-pattern (kills upstream gradient
        # flow at init — see saved memory feedback_dit_head_init). Removing the zero override.
        # nn.init.zeros_(self.head.head.weight)  # REMOVED — was Solaris-INCORRECT
        # FIX 4 (2026-05-08): proj_mouse / proj_keyboard are NOT zero-init in Solaris.
        # solaris_repo/src/models/action_module.py:123,164 use `nnx.Linear(...)` defaults
        # (LeCun normal). Our previous "zero-init for stability" was a misread of the AdaLN
        # final-layer convention. Letting xavier_uniform from the loop above stand.
        # (Action features were being multiplied by zero proj weights at init, so all
        # gradient on the action MLP/attn was second-order and the path collapsed.)

    def forward(self, x, t, visual_context, cond_concat, position_cond=None):
        """
        Args:
            x: (B, 48, T, X, Y, Z) noisy voxel latent
            t: (B,) integer diffusion timestep [0, 1000]
            visual_context: (B, 257, 1280) CLIP-ViT-H features
            cond_concat: (B, 52, T, X, Y, Z) — 48 first-frame voxel + 4 mask channels
            position_cond: (B, T_frames, 3) — v6: raw camera xyz (frame-0-relative,
                already normalized by voxel_dataset.normalize_camera). May be None.
        """
        device = next(self.parameters()).device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Compute grid_sizes BEFORE patch embedding so we know post-patch shape
        B, _, T, X, Y, Z = x.shape
        pT, pX, pY, pZ = self.patch_size
        assert T % pT == 0 and X % pX == 0 and Y % pY == 0 and Z % pZ == 0, \
            f"Voxel dims ({T},{X},{Y},{Z}) not divisible by patch ({pT},{pX},{pY},{pZ})"
        grid_sizes = (T // pT, X // pX, Y // pY, Z // pZ)

        # Channel concat with cond_concat (always 52ch: 48 voxel + 4 mask)
        x = torch.cat([x, cond_concat], dim=1)  # (B, 100, T, X, Y, Z)
        x = self.patch_embedding(x)  # (B, L, dim) where L = T*X*Y*Z / patch_volume

        # Time embedding (identical to Solaris)
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).type_as(x))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        # CLIP context
        context = self.img_emb(visual_context)

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # Solaris-style checkpoint wrapper
                def _ckpt(blk, x_, e_, gs_, freqs_, ctx_, pc_):
                    return blk(x_, e_, gs_, freqs_, ctx_, pc_)
                x = torch.utils.checkpoint.checkpoint(
                    _ckpt, block, x, e0, grid_sizes, self.freqs, context,
                    position_cond, use_reentrant=False)
            else:
                x = block(x, e0, grid_sizes, self.freqs, context, position_cond)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return x.float()

    def unpatchify(self, x, grid_sizes):
        """Inverse of patch_embedding. (B, L, out_dim*patch_volume) -> (B, out_dim, T, X, Y, Z)."""
        T_p, X_p, Y_p, Z_p = grid_sizes
        pT, pX, pY, pZ = self.patch_size
        c = self.out_dim
        x = x.view(x.size(0), T_p, X_p, Y_p, Z_p, c, pT, pX, pY, pZ)
        x = torch.einsum("bTXYZcijkl->bcTiXjYkZl", x)
        x = x.reshape(x.size(0), c, T_p * pT, X_p * pX, Y_p * pY, Z_p * pZ)
        return x

    def _set_gradient_checkpointing(self, value=False):
        self.gradient_checkpointing = value


# =============================================================================
# Builder for 270M-equivalent (with Solaris ratios)
# =============================================================================

def build_voxel_dit_solaris_270m():
    """270M-equivalent voxel-DiT with Solaris ratios:
    - dim=640, num_heads=5, head_dim=128 (Solaris-exact head_dim)
    - ffn_dim=3744 (5.85x dim, Solaris ratio is 5.833x)
    - num_layers=30 (matches Solaris exactly)
    - patch_size=(1,2,2,2) -> 33x6x6x6 = 7128 tokens per chunk
    - Action module scaled proportionally (V6 position-only):
        hidden_size=64 (Solaris 128 / 2 dim ratio)
        position_hidden_dim=384 (was mouse_hidden_dim, Solaris 1024 * 640/1536 ~= 427, rounded to 384)
        heads_num=8 (action attention heads)
        vae_time_compression_ratio=1 (our voxel VAE)
        windows_size=12 (Solaris vae_tc=4 * W=3 = 12 effective)
    Estimated total params: ~270M (slightly lower than v5 since keyboard branch dropped)
    """
    # windows_size=12 matches Solaris's effective frames_window_size = vae_tc * W = 4 * 3 = 12.
    # Our vae_tc=1, so W=12 to get 12-frame position history per voxel frame.
    action_config = dict(
        position_dim_in=3,   # v6: raw camera xyz (was mouse_dim_in=4)
        hidden_size=64,
        img_hidden_size=640,
        position_hidden_dim=384,
        vae_time_compression_ratio=1,
        windows_size=12,
        heads_num=8,
        qk_norm=True,
        qkv_bias=False,
        enable_position=True,
    )
    return VoxelDiTSolaris(
        patch_size=(1, 2, 2, 2),
        in_dim=100,        # 48 (voxel x_t) + 48 (first-frame) + 4 (mask)
        dim=640,
        ffn_dim=3744,
        freq_dim=256,
        out_dim=48,
        num_heads=5,
        num_layers=30,
        qk_norm=True,
        cross_attn_norm=True,
        action_config=action_config,
        clip_in_dim=1280,
        eps=1e-6,
    )


# =============================================================================
# Builder for Solaris-FAITHFUL full scale (~1.5B params, mirrors single_player.yaml)
# =============================================================================

def build_voxel_dit_solaris_1_5b():
    """Solaris-faithful full-scale voxel-DiT.

    Backbone matches Solaris's `SolarisSPModel` defaults exactly
    (solaris_repo/src/models/singleplayer/world_model.py:264-272 +
    solaris_repo/config/model/single_player.yaml):
        dim=1536, ffn_dim=8960, num_heads=12 (head_dim=128), num_layers=30

    Action module matches Solaris single_player.yaml exactly:
        hidden_size=128
        img_hidden_size=1536          (== backbone dim)
        mouse_hidden_dim=1024
        keyboard_hidden_dim=1024
        heads_num=16                  (action-attn head_dim = 1024/16 = 64,
                                        which is what Solaris's mouse_qk_dim_list
                                        [8,28,28] sums to — match)
        vae_time_compression_ratio=1  (our voxel VAE; Solaris uses 4 for RGB)
        windows_size=3                (matches Solaris)

    Backbone + action_module_zipper (FIX 16) keeps action injection on
    blocks 0-14 only; same as 270m.

    Estimated total params: ~1.5B (backbone ~1.1B + action 15× ~25M = ~1.4B + heads/embed)
    """
    # windows_size=12 (see 270m builder note).
    action_config = dict(
        position_dim_in=3,   # v6: raw camera xyz (was mouse_dim_in=4)
        hidden_size=128,
        img_hidden_size=1536,
        position_hidden_dim=1024,
        vae_time_compression_ratio=1,
        windows_size=12,
        heads_num=16,
        qk_norm=True,
        qkv_bias=False,
        enable_position=True,
    )
    return VoxelDiTSolaris(
        patch_size=(1, 2, 2, 2),
        in_dim=100,        # 48 (voxel x_t) + 48 (first-frame) + 4 (mask)
        dim=1536,          # Solaris-exact
        ffn_dim=8960,      # Solaris-exact
        freq_dim=256,      # Solaris-exact
        out_dim=48,
        num_heads=12,      # Solaris-exact (head_dim = 1536/12 = 128)
        num_layers=30,     # Solaris-exact
        qk_norm=True,
        cross_attn_norm=True,
        action_config=action_config,
        clip_in_dim=1280,
        eps=1e-6,
    )
