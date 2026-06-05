"""
3D Voxel VAE (Section 5 / Table 2 / Appendix A.1) -- SCALED-DOWN COPY.

Architecture family is identical to persist_reference_v4/vae_3d.py. Only the
middle channel widths are scaled to target ~3M parameters (paper: 138M).

Invariants preserved (see README.md invariant cross-reference):
  - Input shape:  [B, 48, 48, 48] long (voxel class IDs)
  - Latent shape: [B, 48, 12, 12, 12]   (48 channels, 12^3 grid)
  - Downsample factor: 4 (48 -> 12 via two stride-2 stages)
  - KL weight: 1e-6
  - Loss: cross-entropy (voxel-class prediction) + KL
  - Number of voxel classes: 2517 (Luanti vocab; paper uses 2138)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int, groups: int = 8) -> nn.GroupNorm:
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.ConvTranspose3d(ch, ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class MidBlock3D(nn.Module):
    """Bottleneck: two residual blocks at the deepest channel width."""

    def __init__(self, ch: int):
        super().__init__()
        self.res1 = ResBlock3D(ch, ch)
        self.res2 = ResBlock3D(ch, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res2(self.res1(x))


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_classes: int,
        middle_channels: tuple[int, ...],
        num_res_blocks: int,
        latent_ch: int,
    ):
        super().__init__()
        self.in_proj = nn.Conv3d(in_classes, middle_channels[0], kernel_size=3, padding=1)

        blocks: list[nn.Module] = []
        prev = middle_channels[0]
        downsample_at = {0, 1}  # 48 -> 24 -> 12
        for stage_i, ch in enumerate(middle_channels):
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(prev, ch))
                prev = ch
            if stage_i in downsample_at:
                blocks.append(Downsample3D(prev))
        self.blocks = nn.Sequential(*blocks)

        self.mid = MidBlock3D(prev)
        self.norm_out = _gn(prev)
        self.to_latent = nn.Conv3d(prev, 2 * latent_ch, kernel_size=1)

    def forward(self, x_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x_onehot)
        h = self.blocks(h)
        h = self.mid(h)
        h = F.silu(self.norm_out(h))
        stats = self.to_latent(h)
        mean, logvar = stats.chunk(2, dim=1)
        return mean, logvar


class Decoder3D(nn.Module):
    def __init__(
        self,
        out_classes: int,
        middle_channels: tuple[int, ...],
        num_res_blocks: int,
        latent_ch: int,
    ):
        super().__init__()
        rev = list(reversed(middle_channels))
        self.from_latent = nn.Conv3d(latent_ch, rev[0], kernel_size=1)
        self.mid = MidBlock3D(rev[0])

        blocks: list[nn.Module] = []
        prev = rev[0]
        upsample_at = {0, 1}  # 12 -> 24 -> 48
        for stage_i, ch in enumerate(rev):
            for _ in range(num_res_blocks):
                blocks.append(ResBlock3D(prev, ch))
                prev = ch
            if stage_i in upsample_at:
                blocks.append(Upsample3D(prev))
        self.blocks = nn.Sequential(*blocks)

        self.norm_out = _gn(prev)
        self.out_proj = nn.Conv3d(prev, out_classes, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = self.mid(h)
        h = self.blocks(h)
        h = F.silu(self.norm_out(h))
        return self.out_proj(h)  # logits over voxel classes


class VAE3D(nn.Module):
    """3D Voxel VAE (Table 2) -- scaled-down.

    Paper: middle_channels=(32, 128, 512), ~138M params.
    Scaled-down: middle_channels=(8, 24, 64), ~3M params target.

    The input/output channel dimensions of 2517 classes (Luanti vocab)
    dominate the first and last conv's cost, so the aggressive scale-down
    of the first/last stage (32 -> 8) is the main lever.
    """

    def __init__(
        self,
        num_classes: int = 2517,
        grid_size: int = 48,
        latent_size: int = 12,
        latent_channels: int = 48,
        middle_channels: tuple[int, ...] = (8, 24, 64),
        num_res_blocks: int = 2,
        kl_weight: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.latent_size = latent_size
        self.latent_channels = latent_channels
        self.kl_weight = kl_weight

        self.encoder = Encoder3D(
            in_classes=num_classes,
            middle_channels=middle_channels,
            num_res_blocks=num_res_blocks,
            latent_ch=latent_channels,
        )
        self.decoder = Decoder3D(
            out_classes=num_classes,
            middle_channels=middle_channels,
            num_res_blocks=num_res_blocks,
            latent_ch=latent_channels,
        )

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def encode(self, voxel_ids: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(voxel_ids.long(), num_classes=self.num_classes)
        onehot = onehot.permute(0, 4, 1, 2, 3).float()
        mean, logvar = self.encoder(onehot)
        return self.reparameterize(mean, logvar)

    def encode_mean(self, voxel_ids: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(voxel_ids.long(), num_classes=self.num_classes)
        onehot = onehot.permute(0, 4, 1, 2, 3).float()
        mean, _ = self.encoder(onehot)
        return mean

    def encode_mean_fast(self, voxel_ids: torch.Tensor) -> torch.Tensor:
        """Memory-efficient encoding using embedding lookups instead of one-hot.

        Mathematically identical to encode_mean but uses ~143x less memory
        by decomposing Conv3d(2517, 8, k=3, p=1) on one-hot input into
        27 embedding lookups (one per 3x3x3 kernel position).

        Uses sentinel ID (num_classes) with zero-row for boundary padding.
        """
        from itertools import product as iprod

        ids = voxel_ids.long()
        B, D, H, W = ids.shape
        device = ids.device

        # Pad with sentinel ID (num_classes) — contributes zero via zero-row
        SENTINEL = self.num_classes
        padded = F.pad(ids, (1, 1, 1, 1, 1, 1), value=SENTINEL)

        # Extract Conv3d weights: [out_ch, in_ch, 3, 3, 3]
        weight = self.encoder.in_proj.weight  # [8, 2517, 3, 3, 3]
        bias = self.encoder.in_proj.bias      # [8]
        out_ch = weight.shape[0]

        # Build 27 embedding tables from weight, each [2517+1, out_ch]
        # Append zero-row for sentinel ID
        zero_row = torch.zeros(1, out_ch, device=device, dtype=weight.dtype)

        result = torch.zeros(B, D, H, W, out_ch, device=device, dtype=weight.dtype)
        for di, dj, dk in iprod(range(3), repeat=3):
            # Embedding table for this offset: W[:, :, di, dj, dk].T = [2517, 8]
            table = weight[:, :, di, dj, dk].T.contiguous()  # [2517, out_ch]
            table_padded = torch.cat([table, zero_row], dim=0)  # [2518, out_ch]
            # Gather neighbor IDs at this offset
            neighbor_ids = padded[:, di:di+D, dj:dj+H, dk:dk+W]  # [B, D, H, W]
            result += F.embedding(neighbor_ids, table_padded)

        if bias is not None:
            result += bias

        # Permute to [B, out_ch, D, H, W] — same as Conv3d output
        h = result.permute(0, 4, 1, 2, 3).contiguous()

        # Continue through rest of encoder (skip in_proj, already done)
        h = self.encoder.blocks(h)
        h = self.encoder.mid(h)
        h = F.silu(self.encoder.norm_out(h))
        stats = self.encoder.to_latent(h)
        mean, _ = stats.chunk(2, dim=1)
        return mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, voxel_ids: torch.Tensor) -> dict:
        onehot = F.one_hot(voxel_ids.long(), num_classes=self.num_classes)
        onehot = onehot.permute(0, 4, 1, 2, 3).float()
        mean, logvar = self.encoder(onehot)
        z = self.reparameterize(mean, logvar)
        logits = self.decoder(z)

        ce = F.cross_entropy(logits, voxel_ids.long())
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss = ce + self.kl_weight * kl
        return {
            "loss": loss,
            "ce": ce.detach(),
            "kl": kl.detach(),
            "logits": logits,
            "z": z,
        }


if __name__ == "__main__":
    vae = VAE3D()
    n = sum(p.numel() for p in vae.parameters())
    print(f"VAE3D params: {n/1e6:.3f}M  (target: ~3M, paper: 138M)")
