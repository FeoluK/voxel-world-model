"""v7: same architecture as v6_minimal (position-only action stream).

The only difference vs v6 is at the trainer/dataset layer: v7 feeds per-frame
CLIP CLS tokens (33 tokens per chunk) into the cross-attention path instead of
a single pooled token. The cross-attn module handles variable-length context
natively, so the underlying model code is identical.

Re-exports v6 builders so `import voxel_dit_solaris_v7` Just Works.
"""
from voxel_dit_solaris_v6 import (  # noqa: F401
    build_voxel_dit_solaris_270m,
)
