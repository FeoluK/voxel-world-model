"""Voxel-DiT training dataset.

Yields aligned 33-frame clips from a single random episode. Each clip contains
the 5 input streams the model needs:

    voxel_lat       (48, 33, 12, 12, 12) fp16   — denoising target
    actions_mouse   (33, 2)              fp32   — dx, dy from JSONL (Solaris CAMERA_SCALER)
    actions_keyboard(33, 23)             fp32   — 23 binary keys from JSONL
    camera          (33, 5)              fp32   — (xpos, ypos, zpos, yaw, pitch) from JSONL
    rgb_lat         (16, 9, 27, 48)      fp16   — Wan VAE latent for the same temporal slice

# Important alignment note (RGB latents)
# ---------------------------------------
# `encode_rgb_latents.py` encodes the source mp4 in independent 33-frame
# CHUNKS. Each chunk produces F_lat = 9 latent frames (Wan VAE temporal
# compression: F_lat = (F-1)//4 + 1; for F=33 → 9). Because the chunks are
# encoded independently, the per-frame latent index for source frame `t`
# is well-defined ONLY when the slice we want lies entirely within ONE
# chunk. Therefore we restrict valid clip starts to chunk boundaries:
#   t0 ∈ {0, 33, 66, ...}
# and pull RGB latents as `rgb[:, c*9:(c+1)*9]` where c = t0 // 33.
#
# This keeps RGB ↔ voxel temporal alignment exact at zero ambiguity cost.
# The 36 chunk-boundaries × ~5k episodes ≈ 180k unique clips is plenty.

# Caching
# -------
# Each worker keeps the most recently loaded episode in memory and pulls
# `samples_per_episode` random clips from it before reloading. Amortizes
# NFS read cost like the voxel-VAE trainer.

# Robustness
# ----------
# If any required file is missing (rgb latent for the chosen episode, jsonl,
# malformed line, …) we skip that episode and reload another. The
# constructor builds an index of episodes that have all required files
# present at startup so we don't repeatedly try and fail.
"""
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


# ---------------------------------------------------------------------------
# Paths (on farmshare).
# ---------------------------------------------------------------------------
VOXEL_LAT_DIR = "/scratch/users/flukol/mg_pt_voxel/extract/latents"
RGB_LAT_DIR = "/scratch/users/flukol/mg_pt_voxel/extract/rgb_latents"
JSONL_DIR = "/scratch/users/flukol/vpt_solaris/vpt/v6"
CLIP_DIR = "/scratch/users/flukol/mg_pt_voxel/extract/clip_features_h"


# ---------------------------------------------------------------------------
# Constants from Solaris's `minecraft.py`.
# Mirroring those values verbatim so action conditioning matches the data
# convention that was used to train the rest of the pipeline.
# ---------------------------------------------------------------------------
KEYBOARD_BUTTON_MAPPING = {
    "key.keyboard.escape": "ESC",
    "key.keyboard.s": "back",
    "key.keyboard.q": "drop",
    "key.keyboard.w": "forward",
    "key.keyboard.1": "hotbar.1",
    "key.keyboard.2": "hotbar.2",
    "key.keyboard.3": "hotbar.3",
    "key.keyboard.4": "hotbar.4",
    "key.keyboard.5": "hotbar.5",
    "key.keyboard.6": "hotbar.6",
    "key.keyboard.7": "hotbar.7",
    "key.keyboard.8": "hotbar.8",
    "key.keyboard.9": "hotbar.9",
    "key.keyboard.e": "inventory",
    "key.keyboard.space": "jump",
    "key.keyboard.a": "left",
    "key.keyboard.d": "right",
    "key.keyboard.left.shift": "sneak",
    "key.keyboard.left.control": "sprint",
    "key.keyboard.f": "swapHands",
}

# 23 keyboard slots, in the same order Solaris's ACTION_KEYS uses (indices 0..22).
KEYBOARD_KEYS = [
    "inventory", "ESC",
    "hotbar.1", "hotbar.2", "hotbar.3", "hotbar.4", "hotbar.5",
    "hotbar.6", "hotbar.7", "hotbar.8", "hotbar.9",
    "forward", "back", "left", "right",
    "jump", "sneak", "sprint",
    "swapHands", "attack", "use", "pickItem", "drop",
]
assert len(KEYBOARD_KEYS) == 23
KEY_INDEX = {k: i for i, k in enumerate(KEYBOARD_KEYS)}

CAMERA_SCALER = 360.0 / 2400.0  # Solaris constant: pixels → degrees

# Linear compression matching CameraLinearConverterMatrixGame2 from Solaris.
# 1/15 scaling, clipped to ±20°.
MOUSE_LINEAR_SCALE = 1.0 / 15.0
MOUSE_LINEAR_CLIP = 20.0


# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------
def _read_jsonl(path: str):
    """Parse a VPT jsonl line-by-line. Returns list[dict]. Tolerates blank
    lines but not malformed JSON."""
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _extract_actions_camera(records: list, num_frames: int):
    """From a list of jsonl dicts, build:
        mouse  : (F, 2)  fp32   — linear-compressed dx/dy
        keyboard: (F, 23) fp32   — binary
        camera : (F, 5)  fp32   — (xpos, ypos, zpos, yaw, pitch)

    Args:
        records: list of jsonl dicts (full episode).
        num_frames: clip the output to this many records (we ask for the
            full episode in the dataset, so this should equal len(records)).
    """
    F_ = num_frames
    mouse = np.zeros((F_, 2), dtype=np.float32)
    keyboard = np.zeros((F_, 23), dtype=np.float32)
    camera = np.zeros((F_, 5), dtype=np.float32)

    last_hotbar = 0
    attack_is_stuck = False
    for i in range(F_):
        r = records[i]

        # ---- mouse (Solaris compress_mouse_linear) ------------------------
        m = r["mouse"]
        dx_deg = float(m["dx"]) * CAMERA_SCALER
        dy_deg = float(m["dy"]) * CAMERA_SCALER
        # clip + scale
        dx_lin = float(np.clip(dx_deg, -MOUSE_LINEAR_CLIP, MOUSE_LINEAR_CLIP)) * MOUSE_LINEAR_SCALE
        dy_lin = float(np.clip(dy_deg, -MOUSE_LINEAR_CLIP, MOUSE_LINEAR_CLIP)) * MOUSE_LINEAR_SCALE
        mouse[i, 0] = dx_lin
        mouse[i, 1] = dy_lin

        # ---- keyboard buttons --------------------------------------------
        # Stuck-attack detection (Solaris read_act_slice_vpt does this):
        # if frame 0 already shows attack as a NEW press, treat it as stuck
        # and ignore until a fresh attack press appears.
        new_buttons = m.get("newButtons", []) or []
        if i == 0 and 0 in new_buttons:
            attack_is_stuck = True
        elif attack_is_stuck and 0 in new_buttons:
            attack_is_stuck = False
        buttons = list(m.get("buttons", []) or [])
        if attack_is_stuck and 0 in buttons:
            buttons = [b for b in buttons if b != 0]
        if 0 in buttons:
            keyboard[i, KEY_INDEX["attack"]] = 1.0
        if 1 in buttons:
            keyboard[i, KEY_INDEX["use"]] = 1.0
        if 2 in buttons:
            keyboard[i, KEY_INDEX["pickItem"]] = 1.0

        for k in r["keyboard"].get("keys", []) or []:
            name = KEYBOARD_BUTTON_MAPPING.get(k)
            if name and name in KEY_INDEX:
                keyboard[i, KEY_INDEX[name]] = 1.0

        # hotbar selection
        cur_hotbar = int(r.get("hotbar", 0))
        if cur_hotbar != last_hotbar:
            keyboard[i, KEY_INDEX[f"hotbar.{cur_hotbar + 1}"]] = 1.0
        last_hotbar = cur_hotbar

        # ---- camera (xyz + yaw/pitch) ------------------------------------
        camera[i, 0] = float(r.get("xpos", 0.0))
        camera[i, 1] = float(r.get("ypos", 0.0))
        camera[i, 2] = float(r.get("zpos", 0.0))
        camera[i, 3] = float(r.get("yaw", 0.0))
        camera[i, 4] = float(r.get("pitch", 0.0))

    return mouse, keyboard, camera


# ---------------------------------------------------------------------------
# Camera normalization
# ---------------------------------------------------------------------------
# (x, y, z) are world coordinates and can be very large (e.g. ±10000). yaw is
# unbounded (radians/degrees), pitch is in [-90, 90]. We normalize by:
#   - subtracting the FIRST-FRAME camera position from xyz so the model sees
#     deltas (translation invariant);
#   - dividing yaw/pitch by 180 (degrees) to bring them into ~[-1, 1] range
#     (with yaw periodic mod 360, just dividing keeps relative changes small).
# This matches PERSIST's "agent-centered window" treatment in spirit.
def normalize_camera(camera: np.ndarray) -> np.ndarray:
    """camera: (F, 5)  → normalized fp32 (F, 5)."""
    out = camera.astype(np.float32, copy=True)
    out[:, 0] -= camera[0, 0]
    out[:, 1] -= camera[0, 1]
    out[:, 2] -= camera[0, 2]
    # yaw/pitch are degrees → /180 brings into ~[-1,1] (yaw can wrap, that's fine)
    out[:, 3] /= 180.0
    out[:, 4] /= 180.0
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VoxelDiTDataset(Dataset):
    """Sample aligned 33-frame clips from VPT episodes.

    Args:
        clip_len: number of source frames per clip (default 33).
        rgb_chunk: chunk size used when encoding RGB latents (default 33,
            matching encode_rgb_latents.py CHUNK).
        rgb_lat_per_chunk: number of latent frames per chunk (default 9,
            matching Wan VAE temporal compression of 33 source frames).
        samples_per_epoch: nominal length used for DistributedSampler.
            Each __getitem__ resamples uniformly from valid (episode, start)
            so the actual coverage is uniform across all valid clips.
        samples_per_episode: how many clips to draw before re-loading the
            current episode (NFS amortization).
        require_rgb: if True, require both voxel and rgb latents to exist.
            If False, return zero rgb_lat for episodes without it (useful for
            smoke tests before encoding finishes).
    """

    def __init__(
        self,
        clip_len: int = 33,
        rgb_chunk: int = 33,
        rgb_lat_per_chunk: int = 9,
        samples_per_epoch: int = 200_000,
        samples_per_episode: int = 1,  # FIX 26 (2026-05-08): Solaris batch_sampler.py draws fresh episode per batch element (=1). 50 caused intra-episode action correlation across consecutive grad steps.
        voxel_lat_dir: str = VOXEL_LAT_DIR,
        rgb_lat_dir: str = RGB_LAT_DIR,
        jsonl_dir: str = JSONL_DIR,
        clip_dir: Optional[str] = None,
        clip_pf_dir: Optional[str] = None,         # v7: per-frame CLIP CLS (n_frames, 1280) — used if set
        require_rgb: bool = True,
        require_clip: bool = False,
        require_clip_pf: bool = False,             # v7: filter to episodes with per-frame CLIP
        verbose: bool = False,
        rank: int = 0,         # FIX 31 (2026-05-08): Solaris batch_sampler.py:50
        world_size: int = 1,   # partitions episodes by rank to avoid cross-rank
                               # duplicates (np.arange(rank, num_episodes, world_size)).
    ):
        super().__init__()
        assert clip_len % rgb_chunk == 0 or clip_len == rgb_chunk, (
            f"clip_len ({clip_len}) must equal rgb_chunk ({rgb_chunk}) so "
            "the slice fits inside one RGB chunk; otherwise temporal "
            "alignment with chunked-VAE latents breaks."
        )
        self.clip_len = clip_len
        self.rgb_chunk = rgb_chunk
        self.rgb_lat_per_chunk = rgb_lat_per_chunk
        self.samples_per_epoch = samples_per_epoch
        self.samples_per_episode = samples_per_episode
        self.voxel_lat_dir = voxel_lat_dir
        self.rgb_lat_dir = rgb_lat_dir
        self.jsonl_dir = jsonl_dir
        self.clip_dir = clip_dir
        self.clip_pf_dir = clip_pf_dir
        self.require_rgb = require_rgb
        self.require_clip = require_clip
        self.require_clip_pf = require_clip_pf

        # ---- build episode index --------------------------------------------
        # Each entry: (base, n_frames_voxel, n_chunks). We record n_chunks =
        # n_frames // rgb_chunk so we know how many valid clip starts exist.
        # We do NOT verify the JSONL line count (most of them match; the
        # dataset is robust to mismatch via try/except in _load_episode).
        # ---- index cache --------------------------------------------------
        # The per-episode np.load(vf)["latents"].shape[0] read costs ~100-300ms
        # on NFS — at 4947 episodes that's 8-15 minutes per launch per rank.
        # We cache the (n_frames_voxel) mapping to a small json beside the dir.
        # The cache key is "voxel_lat_dir + count of files"; if anything changes
        # we rebuild silently.
        cache_path = os.path.join(voxel_lat_dir, ".n_frames_index.json")
        n_frames_cache = {}
        try:
            import json
            with open(cache_path) as fp:
                n_frames_cache = json.load(fp)
        except Exception:
            pass

        index: List[Tuple[str, int, int]] = []
        vox_files = sorted(glob.glob(os.path.join(voxel_lat_dir, "*.npz")))
        rgb_set = set(
            os.path.basename(p) for p in
            glob.glob(os.path.join(rgb_lat_dir, "*.npz"))
        )
        clip_set = set()
        if clip_dir is not None:
            clip_set = set(
                os.path.basename(p) for p in
                glob.glob(os.path.join(clip_dir, "*.npz"))
            )
        clip_pf_set = set()
        if clip_pf_dir is not None:
            clip_pf_set = set(
                os.path.basename(p) for p in
                glob.glob(os.path.join(clip_pf_dir, "*.npz"))
            )
        n_added = 0
        for vf in vox_files:
            base = Path(vf).stem
            jsonl_path = os.path.join(jsonl_dir, base + ".jsonl")
            if not os.path.exists(jsonl_path):
                continue
            has_rgb = (base + ".npz") in rgb_set
            if require_rgb and not has_rgb:
                continue
            has_clip = (base + ".npz") in clip_set
            if require_clip and not has_clip:
                continue
            has_clip_pf = (base + ".npz") in clip_pf_set
            if require_clip_pf and not has_clip_pf:
                continue
            # cheap header read for n_frames (cached after first run)
            if base in n_frames_cache:
                n_frames_voxel = n_frames_cache[base]
            else:
                try:
                    with np.load(vf) as z:
                        n_frames_voxel = int(z["latents"].shape[0])
                    n_frames_cache[base] = n_frames_voxel
                    n_added += 1
                except Exception:
                    continue
            n_chunks = n_frames_voxel // rgb_chunk
            if n_chunks < 1:
                continue
            index.append((base, n_frames_voxel, n_chunks))
        # Persist cache (rank-0-only writes via best-effort write)
        if n_added > 0:
            try:
                import json, tempfile
                tmp = cache_path + f".{os.getpid()}.tmp"
                with open(tmp, "w") as fp:
                    json.dump(n_frames_cache, fp)
                os.replace(tmp, cache_path)
            except Exception:
                pass
        # FIX 31 (2026-05-08): partition by DDP rank to match Solaris's
        # batch_sampler.py:50 (np.arange(rank, num_episodes, world_size)).
        # Without this, every rank samples from the same full pool, causing
        # cross-rank duplicates and reducing effective per-step diversity.
        self.rank = rank
        self.world_size = world_size
        if world_size > 1:
            index = index[rank::world_size]
        self.index = index
        if verbose:
            print(
                f"[VoxelDiTDataset] {len(self.index)} valid episodes "
                f"(rank={rank}/{world_size}, require_rgb={require_rgb}, "
                f"require_clip={require_clip}, "
                f"clip_dir={'set' if clip_dir else 'None'})"
            )

        # Worker-local cache: most-recently-loaded episode.
        self._cache_base = None
        self._cache_data = None  # dict with vox_lat (np), rgb_lat (np), mouse, kb, cam
        self._cache_used = 0
        self._rng = None

    def __len__(self):
        return self.samples_per_epoch

    # ------------------------------------------------------------------
    def _ensure_rng(self):
        if self._rng is None:
            wi = get_worker_info()
            seed = (wi.id if wi else 0) * 1_000_003 + (int(time.time() * 1e6) & 0x7FFFFFFF)
            self._rng = np.random.default_rng(seed)

    def _load_episode(self, base: str):
        """Return (vox_lat, rgb_lat or None, mouse, keyboard, camera) for
        a full episode, or None if any required file is missing/malformed."""
        try:
            vf = os.path.join(self.voxel_lat_dir, base + ".npz")
            with np.load(vf) as z:
                vox_lat = np.asarray(z["latents"])  # (N, 48, 12, 12, 12) fp16
            n_frames_voxel = vox_lat.shape[0]

            jsonl_path = os.path.join(self.jsonl_dir, base + ".jsonl")
            records = _read_jsonl(jsonl_path)
            n_use = min(n_frames_voxel, len(records))
            if n_use < self.clip_len:
                return None
            mouse, keyboard, camera = _extract_actions_camera(records, n_use)

            rgb_lat = None
            rgb_path = os.path.join(self.rgb_lat_dir, base + ".npz")
            if os.path.exists(rgb_path):
                with np.load(rgb_path) as z:
                    rgb_lat = np.asarray(z["latents"])  # (16, F_lat, 27, 48) fp16
            else:
                if self.require_rgb:
                    return None

            clip_lat = None
            if self.clip_dir is not None:
                clip_path = os.path.join(self.clip_dir, base + ".npz")
                if os.path.exists(clip_path):
                    with np.load(clip_path) as z:
                        clip_lat = np.asarray(z["clip"])  # (N_chunks, 257, 1280) fp16
                elif self.require_clip:
                    return None

            # v7: per-frame CLIP CLS — (n_frames, 1280) fp16, sliced per-chunk in __getitem__
            clip_pf_lat = None
            if self.clip_pf_dir is not None:
                clip_pf_path = os.path.join(self.clip_pf_dir, base + ".npz")
                if os.path.exists(clip_pf_path):
                    with np.load(clip_pf_path) as z:
                        clip_pf_lat = np.asarray(z["clip_cls"])  # (n_frames, 1280) fp16
                elif self.require_clip_pf:
                    return None

            # Trim everything to n_use frames so indexing is consistent.
            vox_lat = vox_lat[:n_use]
            return dict(
                vox=vox_lat, rgb=rgb_lat, clip=clip_lat, clip_pf=clip_pf_lat,
                mouse=mouse, keyboard=keyboard, camera=camera, n_frames=n_use,
            )
        except Exception:
            return None

    def _refresh_cache(self):
        for _ in range(8):
            i = int(self._rng.integers(len(self.index)))
            base, _, _ = self.index[i]
            data = self._load_episode(base)
            if data is None:
                continue
            self._cache_base = base
            self._cache_data = data
            self._cache_used = 0
            return True
        return False

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        self._ensure_rng()
        if (self._cache_data is None or
                self._cache_used >= self.samples_per_episode):
            ok = self._refresh_cache()
            if not ok:
                # Should be very rare; emit a sentinel so the trainer can skip
                return self._dummy_sample()

        d = self._cache_data
        n_frames = d["n_frames"]
        # FIX 20 (2026-05-08): shift voxel slice by +1 so voxel[i] is the result
        # of action[i] (Solaris minecraft.py:211 explicitly does
        # `frames = video.get_batch(range(start+1, stop+1))` with the comment
        # "Shift frames to the right by one so that frame at t is influenced by
        # action at t."). Previously voxel[i] preceded action[i], making action
        # conditioning causally meaningless during training. Reduce n_chunks by
        # one slot to keep `t1+1 <= n_frames`.
        n_chunks = (n_frames - 1) // self.rgb_chunk
        if n_chunks < 1:
            # Shouldn't happen — index excludes these — but guard anyway.
            ok = self._refresh_cache()
            if not ok:
                return self._dummy_sample()
            d = self._cache_data
            n_chunks = (d["n_frames"] - 1) // self.rgb_chunk

        # FIX 29 (2026-05-08): skip first START_OFFSET frames per Solaris
        # batch_sampler.py:2 (skip first 10s = 200 frames @ 20fps loading screen).
        # Floor to chunk-aligned start to keep RGB indexing exact.
        START_OFFSET = 200
        c_min = (START_OFFSET + self.rgb_chunk - 1) // self.rgb_chunk  # ceil
        c_min = min(c_min, n_chunks - 1)  # don't filter all chunks if episode short
        c = int(self._rng.integers(c_min, n_chunks)) if n_chunks > c_min else int(self._rng.integers(n_chunks))
        t0 = c * self.rgb_chunk
        t1 = t0 + self.clip_len

        # voxel: (48, 33, 12, 12, 12) — sliced at [t0+1:t1+1] (post-action).
        # actions stay at [t0:t1] so action[i] is the cause of voxel[i].
        # rgb_clip and clip_ctx remain at chunk-aligned (t0) raw indices —
        # this leaves a 1-raw-frame visual mismatch on those (rgb is render-only
        # output; clip is a coarse single-feature scene anchor) which is much
        # smaller than the action↔voxel pairing bug it fixes.
        vox_clip = d["vox"][t0 + 1:t1 + 1]  # (33, 48, 12, 12, 12) fp16
        # permute (F, C, X, Y, Z) -> (C, F, X, Y, Z)
        vox_clip = np.transpose(vox_clip, (1, 0, 2, 3, 4))  # (48, 33, 12, 12, 12)

        # rgb: (16, F_lat, 27, 48); take chunk c → frames [c*9 : c*9+9]
        if d["rgb"] is not None:
            f_lat_start = c * self.rgb_lat_per_chunk
            f_lat_stop = f_lat_start + self.rgb_lat_per_chunk
            rgb_clip = d["rgb"][:, f_lat_start:f_lat_stop]  # (16, 9, 27, 48)
            if rgb_clip.shape[1] != self.rgb_lat_per_chunk:
                # last partial chunk — skip and resample
                self._cache_used += 1
                return self.__getitem__(idx)
        else:
            rgb_clip = np.zeros(
                (16, self.rgb_lat_per_chunk, 27, 48), dtype=np.float16,
            )

        mouse_clip = d["mouse"][t0:t1].copy()       # (33, 2)
        keyboard_clip = d["keyboard"][t0:t1].copy() # (33, 23)
        camera_clip = normalize_camera(d["camera"][t0:t1])  # (33, 5)

        # v7: prefer per-frame CLIP CLS if available. Slice [t0+1:t1+1] to match
        # the voxel slicing (FIX 20 post-action alignment). Returns (33, 1280).
        # Otherwise fall back to the chunk-start 257-spatial-token form.
        if d.get("clip_pf") is not None:
            clip_pf = d["clip_pf"]
            if t1 + 1 <= clip_pf.shape[0]:
                clip_t = torch.from_numpy(clip_pf[t0 + 1:t1 + 1].copy())  # (33, 1280)
            else:
                clip_t = torch.zeros(self.clip_len, 1280, dtype=torch.float16)
        elif d.get("clip") is not None and c < d["clip"].shape[0]:
            clip_ctx = d["clip"][c].copy()  # (257, 1280) fp16
            clip_t = torch.from_numpy(clip_ctx)
        else:
            clip_t = torch.zeros(257, 1280, dtype=torch.float16)

        self._cache_used += 1

        return dict(
            voxel_lat=torch.from_numpy(vox_clip.copy()),       # fp16
            rgb_lat=torch.from_numpy(rgb_clip.copy()),          # fp16
            clip_context=clip_t,                                  # fp16 (257, 1280)
            actions_mouse=torch.from_numpy(mouse_clip),         # fp32
            actions_keyboard=torch.from_numpy(keyboard_clip),    # fp32
            camera=torch.from_numpy(camera_clip),                # fp32
        )

    # ------------------------------------------------------------------
    def _dummy_sample(self):
        """Returned only when refresh fails — keeps DataLoader alive."""
        return dict(
            voxel_lat=torch.zeros(48, self.clip_len, 12, 12, 12, dtype=torch.float16),
            rgb_lat=torch.zeros(16, self.rgb_lat_per_chunk, 27, 48, dtype=torch.float16),
            clip_context=torch.zeros(257, 1280, dtype=torch.float16),
            actions_mouse=torch.zeros(self.clip_len, 2),
            actions_keyboard=torch.zeros(self.clip_len, 23),
            camera=torch.zeros(self.clip_len, 5),
        )


# ---------------------------------------------------------------------------
# CLI sanity check.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--require_rgb", action="store_true")
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()

    ds = VoxelDiTDataset(
        require_rgb=args.require_rgb,
        samples_per_epoch=10,
        samples_per_episode=2,
        verbose=True,
    )
    print(f"valid episodes: {len(ds.index)}")
    for i in range(args.samples):
        s = ds[i]
        print(f"sample {i}:")
        for k, v in s.items():
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} "
                  f"min={v.float().min().item():.3f} "
                  f"max={v.float().max().item():.3f}")
