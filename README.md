# Voxel World Model — v7 checkpoint

A generative 3D world model for embodied agents. A diffusion transformer
predicts how a voxel world evolves over time, conditioned on a single
first-frame 3D anchor plus the agent's action/position trajectory — operating
directly in 3D rather than on rendered 2D frames.

This repository is a **progress checkpoint**, not a turnkey runnable repo. The
training data (~155 GB of extracted Minecraft-style voxel sequences), encoded
latents, and model weights (~14 GB per checkpoint) live on cluster storage and
are intentionally **not** committed here. What is here is the model/training/
evaluation code and demonstration media for our best model to date ("v7").

## Result (v7, 270M params, v-prediction)

Evaluated on held-out chunks the model never trained on. Numbers below are for
a representative walking chunk (`renders/v7_ablation_c3/metrics.json`):

| condition            | acc_all | acc_nonair | note |
|----------------------|---------|------------|------|
| **full conditioning**| **0.836** | **0.704** | model vs ground truth |
| zero 3D anchor       | 0.241   | 0.437      | **−0.59 acc_all** — anchor is load-bearing |
| zero CLIP context    | 0.833   | 0.699      | small effect |
| zero position        | 0.852   | 0.735      | position not dominant on low-motion chunks |

Across the broader held-out pool, per-voxel accuracy reaches **~80% (acc_all)**
and **0.69–0.84 (non-air)**, up to ~0.99 on low-motion sequences. The large
collapse when the 3D anchor is removed confirms the model learns real world
structure, not a trivial prior.

See `renders/v7_ablation_c3/` for side-by-side videos:
- `GT.mp4` — ground-truth voxel evolution
- `normal.mp4` — model prediction (full conditioning)
- `compare_zero_cond_concat.mp4` — prediction with the 3D anchor removed (collapses)
- `compare_zero_position.mp4` — prediction with the position trajectory removed

`results/v7_vs_v12_loss.png` — training loss curves (v7 vs the in-progress
simplified v12 variant).

## Code (`src/`)

| file | role |
|------|------|
| `voxel_dit_solaris_v7.py` | v7 entry point — re-exports the v6 architecture |
| `voxel_dit_solaris_v6.py` | the actual diffusion-transformer model |
| `train_voxel_dit_solaris_vpred_v7.py` | training loop (v-prediction, DDP) |
| `ablate_solaris_pool_v7.py` | evaluation/ablation harness (produces the demo media + metrics) |
| `voxel_dataset.py` | dataset loader for extracted voxel-latent sequences |
| `vae_3d.py` | 3D VAE used to decode predicted latents back to voxel ids |
| `global_palette.json` | voxel-id ↔ block-name mapping |
| `run_fulldata_v_v7.sh` | exact training launch (hyperparameters: 4×GPU, eff. batch 32, lr 5e-4, warmup 2000, v-pred) |

## Architecture (brief)

- 270M-parameter diffusion transformer operating on a 3D voxel latent space
  (a 48³ voxel grid encoded to 12³ latents by a 3D VAE).
- Conditioning: first-frame voxel anchor (`cond_concat`), agent position
  trajectory (`position_cond`), and pooled visual context.
- v-prediction objective; bidirectional (full-sequence) training. Making the
  model autoregressive and then self-forcing is the next phase of work.

## Status / next steps

The model is currently **bidirectional** — it sees the whole sequence at once,
which is good for learning but cannot generate forward in time. Planned work:
make it **autoregressive** (predict each step from only the past), then add
**self-forcing** (train on the model's own rollouts to stop error
accumulation), then build a **renderer** that projects the predicted 3D world
back into an agent's first-person view — enabling the multi-agent setting where
separate agents act in and observe one shared predicted world.
