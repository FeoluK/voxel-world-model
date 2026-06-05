#!/bin/bash
#SBATCH --partition=small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=v14stageC
#SBATCH --output=/home/flukol/v14_tc_stageC_ckpts/stageC_%j.out

# v14 Stage C — TEMPORAL multi-frame self-forcing, 4-GPU production run.
# Single-node only (cluster MaxNodes=1); torchrun --standalone.
set -euo pipefail

export V12_CODE_DIR=/home/flukol/v12/code
export V12_VOXEL_LAT_DIR=/home/flukol/treechop_gen/latents
export V12_JSONL_DIR=/home/flukol/treechop_gen/jsonl
export V12_VAE_PT=/home/flukol/v12/code/vae_final.pt
export V14_VOCAB=/home/flukol/v12/code/voxel_vocab_treechop.npz   # corrected vocab, K=158
export PYTHONUNBUFFERED=1

mkdir -p /home/flukol/v14_tc_stageC_ckpts

# Warm-start from the NEWEST old-SF generator ckpt, resolved at submit time.
# (Self-resume from v14_tc_stageC_ckpts takes precedence if a Stage C ckpt already exists.)
RESUME=$(ls -t /home/flukol/v14_tc_oldSF_ckpts/v14_maskgit_dim192_step*.pt | head -1)
echo "warm-start resume = $RESUME"

srun --gres=gpu:4 --cpus-per-task=32 \
  --container-image='nvcr.io#nvidia/pytorch:24.12-py3' \
  --container-mounts=/home/flukol:/home/flukol \
  --container-workdir=/home/flukol/v12/code \
  torchrun --standalone --nproc-per-node=4 train_v14_stageC.py \
    --dim 192 --batch 8 --frames_per_clip 4 --grad_ckpt 1 \
    --resume "$RESUME" \
    --horizon_lo 2 --horizon_hi 64 --horizon_ramp_steps 4000 \
    --self_force_prob 0.8 --sf_gen_steps 5 \
    --slab_frac_lo 0.20 --slab_frac_hi 0.40 --curriculum 1 \
    --lr 2e-4 --warmup 500 \
    --ckpt_dir /home/flukol/v14_tc_stageC_ckpts --ckpt_minutes 20
