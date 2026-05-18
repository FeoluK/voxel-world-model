#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/flukol/mg_pt_voxel/fulldata_v7_%j.out
#SBATCH --error=/scratch/users/flukol/mg_pt_voxel/fulldata_v7_%j.err
set -e
source /scratch/users/flukol/miniconda_fresh/etc/profile.d/conda.sh
conda activate solaris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

VOX=/scratch/users/flukol/mg_pt_voxel
SCRIPT=$VOX/train_voxel_dit_solaris_vpred_v7.py
CKPT_DIR=$VOX/dit_ckpts_solaris_vpred_v7_full
mkdir -p $CKPT_DIR

# v7: same arch as v6 (position-only action stream), but per-frame CLIP CLS
# (33 tokens/chunk) instead of v6's single pooled token. Same lr/warmup/batch.
torchrun --nproc_per_node=4 --master_port=29561 $SCRIPT \
  --steps 30000 \
  --batch 2 \
  --grad_accum 4 \
  --lr 5e-4 \
  --warmup 2000 \
  --grad_clip 1.0 \
  --ckpt_dir $CKPT_DIR \
  --ckpt_minutes 30 \
  --log_every 20
