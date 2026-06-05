#!/bin/bash
#SBATCH -p small
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -t 02:00:00
#SBATCH -J buildvocab
#SBATCH -o /home/flukol/v12/code/buildvocab_%j.out
# Rebuild a CORRECTED compact vocab from the treechop latents.
# 1 GPU, ~minutes (decode-only histogram over 60 clips). Run AFTER the current
# tcgen job; this does NOT train, it only writes voxel_vocab_treechop.npz.
srun --container-image="nvcr.io#nvidia/pytorch:24.12-py3" \
     --container-mounts=/home/flukol:/home/flukol \
     --container-workdir=/home/flukol/v12/code bash -lc "
export PYTHONUNBUFFERED=1
python build_voxel_vocab.py \
  --latents /home/flukol/treechop_gen/latents \
  --vae     /home/flukol/v12/code/vae_final.pt \
  --palette /home/flukol/v12/code/global_palette.json \
  --out     /home/flukol/v12/code/voxel_vocab_treechop.npz \
  --hist_cache /home/flukol/v12/code/treechop_id_hist.npy \
  --frames_per_clip 24 --coverage 0.995
"
echo BUILDVOCAB_DONE
