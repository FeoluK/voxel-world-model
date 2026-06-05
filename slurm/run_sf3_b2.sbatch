#!/bin/bash
#SBATCH -p small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH -J v14b2
#SBATCH -o /home/flukol/v14_tc_run2_ckpts/v14b2_%j.out
mkdir -p /home/flukol/v14_tc_run2_ckpts
srun --container-image="nvcr.io#nvidia/pytorch:24.12-py3" --container-mounts=/home/flukol:/home/flukol --container-workdir=/home/flukol/v12/code bash -lc "
export V12_VOXEL_LAT_DIR=/home/flukol/treechop_gen/latents
export V12_JSONL_DIR=/home/flukol/treechop_gen/jsonl
export V12_VAE_PT=/home/flukol/v12/code/vae_final.pt
export V14_VOCAB=/home/flukol/v12/code/voxel_vocab_treechop.npz
export V12_CODE_DIR=/home/flukol/v12/code PYTHONUNBUFFERED=1
torchrun --standalone --nproc-per-node=4 train_v14_sf3.py \
  --phase B --dim 192 --batch 8 --frames_per_clip 4 --grad_ckpt 1 --lr 2e-4 --warmup 500 --curriculum 1 \
  --phase_steps 2500 --sf_prob_lo 0.0 --sf_prob_hi 0.6 --K_lo 1 --K_hi 24 --max_horizon 64 \
  --sf_gen_steps 8 --sf_temp 0.6 --buffer_size 256 --buffer_refresh 4 --buffer_gen_clips 4 \
  --frontier_weight 3.0 --balanced_ce 1.0 --kl_weight 0.0 \
  --ckpt_dir /home/flukol/v14_tc_run2_ckpts --ckpt_minutes 20 \
  --resume /home/flukol/v14_treechop_fixed_ckpts/v14sf3_A_dim192_step004703.pt
"
