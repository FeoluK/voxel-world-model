#!/bin/bash
#SBATCH -p small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH -t 24:00:00
#SBATCH -J v14oldsf
#SBATCH -o /home/flukol/v14_tc_oldSF_ckpts/oldsf_%j.out
mkdir -p /home/flukol/v14_tc_oldSF_ckpts
srun --container-image="nvcr.io#nvidia/pytorch:24.12-py3" --container-mounts=/home/flukol:/home/flukol --container-workdir=/home/flukol/v12/code bash -lc "
export V12_VOXEL_LAT_DIR=/home/flukol/treechop_gen/latents
export V12_JSONL_DIR=/home/flukol/treechop_gen/jsonl
export V12_VAE_PT=/home/flukol/v12/code/vae_final.pt
export V14_VOCAB=/home/flukol/v12/code/voxel_vocab_treechop.npz
export V12_CODE_DIR=/home/flukol/v12/code PYTHONUNBUFFERED=1
torchrun --standalone --nproc-per-node=4 train_v14_sf.py \
  --dim 192 --batch 8 --frames_per_clip 4 --grad_ckpt 1 --curriculum 1 \
  --self_force_prob 0.4 --sf_gen_steps 5 --lr 2e-4 --warmup 500 \
  --ckpt_dir /home/flukol/v14_tc_oldSF_ckpts --ckpt_minutes 20 \
  --resume /home/flukol/v14_treechop_fixed_ckpts/v14sf3_A_dim192_step004703.pt
"
