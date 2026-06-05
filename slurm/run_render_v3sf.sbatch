#!/bin/bash
# render_v3 SELF-FORCING (DAgger) fine-tune. 4-GPU production launcher.
# Warm-starts from the BEST available v3 BASE ckpt (v3_latest.pt).
# Override at submit: sbatch --export=ALL,NGPU=4,STEPS=20000 run_render_v3sf.sbatch
#SBATCH -p small
#SBATCH -J rv3sf
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH -t 24:00:00
#SBATCH -o /home/flukol/render_v2_treechop/ckpt_v3sf/v3sf_%j.out

set -euo pipefail

NGPU=${NGPU:-4}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-4}              # per-GPU batch (short clips are heavier)
RAST=${RAST:-96}
BASE=${BASE:-160}
PEEL_K=${PEEL_K:-2}
CLIP_FRAMES=${CLIP_FRAMES:-9}
W_TEMPORAL=${W_TEMPORAL:-0.1}
W_PERCEP=${W_PERCEP:-0.05}
PERCEP_FRAMES=${PERCEP_FRAMES:-2}
# ---- self-forcing schedule ----
SF_PROB_HI=${SF_PROB_HI:-0.5}
PHASE_STEPS=${PHASE_STEPS:-4000}
SF_ODE_STEPS=${SF_ODE_STEPS:-12}
# ---- ckpt ----
KEEP_EVERY=${KEEP_EVERY:-2000}
CKPT_MIN=${CKPT_MIN:-20}
CKPT_DIR=${CKPT_DIR:-/home/flukol/render_v2_treechop/ckpt_v3sf}
# warm-start from the BEST available v3 BASE ckpt (auto-self-resumes v3sf_latest.pt if present)
RESUME=${RESUME:-/home/flukol/render_v2_treechop/ckpt_v3/v3_latest.pt}
EXTRA=${EXTRA:-}              # e.g. "--no_percep" or "--grad_ckpt"

# SAME env exports as the base run_render_v3.sbatch
export V12_VOXEL_LAT_DIR=${V12_VOXEL_LAT_DIR:-/home/flukol/treechop_gen/latents}
export V12_RGB_LAT_DIR=${V12_RGB_LAT_DIR:-/home/flukol/treechop_gen/rgb_latents}
export V12_JSONL_DIR=${V12_JSONL_DIR:-/home/flukol/treechop_gen/jsonl}
export VOX_VAE=${VOX_VAE:-/home/flukol/v12/code/vae_final.pt}
export VOX_VOCAB=${VOX_VOCAB:-/home/flukol/v12/code/voxel_vocab.npz}
export WAN_MODULE=${WAN_MODULE:-/home/flukol/v12/wan/modules/vae.py}
export WAN_PTH=${WAN_PTH:-/home/flukol/v12/mg2_weights/Wan2.1_VAE.pth}
export USE_RASTER=1

mkdir -p "${CKPT_DIR}"

srun --container-image="nvcr.io#nvidia/pytorch:24.12-py3" \
     --container-mounts=/home/flukol:/home/flukol \
     --container-workdir=/home/flukol/render_v2 \
     bash -lc "export PYTHONPATH=/home/flukol/v12/code:\$PYTHONPATH; \
       torchrun --standalone --nproc-per-node=${NGPU} \
         /home/flukol/render_v2/train_render_v3_sf.py \
         --steps ${STEPS} --batch ${BATCH} --rast ${RAST} --base ${BASE} \
         --peel_k ${PEEL_K} --clip_frames ${CLIP_FRAMES} \
         --w_temporal ${W_TEMPORAL} --w_percep ${W_PERCEP} --percep_frames ${PERCEP_FRAMES} \
         --sf_prob_hi ${SF_PROB_HI} --phase_steps ${PHASE_STEPS} --sf_ode_steps ${SF_ODE_STEPS} \
         --keep_every ${KEEP_EVERY} --ckpt_min ${CKPT_MIN} \
         --num_workers 14 --log_every 50 \
         --ckpt_dir ${CKPT_DIR} --resume ${RESUME} ${EXTRA}"
echo RV3SF_TRAIN_DONE
