#!/usr/bin/env bash
set -euo pipefail

STEP="4000000"
MODEL_SIZE='XL'
CFG_SCALE=2.3
CLS_CFG_SCALE=2.3
GH=0.85
NUM_STEP=250

SAMPLE_ROOT="/yourpath"
IMAGE_DIR="${SAMPLE_ROOT}/SiT-${MODEL_SIZE}-2-${STEP}_jt-size-256-vae-jt-cfg-${CFG_SCALE}-seed-0-sde-${GH}-${CLS_CFG_SCALE}"
SAMPLE_NPZ="${IMAGE_DIR}.npz"
REF_BATCH="/yourpath/imagenet.npz"

python tools/folder_to_npz.py \
  --input-dir "${IMAGE_DIR}" \
  --output-npz "${SAMPLE_NPZ}"

python evaluations/evaluator.py \
  --ref_batch "${REF_BATCH}" \
  --sample_batch "${SAMPLE_NPZ}" \
  --save_path "${SAMPLE_ROOT}/" \
  --cfg_cond 1 \
  --step "${STEP}" \
  --num_steps "${NUM_STEP}" \
  --cfg "${CFG_SCALE}" \
  --cls_cfg "${CLS_CFG_SCALE}" \
  --gh "${GH}"