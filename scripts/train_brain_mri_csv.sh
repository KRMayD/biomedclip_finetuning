#!/bin/bash
set -euo pipefail

LOSS="${LOSS:-clip}"
GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-brain_mri_${LOSS}_$(date +%Y%m%d_%H%M%S)}"

TRAIN_CSV="${TRAIN_CSV:-/home/msko021220/finegrained-vlm-training/data/brain_mri_dpo_sd_no_figshare_12856.csv}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:-hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/msko021220/biomedclip_finetuning/checkpoints}"

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONNOUSERSITE=1

CMD=(
  /home/msko021220/.conda/envs/busi2/bin/python train.py
  --loss "${LOSS}"
  --train-csv "${TRAIN_CSV}"
  --csv-img-key filename
  --csv-caption-key Caption
  --model-checkpoint "${MODEL_CHECKPOINT}"
  --output-dir "${OUTPUT_DIR}"
  --batch-size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --num-workers "${NUM_WORKERS}"
  --save-freq "${SAVE_FREQ}"
  --gpu 0
)

if [ -n "${MAX_SAMPLES}" ]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi

printf 'Running command:\n' | tee "${OUTPUT_DIR}/command.log"
printf '  %q' "${CMD[@]}" | tee -a "${OUTPUT_DIR}/command.log"
printf '\n' | tee -a "${OUTPUT_DIR}/command.log"

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
