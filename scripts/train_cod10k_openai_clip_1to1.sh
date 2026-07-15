#!/usr/bin/env bash
# Post-train OpenAI CLIP on one positive image-caption pair per COD10K CAM image.
set -euo pipefail
export PYTHONNOUSERSITE=1

if [ $# -ne 1 ]; then
  echo "usage: bash scripts/train_cod10k_openai_clip_1to1.sh <clip|cliprefine>" >&2
  exit 1
fi

LOSS="$1"
case "${LOSS}" in
  clip|cliprefine) ;;
  *)
    echo "unsupported loss: ${LOSS}; expected clip or cliprefine" >&2
    exit 1
    ;;
esac

REPO_DIR="/home/msko021220/biomedclip_finetuning"
PYTHON_BIN="${PYTHON_BIN:-/home/msko021220/.conda/envs/busi2/bin/python}"
OPENCLIP_SRC="${OPENCLIP_SRC:-/home/msko021220/finegrained-vlm-training/biomedclip_finetuning/open_clip/src}"
TRAIN_CSV="${TRAIN_CSV:-/home/msko021220/dataset/cod10k-dataset/COD10K-v3/metadata/cod10k_train_cam_dpo_1caption_diffusion_bgpos_vs_bgonly.csv}"
BASE_CKPT="${BASE_CKPT:-/home/msko021220/dataset/clip_reference_checkpoints/openai_clip_vit_b_32_quickgelu_openclip_state_dict.pt}"
OPENCLIP_MODEL="${OPENCLIP_MODEL:-ViT-B-32-quickgelu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/checkpoints}"
RUN_NAME="${RUN_NAME:-cod10k_openai_${LOSS}_1to1_image_specific_$(date +%Y%m%d_%H%M%S)}"
GPU="${GPU:-0}"

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_FREQ="${SAVE_FREQ:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

for required in "${PYTHON_BIN}" "${TRAIN_CSV}" "${BASE_CKPT}"; do
  if [ ! -e "${required}" ]; then
    echo "missing required path: ${required}" >&2
    exit 1
  fi
done

OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU}"

CMD=(
  "${PYTHON_BIN}" train.py
  --loss "${LOSS}"
  --train-csv "${TRAIN_CSV}"
  --csv-img-key filename
  --csv-caption-key Caption
  --model-backend openclip
  --openclip-model "${OPENCLIP_MODEL}"
  --openclip-src "${OPENCLIP_SRC}"
  --model-checkpoint "${BASE_CKPT}"
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
