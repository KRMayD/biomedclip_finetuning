#!/usr/bin/env bash
set -euo pipefail

# Evaluate the two 1:1 image-specific post-trained models with the exact
# COD10K saliency configuration used for the corresponding DPO experiment.
STAMP="${1:-$(date +%m%d_%H%M)}"
PYTHON_BIN="/home/msko021220/.conda/envs/busi2/bin/python"
FG_ROOT="/home/msko021220/finegrained-vlm-training"
OPENCLIP_SRC="${FG_ROOT}/biomedclip_finetuning/open_clip/src"
TOOLS_DIR="${FG_ROOT}/tools"
IMAGE_DIR="/home/msko021220/dataset/cod10k-dataset/COD10K-v3/Test/Image"
MASK_DIR="/home/msko021220/dataset/cod10k-dataset/COD10K-v3/Test/GT_Object"
RESULT_ROOT="${FG_ROOT}/outputs/cod10k_openai_clip_1to1_posttrain_vb1_vv03_l8_contour1_${STAMP}"

CLIP_CKPT="/home/msko021220/biomedclip_finetuning/checkpoints/cod10k_openai_clip_1to1_image_specific_0711_0702/final_model.pt"
REFINE_CKPT="/home/msko021220/biomedclip_finetuning/checkpoints/cod10k_openai_cliprefine_1to1_image_specific_0711_0702/final_model.pt"
DPO_METRICS="${FG_ROOT}/outputs/cod10k_diffusion_bg1to1_basepos_vs_bgonly_w0500_0000_1000_0250_vb1_vv03_l8_contour1_0703_0259/metrics_comparison.json"
CPU_THREADS="${CPU_THREADS:-8}"
BRAIN_POSTPROCESS_PID="${BRAIN_POSTPROCESS_PID:-}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${OPENCLIP_SRC}:${FG_ROOT}/saliency_maps:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export VECLIB_MAXIMUM_THREADS="${CPU_THREADS}"
export BLIS_NUM_THREADS="${CPU_THREADS}"

for path in "${CLIP_CKPT}" "${REFINE_CKPT}" "${IMAGE_DIR}" "${MASK_DIR}" "${DPO_METRICS}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${RESULT_ROOT}/logs"
echo "===== START ${STAMP} $(date -Is) ====="
echo "result root: ${RESULT_ROOT}"
echo "cpu threads per worker: ${CPU_THREADS}"

brain_paused=0
resume_brain_postprocess() {
  if [[ "${brain_paused}" -eq 1 ]] && kill -0 "${BRAIN_POSTPROCESS_PID}" 2>/dev/null; then
    kill -CONT "${BRAIN_POSTPROCESS_PID}"
    echo "resumed Brain MRI postprocess pid=${BRAIN_POSTPROCESS_PID} at $(date -Is)"
  fi
}
trap resume_brain_postprocess EXIT

if [[ -n "${BRAIN_POSTPROCESS_PID}" ]] && kill -0 "${BRAIN_POSTPROCESS_PID}" 2>/dev/null; then
  kill -STOP "${BRAIN_POSTPROCESS_PID}"
  brain_paused=1
  echo "paused Brain MRI postprocess pid=${BRAIN_POSTPROCESS_PID} at $(date -Is)"
fi

# Each GPU processes one deterministic quarter of the 2,026-image test set.
# Four persistent queues avoid CPU and GPU oversubscription while keeping all GPUs busy.
run_shard() {
  local model_key="$1"
  local checkpoint="$2"
  local shard="$3"
  local gpu="$4"
  local out_dir="${RESULT_ROOT}/${model_key}"
  local log="${RESULT_ROOT}/logs/${model_key}.image_shard${shard}.eval.log"

  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${PYTHON_BIN}" "${TOOLS_DIR}/cod10k_saliency_hparam_search.py" \
      --image-dir "${IMAGE_DIR}" \
      --mask-dir "${MASK_DIR}" \
      --checkpoint "${checkpoint}" \
      --output-dir "${out_dir}" \
      --model "ViT-B-32-quickgelu" \
      --sample-count 0 \
      --seed 42 \
      --device cuda \
      --fixed-vbeta 1.0 \
      --fixed-vvar 0.3 \
      --fixed-vlayer 8 \
      --image-shard-index "${shard}" \
      --image-shard-count 4 \
      --cpu-threads "${CPU_THREADS}" \
      --save-preds
  ) > "${log}" 2>&1
}

queue_pids=()
for shard in 0 1 2 3; do
  (
    run_shard "clip_1to1_image_specific" "${CLIP_CKPT}" "${shard}" "${shard}"
    run_shard "cliprefine_1to1_image_specific" "${REFINE_CKPT}" "${shard}" "${shard}"
  ) &
  pid=$!
  queue_pids+=("${pid}")
  echo "${pid}" > "${RESULT_ROOT}/logs/gpu${shard}.queue.pid"
  echo "launched gpu_queue=${shard} pid=${pid} models=clip,cliprefine"
done

failed=0
for pid in "${queue_pids[@]}"; do
  if ! wait "${pid}"; then
    echo "Evaluation queue failed: pid=${pid}" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

export RESULT_ROOT DPO_METRICS CLIP_CKPT REFINE_CKPT
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["RESULT_ROOT"])
dpo_metrics = json.loads(Path(os.environ["DPO_METRICS"]).read_text(encoding="utf-8"))

models = {
    "clip_1to1_image_specific": os.environ["CLIP_CKPT"],
    "cliprefine_1to1_image_specific": os.environ["REFINE_CKPT"],
}
results = {}
for key, checkpoint in models.items():
    model_dir = root / key
    details = sorted(model_dir.glob("details_shard*_image*.csv"))
    if len(details) != 4:
        raise SystemExit(f"{key}: expected 4 shards, found {len(details)}")
    df = pd.concat(
        [pd.read_csv(path).assign(source_detail_file=str(path)) for path in details],
        ignore_index=True,
    ).sort_values("sample").reset_index(drop=True)
    if len(df) != 2026 or df["sample"].nunique() != 2026:
        raise SystemExit(
            f"{key}: expected 2,026 unique test samples, got {len(df)} rows / {df['sample'].nunique()} unique"
        )
    for col in ("dsc", "nsd", "gt_area", "pred_area"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    details_all = model_dir / "details_all.csv"
    df.to_csv(details_all, index=False)
    results[key] = {
        "checkpoint": checkpoint,
        "details_all": str(details_all),
        "prediction_dir": str(model_dir / "preds" / "vb1.0_vv0.3_l8"),
        "num_samples": int(len(df)),
        "num_unique_samples": int(df["sample"].nunique()),
        "mean_dsc": float(df["dsc"].mean()),
        "std_dsc": float(df["dsc"].std(ddof=0)),
        "median_dsc": float(df["dsc"].median()),
        "mean_nsd": float(df["nsd"].mean()),
        "std_nsd": float(df["nsd"].std(ddof=0)),
        "median_nsd": float(df["nsd"].median()),
    }

baseline = dpo_metrics["baseline_clip"]
dpo = {
    "mean_dsc": dpo_metrics["mean_dsc"],
    "mean_nsd": dpo_metrics["mean_nsd"],
    "checkpoint": dpo_metrics["checkpoint"],
}
for result in results.values():
    result["delta_vs_baseline_clip"] = {
        "dsc": result["mean_dsc"] - baseline["mean_dsc"],
        "nsd": result["mean_nsd"] - baseline["mean_nsd"],
    }
    result["delta_vs_dpo_1to1_image_specific"] = {
        "dsc": result["mean_dsc"] - dpo["mean_dsc"],
        "nsd": result["mean_nsd"] - dpo["mean_nsd"],
    }

report = {
    "test_protocol": {
        "dataset": "COD10K CAM held-out test split",
        "num_samples": 2026,
        "prompt": "animal-specific prompt parsed from each filename",
        "vbeta": 1.0,
        "vvar": 0.3,
        "vlayer": 8,
        "contour": 1,
        "postprocess": "2-cluster kmeans plus connected component containing the saliency maximum",
        "nsd_tolerance_px": 2.0,
    },
    "baseline_openai_clip": baseline,
    "dpo_1to1_image_specific": dpo,
    **results,
}
(root / "metrics_comparison.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
rows = [
    {"model": "baseline_openai_clip", **baseline},
    {"model": "dpo_1to1_image_specific", **dpo},
]
for key, result in results.items():
    rows.append({"model": key, **result})
pd.DataFrame(rows).to_csv(root / "metrics_comparison.csv", index=False)
print(json.dumps({key: {"mean_dsc": value["mean_dsc"], "mean_nsd": value["mean_nsd"]} for key, value in results.items()}, indent=2))
PY

echo "===== COMPLETE $(date -Is) ====="
echo "results: ${RESULT_ROOT}/metrics_comparison.json"
