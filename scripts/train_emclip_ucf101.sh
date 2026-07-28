#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

DATASET="ucf101_mpeg4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =========================================================
# GPU configuration
# 固定使用物理 GPU 0、1、2
# =========================================================

EMCLIP_SCRIPT_GPU_IDS="0,1,2"

# 自动设置：
# CUDA_VISIBLE_DEVICES=0,1,2
# NPROC_PER_NODE=3
source "${SCRIPT_DIR}/_gpu_env.sh"

# =========================================================
# Checkpoints
# =========================================================

# 原始 CLIP ViT-B/16 权重，用于初始化视觉和文本编码器
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/home/fuh/CLIP-models/ViT-B-16.pt}"

# 如果指定的原始 CLIP 权重不存在，则尝试仓库中的备用权重
if [[ ! -f "${CLIP_CHECKPOINT}" &&
      -f "${REPO_ROOT}/clip_vit_b_16.pth" ]]; then
  CLIP_CHECKPOINT="${REPO_ROOT}/clip_vit_b_16.pth"
fi

# K400 上训练完成的 EMCLIP 权重
K400_CHECKPOINT="${K400_CHECKPOINT:-/home/fuh/Efficient-CLIP-codex/output_dir/emclip/k400_emclip_T16_K8_20260721_001418/model_best.pth}"

# 默认加载 K400 权重
# K400 is a transfer-learning source: load model weights only and reset all
# target-dataset training state. An explicitly empty value disables it.
INIT_CHECKPOINT="${INIT_CHECKPOINT-${K400_CHECKPOINT}}"

# RESUME is only for continuing a UCF101 latest.pth/model_best.pth run.
RESUME="${RESUME:-}"

# =========================================================
# Training configuration
# =========================================================

BATCH_SIZE=${BATCH_SIZE:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-${BATCH_SIZE}}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}
PROFILE_COMPUTE=${PROFILE_COMPUTE:-1}

if [[ -z "${COVIAR_DATA_LOADER_DIR:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../pytorch-coviar/data_loader" ]]; then
    COVIAR_DATA_LOADER_DIR="$(
      cd "${SCRIPT_DIR}/../pytorch-coviar/data_loader" &&
      pwd
    )"
  else
    COVIAR_DATA_LOADER_DIR="/home/fuh/m2clip/Coviar/data_loader"
  fi
fi

MASTER_PORT=${MASTER_PORT:-29502}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_dir/emclip}
VARIANT=${VARIANT:-emclip}
T=${T:-16}
K=${K:-8}

STAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_${VARIANT}_T${T}_K${K}_${STAMP}"

mkdir -p "${OUTPUT_DIR}"

# =========================================================
# Build training command
# =========================================================

CMD=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"
  "${REPO_ROOT}/main_emclip.py"

  --model emclip_b16
  --dataset "${DATASET}"
  --emclip-variant "${VARIANT}"

  --candidate-frames "${T}"
  --selected-frames "${K}"
  --mgse-text-mode class_bank

  --epochs 30
  --lr 8e-6
  --input-size 256

  --batch-size "${BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"

  --coviar-data-loader-dir "${COVIAR_DATA_LOADER_DIR}"

  --amp
  --output-dir "${OUTPUT_DIR}"
)

# 是否关闭 pin memory
if [[ "${PIN_MEMORY}" == "0" ||
      "${PIN_MEMORY}" == "false" ||
      "${PIN_MEMORY}" == "False" ]]; then
  CMD+=(--no-pin-memory)
fi

if [[ "${PROFILE_COMPUTE}" == "1" ||
      "${PROFILE_COMPUTE}" == "true" ||
      "${PROFILE_COMPUTE}" == "True" ]]; then
  CMD+=(--profile-compute)
fi

# =========================================================
# Original CLIP checkpoint
# =========================================================

if [[ -n "${CLIP_CHECKPOINT:-}" ]]; then
  [[ -f "${CLIP_CHECKPOINT}" ]] || {
    echo "[emclip] Original CLIP checkpoint not found:" >&2
    echo "[emclip] ${CLIP_CHECKPOINT}" >&2
    exit 1
  }

  CMD+=(--clip-checkpoint "${CLIP_CHECKPOINT}")
fi

# =========================================================
# K400 EMCLIP checkpoint
# =========================================================

if [[ -n "${RESUME:-}" ]]; then
  [[ -f "${RESUME}" ]] || {
    echo "[emclip] UCF101 resume checkpoint not found:" >&2
    echo "[emclip] ${RESUME}" >&2
    exit 1
  }

  CMD+=(--resume "${RESUME}")
elif [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  [[ -f "${INIT_CHECKPOINT}" ]] || {
    echo "[emclip] K400 initialization checkpoint not found:" >&2
    echo "[emclip] ${INIT_CHECKPOINT}" >&2
    exit 1
  }

  CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

# =========================================================
# Optional class names
# =========================================================

if [[ -n "${CLASS_NAMES:-}" ]]; then
  CMD+=(--class-names "${CLASS_NAMES}")
elif [[ -n "${LABEL_CSV:-}" ]]; then
  CMD+=(--label-csv "${LABEL_CSV}")
fi

# =========================================================
# Save and execute command
# =========================================================

printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"

if [[ "${EMCLIP_PRINT_CMD_ONLY:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

echo
echo "============================================================"
echo "[emclip] Starting UCF101 training"
echo "[emclip] physical GPUs       : ${CUDA_VISIBLE_DEVICES}"
echo "[emclip] process count       : ${NPROC_PER_NODE}"
echo "[emclip] CLIP checkpoint     : ${CLIP_CHECKPOINT}"
echo "[emclip] init checkpoint     : ${INIT_CHECKPOINT}"
echo "[emclip] resume checkpoint   : ${RESUME}"
echo "[emclip] output directory    : ${OUTPUT_DIR}"
echo "============================================================"
echo

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train_${STAMP}.log"
