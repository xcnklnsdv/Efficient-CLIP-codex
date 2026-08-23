#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

DATASET="k400"

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

# 原始 CLIP ViT-B/16 权重
# K400 从 CLIP 预训练权重开始训练
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/home/fuh/CLIP-models/ViT-B-16.pt}"

# 如果指定的 CLIP 权重不存在，则尝试仓库中的备用权重
if [[ ! -f "${CLIP_CHECKPOINT}" &&
      -f "${REPO_ROOT}/clip_vit_b_16.pth" ]]; then
  CLIP_CHECKPOINT="${REPO_ROOT}/clip_vit_b_16.pth"
fi

# RESUME 只用于继续之前的 K400 训练
# 例如：
# RESUME=/path/to/latest.pth bash scripts/train_k400.sh
RESUME="${RESUME:-}"

# =========================================================
# K400 label configuration
# =========================================================

LABEL_CSV="${LABEL_CSV:-${REPO_ROOT}/configs/kinetics_400_labels.csv}"

# =========================================================
# Training configuration
# =========================================================

# 注意：这里的 BATCH_SIZE 是每个 GPU 的 batch size
# 默认 3 张 GPU，因此全局 batch size = 16 × 3 = 48
BATCH_SIZE=${BATCH_SIZE:-16}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-${BATCH_SIZE}}

NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}
PROFILE_COMPUTE=${PROFILE_COMPUTE:-1}

FIND_UNUSED_PARAMETERS=${FIND_UNUSED_PARAMETERS:-false}
DEBUG_UNUSED_PARAMETERS=${DEBUG_UNUSED_PARAMETERS:-false}

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

MASTER_PORT=${MASTER_PORT:-29501}

OUTPUT_ROOT=${OUTPUT_ROOT:-output_dir/emclip}
VARIANT=${VARIANT:-emclip}

T=${T:-16}
K=${K:-8}

STAMP="$(date +"%Y%m%d_%H%M%S")"

OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_${VARIANT}_T${T}_K${K}_${STAMP}"
TORCHRUN_LOG_DIR="${OUTPUT_DIR}/torchrun_logs"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${TORCHRUN_LOG_DIR}"

# =========================================================
# Build training command
# =========================================================

CMD=(
  torchrun
  --nproc_per_node="${NPROC_PER_NODE}"
  --master_port="${MASTER_PORT}"

  --log-dir "${TORCHRUN_LOG_DIR}"
  --redirects 3
  --tee 3

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

# =========================================================
# Optional runtime configuration
# =========================================================

# 是否关闭 pin memory
if [[ "${PIN_MEMORY}" == "0" ||
      "${PIN_MEMORY}" == "false" ||
      "${PIN_MEMORY}" == "False" ]]; then
  CMD+=(--no-pin-memory)
fi

# 是否统计计算量
if [[ "${PROFILE_COMPUTE}" == "1" ||
      "${PROFILE_COMPUTE}" == "true" ||
      "${PROFILE_COMPUTE}" == "True" ]]; then
  CMD+=(--profile-compute)
fi

# DDP 是否查找未使用参数
if [[ "${FIND_UNUSED_PARAMETERS}" == "1" ||
      "${FIND_UNUSED_PARAMETERS}" == "true" ||
      "${FIND_UNUSED_PARAMETERS}" == "True" ]]; then
  CMD+=(--find-unused-parameters)
fi

# 是否输出未使用参数调试信息
if [[ "${DEBUG_UNUSED_PARAMETERS}" == "1" ||
      "${DEBUG_UNUSED_PARAMETERS}" == "true" ||
      "${DEBUG_UNUSED_PARAMETERS}" == "True" ]]; then
  CMD+=(--debug-unused-parameters)
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
# Resume K400 training
# =========================================================

if [[ -n "${RESUME:-}" ]]; then
  [[ -f "${RESUME}" ]] || {
    echo "[emclip] K400 resume checkpoint not found:" >&2
    echo "[emclip] ${RESUME}" >&2
    exit 1
  }

  CMD+=(--resume "${RESUME}")
fi

# =========================================================
# K400 class names
# =========================================================

if [[ -n "${CLASS_NAMES:-}" ]]; then
  [[ -f "${CLASS_NAMES}" ]] || {
    echo "[emclip] K400 class names file not found:" >&2
    echo "[emclip] ${CLASS_NAMES}" >&2
    exit 1
  }

  CMD+=(--class-names "${CLASS_NAMES}")

elif [[ -n "${LABEL_CSV:-}" ]]; then
  [[ -f "${LABEL_CSV}" ]] || {
    echo "[emclip] K400 label CSV not found:" >&2
    echo "[emclip] ${LABEL_CSV}" >&2
    exit 1
  }

  CMD+=(--label-csv "${LABEL_CSV}")
fi

# =========================================================
# Save command
# =========================================================

printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"

# 仅打印命令，不实际执行
if [[ "${EMCLIP_PRINT_CMD_ONLY:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

# =========================================================
# Execute
# =========================================================

echo
echo "============================================================"
echo "[emclip] Starting K400 training"
echo "[emclip] physical GPUs       : ${CUDA_VISIBLE_DEVICES}"
echo "[emclip] process count       : ${NPROC_PER_NODE}"
echo "[emclip] CLIP checkpoint     : ${CLIP_CHECKPOINT}"
echo "[emclip] resume checkpoint   : ${RESUME}"
echo "[emclip] label CSV           : ${LABEL_CSV}"
echo "[emclip] batch size per GPU  : ${BATCH_SIZE}"
echo "[emclip] global batch size   : $((BATCH_SIZE * NPROC_PER_NODE))"
echo "[emclip] micro batch size    : ${MICRO_BATCH_SIZE}"
echo "[emclip] output directory    : ${OUTPUT_DIR}"
echo "============================================================"
echo

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train_${STAMP}.log"