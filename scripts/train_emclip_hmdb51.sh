#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

DATASET="hmdb51_mpeg4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =========================================================
# =========================================================

source "${SCRIPT_DIR}/_gpu_env.sh"

# =========================================================
# Checkpoints
# =========================================================

CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/home/fuh/CLIP-models/ViT-B-16.pt}"


# Default: original CLIP. Set INIT_CHECKPOINT/K400_CHECKPOINT explicitly for a transfer ablation.
INIT_CHECKPOINT="${INIT_CHECKPOINT-${K400_CHECKPOINT:-}}"

# RESUME is only for continuing an HMDB51 latest.pth/model_best.pth run.
RESUME="${RESUME:-}"

# =========================================================
# Data and training parameters
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

MASTER_PORT=${MASTER_PORT:-29501}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_dir/emclip}
IMPLEMENTATION=${IMPLEMENTATION:-auto}
MGSE_TRAIN_TEXT_MODE=${MGSE_TRAIN_TEXT_MODE:-}
MGSE_EVAL_TEXT_MODE=${MGSE_EVAL_TEXT_MODE:-class_bank}
VARIANT=${VARIANT:-emclip}
if [[ "${VARIANT}" == "diamond" || "${VARIANT}" == "emclip_diamond" ]]; then
  T=${T:-${K:-8}}
else
  T=${T:-$((2 * ${K:-8}))}
fi
K=${K:-8}

STAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_${VARIANT}_T${T}_K${K}_${STAMP}"

mkdir -p "${OUTPUT_DIR}"

# =========================================================
# Build command
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
  --mgse-text-mode "${MGSE_EVAL_TEXT_MODE}"

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

CMD+=(--emclip-implementation "${IMPLEMENTATION}")
if [[ -n "${MGSE_TRAIN_TEXT_MODE}" ]]; then
  CMD+=(--mgse-train-text-mode "${MGSE_TRAIN_TEXT_MODE}")
fi

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

# 原始 CLIP 权重
if [[ -z "${RESUME:-}" && -z "${INIT_CHECKPOINT:-}" && -n "${CLIP_CHECKPOINT:-}" ]]; then
  [[ -f "${CLIP_CHECKPOINT}" ]] || {
    echo "Original CLIP checkpoint not found: ${CLIP_CHECKPOINT}" >&2
    exit 1
  }

  CMD+=(--clip-checkpoint "${CLIP_CHECKPOINT}")
fi

# K400 EMCLIP 权重
if [[ -n "${RESUME:-}" ]]; then
  [[ -f "${RESUME}" ]] || {
    echo "HMDB51 resume checkpoint not found: ${RESUME}" >&2
    exit 1
  }

  CMD+=(--resume "${RESUME}")
elif [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  [[ -f "${INIT_CHECKPOINT}" ]] || {
    echo "K400 initialization checkpoint not found: ${INIT_CHECKPOINT}" >&2
    exit 1
  }

  CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

if [[ -n "${CLASS_NAMES:-}" ]]; then
  CMD+=(--class-names "${CLASS_NAMES}")
elif [[ -n "${LABEL_CSV:-}" ]]; then
  CMD+=(--label-csv "${LABEL_CSV}")
fi

# =========================================================
# Save and run command
# =========================================================

CMD+=("$@")
printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"

if [[ "${EMCLIP_PRINT_CMD_ONLY:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train_${STAMP}.log"
