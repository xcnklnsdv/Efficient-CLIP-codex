#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

DATASET="ssv2_mpeg4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LABEL_CSV="${LABEL_CSV:-${REPO_ROOT}/configs/something_v2_labels.csv}"
CLIP_CHECKPOINT="${CLIP_CHECKPOINT:-/home/fuh/CLIP-models/ViT-B-16.pt}"
if [[ ! -f "${CLIP_CHECKPOINT}" && -f "${REPO_ROOT}/clip_vit_b_16.pth" ]]; then
  CLIP_CHECKPOINT="${REPO_ROOT}/clip_vit_b_16.pth"
fi

K400_CHECKPOINT="${K400_CHECKPOINT:-/home/fuh/Efficient-CLIP-codex/output_dir/emclip/k400_emclip_T16_K8_20260721_001418/model_best.pth}"

# K400 is a transfer-learning source: load model weights only and reset all
# SSV2 optimizer/scheduler/scaler/epoch state. An explicitly empty value
# disables K400 initialization and leaves the original CLIP initialization.
INIT_CHECKPOINT="${INIT_CHECKPOINT-${K400_CHECKPOINT}}"

# RESUME is only for continuing an SSV2 latest.pth/model_best.pth run.
RESUME="${RESUME:-}"

# Keep the SSV2 launcher aligned with train_emclip_hmdb51.sh and the current
# _gpu_env.sh contract. PyTorch sees these as cuda:0, cuda:1, and cuda:2.
EMCLIP_SCRIPT_GPU_IDS="0,1,2"
source "${SCRIPT_DIR}/_gpu_env.sh"
BATCH_SIZE=${BATCH_SIZE:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-${BATCH_SIZE}}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}
PROFILE_COMPUTE=${PROFILE_COMPUTE:-1}
if [[ -z "${COVIAR_DATA_LOADER_DIR:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../pytorch-coviar/data_loader" ]]; then
    COVIAR_DATA_LOADER_DIR="$(cd "${SCRIPT_DIR}/../pytorch-coviar/data_loader" && pwd)"
  else
    COVIAR_DATA_LOADER_DIR=/home/fuh/m2clip/Coviar/data_loader
  fi
fi
MASTER_PORT=${MASTER_PORT:-29501}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_dir/emclip}
VARIANT=${VARIANT:-emclip}
T=${T:-16}
K=${K:-8}
STAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_${VARIANT}_T${T}_K${K}_${STAMP}"
mkdir -p "${OUTPUT_DIR}"

CMD=(torchrun
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
  --output-dir "${OUTPUT_DIR}")
if [[ "${PIN_MEMORY}" == "0" || "${PIN_MEMORY}" == "false" || "${PIN_MEMORY}" == "False" ]]; then
  CMD+=(--no-pin-memory)
fi
if [[ "${PROFILE_COMPUTE}" == "1" || "${PROFILE_COMPUTE}" == "true" || "${PROFILE_COMPUTE}" == "True" ]]; then
  CMD+=(--profile-compute)
fi

if [[ -n "${CLIP_CHECKPOINT:-}" ]]; then
  [[ -f "${CLIP_CHECKPOINT}" ]] || { echo "CLIP checkpoint not found: ${CLIP_CHECKPOINT}" >&2; exit 1; }
  CMD+=(--clip-checkpoint "${CLIP_CHECKPOINT}")
fi

if [[ -n "${RESUME:-}" ]]; then
  [[ -f "${RESUME}" ]] || {
    echo "SSV2 resume checkpoint not found: ${RESUME}" >&2
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
  [[ -f "${CLASS_NAMES}" ]] || { echo "SSV2 class-names file not found: ${CLASS_NAMES}" >&2; exit 1; }
  CMD+=(--class-names "${CLASS_NAMES}")
elif [[ -n "${LABEL_CSV:-}" ]]; then
  [[ -f "${LABEL_CSV}" ]] || { echo "SSV2 label CSV not found: ${LABEL_CSV}" >&2; exit 1; }
  CMD+=(--label-csv "${LABEL_CSV}")
fi

printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"
if [[ "${EMCLIP_PRINT_CMD_ONLY:-0}" == "1" ]]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train_${STAMP}.log"
