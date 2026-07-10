#!/usr/bin/env bash
set -euo pipefail

DATASET="k400"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_gpu_env.sh"
BATCH_SIZE=${BATCH_SIZE:-4}
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
  main_emclip.py
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
  --amp
  --output-dir "${OUTPUT_DIR}")

if [[ -n "${RESUME:-}" ]]; then
  CMD+=(--resume "${RESUME}")
fi

printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/command.txt"
printf '\n' >> "${OUTPUT_DIR}/command.txt"
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/train_${STAMP}.log"
