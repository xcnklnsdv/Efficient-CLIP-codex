#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

DATASET="k400"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_gpu_env.sh"
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-0}
PIN_MEMORY=${PIN_MEMORY:-0}
MASTER_PORT=${MASTER_PORT:-29501}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_dir/emclip_eval}
TEMPORAL_VIEWS=${TEMPORAL_VIEWS:-1}
SPATIAL_CROPS=${SPATIAL_CROPS:-1}
STAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_eval_${TEMPORAL_VIEWS}x${SPATIAL_CROPS}_${STAMP}"
mkdir -p "${OUTPUT_DIR}"

CMD=(torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}"
  main_emclip.py --model emclip_b16 --dataset "${DATASET}" --eval --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --test-num-temporal-views "${TEMPORAL_VIEWS}" --test-num-spatial-crops "${SPATIAL_CROPS}"
  --amp --output-dir "${OUTPUT_DIR}")
if [[ "${PIN_MEMORY}" == "0" || "${PIN_MEMORY}" == "false" || "${PIN_MEMORY}" == "False" ]]; then CMD+=(--no-pin-memory); fi
if [[ -n "${RESUME:-}" ]]; then CMD+=(--resume "${RESUME}"); fi
printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/command.txt"; printf '\n' >> "${OUTPUT_DIR}/command.txt"
if [[ "${EMCLIP_PRINT_CMD_ONLY:-0}" == "1" ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval_${STAMP}.log"
