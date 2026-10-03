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


# Evaluate a target SSV2 checkpoint. Transfer evaluation must be requested explicitly.
INIT_CHECKPOINT="${INIT_CHECKPOINT-${K400_CHECKPOINT:-}}"
RESUME="${RESUME:-}"

source "${SCRIPT_DIR}/_gpu_env.sh"
BATCH_SIZE=${BATCH_SIZE:-4}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-8}
PIN_MEMORY=${PIN_MEMORY:-1}
if [[ -z "${COVIAR_DATA_LOADER_DIR:-}" ]]; then
  if [[ -d "${SCRIPT_DIR}/../pytorch-coviar/data_loader" ]]; then
    COVIAR_DATA_LOADER_DIR="$(cd "${SCRIPT_DIR}/../pytorch-coviar/data_loader" && pwd)"
  else
    COVIAR_DATA_LOADER_DIR=/home/fuh/m2clip/Coviar/data_loader
  fi
fi
MASTER_PORT=${MASTER_PORT:-29501}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_dir/emclip_eval}
TEMPORAL_VIEWS=${TEMPORAL_VIEWS:-4}
SPATIAL_CROPS=${SPATIAL_CROPS:-3}
IMPLEMENTATION=${IMPLEMENTATION:-auto}
MGSE_EVAL_TEXT_MODE=${MGSE_EVAL_TEXT_MODE:-class_bank}
VARIANT=${VARIANT:-emclip}
if [[ "${VARIANT}" == "diamond" || "${VARIANT}" == "emclip_diamond" ]]; then
  T=${T:-${K:-8}}
else
  T=${T:-$((2 * ${K:-8}))}
fi
K=${K:-8}
STAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}_${VARIANT}_T${T}_K${K}_eval_${TEMPORAL_VIEWS}x${SPATIAL_CROPS}_${STAMP}"
mkdir -p "${OUTPUT_DIR}"

CMD=(torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}"
  "${REPO_ROOT}/main_emclip.py" --model emclip_b16 --dataset "${DATASET}" --eval --batch-size "${BATCH_SIZE}"
  --micro-batch-size "${MICRO_BATCH_SIZE}" --emclip-variant "${VARIANT}" --candidate-frames "${T}" --selected-frames "${K}"
  --num-workers "${NUM_WORKERS}"
  --coviar-data-loader-dir "${COVIAR_DATA_LOADER_DIR}"
  --test-num-temporal-views "${TEMPORAL_VIEWS}" --test-num-spatial-crops "${SPATIAL_CROPS}"
  --amp --output-dir "${OUTPUT_DIR}")
CMD+=(--emclip-implementation "${IMPLEMENTATION}")
CMD+=(--mgse-text-mode "${MGSE_EVAL_TEXT_MODE}")
if [[ -z "${RESUME}" && -z "${INIT_CHECKPOINT}" ]]; then
  echo "Evaluation requires RESUME=/path/to/target/model_best.pth (or an explicit INIT_CHECKPOINT transfer ablation)." >&2
  exit 1
fi

if [[ "${PIN_MEMORY}" == "0" || "${PIN_MEMORY}" == "false" || "${PIN_MEMORY}" == "False" ]]; then CMD+=(--no-pin-memory); fi
if [[ -z "${RESUME:-}" && -z "${INIT_CHECKPOINT:-}" && -n "${CLIP_CHECKPOINT:-}" ]]; then
  [[ -f "${CLIP_CHECKPOINT}" ]] || { echo "CLIP checkpoint not found: ${CLIP_CHECKPOINT}" >&2; exit 1; }
  CMD+=(--clip-checkpoint "${CLIP_CHECKPOINT}")
fi
if [[ -n "${RESUME:-}" ]]; then
  [[ -f "${RESUME}" ]] || { echo "SSV2 evaluation checkpoint not found: ${RESUME}" >&2; exit 1; }
  CMD+=(--resume "${RESUME}")
elif [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  [[ -f "${INIT_CHECKPOINT}" ]] || { echo "K400 initialization checkpoint not found: ${INIT_CHECKPOINT}" >&2; exit 1; }
  CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi
if [[ -n "${CLASS_NAMES:-}" ]]; then
  [[ -f "${CLASS_NAMES}" ]] || { echo "SSV2 class-names file not found: ${CLASS_NAMES}" >&2; exit 1; }
  CMD+=(--class-names "${CLASS_NAMES}")
elif [[ -n "${LABEL_CSV:-}" ]]; then
  [[ -f "${LABEL_CSV}" ]] || { echo "SSV2 label CSV not found: ${LABEL_CSV}" >&2; exit 1; }
  CMD+=(--label-csv "${LABEL_CSV}")
fi
CMD+=("$@")
printf '%q ' "${CMD[@]}" > "${OUTPUT_DIR}/command.txt"; printf '\n' >> "${OUTPUT_DIR}/command.txt"
if [[ "${EMCLIP_PRINT_CMD_ONLY:-0}" == "1" ]]; then printf '%q ' "${CMD[@]}"; printf '\n'; exit 0; fi
"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval_${STAMP}.log"
