#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/_gpu_env.sh"

python "${REPO_ROOT}/main_emclip.py" --synthetic-smoke --model emclip_b16 --emclip-variant emclip --mgse-text-mode class_bank
python "${REPO_ROOT}/main_emclip.py" --synthetic-smoke --model emclip_diamond_b16_k8 --emclip-variant diamond

CLIP_CHECKPOINT=${CLIP_CHECKPOINT:-/home/fuh/CLIP-models/ViT-B-16.pt}
COVIAR_DATA_LOADER_DIR=${COVIAR_DATA_LOADER_DIR:-${REPO_ROOT}/pytorch-coviar/data_loader}
if [[ ! -f "${CLIP_CHECKPOINT}" ]]; then
  echo "SKIP real-data smoke tests: CLIP checkpoint not found: ${CLIP_CHECKPOINT}"
  exit 0
fi
if ! compgen -G "${COVIAR_DATA_LOADER_DIR}/coviar*.so" >/dev/null && \
   ! compgen -G "${COVIAR_DATA_LOADER_DIR}/coviar*.pyd" >/dev/null; then
  echo "SKIP real-data smoke tests: CoViAR extension not found in ${COVIAR_DATA_LOADER_DIR}"
  exit 0
fi

run_real_smoke() {
  local dataset=$1
  local root=$2
  local list=$3
  shift 3
  if [[ ! -d "${root}" ]]; then
    echo "SKIP ${dataset}: data root not found: ${root}"
    return
  fi
  if [[ ! -f "${list}" ]]; then
    echo "SKIP ${dataset}: validation list not found: ${list}"
    return
  fi
  python "${REPO_ROOT}/main_emclip.py" \
    --dataset "${dataset}" \
    --clip-checkpoint "${CLIP_CHECKPOINT}" \
    --coviar-data-loader-dir "${COVIAR_DATA_LOADER_DIR}" \
    --val-root "${root}" \
    --compressed-video-root "${root}" \
    --val-list "${list}" \
    --eval \
    --preflight-only \
    --num-workers 0 \
    --no-pin-memory \
    "$@"
}

SSV2_ROOT=${SSV2_ROOT:-/mnt/data/sthv2/mpeg4_video/20bn-something-something-v2}
SSV2_VAL_LIST=${SSV2_VAL_LIST:-/home/fuh/CMPT/lists/sthv2/val_rgb.txt}
if [[ -n "${SSV2_CLASS_NAMES:-}" ]]; then
  run_real_smoke ssv2_mpeg4 "${SSV2_ROOT}" "${SSV2_VAL_LIST}" --class-names "${SSV2_CLASS_NAMES}"
else
  run_real_smoke ssv2_mpeg4 "${SSV2_ROOT}" "${SSV2_VAL_LIST}" \
    --label-csv "${SSV2_LABEL_CSV:-${REPO_ROOT}/configs/something_v2_labels.csv}"
fi
run_real_smoke hmdb51_mpeg4 \
  "${HMDB51_ROOT:-/mnt/data/hmdb51/mpeg4_videos}" \
  "${HMDB51_VAL_LIST:-/home/fuh/CMPT/lists/hmdb51/val_rgb_split_1.txt}"
run_real_smoke ucf101_mpeg4 \
  "${UCF101_ROOT:-/mnt/data/ucf101/Cucf101}" \
  "${UCF101_VAL_LIST:-/home/fuh/CMPT/lists/ucf101/val_rgb_split_1.txt}"
run_real_smoke k400 \
  "${K400_ROOT:-/mnt/data/CKinetics}" \
  "${K400_VAL_LIST:-/mnt/data/CKinetics/datalist/k400_val.txt}"
