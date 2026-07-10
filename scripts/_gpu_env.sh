#!/usr/bin/env bash

# Optional GPU selection helper for EM-CLIP scripts.
#
# Usage:
#   GPU_IDS=0,1 bash scripts/train_emclip_hmdb51.sh
#   GPUS=2 bash scripts/eval_emclip_ucf101.sh
#   CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_emclip_k400.sh
#
# If NPROC_PER_NODE is not explicitly set, it is derived from GPU_IDS, GPUS,
# or CUDA_VISIBLE_DEVICES. The local default below pins scripts to GPU 4 and 5.

if [[ -z "${COVIAR_FFMPEG_LIB:-}" ]]; then
  _EMCLIP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -d "${_EMCLIP_SCRIPT_DIR}/../pytorch-coviar/data_loader/ffmpeg/lib" ]]; then
    COVIAR_FFMPEG_LIB="$(cd "${_EMCLIP_SCRIPT_DIR}/../pytorch-coviar/data_loader/ffmpeg/lib" && pwd)"
  else
    COVIAR_FFMPEG_LIB=/home/fuh/ffmpeg_coviar/lib
  fi
fi
export COVIAR_FFMPEG_LIB
export LD_LIBRARY_PATH="${COVIAR_FFMPEG_LIB}:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
GPU_IDS=${GPU_IDS:-${GPUS:-${CUDA_VISIBLE_DEVICES}}}
GPU_IDS="${GPU_IDS//[[:space:]]/}"

if [[ -n "${GPU_IDS}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -r -a _EMCLIP_GPU_ID_ARRAY <<< "${GPU_IDS}"
    NPROC_PER_NODE="${#_EMCLIP_GPU_ID_ARRAY[@]}"
  else
    NPROC_PER_NODE=4
  fi
fi

export NPROC_PER_NODE
