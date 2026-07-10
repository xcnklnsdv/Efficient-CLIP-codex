#!/usr/bin/env bash

# Optional GPU selection helper for EM-CLIP scripts.
#
# Usage:
#   GPU_IDS=0,1 bash scripts/train_emclip_hmdb51.sh
#   GPUS=2 bash scripts/eval_emclip_ucf101.sh
#   CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_emclip_k400.sh
#
# If NPROC_PER_NODE is not explicitly set, it is derived from GPU_IDS, GPUS,
# or CUDA_VISIBLE_DEVICES. If none is set, scripts keep the paper default of 4 GPUs.

GPU_IDS=${GPU_IDS:-${GPUS:-${CUDA_VISIBLE_DEVICES:-}}}
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
