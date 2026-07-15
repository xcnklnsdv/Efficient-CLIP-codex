#!/usr/bin/env bash

# Central GPU selection for every EM-CLIP train/eval script.
#
# Edit this line to choose the default physical GPU ids used by the scripts.
# Examples: "0", "0,1", "2,3,6,7". An empty string disables the default list.
EMCLIP_DEFAULT_GPU_IDS="6,7"
#
# One-off command-line overrides still take priority:
#   GPU_IDS=0,1 bash scripts/train_emclip_hmdb51.sh
#   GPUS=2 bash scripts/eval_emclip_ucf101.sh
#   CUDA_VISIBLE_DEVICES=0,3 bash scripts/train_emclip_k400.sh
#
# Priority: GPU_IDS > GPUS > CUDA_VISIBLE_DEVICES > EMCLIP_DEFAULT_GPU_IDS.
# If NPROC_PER_NODE is not explicitly set, it is derived from the selected list.

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

if [[ -n "${GPU_IDS+x}" ]]; then
  GPU_IDS="${GPU_IDS}"
elif [[ -n "${GPUS+x}" ]]; then
  GPU_IDS="${GPUS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  GPU_IDS="${CUDA_VISIBLE_DEVICES}"
else
  GPU_IDS="${EMCLIP_DEFAULT_GPU_IDS}"
fi
GPU_IDS="${GPU_IDS//[[:space:]]/}"

if [[ -n "${GPU_IDS}" ]]; then
  if [[ ! "${GPU_IDS}" =~ ^[^,]+(,[^,]+)*$ ]]; then
    echo "Invalid GPU list '${GPU_IDS}'. Use a comma-separated list such as 0,1,2,3." >&2
    return 2 2>/dev/null || exit 2
  fi
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
else
  export CUDA_VISIBLE_DEVICES=""
fi

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${GPU_IDS}" ]]; then
    IFS=',' read -r -a _EMCLIP_GPU_ID_ARRAY <<< "${GPU_IDS}"
    NPROC_PER_NODE="${#_EMCLIP_GPU_ID_ARRAY[@]}"
  else
    NPROC_PER_NODE=1
  fi
fi

export NPROC_PER_NODE
