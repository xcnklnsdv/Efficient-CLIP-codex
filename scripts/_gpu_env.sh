#!/usr/bin/env bash

# Respect caller GPU visibility (including UUID/MIG identifiers). Physical GPU
# IDs are never hard-coded by launchers. The historical override remains opt-in.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" && -n "${EMCLIP_SCRIPT_GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EMCLIP_SCRIPT_GPU_IDS}"
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[emclip][gpu] NPROC_PER_NODE must be a positive integer" >&2
  return 2 2>/dev/null || exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _EMCLIP_VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
  if (( NPROC_PER_NODE > ${#_EMCLIP_VISIBLE_DEVICES[@]} )); then
    echo "[emclip][gpu] NPROC_PER_NODE=${NPROC_PER_NODE} exceeds CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
    return 2 2>/dev/null || exit 2
  fi
  unset _EMCLIP_VISIBLE_DEVICES
fi
export NPROC_PER_NODE

if [[ -z "${COVIAR_FFMPEG_LIB:-}" ]]; then
  _EMCLIP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ -d "${_EMCLIP_SCRIPT_DIR}/../pytorch-coviar/data_loader/ffmpeg/lib" ]]; then
    COVIAR_FFMPEG_LIB="$(cd "${_EMCLIP_SCRIPT_DIR}/../pytorch-coviar/data_loader/ffmpeg/lib" && pwd)"
  else
    COVIAR_FFMPEG_LIB="/home/fuh/ffmpeg_coviar/lib"
  fi
  unset _EMCLIP_SCRIPT_DIR
fi
export COVIAR_FFMPEG_LIB
export LD_LIBRARY_PATH="${COVIAR_FFMPEG_LIB}:${LD_LIBRARY_PATH:-}"
echo "[emclip][gpu] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<caller default>} processes=${NPROC_PER_NODE}"
