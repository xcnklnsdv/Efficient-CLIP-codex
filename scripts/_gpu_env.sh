#!/usr/bin/env bash

# EM-CLIP GPU environment configuration.
#
# 每个训练/测试脚本必须在 source 本文件之前指定：
#
#   EMCLIP_SCRIPT_GPU_IDS="3,4,5,6,7"
#   source "${SCRIPT_DIR}/_gpu_env.sh"
#
# PyTorch 内部会重新编号：
#   cuda:0 -> 物理 GPU 3
#   cuda:1 -> 物理 GPU 4
#   cuda:2 -> 物理 GPU 5
#   cuda:3 -> 物理 GPU 6
#   cuda:4 -> 物理 GPU 7

# ---------------------------------------------------------
# CoViAR FFmpeg libraries
# ---------------------------------------------------------

if [[ -z "${COVIAR_FFMPEG_LIB:-}" ]]; then
  _EMCLIP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

  if [[ -d "${_EMCLIP_SCRIPT_DIR}/../pytorch-coviar/data_loader/ffmpeg/lib" ]]; then
    COVIAR_FFMPEG_LIB="$(
      cd "${_EMCLIP_SCRIPT_DIR}/../pytorch-coviar/data_loader/ffmpeg/lib" &&
      pwd
    )"
  else
    COVIAR_FFMPEG_LIB="/home/fuh/ffmpeg_coviar/lib"
  fi
fi

export COVIAR_FFMPEG_LIB
export LD_LIBRARY_PATH="${COVIAR_FFMPEG_LIB}:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------
# GPU selection
# ---------------------------------------------------------

if [[ -z "${EMCLIP_SCRIPT_GPU_IDS+x}" ]]; then
  echo "[emclip][gpu] EMCLIP_SCRIPT_GPU_IDS is not defined." >&2
  echo "[emclip][gpu] Add this before sourcing _gpu_env.sh:" >&2
  echo 'EMCLIP_SCRIPT_GPU_IDS="0,1,2"' >&2
  return 2 2>/dev/null || exit 2
fi

SELECTED_GPU_IDS="${EMCLIP_SCRIPT_GPU_IDS}"
SELECTED_GPU_IDS="${SELECTED_GPU_IDS//[[:space:]]/}"

if [[ -z "${SELECTED_GPU_IDS}" ]]; then
  echo "[emclip][gpu] GPU list cannot be empty." >&2
  return 2 2>/dev/null || exit 2
fi

if [[ ! "${SELECTED_GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "[emclip][gpu] Invalid GPU list: '${SELECTED_GPU_IDS}'" >&2
  echo "[emclip][gpu] Expected format: 0 or 0,1 or 0,1,2" >&2
  return 2 2>/dev/null || exit 2
fi

# 强制覆盖终端中遗留的显卡环境变量。
export GPU_IDS="${SELECTED_GPU_IDS}"
export CUDA_VISIBLE_DEVICES="${SELECTED_GPU_IDS}"

# 始终根据脚本指定的显卡数量计算 torchrun 进程数，
# 不使用终端中遗留的 NPROC_PER_NODE。
IFS=',' read -r -a _EMCLIP_GPU_ID_ARRAY <<< "${SELECTED_GPU_IDS}"
NPROC_PER_NODE="${#_EMCLIP_GPU_ID_ARRAY[@]}"
export NPROC_PER_NODE

echo "[emclip][gpu] physical GPU ids : ${CUDA_VISIBLE_DEVICES}"
echo "[emclip][gpu] visible GPU count: ${NPROC_PER_NODE}"

for ((i = 0; i < NPROC_PER_NODE; i++)); do
  echo "[emclip][gpu] cuda:${i} -> physical GPU ${_EMCLIP_GPU_ID_ARRAY[$i]}"
done

unset SELECTED_GPU_IDS
unset _EMCLIP_GPU_ID_ARRAY
unset _EMCLIP_SCRIPT_DIR