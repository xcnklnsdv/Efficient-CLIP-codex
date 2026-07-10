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
