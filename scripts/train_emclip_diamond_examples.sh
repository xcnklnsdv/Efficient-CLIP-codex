#!/usr/bin/env bash
set -euo pipefail

echo "Example EM-CLIP-diamond K=8:"
echo "VARIANT=diamond T=8 K=8 bash scripts/train_emclip_hmdb51.sh"
echo
echo "Example EM-CLIP-diamond K=16:"
echo "VARIANT=diamond T=16 K=16 bash scripts/train_emclip_ucf101.sh"
