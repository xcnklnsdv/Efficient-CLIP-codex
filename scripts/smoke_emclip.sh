#!/usr/bin/env bash
set -euo pipefail

python main_emclip.py --synthetic-smoke --model emclip_b16 --emclip-variant emclip --mgse-text-mode class_bank
python main_emclip.py --synthetic-smoke --model emclip_diamond_b16_k8 --emclip-variant diamond
