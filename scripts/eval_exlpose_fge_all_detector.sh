#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="ignore"

# Unified detector-box eval: ExLPose splits + OCN.
#
# Usage:
#   bash scripts/eval_exlpose_fge_all_detector.sh <pose_ckpt.pth> [yolo_weights.pt] [res50|hrnet]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSE_CKPT="${1:-}"
YOLO_WEIGHTS="${2:-}"
MODEL="${3:-auto}"

if [[ -z "$POSE_CKPT" ]]; then
  echo "Usage: bash scripts/eval_exlpose_fge_all_detector.sh <pose_ckpt.pth> [yolo_weights.pt] [res50|hrnet]"
  exit 1
fi

cd "$ROOT_DIR"
echo "=== ExLPose splits with detector boxes ==="
bash scripts/eval_exlpose_fge_splits_detector.sh "$POSE_CKPT" "$YOLO_WEIGHTS" "$MODEL"
echo
echo "=== OCN with detector boxes ==="
bash scripts/eval_exlpose_fge_ocn_detector.sh "$POSE_CKPT" "$YOLO_WEIGHTS" "$MODEL"

