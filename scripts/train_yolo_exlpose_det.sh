#!/usr/bin/env bash
set -euo pipefail

# One-command YOLO detector training on ExLPose LL+WL person boxes.
# Usage:
#   bash scripts/train_yolo_exlpose_det.sh      # stage 1 (safe, default)
#   bash scripts/train_yolo_exlpose_det.sh 2    # stage 2 (safe)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEP="${1:-1}"

cd "$ROOT_DIR"

python scripts/prepare_exlpose_yolo_det.py

case "$STEP" in
  1|m|M)
    yolo detect train \
      model=yolo11m.pt \
      data=data/ExLPoseDetector/exlpose_det.yaml \
      imgsz=960 \
      batch=20 \
      workers=8 \
      epochs=50 \
      device=0 \
      project=work_dirs/yolo_exlpose \
      name=m_fast_e50_v3
    ;;
  2|l|L)
    yolo detect train \
      model=yolo11l.pt \
      data=data/ExLPoseDetector/exlpose_det.yaml \
      imgsz=960 \
      batch=14 \
      workers=8 \
      epochs=50 \
      device=0 \
      project=work_dirs/yolo_exlpose \
      name=l_fast_e50_v3
    ;;
  *)
    echo "Usage: bash scripts/train_yolo_exlpose_det.sh [1|2]"
    echo "  1 -> yolo11m safe (default)"
    echo "  2 -> yolo11l safe"
    exit 1
    ;;
esac
