#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Fail to import ``MultiScaleDeformableAttention``:UserWarning:mmcv.cnn.bricks.transformer,ignore:Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected:UserWarning:mmengine.runner.checkpoint}"

# Clean command: HRNet-W32 + FGE (mixed LL+WL, 40 epochs)
# Usage:
#   conda activate freqpose
#   cd /path/to/mmpose
#   bash scripts/train_exlpose_fge_hrnet.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f "tools/train.py" ]]; then
  echo "[ERROR] This script must run inside an upstream MMPose checkout with the FreqPose overlay applied."
  echo "Current dir: $ROOT_DIR"
  exit 2
fi

export EXLPOSE_DATA_ROOT="${EXLPOSE_DATA_ROOT:-$ROOT_DIR/data/ExLPose}"

python scripts/prepare_exlpose_train_mixed.py

python tools/train.py \
  configs/body_2d_keypoint/topdown_heatmap/exlpose/td-hm_hrnet-w32_8xb64-40e_exlpose-256x192_fge_pilot_mixed.py \
  --amp
