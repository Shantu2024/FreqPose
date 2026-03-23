#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Fail to import ``MultiScaleDeformableAttention``:UserWarning:mmcv.cnn.bricks.transformer,ignore:Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected:UserWarning:mmengine.runner.checkpoint}"

# Evaluate LL/WL splits with GT-box protocol.
# Usage:
#   conda activate freqpose
#   cd /path/to/mmpose
#   bash scripts/eval_exlpose_fge_splits.sh <checkpoint.pth> [res50|hrnet]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:-}"
MODEL="${2:-auto}"
DATA_ROOT="${EXLPOSE_DATA_ROOT:-$ROOT_DIR/data/ExLPose}"
ANN_ROOT="${DATA_ROOT}/Annotations"
TMP_DIR="${ROOT_DIR}/.tmp_eval_ann"

if [[ ! -f "$ROOT_DIR/tools/test.py" ]]; then
  echo "[ERROR] This script must run inside an upstream MMPose checkout with the FreqPose overlay applied."
  echo "Current dir: $ROOT_DIR"
  exit 2
fi

if [[ -z "$CKPT" ]]; then
  echo "Usage: bash scripts/eval_exlpose_fge_splits.sh <checkpoint.pth> [res50|hrnet]"
  exit 1
fi

if [[ "$MODEL" == "auto" ]]; then
  if [[ "$CKPT" == *"hrnet"* ]]; then
    MODEL="hrnet"
  else
    MODEL="res50"
  fi
fi

if [[ "$MODEL" == "hrnet" ]]; then
  CFG="configs/body_2d_keypoint/topdown_heatmap/exlpose/td-hm_hrnet-w32_8xb64-40e_exlpose-256x192_fge_pilot_mixed.py"
else
  CFG="configs/body_2d_keypoint/topdown_heatmap/exlpose/td-hm_res50_8xb64-40e_exlpose-256x192_fge_pilot_mixed.py"
fi

cd "$ROOT_DIR"
mkdir -p "$TMP_DIR"
for SPLIT in LL-N LL-H LL-E WL LL-A; do
  ANN="ExLPose_test_${SPLIT}.json"
  [[ "$SPLIT" == "WL" ]] && ANN="ExLPose_test_WL.json"
  AREA_ANN="${TMP_DIR}/${ANN%.json}.with_area.json"

  python scripts/ensure_coco_area.py --src "$ANN_ROOT/$ANN" --dst "$AREA_ANN" >/dev/null

  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py "$CFG" "$CKPT" \
    --cfg-options \
      test_dataloader.dataset.ann_file="${AREA_ANN}" \
      test_evaluator.ann_file="${AREA_ANN}" \
      test_dataloader.num_workers=0 \
      test_dataloader.prefetch_factor=None \
      test_dataloader.persistent_workers=False \
    | grep "coco/AP:" | tail -n1 | sed "s/^/[${SPLIT}] /"
done
