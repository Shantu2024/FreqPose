#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Fail to import ``MultiScaleDeformableAttention``:UserWarning:mmcv.cnn.bricks.transformer,ignore:Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected:UserWarning:mmengine.runner.checkpoint}"

# Evaluate one FGE checkpoint on ExLPose-OCN splits (GT-box protocol).
#
# Usage:
#   conda activate freqpose
#   cd /path/to/mmpose
#   bash scripts/eval_exlpose_fge_ocn.sh <checkpoint.pth> [res50|hrnet]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:-}"
MODEL="${2:-auto}"
DATA_ROOT="${EXLPOSE_DATA_ROOT:-$ROOT_DIR/data/ExLPose}"
ANN_ROOT="${DATA_ROOT}/Annotations"
TMP_DIR="${ROOT_DIR}/.tmp_eval_ann"

if [[ ! -f "$ROOT_DIR/tools/test.py" ]]; then
  echo "[ERROR] This script must run inside a full mmpose repo."
  echo "Current dir: $ROOT_DIR"
  exit 2
fi

if [[ -z "$CKPT" ]]; then
  echo "Usage: bash scripts/eval_exlpose_fge_ocn.sh <checkpoint.pth> [res50|hrnet]"
  exit 1
fi
if [[ -f "$ROOT_DIR/$CKPT" ]]; then
  CKPT="$ROOT_DIR/$CKPT"
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  exit 2
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

SPLITS=("A7M3" "RICOH3")
declare -A AP

for s in "${SPLITS[@]}"; do
  ANN_FILE="ExLPose-OC_test_${s}.json"
  AREA_ANN="${TMP_DIR}/${ANN_FILE%.json}.with_area.json"
  python scripts/ensure_coco_area.py --src "$ANN_ROOT/$ANN_FILE" --dst "$AREA_ANN" >/dev/null
  OUT="$(TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py \
    "$CFG" "$CKPT" \
    --cfg-options \
      "test_dataloader.dataset.ann_file=${AREA_ANN}" \
      "test_dataloader.dataset.data_prefix.img=." \
      "test_dataloader.num_workers=0" \
      "test_dataloader.prefetch_factor=None" \
      "test_dataloader.persistent_workers=False" \
      "test_evaluator.ann_file=${AREA_ANN}" 2>&1)"
  LINE="$(echo "$OUT" | grep "coco/AP:" | tail -n1 || true)"
  VAL="$(echo "$LINE" | sed -n "s/.*coco\/AP: \([0-9.]*\).*/\1/p")"
  AP["$s"]="${VAL:-NA}"
  echo "[$s] $LINE"
done

echo
echo "Summary (AP@0.5:0.95)"
for s in "${SPLITS[@]}"; do
  v="${AP[$s]}"
  if [[ "$v" == "NA" ]]; then
    echo "$s: NA"
  else
    python - "$s" "$v" <<'PY'
import sys
s,v=sys.argv[1],float(sys.argv[2])
print(f"{s}: {v:.6f} ({v*100:.2f}%)")
PY
  fi
done
