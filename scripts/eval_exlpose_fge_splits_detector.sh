#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Fail to import ``MultiScaleDeformableAttention``:UserWarning:mmcv.cnn.bricks.transformer,ignore:Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected:UserWarning:mmengine.runner.checkpoint}"

# Evaluate pose checkpoint with detector-predicted boxes (YOLO).
#
# Usage:
#   bash scripts/eval_exlpose_fge_splits_detector.sh <pose_ckpt.pth> [yolo_weights.pt] [res50|hrnet]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSE_CKPT="${1:-}"
YOLO_WEIGHTS="${2:-}"
MODEL="${3:-auto}"
DATA_ROOT="${EXLPOSE_DATA_ROOT:-$ROOT_DIR/data/ExLPose}"
ANN_ROOT="${DATA_ROOT}/Annotations"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/release/protocol_detector_eval/$TS"
TMP_DIR="$ROOT_DIR/.tmp_eval_ann"

if [[ ! -f "$ROOT_DIR/tools/test.py" ]]; then
  echo "[ERROR] This script must run inside an upstream MMPose checkout with the FreqPose overlay applied."
  echo "Current dir: $ROOT_DIR"
  exit 2
fi

if [[ -z "$POSE_CKPT" ]]; then
  echo "Usage: bash scripts/eval_exlpose_fge_splits_detector.sh <pose_ckpt.pth> [yolo_weights.pt] [res50|hrnet]"
  exit 1
fi

if [[ -f "$ROOT_DIR/$POSE_CKPT" ]]; then
  POSE_CKPT="$ROOT_DIR/$POSE_CKPT"
fi
if [[ ! -f "$POSE_CKPT" ]]; then
  echo "Pose checkpoint not found: $POSE_CKPT"
  exit 2
fi

if [[ "$MODEL" == "auto" ]]; then
  if [[ "$POSE_CKPT" == *"hrnet"* ]]; then
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

if [[ -z "$YOLO_WEIGHTS" ]]; then
  # Prefer the detector path used by scripts/train_yolo_exlpose_det.sh
  YOLO_WEIGHTS="$(ls -td "$ROOT_DIR"/work_dirs/yolo_exlpose/*/weights/best.pt 2>/dev/null | head -n 1 || true)"
  # Backward-compatible fallback for legacy runs.
  if [[ -z "$YOLO_WEIGHTS" ]]; then
    YOLO_WEIGHTS="$(ls -td "$ROOT_DIR"/runs/detect/work_dirs/yolo_exlpose/*/weights/best.pt 2>/dev/null | head -n 1 || true)"
  fi
fi
if [[ -f "$ROOT_DIR/$YOLO_WEIGHTS" ]]; then
  YOLO_WEIGHTS="$ROOT_DIR/$YOLO_WEIGHTS"
fi
if [[ ! -f "$YOLO_WEIGHTS" ]]; then
  echo "YOLO weights not found. Provide path explicitly."
  exit 3
fi

mkdir -p "$OUT_DIR"
mkdir -p "$TMP_DIR"
cd "$ROOT_DIR"

{
  echo "timestamp=$TS"
  echo "pose_ckpt=$POSE_CKPT"
  echo "yolo_weights=$YOLO_WEIGHTS"
  echo "cfg=$CFG"
  echo "bbox_policy=YOLO_predicted"
} > "$OUT_DIR/metadata.txt"

SPLITS=("LL-N" "LL-H" "LL-E" "WL" "LL-A")
for s in "${SPLITS[@]}"; do
  ANN_FILE="ExLPose_test_${s}.json"
  [[ "$s" == "WL" ]] && ANN_FILE="ExLPose_test_WL.json"
  AREA_ANN="$TMP_DIR/${ANN_FILE%.json}.with_area.json"
  DET_JSON="$OUT_DIR/det_${s}.json"
  LOG="$OUT_DIR/${s}.log"

  python scripts/ensure_coco_area.py --src "$ANN_ROOT/$ANN_FILE" --dst "$AREA_ANN" >/dev/null

  python scripts/gen_yolo_bbox_json.py \
    --ann-file "$AREA_ANN" \
    --data-root "$DATA_ROOT" \
    --weights "$YOLO_WEIGHTS" \
    --out-json "$DET_JSON" \
    --imgsz 960 --conf 0.10 --iou 0.70 --device 0 --batch 16 > "$OUT_DIR/det_${s}.build.log" 2>&1

  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python tools/test.py \
    "$CFG" "$POSE_CKPT" \
    --cfg-options \
      "test_dataloader.dataset.ann_file=${AREA_ANN}" \
      "test_dataloader.dataset.bbox_file=${DET_JSON}" \
      "test_dataloader.num_workers=0" \
      "test_dataloader.prefetch_factor=None" \
      "test_dataloader.persistent_workers=False" \
      "test_evaluator.ann_file=${AREA_ANN}" \
    > "$LOG" 2>&1

  LINE="$(grep "coco/AP:" "$LOG" | tail -n1 || true)"
  echo "[$s] $LINE"
done

echo "Saved detector-box eval report: $OUT_DIR"
