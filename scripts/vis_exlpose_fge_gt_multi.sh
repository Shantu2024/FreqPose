#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="ignore"

# GT multi-person visualization in one image for FGE checkpoints.
#
# Usage:
#   bash scripts/vis_exlpose_fge_gt_multi.sh <checkpoint.pth> [SPLIT] [N|all]
#
# Split:
#   LL-N | LL-H | LL-E | LL-A | WL | A7M3 | RICOH3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CKPT="${1:-}"
SPLIT="${2:-LL-A}"
N="${3:-all}"
DEVICE="${DEVICE:-cuda:0}"
KPT_THR="${KPT_THR:-0.35}"
BBOX_THR="${BBOX_THR:-0.0}"
RADIUS="${RADIUS:-4}"
THICKNESS="${THICKNESS:-2}"
OUT_DIR="release/protocol_gt_vis_multi_fge/$(date +%Y%m%d_%H%M%S)"

if [[ -z "$CKPT" ]]; then
  echo "Usage: bash scripts/vis_exlpose_fge_gt_multi.sh <checkpoint.pth> [SPLIT] [N|all]"
  exit 1
fi

if [[ -f "$ROOT_DIR/$CKPT" ]]; then
  CKPT="$ROOT_DIR/$CKPT"
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  exit 2
fi

if [[ "$CKPT" == *"hrnet"* ]]; then
  CFG="configs/body_2d_keypoint/topdown_heatmap/exlpose/td-hm_hrnet-w32_8xb64-40e_exlpose-256x192_fge_pilot_mixed.py"
else
  CFG="configs/body_2d_keypoint/topdown_heatmap/exlpose/td-hm_res50_8xb64-40e_exlpose-256x192_fge_pilot_mixed.py"
fi

if [[ "$N" == "all" ]]; then
  N=-1
fi

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

python scripts/vis_exlpose_fge_gt_multi.py \
  --ckpt "$CKPT" \
  --cfg "$CFG" \
  --split "$SPLIT" \
  --n "$N" \
  --device "$DEVICE" \
  --kpt-thr "$KPT_THR" \
  --bbox-thr "$BBOX_THR" \
  --radius "$RADIUS" \
  --thickness "$THICKNESS" \
  --out-dir "$OUT_DIR"

echo "Saved GT multi visualization: $OUT_DIR/$SPLIT"
