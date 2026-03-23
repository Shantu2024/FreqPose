#!/usr/bin/env bash
set -euo pipefail
# User requested a clean terminal: suppress Python warnings for this wrapper.
export PYTHONWARNINGS="ignore"

# Unified eval: ExLPose splits + OCN (GT-box protocol).
#
# Usage:
#   bash scripts/eval_exlpose_fge_all.sh <checkpoint.pth> [res50|hrnet]

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT="${1:-}"
MODEL="${2:-auto}"

if [[ -z "$CKPT" ]]; then
  echo "Usage: bash scripts/eval_exlpose_fge_all.sh <checkpoint.pth> [res50|hrnet]"
  exit 1
fi

cd "$ROOT_DIR"
echo "=== ExLPose splits (LL/WL) ==="
bash scripts/eval_exlpose_fge_splits.sh "$CKPT" "$MODEL"
echo
echo "=== OCN (A7M3/RICOH3) ==="
bash scripts/eval_exlpose_fge_ocn.sh "$CKPT" "$MODEL"
