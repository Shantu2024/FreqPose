#!/usr/bin/env bash
set -euo pipefail

# Launch TensorBoard with stable labels for FGE runs.
# Usage:
#   bash scripts/tb_fge.sh
#   bash scripts/tb_fge.sh 16006

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-6006}"

latest_vis_dir() {
  local pattern="$1"
  ls -dt "$ROOT_DIR"/work_dirs/"$pattern"/*/vis_data 2>/dev/null | head -n1 || true
}

RES50_DIR="$(latest_vis_dir tdhm_res50_exlpose_fge_pilot40_mixed)"
HRNET_DIR="$(latest_vis_dir tdhm_hrnet_w32_exlpose_fge_pilot40_mixed)"

if [[ -z "$RES50_DIR" && -z "$HRNET_DIR" ]]; then
  echo "No vis_data directories found under $ROOT_DIR/work_dirs"
  exit 1
fi

SPECS=()
if [[ -n "$RES50_DIR" ]]; then
  SPECS+=("res50:${RES50_DIR}")
fi
if [[ -n "$HRNET_DIR" ]]; then
  SPECS+=("hrnet:${HRNET_DIR}")
fi

LOGDIR_SPEC="$(IFS=,; echo "${SPECS[*]}")"
echo "Launching TensorBoard with:"
echo "  $LOGDIR_SPEC"
echo "Open: http://localhost:${PORT}"

exec tensorboard --logdir_spec "$LOGDIR_SPEC" --port "$PORT" --bind_all --load_fast=false
