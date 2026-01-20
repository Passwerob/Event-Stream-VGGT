#!/bin/bash

# Script to run pose evaluation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default paths
GT_POSES="${1:-$SRC_DIR/evencoder/data/processed_data/screen-1/meta/poses_selected.npz}"
PRED_POSES="${2:-$SRC_DIR/output/cameras}"
OUTPUT="${3:-$SRC_DIR/output/pose_eval_metrics.json}"

echo "Running pose evaluation..."
echo "  GT poses: $GT_POSES"
echo "  Pred poses: $PRED_POSES"
echo "  Output: $OUTPUT"
echo ""

cd "$SRC_DIR"
python3 eval/event_eval/eval_pose.py \
    --gt_poses "$GT_POSES" \
    --pred_poses "$PRED_POSES" \
    --output "$OUTPUT"


python3 eval/event_eval/eval_pose.py \
    --gt_poses evencoder/data/processed_data/screen-1/meta/poses_selected.npz \
    --pred_poses output/screen-1/cameras \
    --output output/pose_eval_metrics.json

