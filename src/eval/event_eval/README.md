# Event Pose Evaluation

This directory contains scripts for evaluating pose accuracy by comparing predicted poses with ground truth poses.

## Files

- `eval_pose.py`: Main evaluation script
- `run_eval.sh`: Convenience script to run evaluation
- `README.md`: This file

## Usage

### Basic Usage

```bash
cd /share/magic_group/aigc/fcr/EventVGGT/StreamVGGT/src
python3 eval/event_eval/eval_pose.py \
    --gt_poses evencoder/data/processed_data/screen-1/meta/poses_selected.npz \
    --pred_poses output/cameras
```

### Using the Shell Script

```bash
cd /share/magic_group/aigc/fcr/EventVGGT/StreamVGGT/src
./eval/event_eval/run_eval.sh [gt_poses_path] [pred_poses_path] [output_json_path]
```

### Save Results to JSON

```bash
python3 eval/event_eval/eval_pose.py \
    --gt_poses evencoder/data/processed_data/screen-1/meta/poses_selected.npz \
    --pred_poses output/cameras \
    --output output/pose_eval_metrics.json
```

### Without Normalization

By default, poses are normalized (translation vectors are normalized to unit length) before comparison. To disable normalization:

```bash
python3 eval/event_eval/eval_pose.py \
    --gt_poses <gt_path> \
    --pred_poses <pred_path> \
    --no_normalize
```

## Input Format

### Ground Truth Poses

- Format: NPZ file containing a key `'pose'`
- Shape: `(N, 4, 4)` - N poses as 4x4 homogeneous transformation matrices

### Predicted Poses

- Format: Directory containing `cameras.npz`
- The `cameras.npz` should contain:
  - `'extrinsic'`: Shape `(N, 3, 4)` - Extrinsic matrices
  - `'intrinsic'`: Shape `(N, 3, 3)` - Intrinsic matrices (optional)
  - `'frame_names'`: List of frame names (optional)

The script automatically converts 3x4 extrinsic matrices to 4x4 homogeneous transformation matrices.

## Output Metrics

The evaluation computes the following metrics:

1. **Rotation Error (degrees)**
   - Mean, median, std, min, max
   - Computed as angular difference between rotation matrices

2. **Translation Error (normalized)**
   - Mean, median, std, min, max
   - Computed after normalizing translation vectors to unit length

3. **Translation Error (unnormalized)**
   - Mean, median, std, min, max
   - Raw translation error without normalization (for reference)

## Normalization

By default, both ground truth and predicted poses are normalized before comparison:
- Translation vectors are normalized to unit length (L2 norm)
- This allows comparison of poses regardless of their absolute scale

This is important when comparing poses from different coordinate systems or scales.




