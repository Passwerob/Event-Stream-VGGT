"""
Evaluate pose accuracy by comparing predicted poses with ground truth poses.
Both poses are normalized before comparison.
"""

import numpy as np
import os
import argparse
from pathlib import Path
import json


def extrinsic_3x4_to_4x4(extrinsic_3x4):
    """
    Convert 3x4 extrinsic matrix to 4x4 homogeneous transformation matrix.
    
    Args:
        extrinsic_3x4: [3, 4] or [N, 3, 4] array
        
    Returns:
        [4, 4] or [N, 4, 4] array
    """
    if extrinsic_3x4.ndim == 2:
        extrinsic_3x4 = extrinsic_3x4[np.newaxis, ...]
    
    N = extrinsic_3x4.shape[0]
    extrinsic_4x4 = np.zeros((N, 4, 4), dtype=extrinsic_3x4.dtype)
    extrinsic_4x4[:, :3, :] = extrinsic_3x4
    extrinsic_4x4[:, 3, 3] = 1.0
    
    if N == 1:
        return extrinsic_4x4[0]
    return extrinsic_4x4


def normalize_pose_translation(pose_4x4, scale=None):
    """
    Normalize the translation part of a pose matrix.
    
    Args:
        pose_4x4: [4, 4] or [N, 4, 4] pose matrix
        scale: Optional scale factor. If None, normalize by L2 norm of translation
        
    Returns:
        Normalized pose matrix with same shape
    """
    if pose_4x4.ndim == 2:
        pose_4x4 = pose_4x4[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False
    
    N = pose_4x4.shape[0]
    normalized = pose_4x4.copy()
    
    for i in range(N):
        translation = pose_4x4[i, :3, 3]
        if scale is None:
            # Normalize by L2 norm
            trans_norm = np.linalg.norm(translation)
            if trans_norm > 1e-8:
                normalized[i, :3, 3] = translation / trans_norm
            else:
                normalized[i, :3, 3] = translation
        else:
            # Normalize by given scale
            normalized[i, :3, 3] = translation / scale
    
    if squeeze:
        return normalized[0]
    return normalized


def compute_rotation_error(R1, R2):
    """
    Compute rotation error between two rotation matrices.
    
    Args:
        R1, R2: [3, 3] rotation matrices
        
    Returns:
        Angular error in degrees
    """
    R_diff = R1 @ R2.T
    trace = np.trace(R_diff)
    # Clamp to [-1, 1] for numerical stability
    trace = np.clip(trace, -1.0, 3.0)
    angle_rad = np.arccos((trace - 1.0) / 2.0)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def compute_translation_error(t1, t2, normalized=True):
    """
    Compute translation error between two translation vectors.
    
    Args:
        t1, t2: [3] translation vectors
        normalized: If True, compute relative error (after normalization)
        
    Returns:
        Translation error (L2 distance)
    """
    if normalized:
        # Normalize both translations
        t1_norm = t1 / (np.linalg.norm(t1) + 1e-8)
        t2_norm = t2 / (np.linalg.norm(t2) + 1e-8)
        error = np.linalg.norm(t1_norm - t2_norm)
    else:
        error = np.linalg.norm(t1 - t2)
    return error


def evaluate_poses(gt_poses, pred_poses, normalize=True):
    """
    Evaluate pose accuracy by comparing ground truth and predicted poses.
    
    Args:
        gt_poses: [N, 4, 4] ground truth poses
        pred_poses: [N, 4, 4] predicted poses
        normalize: Whether to normalize poses before comparison
        
    Returns:
        Dictionary with evaluation metrics
    """
    N = len(gt_poses)
    assert len(pred_poses) == N, f"Mismatch: {len(gt_poses)} GT vs {len(pred_poses)} pred"
    
    # Normalize poses if requested
    if normalize:
        gt_poses_norm = normalize_pose_translation(gt_poses)
        pred_poses_norm = normalize_pose_translation(pred_poses)
    else:
        gt_poses_norm = gt_poses
        pred_poses_norm = pred_poses
    
    rotation_errors = []
    translation_errors = []
    translation_errors_unnormalized = []
    
    for i in range(N):
        gt_pose = gt_poses_norm[i]
        pred_pose = pred_poses_norm[i]
        
        # Extract rotation and translation
        R_gt = gt_pose[:3, :3]
        t_gt = gt_pose[:3, 3]
        R_pred = pred_pose[:3, :3]
        t_pred = pred_pose[:3, 3]
        
        # Compute rotation error
        rot_err = compute_rotation_error(R_gt, R_pred)
        rotation_errors.append(rot_err)
        
        # Compute translation error (normalized)
        trans_err = compute_translation_error(t_gt, t_pred, normalized=True)
        translation_errors.append(trans_err)
        
        # Also compute unnormalized translation error for reference
        trans_err_unnorm = compute_translation_error(
            gt_poses[i, :3, 3], 
            pred_poses[i, :3, 3], 
            normalized=False
        )
        translation_errors_unnormalized.append(trans_err_unnorm)
    
    rotation_errors = np.array(rotation_errors)
    translation_errors = np.array(translation_errors)
    translation_errors_unnormalized = np.array(translation_errors_unnormalized)
    
    metrics = {
        'num_poses': N,
        'rotation_error_deg': {
            'mean': float(np.mean(rotation_errors)),
            'median': float(np.median(rotation_errors)),
            'std': float(np.std(rotation_errors)),
            'min': float(np.min(rotation_errors)),
            'max': float(np.max(rotation_errors)),
        },
        'translation_error_normalized': {
            'mean': float(np.mean(translation_errors)),
            'median': float(np.median(translation_errors)),
            'std': float(np.std(translation_errors)),
            'min': float(np.min(translation_errors)),
            'max': float(np.max(translation_errors)),
        },
        'translation_error_unnormalized': {
            'mean': float(np.mean(translation_errors_unnormalized)),
            'median': float(np.median(translation_errors_unnormalized)),
            'std': float(np.std(translation_errors_unnormalized)),
            'min': float(np.min(translation_errors_unnormalized)),
            'max': float(np.max(translation_errors_unnormalized)),
        },
    }
    
    return metrics


def load_gt_poses(gt_path):
    """Load ground truth poses from npz file."""
    data = np.load(gt_path, allow_pickle=True)
    if 'pose' in data:
        poses = data['pose']
    else:
        raise ValueError(f"Key 'pose' not found in {gt_path}. Available keys: {list(data.keys())}")
    
    # Ensure poses are 4x4
    if poses.shape[-2:] == (4, 4):
        return poses
    else:
        raise ValueError(f"Unexpected pose shape: {poses.shape}")


def load_pred_poses(pred_dir):
    """Load predicted poses from output directory."""
    pred_path = os.path.join(pred_dir, 'cameras.npz')
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predicted poses not found: {pred_path}")
    
    data = np.load(pred_path, allow_pickle=True)
    if 'extrinsic' in data:
        extrinsic_3x4 = data['extrinsic']
        # Convert 3x4 to 4x4
        poses = extrinsic_3x4_to_4x4(extrinsic_3x4)
        return poses
    else:
        raise ValueError(f"Key 'extrinsic' not found in {pred_path}. Available keys: {list(data.keys())}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pose accuracy")
    parser.add_argument(
        '--gt_poses',
        type=str,
        default='/share/magic_group/aigc/fcr/EventVGGT/StreamVGGT/src/evencoder/data/processed_data/screen-1/meta/poses_selected.npz',
        help='Path to ground truth poses npz file'
    )
    parser.add_argument(
        '--pred_poses',
        type=str,
        default='/share/magic_group/aigc/fcr/EventVGGT/StreamVGGT/src/output/cameras',
        help='Path to predicted poses directory (should contain cameras.npz)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file to save metrics (optional)'
    )
    parser.add_argument(
        '--no_normalize',
        action='store_true',
        help='Do not normalize poses before comparison'
    )
    args = parser.parse_args()
    
    print("Loading ground truth poses...")
    gt_poses = load_gt_poses(args.gt_poses)
    print(f"  Loaded {len(gt_poses)} GT poses, shape: {gt_poses.shape}")
    
    print("Loading predicted poses...")
    pred_poses = load_pred_poses(args.pred_poses)
    print(f"  Loaded {len(pred_poses)} predicted poses, shape: {pred_poses.shape}")
    
    # Ensure same number of poses
    min_len = min(len(gt_poses), len(pred_poses))
    if len(gt_poses) != len(pred_poses):
        print(f"Warning: Mismatch in number of poses ({len(gt_poses)} vs {len(pred_poses)}). Using first {min_len}.")
        gt_poses = gt_poses[:min_len]
        pred_poses = pred_poses[:min_len]
    
    print(f"\nEvaluating {min_len} poses (normalize={not args.no_normalize})...")
    metrics = evaluate_poses(gt_poses, pred_poses, normalize=not args.no_normalize)
    
    print("\n" + "="*60)
    print("POSE EVALUATION RESULTS")
    print("="*60)
    print(f"Number of poses: {metrics['num_poses']}")
    print(f"\nRotation Error (degrees):")
    rot = metrics['rotation_error_deg']
    print(f"  Mean:   {rot['mean']:.4f}°")
    print(f"  Median: {rot['median']:.4f}°")
    print(f"  Std:    {rot['std']:.4f}°")
    print(f"  Min:    {rot['min']:.4f}°")
    print(f"  Max:    {rot['max']:.4f}°")
    
    print(f"\nTranslation Error (normalized):")
    trans_norm = metrics['translation_error_normalized']
    print(f"  Mean:   {trans_norm['mean']:.6f}")
    print(f"  Median: {trans_norm['median']:.6f}")
    print(f"  Std:    {trans_norm['std']:.6f}")
    print(f"  Min:    {trans_norm['min']:.6f}")
    print(f"  Max:    {trans_norm['max']:.6f}")
    
    print(f"\nTranslation Error (unnormalized, for reference):")
    trans_unnorm = metrics['translation_error_unnormalized']
    print(f"  Mean:   {trans_unnorm['mean']:.6f}")
    print(f"  Median: {trans_unnorm['median']:.6f}")
    print(f"  Std:    {trans_unnorm['std']:.6f}")
    print(f"  Min:    {trans_unnorm['min']:.6f}")
    print(f"  Max:    {trans_unnorm['max']:.6f}")
    print("="*60)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == '__main__':
    main()




