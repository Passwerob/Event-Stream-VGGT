import os
import cv2
import torch
import numpy as np
import glob
import argparse
import sys
import json

# sys.path.append("src/")
from streamvggt.models.streamvggt import StreamVGGT
from streamvggt.utils.load_fn import load_and_preprocess_images
from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
from typing import List


class StreamVGGTInference:  
    """StreamVGGT inference wrapper for images or video input."""
    
    def __init__(
        self,
        checkpoint_path="ckpt/checkpoints.pth",
        device=None,
        extractor="dino",
        evencoder_checkpoint=None,
        ev_in_channels=8,
        ev_out_channels=16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if extractor == "evencoder" and evencoder_checkpoint is None:
            default_ev_ckpt = os.path.join(
                os.path.dirname(__file__),
                "..",
                "checkpoints",
                "evencoder",
                "20251223-1503",
                "best_evencoder.pth",
            )
            evencoder_checkpoint = default_ev_ckpt
        self.extractor = extractor
        self.evencoder_checkpoint = evencoder_checkpoint
        self.ev_in_channels = ev_in_channels
        self.ev_out_channels = ev_out_channels
        self.model = self._load_model(checkpoint_path)
        
    def _load_model(self, checkpoint_path):
        """Load StreamVGGT model from checkpoint."""
        # Expand ~ to home directory
        checkpoint_path = os.path.expanduser(checkpoint_path.strip())
        print(f"Loading model from {checkpoint_path}...")
        
        if os.path.exists(checkpoint_path):
            model = StreamVGGT(
                extractor_type=self.extractor,
                evencoder_ckpt_path=self.evencoder_checkpoint,
                evencoder_in_channels=self.ev_in_channels,
                evencoder_out_channels=self.ev_out_channels,
            )
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            ckpt = self._strip_patch_embed_keys_if_needed(ckpt)
            model.load_state_dict(ckpt, strict=self.extractor != "evencoder")
            del ckpt
        else:
            print("Local checkpoint not found, downloading from Hugging Face...")
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="lch01/StreamVGGT",
                filename="checkpoints.pth",
                revision="main",
                force_download=True
            )
            model = StreamVGGT(
                extractor_type=self.extractor,
                evencoder_ckpt_path=self.evencoder_checkpoint,
                evencoder_in_channels=self.ev_in_channels,
                evencoder_out_channels=self.ev_out_channels,
            )
            ckpt = torch.load(path, map_location="cpu")
            ckpt = self._strip_patch_embed_keys_if_needed(ckpt)
            model.load_state_dict(ckpt, strict=self.extractor != "evencoder")
            del ckpt
        
        model.to(self.device)
        model.eval()
        print(f"Model loaded on {self.device}")
        return model

    def _load_event_voxels(self, event_files: List[str]) -> torch.Tensor:
        """Load stacked event voxel grids from .pt files."""
        voxel_list = []
        for path in sorted(event_files):
            voxel = torch.load(path, map_location="cpu", weights_only=True)
            voxel_list.append(voxel)
        if len(voxel_list) == 0:
            raise ValueError("No event tensors found.")
        return torch.stack(voxel_list, dim=0)  # [S, C, H, W]

    def _strip_patch_embed_keys_if_needed(self, state_dict):
        """Remove DINO-specific patch embed weights when using EvEncoder extractor."""
        if not isinstance(state_dict, dict):
            return state_dict

        # Unwrap common checkpoint formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict and isinstance(state_dict["model"], dict):
            state_dict = state_dict["model"]

        if self.extractor != "evencoder":
            return state_dict
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("aggregator.patch_embed")}
        dropped = len(state_dict) - len(filtered)
        if dropped > 0:
            print(f"Removed {dropped} DINO extractor weights for EvEncoder mode.")
        return filtered
    
    def _extract_frames_from_video(self, video_path, output_dir, fps_interval=1.0):
        """Extract frames from video at specified interval."""
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * fps_interval)
        
        frame_paths = []
        count = 0
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if count % frame_interval == 0:
                frame_path = os.path.join(output_dir, f"{frame_idx:06d}.png")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                frame_idx += 1
        
        cap.release()
        print(f"Extracted {len(frame_paths)} frames from video")
        return sorted(frame_paths)
    
    def _get_image_paths(self, image_folder):
        """Get sorted list of image paths from folder."""
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(os.path.join(image_folder, ext)))
        image_paths = sorted(list(set(image_paths)))
        print(f"Found {len(image_paths)} images")
        return image_paths
    
    @torch.no_grad()
    def inference(self, image_paths):
        """Run inference on list of image paths."""
        if len(image_paths) == 0:
            raise ValueError("No images provided")
        
        images = load_and_preprocess_images(image_paths).to(self.device)  # [S, 3, H, W]
        print(f"Input images shape: {images.shape}")
        
        frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]
        
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            output = self.model.inference(frames)
        
        all_pts3d, all_conf, all_depth, all_depth_conf, all_pose = [], [], [], [], []
        for res in output.ress:
            all_pts3d.append(res['pts3d_in_other_view'].squeeze(0))
            all_conf.append(res['conf'].squeeze(0))
            all_depth.append(res['depth'].squeeze(0))
            all_depth_conf.append(res['depth_conf'].squeeze(0))
            all_pose.append(res['camera_pose'].squeeze(0))
        
        predictions = {
            "images": images,
            "world_points": torch.stack(all_pts3d, dim=0),
            "world_points_conf": torch.stack(all_conf, dim=0),
            "depth": torch.stack(all_depth, dim=0),
            "depth_conf":  torch.stack(all_depth_conf, dim=0),
            "pose_enc": torch.stack(all_pose, dim=0),
        }
        
        pose_enc = predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        predictions["extrinsic"] = extrinsic.squeeze(0)
        predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None
        
        print(f"Output shapes:")
        print(f"  world_points: {predictions['world_points'].shape}")
        print(f"  depth: {predictions['depth'].shape}")
        print(f"  extrinsic: {predictions['extrinsic'].shape}")
        print(f"  intrinsic: {predictions['intrinsic'].shape}")
        
        return predictions, image_paths
    
    def inference_from_folder(self, image_folder):
        """Run inference on images from folder."""
        image_paths = self._get_image_paths(image_folder)
        return self.inference(image_paths)

    @torch.no_grad()
    def inference_from_events(self, event_paths: List[str]):
        """Run inference on pre-binned event voxel grids stored as .pt files."""
        if len(event_paths) == 0:
            raise ValueError("No event tensors provided")

        voxels = self._load_event_voxels(event_paths).to(self.device)  # [S, C, H, W]
        print(f"Loaded {voxels.shape[0]} event frames, shape per frame: {voxels.shape[1:]}")

        frames = [{"img": voxels[i].unsqueeze(0)} for i in range(voxels.shape[0])]

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.cuda.amp.autocast(dtype=dtype):
            output = self.model.inference(frames)

        all_pts3d, all_conf, all_depth, all_depth_conf, all_pose = [], [], [], [], []
        for res in output.ress:
            all_pts3d.append(res['pts3d_in_other_view'].squeeze(0))
            all_conf.append(res['conf'].squeeze(0))
            all_depth.append(res['depth'].squeeze(0))
            all_depth_conf.append(res['depth_conf'].squeeze(0))
            all_pose.append(res['camera_pose'].squeeze(0))

        predictions = {
            "images": voxels,
            "world_points": torch.stack(all_pts3d, dim=0),
            "world_points_conf": torch.stack(all_conf, dim=0),
            "depth": torch.stack(all_depth, dim=0),
            "depth_conf":  torch.stack(all_depth_conf, dim=0),
            "pose_enc": torch.stack(all_pose, dim=0),
        }

        pose_enc = predictions["pose_enc"].unsqueeze(0) if predictions["pose_enc"].ndim == 2 else predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, voxels.shape[-2:])
        predictions["extrinsic"] = extrinsic.squeeze(0)
        predictions["intrinsic"] = intrinsic.squeeze(0) if intrinsic is not None else None

        print(f"Output shapes:")
        print(f"  world_points: {predictions['world_points'].shape}")
        print(f"  depth: {predictions['depth'].shape}")
        print(f"  extrinsic: {predictions['extrinsic'].shape}")
        print(f"  intrinsic: {predictions['intrinsic'].shape}")

        return predictions, event_paths
    
    def inference_from_video(self, video_path, temp_dir="temp_frames", fps_interval=1.0):
        """Run inference on video."""
        image_paths = self._extract_frames_from_video(video_path, temp_dir, fps_interval)
        return self.inference(image_paths)


def save_point_cloud_ply(points, colors, confidences, save_path, conf_threshold=0.5):
    """Save point cloud as PLY file."""
    mask = confidences > conf_threshold
    points = points[mask]
    colors = colors[mask]
    
    num_points = points.shape[0]
    print(f"Saving {num_points} points to {save_path}")
    
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(save_path, 'w') as f:
        f.write(header)
        for i in range(num_points):
            x, y, z = points[i]
            r, g, b = colors[i].astype(np.uint8)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


def save_depth_map(depth, save_path, colormap=True):
    """Save depth map as PNG (16-bit) and optionally colored visualization."""
    depth_min, depth_max = depth.min(), depth.max()
    if depth_max - depth_min > 0:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth)
    
    depth_16bit = (depth_normalized * 65535).astype(np.uint16)
    cv2.imwrite(f"{save_path}.png", depth_16bit)
    np.save(f"{save_path}.npy", depth)
    
    if colormap: 
        depth_vis = (depth_normalized * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        cv2.imwrite(f"{save_path}_vis.png", depth_colored)
    
    meta = {"min": float(depth_min), "max": float(depth_max)}
    with open(f"{save_path}_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)


def save_camera_params(extrinsic, intrinsic, save_dir, frame_names):
    """Save camera parameters in multiple common formats."""
    os.makedirs(save_dir, exist_ok=True)
    num_frames = extrinsic.shape[0]
    
    for i in range(num_frames):
        frame_name = os.path.splitext(os.path.basename(frame_names[i]))[0]
        ext_3x4 = extrinsic[i]
        ext_4x4 = np.eye(4)
        ext_4x4[: 3, :] = ext_3x4
        np.savetxt(os.path.join(save_dir, f"{frame_name}_extrinsic.txt"), ext_4x4, fmt="%.8f")
        np.savetxt(os.path.join(save_dir, f"{frame_name}_intrinsic.txt"), intrinsic[i], fmt="%.8f")
    
    save_colmap_format(extrinsic, intrinsic, save_dir, frame_names)
    save_nerf_format(extrinsic, intrinsic, save_dir, frame_names)
    np.savez(os.path.join(save_dir, "cameras.npz"), extrinsic=extrinsic, intrinsic=intrinsic, frame_names=frame_names)


def save_colmap_format(extrinsic, intrinsic, save_dir, frame_names):
    """Save camera params in COLMAP text format."""
    colmap_dir = os.path.join(save_dir, "colmap")
    os.makedirs(colmap_dir, exist_ok=True)
    
    num_frames = extrinsic.shape[0]
    H, W = 518, 518
    
    with open(os.path.join(colmap_dir, "cameras.txt"), 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i in range(num_frames):
            K = intrinsic[i]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            f.write(f"{i+1} PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")
    
    with open(os.path.join(colmap_dir, "images.txt"), 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i in range(num_frames):
            ext = extrinsic[i]
            R = ext[:3, : 3]
            t = ext[: 3, 3]
            quat = rotation_matrix_to_quaternion(R)
            qw, qx, qy, qz = quat
            tx, ty, tz = t
            frame_name = os.path.basename(frame_names[i])
            f.write(f"{i+1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} {tx:.8f} {ty:.8f} {tz:.8f} {i+1} {frame_name}\n")
            f.write("\n")
    
    with open(os.path.join(colmap_dir, "points3D.txt"), 'w') as f:
        f.write("# 3D point list (empty)\n")


def save_nerf_format(extrinsic, intrinsic, save_dir, frame_names):
    """Save camera params in NeRF/3DGS transforms.json format."""
    num_frames = extrinsic.shape[0]
    K = intrinsic[0]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    H, W = 518, 518
    fov_x = 2 * np.arctan(W / (2 * fx))
    fov_y = 2 * np.arctan(H / (2 * fy))
    
    transforms = {
        "camera_angle_x": float(fov_x),
        "camera_angle_y": float(fov_y),
        "fl_x": fx, "fl_y": fy, "cx": cx, "cy":  cy,
        "w": W, "h": H,
        "frames": []
    }
    
    for i in range(num_frames):
        ext_3x4 = extrinsic[i]
        c2w = np.eye(4)
        c2w[:3, :] = ext_3x4
        c2w = np.linalg.inv(c2w)
        c2w[:, 1:3] *= -1
        frame = {"file_path": os.path.basename(frame_names[i]), "transform_matrix": c2w.tolist()}
        transforms["frames"].append(frame)
    
    with open(os.path.join(save_dir, "transforms.json"), 'w') as f:
        json.dump(transforms, f, indent=2)


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]: 
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def save_results(predictions, image_paths, output_dir, conf_threshold=0.5):
    """Save all results in standard formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    world_points = predictions["world_points"].cpu().numpy()
    world_points_conf = predictions["world_points_conf"].cpu().numpy()
    depth = predictions["depth"].cpu().numpy()
    images = predictions["images"].cpu().numpy()
    extrinsic = predictions["extrinsic"].cpu().numpy()
    intrinsic = predictions["intrinsic"].cpu().numpy()
    
    num_frames = world_points.shape[0]
    
    pc_dir = os.path.join(output_dir, "point_cloud")
    depth_dir = os.path.join(output_dir, "depth")
    cam_dir = os.path.join(output_dir, "cameras")
    img_dir = os.path.join(output_dir, "images")
    
    os.makedirs(pc_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    all_points, all_colors, all_confs = [], [], []
    
    print(f"Saving results for {num_frames} frames...")
    
    for i in range(num_frames):
        frame_name = f"frame_{i:06d}"
        pts = world_points[i].reshape(-1, 3)
        conf = world_points_conf[i].reshape(-1)
        img = images[i].transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        colors = img.reshape(-1, 3)
        
        save_point_cloud_ply(pts, colors, conf, os.path.join(pc_dir, f"{frame_name}.ply"), conf_threshold)
        all_points.append(pts)
        all_colors.append(colors)
        all_confs.append(conf)
        
        depth_frame = depth[i, :, : , 0]
        save_depth_map(depth_frame, os.path.join(depth_dir, frame_name))
        cv2.imwrite(os.path.join(img_dir, f"{frame_name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    merged_confs = np.concatenate(all_confs, axis=0)
    save_point_cloud_ply(merged_points, merged_colors, merged_confs, os.path.join(pc_dir, "merged.ply"), conf_threshold)
    
    frame_names = [f"frame_{i:06d}.png" for i in range(num_frames)]
    save_camera_params(extrinsic, intrinsic, cam_dir, frame_names)
    
    print(f"Results saved to {output_dir}")


def _get_event_paths(event_input: str) -> List[str]:
    """Collect .pt event tensors from a single file or directory."""
    if os.path.isfile(event_input) and event_input.endswith(".pt"):
        return [event_input]
    elif os.path.isdir(event_input):
        return sorted(glob.glob(os.path.join(event_input, "*.pt")))
    else:
        return []


def main():
    parser = argparse.ArgumentParser(description="StreamVGGT Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to image folder or video file")
    parser.add_argument("--checkpoint", type=str, 
                        default="",
                        help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--fps_interval", type=float, default=1.0, help="For video: extract one frame every N seconds")
    parser.add_argument("--conf_threshold", type=float, default=0.5, help="Confidence threshold for point cloud filtering (0-1)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--extractor", type=str, default="dino", choices=["dino", "evencoder"],
                        help="Feature extractor to use for patch tokens")
    parser.add_argument("--evencoder_checkpoint", type=str, default=None,
                        help="Path to EvEncoder checkpoint (expects keys 'student' and 'projector')")
    parser.add_argument("--ev_in_channels", type=int, default=8, help="EvEncoder input channels (voxel bins)")
    parser.add_argument("--ev_out_channels", type=int, default=16, help="EvEncoder output channels before projection")
    parser.add_argument("--input_type", type=str, default="auto", choices=["auto", "images", "video", "events"],
                        help="Force input interpretation; auto tries images/video/events by heuristics")
    args = parser.parse_args()
    
    inferencer = StreamVGGTInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        extractor=args.extractor,
        evencoder_checkpoint=args.evencoder_checkpoint,
        ev_in_channels=args.ev_in_channels,
        ev_out_channels=args.ev_out_channels,
    )
    
    # Decide input mode
    mode = args.input_type
    if mode == "auto":
        if os.path.isdir(args.input):
            # Prefer events if .pt files exist
            event_paths = _get_event_paths(args.input)
            if len(event_paths) > 0:
                mode = "events"
            else:
                mode = "images"
        elif os.path.isfile(args.input):
            if args.input.endswith(".pt"):
                mode = "events"
            else:
                mode = "video"
        else:
            raise ValueError(f"Input path not found: {args.input}")

    if mode == "events":
        event_paths = _get_event_paths(args.input)
        if len(event_paths) == 0:
            raise ValueError(f"No .pt event tensors found at {args.input}")
        print(f"Processing events: {args.input}")
        predictions, image_paths = inferencer.inference_from_events(event_paths)
    elif mode == "images":
        if not os.path.isdir(args.input):
            raise ValueError(f"Image mode expects a directory, got {args.input}")
        print(f"Processing image folder: {args.input}")
        predictions, image_paths = inferencer.inference_from_folder(args.input)
    elif mode == "video":
        if not os.path.isfile(args.input):
            raise ValueError(f"Video mode expects a file, got {args.input}")
        print(f"Processing video: {args.input}")
        predictions, image_paths = inferencer.inference_from_video(args.input, fps_interval=args.fps_interval)
    else:
        raise ValueError(f"Unsupported input_type: {mode}")
    
    save_results(predictions, image_paths, args.output, args.conf_threshold)
    torch.cuda.empty_cache()
    print("Done!")


if __name__ == "__main__":
    main()
