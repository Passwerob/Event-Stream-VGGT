# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any
from torch import serialization as torch_serialization

from streamvggt.layers import PatchEmbed
from streamvggt.layers.block import Block
from streamvggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from streamvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class EvEncoderPatchEmbed(nn.Module):
    """Wrap EvEncoder + projector to produce patch tokens matching StreamVGGT expectations."""

    def __init__(self, encoder: nn.Module, projector: nn.Module, patch_size: int, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel: [B, C, H, W] event voxel grid for a single frame.
        Returns:
            tokens: [B, N_patches, embed_dim]
        """
        B, C, H, W = voxel.shape
        # EvEncoder expects [B, T, C, H, W], treat each frame independently.
        encoder_out, _ = self.encoder(voxel.unsqueeze(1))
        feats = encoder_out[:, 0]  # [B, C_out, H', W']
        proj = self.projector(feats)  # [B, embed_dim, H', W']

        # Resize to match StreamVGGT patch grid (img_size // patch_size)
        grid_h = max(1, H // self.patch_size)
        grid_w = max(1, W // self.patch_size)
        proj = F.interpolate(proj, size=(grid_h, grid_w), mode="bilinear", align_corners=False)

        tokens = proj.flatten(2).transpose(1, 2)  # [B, N_patches, embed_dim]
        return tokens


class EvEncoderV2PatchEmbed(nn.Module):
    """Wrap EvEncoder-v2 (DINO tokens) to match StreamVGGT patch token shape."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, voxel_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_seq: [B, S, C, H, W] event voxel sequence.
        Returns:
            tokens: [B*S, N_patches, embed_dim]
        """
        if voxel_seq.dim() != 5:
            raise ValueError(f"Expected 5D input [B, S, C, H, W], got {voxel_seq.shape}")

        encoder_out, _ = self.encoder(voxel_seq)
        B, S, P, D = encoder_out.shape
        return encoder_out.reshape(B * S, P, D)


class EvEncoderV3PatchEmbed(nn.Module):
    """Wrap EvEncoder-v3 (E2VID) to match StreamVGGT patch token shape."""

    def __init__(self, encoder: nn.Module, patch_size: int, embed_dim: int):
        super().__init__()
        self.encoder = encoder
        self.patch_size = patch_size
        self.embed_dim = embed_dim

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel: [B, C, H, W] event voxel grid for a single frame.
        Returns:
            tokens: [B, N_patches, embed_dim]
        """
        B, C, H, W = voxel.shape
        encoder_out, _ = self.encoder(voxel.unsqueeze(1), use_projector=True)
        feats = encoder_out[:, 0]  # [B, embed_dim, H', W']

        grid_h = max(1, H // self.patch_size)
        grid_w = max(1, W // self.patch_size)
        feats = F.interpolate(feats, size=(grid_h, grid_w), mode="bilinear", align_corners=False)

        tokens = feats.flatten(2).transpose(1, 2)
        return tokens


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        evencoder_ckpt_path (str): Optional checkpoint path for EvEncoder extractor.
        evencoder_in_channels (int): EvEncoder input channels (voxel bins).
        evencoder_out_channels (int): EvEncoder output channels before projection.
        evencoder_student_key (str): Key used to find EvEncoder weights inside checkpoint dict.
        evencoder_projector_key (str): Key used to find projector weights inside checkpoint dict.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        evencoder_ckpt_path=None,
        evencoder_in_channels=8,
        evencoder_out_channels=16,
        evencoder_student_key="student",
        evencoder_projector_key="projector",
    ):
        super().__init__()

        self.patch_embed_type = patch_embed
        self.use_evencoder = patch_embed in {"evencoder", "evencoder-v2", "evencoder-v3"}
        self.use_evencoder_v2 = patch_embed == "evencoder-v2"

        self.__build_patch_embed__(
            patch_embed,
            img_size,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
            evencoder_ckpt_path=evencoder_ckpt_path,
            evencoder_in_channels=evencoder_in_channels,
            evencoder_out_channels=evencoder_out_channels,
            evencoder_student_key=evencoder_student_key,
            evencoder_projector_key=evencoder_projector_key,
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )


    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
        evencoder_ckpt_path=None,
        evencoder_in_channels=8,
        evencoder_out_channels=16,
        evencoder_student_key="student",
        evencoder_projector_key="projector",
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        elif patch_embed == "evencoder-v2":
            from evencoder.models.evencoder_dinov2 import EvEncoder

            # Pass evencoder_ckpt_path to EvEncoder so DINOv2 can extract weights from it
            # This avoids downloading DINOv2 weights when they're already in the checkpoint
            encoder = EvEncoder(in_channels=evencoder_in_channels, checkpoint_path=evencoder_ckpt_path)
            encoder.requires_grad_(False)

            self.patch_embed = EvEncoderV2PatchEmbed(encoder=encoder)
            self._load_evencoder_weights(
                evencoder_ckpt_path,
                encoder,
                projector=None,
                student_key=evencoder_student_key,
                projector_key=None,
            )
        elif patch_embed == "evencoder":
            from evencoder.models.evencoder import EvEncoder
            from evencoder.utils.loss_utils import FeatureProjector

            encoder = EvEncoder(in_channels=evencoder_in_channels, out_channels=evencoder_out_channels)
            projector = FeatureProjector(in_channels=evencoder_out_channels, out_channels=embed_dim)

            # Freeze extractor by default to mirror pretrained DINO behavior
            encoder.requires_grad_(False)
            projector.requires_grad_(False)

            self.patch_embed = EvEncoderPatchEmbed(
                encoder=encoder,
                projector=projector,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )

            self._load_evencoder_weights(
                evencoder_ckpt_path,
                encoder,
                projector,
                student_key=evencoder_student_key,
                projector_key=evencoder_projector_key,
            )
        elif patch_embed == "evencoder-v3":
            from evencoder.models.evencoder_e2vid import EvEncoderE2VID

            encoder = EvEncoderE2VID(
                in_channels=evencoder_in_channels,
                out_channels=evencoder_out_channels,
                project_out_channels=embed_dim,
            )
            encoder.requires_grad_(False)

            self.patch_embed = EvEncoderV3PatchEmbed(
                encoder=encoder,
                patch_size=patch_size,
                embed_dim=embed_dim,
            )

            self._load_evencoder_weights(
                evencoder_ckpt_path,
                encoder,
                projector=None,
                student_key=evencoder_student_key,
                projector_key=None,
            )
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )
            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if use_cache and past_key_values[0] is not None:
            _, _, S_true, _, _ = past_key_values[0][0].shape
            S_true += 1
        else:
            S_true = S

        if use_cache and S > 1:
            print(f"Use KV cache expects S=1, got S={S}")

        if (not self.use_evencoder) and C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        if not self.use_evencoder:
            images = (images - self._resnet_mean.to(images.device)) / self._resnet_std.to(images.device)

        # Reshape to [B*S, C, H, W] for patch embedding
        if self.use_evencoder_v2:
            patch_tokens = self.patch_embed(images)
        else:
            images = images.reshape(B * S, C_in, H, W)
            patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        if use_cache:
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-1:, :, :]
            
            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full[-1:, :, :]
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        # Concatenate special tokens with patch tokens
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            grid_h = H // self.patch_size
            grid_w = W // self.patch_size

            if self.use_evencoder_v2:
                enc = getattr(self.patch_embed, "encoder", None)
                grid_h = getattr(enc, "grid_h", grid_h)
                grid_w = getattr(enc, "grid_w", grid_w)

            # Align grid to the actual number of patch tokens to avoid reshape errors
            actual_patches = patch_tokens.shape[1]
            expected_patches = grid_h * grid_w
            if expected_patches != actual_patches and actual_patches > 0:
                if grid_w > 0 and actual_patches % grid_w == 0:
                    grid_h = actual_patches // grid_w
                else:
                    grid_h, grid_w = actual_patches, 1

            pos = self.position_getter(B * S, grid_h, grid_w, device=images.device)

        if self.patch_start_idx > 0 and pos is not None:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if use_cache:
                        if past_key_values[global_idx] is not None:
                            k, v = past_key_values[global_idx]
                        tokens, global_idx, global_intermediates, new_kv = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values[global_idx] if past_key_values[global_idx] is not None else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx
                        )
                        past_key_values[global_idx - 1] = new_kv
                    else: 
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        if use_cache:      
            return output_list, self.patch_start_idx, past_key_values
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
                """
        
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.reshape(B, S, P, 2).reshape(B, S * P, 2)
            
        intermediates = []

        for _ in range(self.aa_block_size):
            if not use_cache:
                L = S * P
                frame_ids = torch.arange(L, device=tokens.device) // P  # [0,0,...,1,1,...,S-1]
                future_frame = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)
                attn_mask = future_frame.to(tokens.dtype) * torch.finfo(tokens.dtype).min
            else:
                attn_mask = None
                
            if use_cache:
                tokens, block_kv = self.global_blocks[global_idx](
                    tokens, 
                    pos=pos, 
                    attn_mask=attn_mask, 
                    past_key_values=past_key_values_block,
                    use_cache=True
                )
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos, attn_mask=attn_mask)
            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv
        return tokens, global_idx, intermediates

    def _load_evencoder_weights(
        self,
        ckpt_path: Optional[str],
        encoder: nn.Module,
        projector: Optional[nn.Module],
        student_key: str = "student",
        projector_key: Optional[str] = "projector",
    ) -> None:
        """Load EvEncoder (and optional projector) weights while keeping StreamVGGT compatibility."""
        if ckpt_path is None:
            logger.info("EvEncoder checkpoint path not provided; using random init for extractor.")
            return

        resolved_path = os.path.expanduser(ckpt_path)
        if not os.path.exists(resolved_path):
            logger.warning(f"EvEncoder checkpoint not found at {resolved_path}")
            return

        try:
            add_safe_globals = getattr(torch_serialization, "add_safe_globals", None)
            safe_globals_ctx = getattr(torch_serialization, "safe_globals", None)

            if add_safe_globals is not None:
                add_safe_globals([argparse.Namespace])
                ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
            elif safe_globals_ctx is not None:
                with safe_globals_ctx([argparse.Namespace]):
                    ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
            else:
                ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            logger.warning(f"Failed to load EvEncoder checkpoint {resolved_path}: {exc}")
            return

        student_state = None
        projector_state = None

        if isinstance(ckpt, dict):
            student_state = ckpt.get(student_key) or ckpt.get("state_dict") or ckpt.get("model") or ckpt
            if projector is not None and projector_key is not None:
                projector_state = ckpt.get(projector_key)
        else:
            student_state = ckpt

        if student_state is not None:
            msg = encoder.load_state_dict(student_state, strict=False)
            if msg.missing_keys or msg.unexpected_keys:
                logger.info(f"EvEncoder load_state_dict info - missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)}")
        else:
            logger.warning("No student weights found in EvEncoder checkpoint.")

        if projector is not None and projector_state is not None:
            proj_msg = projector.load_state_dict(projector_state, strict=False)
            if proj_msg.missing_keys or proj_msg.unexpected_keys:
                logger.info(f"Projector load_state_dict info - missing: {len(proj_msg.missing_keys)}, unexpected: {len(proj_msg.unexpected_keys)}")


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined
