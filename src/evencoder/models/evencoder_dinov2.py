import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
        
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(32, out_channels) 
        self.act = nn.SiLU() 
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return h + self.shortcut(x)

class ETFModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_x_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU()
        )
        self.conv_h_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU()
        )
        self.f_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1), 
            nn.Sigmoid()
        )
    def forward(self, x_curr, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros_like(x_curr)
        feat_x = self.conv_x_gate(x_curr)
        feat_h = self.conv_h_gate(h_prev)
        gate_input = torch.cat([feat_x, feat_h], dim=1)
        g_i = self.f_gate(gate_input)
        y_curr = g_i * x_curr + (1 - g_i) * h_prev
        return y_curr

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class EvEncodBlockBone(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res_block = ResBlock(in_channels, in_channels)
        self.etf = ETFModule(in_channels)                   
        self.down = Downsample(in_channels, out_channels)   
        
    def forward(self, x, h_prev=None):
        x = self.res_block(x)
        h_curr = self.etf(x, h_prev)
        out = self.down(h_curr)
        return out, h_curr


class DINOv2(nn.Module):
    def __init__(self, model_name="dinov2_vitl14_reg", device="cuda", checkpoint_path=None):
        super().__init__()
        self.device = device
        print(f"Loading Frozen DINOv2: {model_name}...")      

        # Build model architecture based on variant name
        if model_name == "dinov2_vits14_reg":
            model_fn = vit_small
        elif model_name == "dinov2_vitb14_reg":
            model_fn = vit_base
        elif model_name == "dinov2_vitl14_reg":
            model_fn = vit_large
        elif model_name == "dinov2_vitg2_reg":
            model_fn = vit_giant2
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Initialize backbone with DINOv2-compatible configuration
        self.backbone = model_fn(
            img_size=518,  # Standard DINOv2 size
            patch_size=14,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            block_chunks=0,
            init_values=1.0,
        ).to(device)
        
        # Load weights
        state_dict = self._load_weights(model_name, checkpoint_path)
        
        if state_dict is not None:
            msg = self.backbone.load_state_dict(state_dict, strict=False)
            
            if msg.missing_keys:
                missing_preview = msg.missing_keys[:5]
                print(f"Warning: Missing keys in checkpoint: {missing_preview}...")
            if msg.unexpected_keys:
                unexpected_preview = msg.unexpected_keys[:5]
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_preview}...")
            
            print(f"✓ Weights loaded successfully")
        else:
            print(f"⚠ Warning: No weights loaded for {model_name}, using random initialization!")
        
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        
        # Freeze parameters
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def _load_weights(self, model_name, checkpoint_path=None):
        """
        Load model weights from local checkpoint or download from official URL.
        
        This method gracefully handles multiple checkpoint formats:
        - Direct state_dict: {'layer1.weight': tensor, ...}
        - Checkpoint dict with 'state_dict' key: {'state_dict': {...}, ...}
        - Checkpoint dict with 'model' key: {'model': {...}, ...}
        - Checkpoint dict with 'backbone' key: {'backbone': {...}, ...}
        
        Also handles DataParallel artifacts by removing 'module.' prefixes.
        
        Args:
            model_name: Model variant name for URL fallback when local checkpoint unavailable
            checkpoint_path: Optional path to local checkpoint file (.pt or .pth)
            
        Returns:
            state_dict: Cleaned model state dictionary ready for loading, or None if all attempts fail
        """
        # Official DINOv2 pretrained weights URLs
        DINOV2_WEIGHTS_URLS = {
            "dinov2_vits14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
            "dinov2_vitb14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
            "dinov2_vitl14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
            "dinov2_vitg2_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
        }
        
        # Priority 1: Attempt to load from local checkpoint if path is provided
        if checkpoint_path is not None:
            abs_checkpoint_path = os.path.abspath(checkpoint_path)
            
            if os.path.exists(abs_checkpoint_path):
                print(f"Loading weights from local checkpoint: {abs_checkpoint_path}")
                try:
                    checkpoint = torch.load(abs_checkpoint_path, map_location=self.device)
                    
                    # Extract state_dict from various checkpoint formats
                    if isinstance(checkpoint, dict):
                        # Check for nested state_dict keys
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'backbone' in checkpoint:
                            state_dict = checkpoint['backbone']
                        else:
                            # Assume the entire dict is the state_dict itself
                            state_dict = checkpoint
                    else:
                        # If checkpoint is not a dict, assume it's already a state_dict
                        state_dict = checkpoint
                    
                    # Clean state_dict keys to remove DataParallel artifacts
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        # Remove 'module.' prefix if present (handles DataParallel checkpoints)
                        new_key = key.replace('module.', '') if key.startswith('module.') else key
                        cleaned_state_dict[new_key] = value
                    
                    return cleaned_state_dict
                    
                except Exception as e:
                    print(f"⚠ Error loading checkpoint from {abs_checkpoint_path}: {e}")
                    print(f"Falling back to URL download...")
            else:
                print(f"⚠ Checkpoint file not found: {abs_checkpoint_path}")
                print(f"Falling back to URL download...")
        
        # Priority 2: Download from official DINOv2 repository as fallback
        if model_name in DINOV2_WEIGHTS_URLS:
            url = DINOV2_WEIGHTS_URLS[model_name]
            print(f"Downloading weights from official URL: {url}")
            try:
                state_dict = torch.hub.load_state_dict_from_url(url, map_location=self.device)
                return state_dict
            except Exception as e:
                print(f"⚠ Error downloading weights: {e}")
                return None
        
        return None
    
    def interpolate_pos_encoding(self, x, w, h):
        """
        Interpolate positional embeddings to match grid size (w, h).
        Necessary because Event features might not exactly match pre-training resolution.
        """
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        pos_embed = self.backbone.pos_embed
        N = pos_embed.shape[1] - 1
        
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        
        M = int(math.sqrt(N)) # e.g. 37 for 518 size
        
        # Reshape to spatial for interpolation
        patch_pos_embed = patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed.float(),
            size=(h, w), # Target grid size
            mode='bicubic',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)


    def forward_from_latent(self, x_latent, grid_size):
        """
        Bypass patch embedding, inject features directly into blocks.
        x_latent: [B, N_patches, Dim]
        grid_size: (h, w) patches
        """
        B, N, C = x_latent.shape
        
        # 1. Prepare CLS + Register Tokens
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        tokens = [cls_token]
        if hasattr(self.backbone, 'register_tokens') and self.backbone.register_tokens is not None:
            reg_tokens = self.backbone.register_tokens.expand(B, -1, -1)
            tokens.append(reg_tokens)
            
        # 2. Concat
        x = torch.cat(tokens + [x_latent], dim=1)
        
        # 3. Add Positional Embedding (Interpolated)
        pos_embed = self.interpolate_pos_encoding(x, grid_size[1], grid_size[0])
        
        # Handle register token length mismatch for pos embed
        if x.shape[1] != pos_embed.shape[1]:
             diff = x.shape[1] - pos_embed.shape[1]
             pos_embed = torch.cat([pos_embed[:, :1].expand(-1, diff, -1), pos_embed], dim=1)
             
        x = x + pos_embed
        
        # 4. Pass through Transformer Blocks
        for blk in self.backbone.blocks:
            x = blk(x)
            
        x = self.backbone.norm(x)
        
        # 5. Return Patch Tokens only (remove CLS/Reg)
        num_special = 1 + (self.backbone.num_register_tokens if hasattr(self.backbone, 'num_register_tokens') else 0)
        return x[:, num_special:]


class EvEncoder(nn.Module):
    def __init__(self, 
                 in_channels=8, 
                 base_channels=64, 
                 dino_model="dinov2_vitl14_reg",
                 checkpoint_path=None,
                 target_res=(392, 518),
                 enable_decoder=False,
                 recon_out_channels=1):
        """
        Args:
            in_channels: Event voxel channels.
            base_channels: Channel multiplier for EvEncoder.
            dino_model: Which DINOv2 backbone to use.
            target_res: (H, W) The target RGB resolution we want to mimic.
            enable_decoder: Whether to build a decoder for grayscale reconstruction.
            recon_out_channels: Output channels for reconstruction (default: 1 for grayscale).
        """
        super().__init__()
        
        self.target_res = target_res
        
        # --- Part 1: Event Encoder (Trainable) ---
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.stage1 = EvEncodBlockBone(base_channels, base_channels * 2)     # 64 -> 128
        self.stage2 = EvEncodBlockBone(base_channels * 2, base_channels * 4) # 128 -> 256
        self.stage3 = EvEncodBlockBone(base_channels * 4, base_channels * 4) # 256 -> 256
        
        self.ev_out_channels = base_channels * 4 # 256

        self.enable_decoder = enable_decoder
        if self.enable_decoder:
            self.decoder = nn.Sequential(
                nn.Conv2d(self.ev_out_channels, base_channels * 2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(base_channels // 2, recon_out_channels, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )
        else:
            self.decoder = None
        
        # --- Part 2: Frozen DINOv2 Backbone ---
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dino = DINOv2(dino_model, device=device, checkpoint_path=checkpoint_path)
        
        # Calculate grid size based on target resolution
        # 392 // 14 = 28, 518 // 14 = 37
        self.grid_h = target_res[0] // self.dino.patch_size
        self.grid_w = target_res[1] // self.dino.patch_size
        
        # --- Part 3: Latent Projector (Trainable) ---
        # Projects EvEncoder features (256) to DINO Dim (1024)
        # Also handles "domain gap" between sparse events and dense RGB features
        self.projector = nn.Sequential(
            nn.Conv2d(self.ev_out_channels, self.dino.embed_dim, kernel_size=1),
            nn.GroupNorm(32, self.dino.embed_dim),
            nn.SiLU(),
            # Optional: Add another conv layer if more capacity is needed
        )
        
    def forward(self, voxel_grid, hidden_states=None, return_recon=False):
        """
        Args:
            voxel_grid: [B, T, C_in, H, W]
            hidden_states: List of states for ETF modules
            return_recon: If True and decoder is enabled, return grayscale reconstruction.
        Returns:
            dino_features: [B, T, N_patches, DINO_Dim]
            next_states: Updated hidden states
            recon_out: [B, T, 1, H, W] if return_recon else None
        """
        B, T, C, H, W = voxel_grid.shape
        
        if hidden_states is None:
            hidden_states = [None, None, None]
            
        z_list = []
        next_states = [None, None, None] # Just placeholders
        
        # 1. Process Temporal Sequence with EvEncoder
        # We process frame-by-frame for temporal correctness (ETFModule) Note: To optimize speed, we collect features first then run DINO in parallel
        ev_features_list = []
        current_states = hidden_states
        recon_list = []
        
        for t in range(T):
            x = voxel_grid[:, t]
            
            # Stem
            feat = self.stem(x) 
            # Stages
            feat, h1 = self.stage1(feat, current_states[0])
            feat, h2 = self.stage2(feat, current_states[1])
            feat, h3 = self.stage3(feat, current_states[2])
            
            # feat shape: [B, 256, H/8, W/8]
            ev_features_list.append(feat)

            if return_recon and self.decoder is not None:
                recon_list.append(self.decoder(feat))
            
            current_states = [h1, h2, h3]
            
        # 2. Batch Projection & DINO Injection
        ev_features = torch.stack(ev_features_list, dim=1) # Stack: [B, T, 256, H/8, W/8]
        
        # Flatten Batch and Time for efficient parallel processing
        flat_ev = ev_features.view(B*T, -1, ev_features.shape[-2], ev_features.shape[-1]) # [B*T, 256, H/8, W/8]
        proj_feat = self.projector(flat_ev) # Project: [B*T, 1024, H/8, W/8]
        
        dino_input_map = F.interpolate(proj_feat, size=(self.grid_h, self.grid_w), mode='bilinear', align_corners=False) # Interpolate to DINO Grid Size: [B*T, 1024, 28, 37] TODO: check if this is correct
        tokens = dino_input_map.flatten(2).transpose(1, 2) # Flatten to tokens: [B*T, 1024, 28*37] -> [B*T, N, 1024]
        dino_out = self.dino.forward_from_latent(tokens, grid_size=(self.grid_h, self.grid_w)) # [B*T, N, 1024] -> [B*T, N, 1024]
        dino_out = dino_out.view(B, T, -1, self.dino.embed_dim) # [B*T, N, 1024] -> [B, T, N, 1024]
        
        recon_out = torch.stack(recon_list, dim=1) if recon_list else None
        return dino_out, recon_out, current_states

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Test 1: Load from local checkpoint if available
    checkpoint_path = "/data/fcr/code/fcr/EvEncoder/ckpts/dinov2_stream_vggt.pt"
    abs_checkpoint_path = os.path.abspath(checkpoint_path)
    
    if os.path.exists(abs_checkpoint_path):
        print("=" * 60)
        print("Test 1: Loading from local checkpoint")
        print("=" * 60)
        model = EvEncoder(
            in_channels=8,
            base_channels=64,
            dino_model="dinov2_vitl14_reg",
            checkpoint_path=abs_checkpoint_path,
            target_res=(392, 518)
        ).to(device)
    else:
        print("=" * 60)
        print("Test 1: Loading from official URL (checkpoint not found locally)")
        print("=" * 60)
        # Initialize the complete model
        # Assume input is 8 channels (Event Voxel), target RGB res is 392x518
        model = EvEncoder(
            in_channels=8,
            base_channels=64,
            dino_model="dinov2_vitl14_reg", 
            target_res=(392, 518)
        ).to(device)
    
    print("\nModel Initialized:")
    print(f"  EvEncoder Channels: {model.ev_out_channels} -> Projector -> {model.dino.embed_dim}")
    print(f"  Target Resolution: {model.target_res} -> Grid: {model.grid_h}x{model.grid_w}")
    print(f"  DINO Frozen: {not next(model.dino.backbone.parameters()).requires_grad}")
    
    # Dummy Input: [B=1, T=3, C=8, H=256, W=256]
    # Note: Input H/W can be anything, model will interpolate to 392x518 internal representation
    dummy_voxels = torch.randn(1, 3, 8, 256, 256).to(device)
    
    print(f"\nForward Pass Input: {dummy_voxels.shape}")
    
    # Run
    features, states = model(dummy_voxels)
    
    print("-" * 30)
    print("Output Analysis:")
    print(f"Feature Shape: {features.shape}") 
    # Expect: [1, 3, 1036, 1024]
    # 1036 = (392//14) * (518//14) = 28 * 37
    
    expected_patches = (392 // 14) * (518 // 14)
    print(f"Expected Patches: {expected_patches}")
    print(f"Actual Patches:   {features.shape[2]}")
    
    # Verify gradients
    print("-" * 30)
    loss = features.mean()
    loss.backward()
    print("Backward pass successful.")
    print(f"Projector Grad: {model.projector[0].weight.grad is not None}") # Should be True
    print(f"Stem Grad:      {model.stem.weight.grad is not None}")      # Should be True
    print(f"DINO Grad:      {model.dino.backbone.blocks[0].ls1.gamma.grad is not None}") # Should be False
