import os
import torch
import torch.nn as nn
from .layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

# Official DINOv2 pretrained weights URLs (with register tokens)
DINOV2_WEIGHTS_URLS = {
    "dinov2_vits14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2_vitb14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2_vitl14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2_vitg14_reg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth", 
}

class DINOv2Teacher(nn.Module):
    """
    DINOv2 Teacher model for knowledge distillation.
    
    This class wraps a DINOv2 Vision Transformer backbone and provides
    a clean interface for extracting patch-level features from RGB images.
    The model is frozen by default to serve as a stable teacher during distillation.
    """
    
    def __init__(self, model_name="dinov2_vitl14_reg", img_size=518, device="cuda", checkpoint_path=None):
        """
        Initialize DINOv2 Teacher model.
        
        Args:
            model_name: Model variant name, must be a key in DINOV2_WEIGHTS_URLS.
                       Options: 'dinov2_vits14_reg', 'dinov2_vitb14_reg', 
                                'dinov2_vitl14_reg', 'dinov2_vitg2_reg'
            img_size: Input image size (default: 518, standard DINOv2 size)
            device: Device to run the model on ('cuda' or 'cpu')
            checkpoint_path: Optional path to local checkpoint file. If provided,
                            weights will be loaded from this file instead of downloading.
                            The checkpoint can be either:
                            - A direct state_dict (dict of model parameters)
                            - A checkpoint dict with 'state_dict' or 'model' key
        """
        super().__init__()
        self.device = device
        self.img_size = img_size
        
        print(f"Initializing DINOv2 Teacher Model: {model_name}...")

        # Build model architecture based on variant name
        # Architecture selection must match the checkpoint's original architecture
        # to ensure proper weight loading compatibility
        if model_name == "dinov2_vits14_reg":
            model_fn = vit_small
        elif model_name == "dinov2_vitb14_reg":
            model_fn = vit_base
        elif model_name == "dinov2_vitl14_reg":
            model_fn = vit_large
        elif model_name == "dinov2_vitg2_reg":
            model_fn = vit_giant2
        else:
            raise ValueError(f"Unknown model name: {model_name}. "
                           f"Supported variants: {list(DINOV2_WEIGHTS_URLS.keys())}")

        # Initialize backbone with DINOv2-compatible configuration
        self.backbone = model_fn(
            img_size=img_size,
            patch_size=14,  # Standard patch size for DINOv2 models
            num_register_tokens=4,  # Register tokens improve training stability and feature quality
            interpolate_antialias=True,  # Smooth interpolation for better feature extraction
            interpolate_offset=0.0,
            block_chunks=0,  # No chunking for full attention computation
            init_values=1.0,  # Layer scale initialization value
        ).to(device)
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

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Explicitly freeze mask_token if present (used in masked autoencoder variants)
        if hasattr(self.backbone, "mask_token"):
            self.backbone.mask_token.requires_grad_(False)

        _RESNET_MEAN = [0.485, 0.456, 0.406]  # RGB channel means
        _RESNET_STD = [0.229, 0.224, 0.225]   # RGB channel standard deviations
        
        self.register_buffer(
            "_resnet_mean",
            torch.tensor(_RESNET_MEAN).view(1, 3, 1, 1).to(device),
            persistent=False
        )
        self.register_buffer(
            "_resnet_std",
            torch.tensor(_RESNET_STD).view(1, 3, 1, 1).to(device),
            persistent=False
        )

    def _load_weights(self, model_name, checkpoint_path=None):
        """
        Load model weights from local checkpoint or download from official URL.
        
        This method gracefully handles multiple checkpoint formats commonly used
        in PyTorch model serialization:
        - Direct state_dict: {'layer1.weight': tensor, ...}
        - Checkpoint dict with 'state_dict' key: {'state_dict': {...}, ...}
        - Checkpoint dict with 'model' key: {'model': {...}, ...}
        - Checkpoint dict with 'backbone' key: {'backbone': {...}, ...}
        
        The method also handles DataParallel artifacts by removing 'module.' prefixes
        that may be present in checkpoints saved from multi-GPU training.
        
        Args:
            model_name: Model variant name for URL fallback when local checkpoint unavailable
            checkpoint_path: Optional path to local checkpoint file (.pt or .pth)
            
        Returns:
            state_dict: Cleaned model state dictionary ready for loading, or None if all attempts fail
        """
        # Priority 1: Attempt to load from local checkpoint if path is provided
        if checkpoint_path is not None:
            # Resolve absolute path to handle relative paths correctly
            abs_checkpoint_path = os.path.abspath(checkpoint_path)
            
            if os.path.exists(abs_checkpoint_path):
                print(f"Loading weights from local checkpoint: {abs_checkpoint_path}")
                try:
                    checkpoint = torch.load(abs_checkpoint_path, map_location=self.device)
                    
                    # Extract state_dict from various checkpoint formats
                    # This handles different serialization conventions across projects
                    if isinstance(checkpoint, dict):
                        # Check for nested state_dict keys (common in training checkpoints)
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
                    # Models trained with DataParallel wrap layers with 'module.' prefix
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

    def forward(self, images):
        """
        Forward pass through the DINOv2 teacher model.
        
        Extracts patch-level features from RGB images. The input images are
        normalized using ImageNet statistics, then passed through the frozen
        DINOv2 backbone to obtain normalized patch tokens.
        
        Args:
            images: Input RGB images of shape [B, 3, H, W] with values in [0, 1].
                   Images should be pre-resized to match self.img_size.
        
        Returns:
            patch_tokens: Normalized patch tokens of shape [B, N_patches, Embed_Dim].
                         Contains only image patch tokens, excluding CLS and register tokens.
                         N_patches = (H // patch_size) * (W // patch_size)
        """
        if images.shape[-2:] != (self.img_size, self.img_size):
            pass 

        x = (images - self._resnet_mean) / self._resnet_std

        with torch.no_grad():
            features_dict = self.backbone(x)
            
        if isinstance(features_dict, dict):
            if "x_norm_patchtokens" in features_dict:
                patch_tokens = features_dict["x_norm_patchtokens"]
            else:
                available_keys = list(features_dict.keys())
                print(f"Warning: 'x_norm_patchtokens' not found. Available keys: {available_keys}")
                for key, value in features_dict.items():
                    if isinstance(value, torch.Tensor):
                        patch_tokens = value
                        break
                else:
                    raise ValueError("Could not extract patch tokens from backbone output")
        else:
            patch_tokens = features_dict
        
        return patch_tokens


if __name__ == "__main__":
    import sys
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Create dummy input matching DINOv2 standard size
    dummy_img = torch.rand(1, 3, 392, 518).to(device)
    
    # Test 1: Load from local checkpoint if available
    # This test verifies that the model can successfully load the custom checkpoint
    checkpoint_path = "/data/fcr/code/fcr/EvEncoder/ckpts/dinov2_stream_vggt.pt"
    abs_checkpoint_path = os.path.abspath(checkpoint_path)
    
    if os.path.exists(abs_checkpoint_path):
        print("=" * 60)
        print("Test 1: Loading from local checkpoint")
        print("=" * 60)
        teacher = DINOv2Teacher(
            model_name="dinov2_vitl14_reg", 
            device=device,
            checkpoint_path=abs_checkpoint_path
        )
        out = teacher(dummy_img)
        print(f"✓ Success! Output shape: {out.shape}\n")
    else:
        print(f"⚠ Checkpoint not found at {abs_checkpoint_path}, skipping local test\n")
    

# all resize to torch.Size([49, 3, 392, 518])
# input image shape: torch.Size([1, 3, 392, 518])
# dino output shape: torch.Size([1, 1036, 1024])