import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    """
    Conv -> Norm -> SiLU -> Conv -> Norm -> (Skip Connection)
    """
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
    """
    Gate Network
    """
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
        """
        Args:
            x_curr: [B, C, H, W]
            h_prev: [B, C, H, W] 
        Returns:
            y_curr: Fusion Feature
        """
        if h_prev is None:
            h_prev = torch.zeros_like(x_curr)
            
        feat_x = self.conv_x_gate(x_curr)
        feat_h = self.conv_h_gate(h_prev)
        
        # compute Gate g_i
        gate_input = torch.cat([feat_x, feat_h], dim=1)
        g_i = self.f_gate(gate_input)
        
        # Fusion
        # y_i = g_i * x_i + (1 - g_i) * h_{i-1}
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

class EvEncoder(nn.Module):
    def __init__(self, in_channels=8, base_channels=64, out_channels=16):
        """
        Args:
            in_channels: Voxel Grid channels
            base_channels: Hidden Layer channels
            out_channels: Latent Feature channels
        """
        super().__init__()
        
        self.stem = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # TODO: Validate the dimension?
        self.stage1 = EvEncodBlockBone(base_channels, base_channels * 2)
        # Stage 2: 2C -> 4C, H/2 -> H/4
        self.stage2 = EvEncodBlockBone(base_channels * 2, base_channels * 4)
        # Stage 3: 4C -> 4C, H/4 -> H/8 
        self.stage3 = EvEncodBlockBone(base_channels * 4, base_channels * 4)
        self.out_proj = nn.Conv2d(base_channels * 4, out_channels, kernel_size=3, padding=1)
        
    def forward(self, voxel_grid, hidden_states=None):
        """
        Args:
            voxel_grid: [B, T, C_in, H, W]
            hidden_states: list of tensors
        Returns:
            z: [B, T, C_out, H/8, W/8]
            last_states: list of tensors
        """
        B, T, C, H, W = voxel_grid.shape
        
        if hidden_states is None:
            hidden_states = [None, None, None]
        z_list = []
        
        for t in range(T):
            x = voxel_grid[:, t] # [B, C, H, W]
            feat = self.stem(x)
            feat, h1 = self.stage1(feat, hidden_states[0]) # feat [B, 64, H, W] -> [B, 128, H/2, W/2]
            feat, h2 = self.stage2(feat, hidden_states[1]) # feat [B, 128, H/2, W/2] -> [B, 256, H/4, W/4]
            feat, h3 = self.stage3(feat, hidden_states[2]) # feat [B, 256, H/4, W/4] -> [B, 256, H/8, W/8]
            
            z_t = self.out_proj(feat) # [B, 16, H/8, W/8]
            z_list.append(z_t)
            hidden_states = [h1, h2, h3]
            
        z = torch.stack(z_list, dim=1) # [B, T, C_out, H/8, W/8]
        return z, hidden_states


if __name__ == "__main__":
    import os
    from pathlib import Path
    
    # Test with real data
    events_dir = Path("/data/fcr/code/fcr/EvEncoder/data/test/zurich_city_02_b_49/events")
    
    print("=" * 60)
    print("Testing EvEncoder with real event data")
    print("=" * 60)
    
    # Load event files
    event_files = sorted([f for f in events_dir.glob("*.pt")])
    print(f"\nFound {len(event_files)} event files")
    
    if len(event_files) == 0:
        print("Error: No event files found!")
        exit(1)
    
    # Load first file to check dimensions
    sample = torch.load(event_files[0], weights_only=True)
    print(f"Sample event shape: {sample.shape}")
    print(f"Sample event dtype: {sample.dtype}")
    print(f"Sample event range: [{sample.min():.2f}, {sample.max():.2f}]")
    print(f"Sample event non-zero: {torch.count_nonzero(sample):,}")
    
    # Load all events
    print(f"\nLoading {len(event_files)} event files...")
    events_list = []
    for i, event_file in enumerate(event_files):
        event_data = torch.load(event_file, weights_only=True)  # [8, 1080, 1440]
        # Use first 5 bins to match model's in_channels=5
        event_data = event_data[:8]  # [5, 1080, 1440]
        events_list.append(event_data)
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1}/{len(event_files)} files...")
    
    # Stack into sequence: [T, C, H, W]
    events_sequence = torch.stack(events_list, dim=0)  # [49, 5, 1080, 1440]
    print(f"\nEvent sequence shape: {events_sequence.shape}")
    
    # Model configuration
    C_in = 8  # Using first 5 bins
    C_out = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Create model
    model = EvEncoder(in_channels=C_in, out_channels=C_out).to(device)
    model.eval()
    
    # Test 1: Batch processing (training mode)
    print("\n" + "=" * 60)
    print("Test 1: Batch Processing (Training Mode)")
    print("=" * 60)
    
    # Use first 8 frames as a batch
    T_batch = 8
    events_batch = events_sequence[:T_batch].unsqueeze(0).to(device)  # [1, 8, 5, 1080, 1440]
    print(f"Input shape: {events_batch.shape}")
    
    with torch.no_grad():
        latents, _ = model(events_batch)
        print(f"Output latent shape: {latents.shape}")
        print(f"Output latent range: [{latents.min():.4f}, {latents.max():.4f}]")
        print(f"Output latent mean: {latents.mean():.4f}, std: {latents.std():.4f}")
    
    # Test 2: Streaming inference mode
    print("\n" + "=" * 60)
    print("Test 2: Streaming Inference Mode")
    print("=" * 60)
    
    states = None
    num_test_frames = min(10, len(events_sequence))
    
    with torch.no_grad():
        for t in range(num_test_frames):
            single_frame = events_sequence[t:t+1].unsqueeze(0).to(device)  # [1, 1, 5, 1080, 1440]
            latent_t, states = model(single_frame, hidden_states=states)
            print(f"Frame {t:2d}: Input {single_frame.shape[2:]} -> Latent {latent_t.shape[2:]}, "
                  f"non-zero: {torch.count_nonzero(latent_t).item():,}")
    
    # Test 3: Full sequence processing
    print("\n" + "=" * 60)
    print("Test 3: Full Sequence Processing")
    print("=" * 60)
    
    # Process in chunks to avoid memory issues
    chunk_size = 8
    all_latents = []
    states = None
    
    with torch.no_grad():
        for i in range(0, len(events_sequence), chunk_size):
            chunk = events_sequence[i:i+chunk_size].unsqueeze(0).to(device)  # [1, T_chunk, 5, 1080, 1440]
            latents_chunk, states = model(chunk, hidden_states=states)
            all_latents.append(latents_chunk.cpu())
            print(f"Processed frames {i:2d}-{min(i+chunk_size-1, len(events_sequence)-1):2d}: "
                  f"latent shape {latents_chunk.shape}")
    
    # Concatenate all latents
    full_latents = torch.cat(all_latents, dim=1)  # [1, 49, 16, H/8, W/8]
    print(f"\nFull sequence latent shape: {full_latents.shape}")
    print(f"Full sequence latent stats: mean={full_latents.mean():.4f}, "
          f"std={full_latents.std():.4f}, non-zero={torch.count_nonzero(full_latents).item():,}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)