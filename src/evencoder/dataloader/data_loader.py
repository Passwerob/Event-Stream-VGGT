import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import time
import json

class PairedEventImageDataset(Dataset):
    """
    Dataset class to load paired Event Voxel Grids and RGB Images.
    
    Expected Data Structure:
        root_dir/
            sequence_name/
                events/
                    000000.pt
                    000001.pt
                    ...
                images/
                    000000.png
                    000001.png
                    ...
    
    Data Format:
        - Events: .pt files containing voxel grids [C_event, H, W] / [Event_bin_num, H, W]
        - Images: .png files [H, W, 3]
    """
    def __init__(self, root_dir, sequence_length=8, transform=None, image_size=(392, 518)):
        """
        Args:
            root_dir: Path to the data directory (e.g., "data/processed_data")
            sequence_length: Number of frames per sequence
            transform: Optional transform to apply to images
            image_size:  Target size (H, W) for resizing images for DINOv2
        """
        self.root_dir = Path(root_dir)
        self.seq_len = sequence_length
        self.image_size = image_size
        
        # Scan all sequences in root_dir
        self.sequences = []
        for seq_dir in sorted(self.root_dir.iterdir()):
            if seq_dir.is_dir():
                events_dir = seq_dir / "events"
                images_dir = seq_dir / "images"
                
                if events_dir.exists() and images_dir.exists():
                    # Get all event files
                    event_files = sorted(events_dir.glob("*.pt"))
                    image_files = sorted(images_dir.glob("*.png"))
                    
                    # Verify matching counts
                    if len(event_files) == len(image_files):
                        self.sequences.append({
                            'name': seq_dir.name,
                            'events': event_files,
                            'images': image_files,
                            'length': len(event_files)
                        })
        
        # Create samples with sliding window
        self.samples = []
        for seq in self.sequences:
            num_frames = seq['length']
            # Create overlapping windows
            for start_idx in range(0, num_frames - self.seq_len + 1):
                self.samples.append({
                    'sequence': seq,
                    'start_idx': start_idx
                })
        
        # Default transform for images
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),  # Resize to DINOv2 input size
                transforms.ToTensor(),  # Convert to tensor [0, 1]
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # #region agent log
        data_load_start = time.time()
        # #endregion
        
        sample = self.samples[idx]
        seq = sample['sequence']
        start_idx = sample['start_idx']
        
        # Load sequence of events and images
        events_list = []
        images_list = []
        
        # #region agent log
        event_load_times = []
        image_load_times = []
        # #endregion
        
        for i in range(self.seq_len):
            frame_idx = start_idx + i
            
            # Load event voxel grid:  [C_event, H, W]
            # #region agent log
            event_load_start = time.time()
            # #endregion
            
            event_path = seq['events'][frame_idx]
            event_voxel = torch.load(event_path)
            
            # #region agent log
            event_load_time = time.time() - event_load_start
            event_load_times.append(event_load_time)
            # #endregion
            
            # Ensure event_voxel is a tensor
            if not isinstance(event_voxel, torch.Tensor):
                event_voxel = torch.tensor(event_voxel)
            
            events_list.append(event_voxel)
            
            # Load image
            # #region agent log
            image_load_start = time.time()
            # #endregion
            
            image_path = seq['images'][frame_idx]
            image = Image.open(image_path).convert('RGB')
            
            # #region agent log
            image_open_time = time.time() - image_load_start
            transform_start = time.time()
            # #endregion
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # #region agent log
            transform_time = time.time() - transform_start
            image_load_times.append(image_open_time + transform_time)
            # #endregion
            
            images_list.append(image)
        
        # Stack into sequences:  [T, C, H, W]
        events = torch.stack(events_list, dim=0)  # [T, C_event, H, W]
        images = torch.stack(images_list, dim=0)  # [T, 3, H_dino, W_dino]
        
        # #region agent log
        total_data_load_time = time.time() - data_load_start
        if idx % 100 == 0:  # Log every 100th sample to avoid too much logging
            try:
                with open('/data/fcr/.cursor/debug.log', 'a') as f:
                    json.dump({
                        'sessionId': 'debug-session',
                        'runId': 'pre-fix',
                        'hypothesisId': 'D',
                        'location': 'data_loader.py:88',
                        'message': 'Data sample loaded',
                        'data': {
                            'idx': idx,
                            'total_data_load_time': total_data_load_time,
                            'avg_event_load_time': sum(event_load_times) / len(event_load_times) if event_load_times else 0,
                            'avg_image_load_time': sum(image_load_times) / len(image_load_times) if image_load_times else 0,
                            'seq_len': self.seq_len,
                            'timestamp': time.time()
                        },
                        'timestamp': int(time.time() * 1000)
                    }, f)
                    f.write('\n')
            except:
                pass  # Ignore logging errors in worker processes
        # #endregion
        
        return events, images


def get_dataloader(root_dir, batch_size=4, num_workers=4, seq_len=8, 
                   image_size=(392, 518), shuffle=True):
    """
    Factory function to create the DataLoader.
    
    Args:
        root_dir: Path to the data directory
        batch_size:  Batch size
        num_workers: Number of worker processes
        seq_len:  Sequence length
        image_size:  Target image size (H, W) for DINOv2
        shuffle:  Whether to shuffle the data
        
    Returns:
        DataLoader instance
    """
    dataset = PairedEventImageDataset(
        root_dir, 
        sequence_length=seq_len,
        image_size=image_size
    )
    
    print(f"Dataset created with {len(dataset)} samples from {len(dataset.sequences)} sequences")
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return loader


# Example usage
if __name__ == "__main__":
    # Test the dataloader
    root_dir = "/data/fcr/code/fcr/EvEncoder/data/processed_data"
    dataloader = get_dataloader(
        root_dir=root_dir,
        batch_size=2,
        num_workers=2,
        seq_len=8,
        image_size=(392, 518)
    )
    
    # Test loading one batch
    for events, images in dataloader:
        print(f"Events shape: {events.shape}")  # [B, T, C_event, H, W]
        print(f"Images shape: {images.shape}")  # [B, T, 3, H_dino, W_dino]
        print(f"Events range: [{events.min():.3f}, {events.max():.3f}]")
        print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
        break