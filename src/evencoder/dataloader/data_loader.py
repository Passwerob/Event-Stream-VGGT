import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


class PairedEventImageDataset(Dataset):
    """
    Dataset for paired Event Voxel Grids and RGB Images. 
    Supports both raw PNG images and preprocessed . pt tensors.
    """
    def __init__(self, root_dir, sequence_length=8, transform=None, image_size=(392, 518)):
        self.root_dir = Path(root_dir)
        self.seq_len = sequence_length
        self.image_size = image_size  # (H, W)
        
        # Scan all sequences
        self.sequences = []
        for seq_dir in sorted(self.root_dir.iterdir()):
            if seq_dir.is_dir():
                events_dir = seq_dir / "events"
                images_dir = seq_dir / "images"
                
                if events_dir.exists() and images_dir.exists():
                    event_files = sorted(events_dir.glob("*.pt"))
                    
                    # ✅ Check if images are preprocessed (. pt) or raw (. png)
                    image_pt_files = sorted(images_dir.glob("*.pt"))
                    image_png_files = sorted(images_dir.glob("*.png"))
                    
                    if len(image_pt_files) > 0:  
                        # Preprocessed data
                        image_files = image_pt_files
                        self.preprocessed = True
                    elif len(image_png_files) > 0:
                        # Raw data
                        image_files = image_png_files
                        self.preprocessed = False
                    else:
                        continue
                    
                    if len(event_files) == len(image_files) and len(event_files) > 0:
                        self.sequences.append({
                            'name': seq_dir.name,
                            'events': event_files,
                            'images': image_files,
                            'length': len(event_files)
                        })
        
        # Detect if preprocessed by checking first sequence
        if self.sequences:
            first_image = self.sequences[0]['images'][0]
            self.preprocessed = first_image.suffix == '.pt'
            print(f"Dataset mode: {'Preprocessed' if self.preprocessed else 'Raw'}")
        
        # Create samples
        self.samples = []
        for seq in self.sequences:
            for start_idx in range(0, seq['length'] - self.seq_len + 1):
                self.samples.append({
                    'sequence': seq,
                    'start_idx':  start_idx
                })
        
        # Transform (only for raw images)
        if not self.preprocessed:
            if transform is None:
                self. transform = transforms.Compose([
                    transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms. ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transform

    def __len__(self):
        return len(self.samples)

    def resize_event_voxel(self, event_voxel):
        """
        Resize event voxel grid using nearest neighbor interpolation. 
        Preserves discrete nature of event data.
        """
        if event_voxel.dim() == 2:
            event_voxel = event_voxel.unsqueeze(0)
        elif event_voxel.dim() == 3:
            if event_voxel.shape[2] < event_voxel.shape[0] and event_voxel.shape[2] < event_voxel. shape[1]:
                event_voxel = event_voxel. permute(2, 0, 1)
        
        event_voxel = event_voxel.unsqueeze(0)
        
        # ✅ Use nearest neighbor instead of bilinear
        resized = F.interpolate(
            event_voxel, 
            size=self.image_size,
            mode='nearest'  
        )
        
        return resized.squeeze(0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        seq = sample['sequence']
        start_idx = sample['start_idx']
        
        events_list = []
        images_list = []
        
        for i in range(self.seq_len):
            frame_idx = start_idx + i
            
            # Load event
            event_voxel = torch.load(seq['events'][frame_idx], weights_only=True)
            if not isinstance(event_voxel, torch. Tensor):
                event_voxel = torch.tensor(event_voxel)
            
            # ✅ Resize event voxel to match image size
            event_voxel = self.resize_event_voxel(event_voxel)
            events_list.append(event_voxel)
            
            # Load image
            if self.preprocessed:
                # ✅ Preprocessed: direct load (FAST!)
                image_tensor = torch.load(seq['images'][frame_idx], weights_only=True)
            else:
                # ✅ Raw:  decode PNG (SLOW)
                with Image.open(seq['images'][frame_idx]) as img:
                    img = img.convert('RGB')
                    image_tensor = self.transform(img)
            
            images_list. append(image_tensor)
        
        events = torch.stack(events_list, dim=0)
        images = torch.stack(images_list, dim=0)
        
        return events, images


def get_dataloader(root_dir, batch_size=4, num_workers=4, seq_len=8, 
                   image_size=(392, 518), shuffle=True):
    """
    Factory function to create the DataLoader.
    
    Args:
        root_dir: Path to the data directory
        batch_size: Batch size
        num_workers: Number of worker processes
        seq_len:  Sequence length
        image_size:  Target image size (H, W) for DINOv2
        shuffle: Whether to shuffle the data
        
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
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=4 if num_workers > 0 else None  # Prefetch more batches
    )
    return loader


# Example usage
if __name__ == "__main__":
    # Test the dataloader
    root_dir = "/data/fcr/code/fcr/EvEncoder/data/processed_data"
    dataloader = get_dataloader(
        root_dir=root_dir,
        batch_size=2,
        num_workers=4,
        seq_len=8,
        image_size=(392, 518)
    )
    
    # Test loading one batch
    import time
    print("\nTesting dataloader...")
    start = time.time()
    for events, images in dataloader: 
        elapsed = time.time() - start
        print(f"Events shape: {events.shape}")  # [B, T, C_event, H, W]
        print(f"Images shape: {images.shape}")  # [B, T, 3, H_dino, W_dino]
        print(f"Events range:  [{events.min():.3f}, {events.max():.3f}]")
        print(f"Images range: [{images. min():.3f}, {images.max():.3f}]")
        print(f"Time to load batch: {elapsed:.3f}s")
        break