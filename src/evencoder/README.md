# EvEncoder - Event Camera Encoder via Knowledge Distillation

This project implements knowledge distillation from DINOv2 (Teacher) to EvEncoder (Student) for event camera representation learning. The student model learns to encode event voxel grids into rich feature representations by mimicking the teacher's knowledge from RGB images.

## Overview

EvEncoder is trained using knowledge distillation, where:
- **Teacher Model**: DINOv2 Vision Transformer (frozen, pre-trained on RGB images)
- **Student Model**: EvEncoder (learnable, processes event voxel grids)
- **Objective**: Align student features with teacher features in the embedding space

## Features

- 🎯 Knowledge distillation from DINOv2 to event-based encoder
- 🚀 Distributed training support (multi-GPU via `torchrun`)
- 📊 Weights & Biases (wandb) integration for experiment tracking
- 🔄 Automatic feature alignment between student and teacher
- 💾 Checkpoint saving and resuming
- 📈 Learning rate scheduling

## Project Structure

```
EvEncoder/
├── distill.py                 # Main training script
├── models/
│   ├── evencoder.py          # EvEncoder student model
│   ├── dino.py               # DINOv2 teacher model wrapper
│   └── layers/               # Vision Transformer layers
├── dataloader/
│   ├── data_loader.py        # Paired event-image dataset
│   └── event_sequence.py     # Event sequence utilities
├── utils/
│   └── loss_utils.py         # Distillation loss and feature projector
└── run_distiller.sh          # Example training script
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+ (with CUDA support for distributed training)
- torchvision
- wandb
- tqdm
- numpy
- PIL/Pillow

### Setup

```bash
# Install dependencies
pip install torch torchvision wandb tqdm numpy pillow

# Or use conda
conda install pytorch torchvision -c pytorch
pip install wandb tqdm numpy pillow
```

## Data Format

The dataset should be organized as follows:

```
data/processed_data/
├── sequence_001/
│   ├── events/
│   │   ├── 000000.pt
│   │   ├── 000001.pt
│   │   └── ...
│   └── images/
│       ├── 000000.png
│       ├── 000001.png
│       └── ...
├── sequence_002/
│   └── ...
└── ...
```

**Data Specifications:**
- **Events**: `.pt` files containing voxel grids with shape `[C_event, H, W]` (typically `C_event=8`)
- **Images**: `.png` files with shape `[H, W, 3]` (RGB)
- Images are automatically resized to `(392, 518)` for DINOv2 compatibility

## Usage

### Single GPU Training

```bash
python distill.py \
    --data_path /path/to/processed_data \
    --save_dir ./checkpoints/experiment_name \
    --batch_size 4 \
    --epochs 50 \
    --lr 1e-4 \
    --seq_len 4 \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --wandb_project evencoder-distillation \
    --wandb_run_name experiment_name
```

### Multi-GPU Distributed Training

```bash
torchrun --nproc_per_node=4 distill.py \
    --data_path /path/to/processed_data \
    --save_dir ./checkpoints/experiment_name \
    --batch_size 1 \
    --epochs 50 \
    --lr 1e-4 \
    --seq_len 4 \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --wandb_project evencoder-distillation \
    --wandb_run_name experiment_name \
    --num_workers 32
```

### Using the Shell Script

Edit `run_distiller.sh` with your parameters and run:

```bash
bash run_distiller.sh
```

## Command Line Arguments

### Required Arguments

- `--data_path`: Path to the processed data directory
- `--save_dir`: Directory to save checkpoints
- `--teacher_checkpoint_path`: Path to the DINOv2 teacher checkpoint

### Training Arguments

- `--batch_size`: Batch size per GPU (default: 4)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--seq_len`: Sequence length for temporal modeling (default: 4)
- `--num_workers`: Number of data loading workers (default: 32)
- `--resume`: Path to checkpoint to resume from (optional)

### Model Arguments

- `--dino_model`: DINOv2 model variant (default: `dinov2_vitl14_reg`)
- `--event_voxel_bin_num`: Number of event voxel bins (default: 8)

### Logging Arguments

- `--wandb_project`: W&B project name (default: `evencoder-distillation`)
- `--wandb_run_name`: W&B run name (optional)
- `--no_wandb`: Disable wandb logging
- `--log_interval`: Logging interval in batches (default: 10)

## Model Architecture

### EvEncoder (Student)

- **Input**: Event voxel grids `[B, T, C_event, H, W]`
- **Architecture**: 
  - ResNet-like encoder with residual blocks
  - ETF (Event Temporal Fusion) modules for temporal modeling
  - Feature projector to align with teacher dimensions
- **Output**: Feature tokens `[B*T, N_patches, Embed_dim]`

### DINOv2 (Teacher)

- **Input**: RGB images `[B*T, 3, H, W]` (resized to 392x518)
- **Architecture**: Vision Transformer (ViT-L/14)
- **Output**: Feature tokens `[B*T, N_patches, Embed_dim]` (1369 patches, 1024 dim)

### Feature Alignment

When student and teacher have different numbers of patches, bilinear interpolation is used to align the spatial dimensions before computing the distillation loss.

## Training Details

### Loss Function

The distillation loss is computed as:

```
Loss = MSE(student_features, teacher_features)
```

Optionally, KL divergence can be added for variational regularization.

### Checkpointing

- **Best model**: Saved as `best_evencoder.pth` when validation loss improves
- **Periodic saves**: Every 5 epochs as `epoch_{N}.pth`
- **Latest**: Always saved as `latest.pth`

### Resuming Training

```bash
python distill.py \
    --resume ./checkpoints/experiment_name/latest.pth \
    --data_path /path/to/data \
    --save_dir ./checkpoints/experiment_name \
    ...
```



## Citation

If you use this code in your research, please cite:

```bibtex
@misc{evencoder2024,
  title={EvEncoder: Event Camera Encoder via Knowledge Distillation},
  author={Your Name},
  year={2024}
}
```

## License

[Specify your license here]

## Acknowledgments

- DINOv2: [Facebook Research](https://github.com/facebookresearch/dinov2)
- PyTorch Distributed Training
- Weights & Biases for experiment tracking
