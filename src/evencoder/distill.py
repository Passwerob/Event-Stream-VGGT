import os
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import time
import json
import threading

from models.dino import DINOv2Teacher
from dataloader.data_loader import PairedEventImageDataset
from utils.loss_utils import DistillationLoss


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is main process."""
    return not dist.is_initialized() or dist.get_rank() == 0


def safe_wandb_init(args, is_main):
    """
    Returns:  wandb_enabled (bool)
    """
    if not is_main or args.no_wandb:
        return False
    
    os.environ['WANDB_START_METHOD'] = 'thread'
    os.environ['WANDB_INIT_TIMEOUT'] = '300' 
    
    try:
        print("Initializing wandb...")
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            resume='allow' if args.resume else None,
            settings=wandb.Settings(
                start_method="thread",
                _service_wait=300,  
            )
        )
        print("✅ Wandb initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️  Wandb initialization failed: {e}")
        print("Continuing training without wandb logging...")
        return False


def safe_wandb_log(data, wandb_enabled):
    if not wandb_enabled:
        return
    
    try:
        wandb.log(data)
    except Exception as e:
        print(f"⚠️  Wandb logging failed: {e}")

def get_distributed_dataloader(root_dir, batch_size, seq_len, num_workers, 
                                split='train', world_size=1, rank=0, 
                                train_ratio=0.9):
    """
    Get dataloader with train/val split at sequence level.
    Ensures frames from same sequence stay together.
    """
    from torch.utils.data import DataLoader
    import numpy as np
    
    # Load full dataset
    full_dataset = PairedEventImageDataset(
        root_dir=root_dir,
        sequence_length=seq_len,
        image_size=(392, 518)
    )
    
    # Get unique sequences
    num_sequences = len(full_dataset.sequences)
    train_seq_count = int(num_sequences * train_ratio)
    
    # Split sequences with fixed seed
    np.random.seed(42)
    seq_indices = np.random.permutation(num_sequences)
    
    if split == 'train':
        selected_seq_indices = set(seq_indices[:train_seq_count])
    else:  # val
        selected_seq_indices = set(seq_indices[train_seq_count:])
    
    # Filter samples to only include selected sequences
    filtered_samples = []
    for sample in full_dataset.samples:
        seq_idx = full_dataset.sequences.index(sample['sequence'])
        if seq_idx in selected_seq_indices: 
            filtered_samples.append(sample)
    
    # Replace dataset samples
    full_dataset.samples = filtered_samples
    
    # Distributed sampler
    if world_size > 1:
        sampler = DistributedSampler(
            full_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == 'train')
        )
        shuffle = None
    else:
        sampler = None
        shuffle = (split == 'train')
    
    loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return loader

# def get_distributed_dataloader(root_dir, batch_size, seq_len, num_workers, 
#                                     split='train', world_size=1, rank=0):
#         """
#         Get dataloader for train or test split. 
#         Train and test data are in separate directories.
        
#         Args:
#             root_dir: Root directory containing 'rgv_interval_train_preprocessed' 
#                     and 'rgv_interval_test_preprocessed'
#             batch_size: Batch size per GPU
#             seq_len:  Sequence length
#             num_workers:  Number of data loading workers
#             split: 'train' or 'test'
#             world_size:  Number of GPUs
#             rank: Current GPU rank
#         """
#         from torch.utils.data import DataLoader
#         from pathlib import Path
        
#         root_dir = Path(root_dir)
        
#         # Select appropriate data directory based on split
#         if split == 'train':
#             data_dir = root_dir / 'train'
#         elif split == 'test':
#             data_dir = root_dir / 'test'
#         else: 
#             raise ValueError(f"split must be 'train' or 'test', got {split}")
        
#         if not data_dir.exists():
#             raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
#         # Load dataset
#         dataset = PairedEventImageDataset(
#             root_dir=data_dir,
#             sequence_length=seq_len,
#             image_size=(392, 518)
#         )
        
#         print(f"Loaded {split} dataset from {data_dir}")
#         print(f"  Sequences: {len(dataset.sequences)}")
#         print(f"  Samples:  {len(dataset.samples)}")
        
#         # Distributed sampler
#         if world_size > 1:
#             sampler = DistributedSampler(
#                 dataset,
#                 num_replicas=world_size,
#                 rank=rank,
#                 shuffle=(split == 'train')
#             )
#             shuffle = None
#         else:
#             sampler = None
#             shuffle = (split == 'train')
        
#         loader = DataLoader(
#             dataset,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             sampler=sampler,
#             num_workers=num_workers,
#             pin_memory=True,
#             drop_last=(split == 'train'),  # Only drop last batch for training
#             persistent_workers=True if num_workers > 0 else False,
#             prefetch_factor=2 if num_workers > 0 else None
#         )
        
#         return loader

class DistillationTrainer:  
    """
    Trainer for distilling DINOv2 (Teacher) to EvEncoder (Student).
    Supports distributed training and wandb logging.
    """
    
    def __init__(self, args, rank, world_size, local_rank):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main = is_main_process()
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device('cpu')

        # Initialize Teacher (Frozen)
        if self.is_main:
            print(f"Initializing Teacher:  {args.dino_model}...")
        self.teacher = DINOv2Teacher(model_name=args.dino_model, device=self.device, checkpoint_path=args.teacher_checkpoint_path)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        self._verify_teacher_dimensions()
        
        # Initialize Student (Trainable) with projector integrated
        if self.is_main:
            print("Initializing Student:  EvEncoder with projector...")
        if args.evencoder_type == "evencoder-v1":
            from models.evencoder import EvEncoder
            self.student = EvEncoder(
                in_channels=args.event_voxel_bin_num, 
                out_channels=16,
                project_out_channels=self.teacher_embed_dim
            ).to(self.device)
        elif args.evencoder_type == "evencoder-v2":
            from models.evencoder_dinov2 import EvEncoder
            self.student = EvEncoder(
                in_channels=args.event_voxel_bin_num, 
                base_channels=64, 
                dino_model=args.dino_model,
                target_res=(392, 518),
                checkpoint_path=args.teacher_checkpoint_path,
            ).to(self.device)
        else:
            raise ValueError(f"Invalid evencoder type: {args.evencoder_type}")

        # Wrap model with DDP
        if world_size > 1:
            self.student = DDP(
                self.student,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )

        # Get actual model for saving (unwrap DDP if needed)
        self.student_module = self.student.module if hasattr(self.student, 'module') else self.student

        # Optimizer (projector is part of student now)
        self.optimizer = optim.AdamW(
            self.student.parameters(),
            lr=args.lr,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01
        )
        
        # Loss Function
        self.criterion = DistillationLoss().to(self.device)
        
        self.wandb_enabled = safe_wandb_init(args, self.is_main)
        if self.wandb_enabled:
            try:
                wandb.watch(self.student, log='all', log_freq=100)
            except Exception as e: 
                print(f"⚠️  wandb.watch failed: {e}")

    def _verify_teacher_dimensions(self):
        """Verify and store teacher output dimensions."""
        if self.is_main:
            print("Verifying teacher output dimensions...")
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 518, 518).to(self.device)
            teacher_output = self.teacher(dummy_input)
            
            self.teacher_n_patches = teacher_output.shape[1]
            self.teacher_embed_dim = teacher_output.shape[2]
            self.teacher_grid_size = int(self.teacher_n_patches ** 0.5)
            
            if self.teacher_grid_size ** 2 != self.teacher_n_patches:
                if self.is_main:
                    print(f"Warning: {self.teacher_n_patches} patches is not a perfect square.")
                self.teacher_grid_size = int(self.teacher_n_patches ** 0.5)
                self.teacher_n_patches = self.teacher_grid_size ** 2
            
            if self.is_main:
                print(f"Teacher:  {self.teacher_n_patches} patches, "
                      f"grid={self.teacher_grid_size}x{self.teacher_grid_size}, "
                      f"embed_dim={self.teacher_embed_dim}")

    def _process_batch(self, events, images):
        """Process single batch:  teacher forward, student forward, compute loss."""
        B, T, C_ev, H_ev, W_ev = events.shape
        B_img, T_img, C_img, H_img, W_img = images.shape
        
        assert B == B_img and T == T_img
        B_T = B * T

        # 1 Teacher Forward (Frozen)
        with torch.no_grad():
            flat_images = images.view(B_T, C_img, H_img, W_img)
            teacher_tokens = self.teacher(flat_images)

        # 2 Student Forward (with projector applied)
        # For evencoder-v1, use_projector=True; for evencoder-v2, projector is integrated
        if self.args.evencoder_type == "evencoder-v1":
            student_output, _ = self.student(events, use_projector=True)
            # evencoder-v1 output: [B, T, C_out, H_enc, W_enc]
            _, _, C_out, H_enc, W_enc = student_output.shape
            student_proj = student_output.view(B_T, C_out, H_enc, W_enc)
            # Convert to token format: [B*T, N_patches, Embed_dim]
            B_T_check, Embed_dim, H_enc, W_enc = student_proj.shape
            student_tokens = student_proj.view(B_T_check, Embed_dim, H_enc * W_enc)
            student_tokens = student_tokens.transpose(1, 2)
        else:  # evencoder-v2
            student_output, _ = self.student(events)
            # evencoder-v2 output: [B, T, N_patches, DINO_Dim] - already in token format
            student_tokens = student_output.view(B_T, student_output.shape[2], student_output.shape[3])
        
        N_teacher = teacher_tokens.shape[1]
        N_student = student_tokens.shape[1]
        
        if N_student != N_teacher:
            student_tokens_temp = student_tokens.transpose(1, 2)
            student_tokens_temp = F.interpolate(
                student_tokens_temp.unsqueeze(-1),
                size=(N_teacher, 1),
                mode='bilinear',
                align_corners=False
            ).squeeze(-1)
            student_tokens = student_tokens_temp.transpose(1, 2)
        
        # 6 Compute Loss in Token Space
        loss = self.criterion(student_feat=student_tokens, teacher_feat=teacher_tokens)
        
        return loss

    def train_epoch(self, dataloader, epoch_idx):
        """Train for one epoch."""
        self.student.train()
        
        epoch_start_time = time.time()
        
        # Set epoch for distributed sampler
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch_idx)
        
        epoch_loss = 0.0
        
        if self.is_main:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Train]")
        else:
            pbar = dataloader
        
        batch_times = []
        data_load_times = []
        batch_start_time = time.time()
        
        for batch_idx, (events, images) in enumerate(pbar):
            
            data_load_time = time.time() - batch_start_time
            data_load_times.append(data_load_time)
            
            compute_start = time.time()
            
            events = events.to(self.device, non_blocking=True)  # ✅ non_blocking
            images = images.to(self.device, non_blocking=True)
            
            loss = self._process_batch(events, images)
            
            self.optimizer.zero_grad(set_to_none=True)  # ✅ 更彻底清理梯度
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                max_norm=1.0
            )
            self.optimizer.step()
            
            # ✅ 立即 detach loss
            loss_value = loss.detach().item()
            epoch_loss += loss_value
            
            compute_time = time.time() - compute_start
            batch_times.append(compute_time)
            
            # Update progress bar
            if self.is_main:
                pbar.set_postfix({
                    "Loss": f"{loss_value:.4f}",
                    "DataT": f"{data_load_time:.2f}s",
                    "CompT": f"{compute_time:.2f}s"
                })
                
                # ✅ Log to wandb with safe wrapper
                if batch_idx % self.args.log_interval == 0:
                    global_step = (epoch_idx - 1) * len(dataloader) + batch_idx
                    safe_wandb_log({
                        'train/loss': loss_value,
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/data_time': data_load_time,
                        'train/compute_time': compute_time,
                        'train/epoch': epoch_idx,
                        'train/step': global_step
                    }, self.wandb_enabled)
            
            del events, images, loss
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            
            batch_start_time = time.time()
        
        # Gather losses from all processes
        if self.world_size > 1:
            epoch_loss_tensor = torch.tensor(epoch_loss).to(self.device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item() / self.world_size
            del epoch_loss_tensor
        
        avg_loss = epoch_loss / len(dataloader)
        epoch_time = time.time() - epoch_start_time
        
        if self.is_main:
            avg_data_time = sum(data_load_times) / len(data_load_times) if data_load_times else 0
            avg_compute_time = sum(batch_times) / len(batch_times) if batch_times else 0
            
            print(f"Epoch {epoch_idx} Avg Loss: {avg_loss:.5f}")
            print(f"  Total Time: {epoch_time:.1f}s")
            print(f"  Avg Data Load:  {avg_data_time:.3f}s")
            print(f"  Avg Compute:  {avg_compute_time:.3f}s")
            print(f"  Data/Compute Ratio: {avg_data_time/avg_compute_time:.2f}")
            
            safe_wandb_log({
                'train/epoch_loss': avg_loss,
                'train/epoch_time': epoch_time,
                'train/avg_data_time': avg_data_time,
                'train/avg_compute_time': avg_compute_time,
                'epoch': epoch_idx
            }, self.wandb_enabled)
        
        return avg_loss

    def validate(self, dataloader, epoch_idx):
        """Validate the model."""
        self.student.eval()
        
        val_loss = 0.0
        
        if self.is_main:
            pbar = tqdm(dataloader, desc="[Validation]")
        else:
            pbar = dataloader
        
        with torch.no_grad():
            for batch_idx, (events, images) in enumerate(pbar):
                events = events.to(self.device, non_blocking=True)
                images = images.to(self.device, non_blocking=True)
                
                loss = self._process_batch(events, images)
                val_loss += loss.item()
                
                if self.is_main:
                    pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
                
                # ✅ 显式删除
                del events, images, loss
                
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Gather losses from all processes
        if self.world_size > 1:
            val_loss_tensor = torch.tensor(val_loss).to(self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / self.world_size
            del val_loss_tensor
        
        avg_loss = val_loss / len(dataloader)
        
        if self.is_main:
            print(f"Validation Avg Loss: {avg_loss:.5f}")
            safe_wandb_log({
                'val/loss': avg_loss,
                'epoch': epoch_idx
            }, self.wandb_enabled)
        
        return avg_loss

    def save_checkpoint(self, path, epoch=None, loss=None):
        """Save checkpoint (only on main process)."""
        if not self.is_main:
            return
        
        checkpoint = {
            'student':  self.student_module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'args': self.args
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        
        # ✅ Save to wandb in background thread (non-blocking)
        if self.wandb_enabled:
            def save_to_wandb():
                try:
                    wandb.save(path, policy='now')
                except Exception as e: 
                    print(f"⚠️  Failed to save checkpoint to wandb: {e}")
            
            wandb_thread = threading.Thread(target=save_to_wandb, daemon=True)
            wandb_thread.start()

    def load_checkpoint(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student_module.load_state_dict(checkpoint['student'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:  
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Handle legacy checkpoints that had separate projector
        if 'projector' in checkpoint:
            if self.is_main:
                print("⚠️  Legacy checkpoint detected with separate projector. "
                      "Projector is now integrated into EvEncoder, skipping separate projector load.")
        
        if self.is_main:
            print(f"Checkpoint loaded from {path}")
        
        return checkpoint.get('epoch', 0)


def main():
    parser = argparse.ArgumentParser(description="EvEncoder Distillation Training")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size per GPU")
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Hardware
    parser.add_argument("--num_workers", type=int, default=4) 
    
    # Data split
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="Random seed for train/val split")    
    # Paths
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_path", type=str, default="./data/precessed_data")
    parser.add_argument("--resume", type=str, default=None)
    
    # Model
    parser.add_argument("--dino_model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--teacher_checkpoint_path", type=str, default=None)
    parser.add_argument("--event_voxel_bin_num", type=int, default=8)
    parser.add_argument("--evencoder_type",type=str,choices=["evencoder-v1", "evencoder-v2"], default="evencoder-v1")
    # Wandb
    parser.add_argument("--no_wandb", action='store_true',
                        help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="evencoder-distillation")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Log interval for wandb")
    
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Only print on main process
    if is_main_process():
        print("=" * 50)
        print("Configuration:")
        for arg, value in vars(args).items():
            print(f"  {arg}: {value}")
        print(f"\nDistributed Training:")
        print(f"  World Size: {world_size}")
        print(f"  Rank: {rank}")
        print(f"  Local Rank:  {local_rank}")
        print("=" * 50)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # DataLoaders with split
    train_loader = get_distributed_dataloader(
        root_dir=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        split='train',
        world_size=world_size,
        rank=rank,
    )
    
    val_loader = get_distributed_dataloader(
        root_dir=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        split='val',
        world_size=world_size,
        rank=rank,
    )
        # random_seed=args.split_seed
        # val_ratio=args.val_ratio,
    
    if is_main_process():
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
    
    # Trainer
    trainer = DistillationTrainer(args, rank, world_size, local_rank)
    
    start_epoch = 1
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1
    
    # Training Loop
    best_val_loss = float('inf')
    
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if is_main_process():
                print(f"\n{'='*50}\nEpoch {epoch}/{args.epochs}\n{'='*50}")
            
            avg_train_loss = trainer.train_epoch(train_loader, epoch)
            avg_val_loss = trainer.validate(val_loader, epoch)
            
            trainer.scheduler.step()
            
            if is_main_process():
                print(f"LR: {trainer.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trainer.save_checkpoint(
                    os.path.join(args.save_dir, "best_evencoder.pth"),
                    epoch=epoch, loss=avg_val_loss
                )
            
            # Save every 5 epochs
            if epoch % 5 == 0:
                trainer.save_checkpoint(
                    os.path.join(args.save_dir, f"epoch_{epoch}.pth"),
                    epoch=epoch, loss=avg_val_loss
                )
            
            # Save latest
            trainer.save_checkpoint(
                os.path.join(args.save_dir, "latest.pth"),
                epoch=epoch, loss=avg_val_loss
            )
        
        if is_main_process():
            print(f"\nTraining Complete!  Best Val Loss: {best_val_loss:.5f}")
            if trainer.wandb_enabled:  # ✅ 使用 wandb_enabled 标志
                try:
                    wandb.finish()
                except Exception as e: 
                    print(f"⚠️  wandb.finish() failed: {e}")
    
    except KeyboardInterrupt:
        if is_main_process():
            print("\nTraining interrupted by user")
            if trainer.wandb_enabled:
                try:
                    wandb.finish()
                except:
                    pass
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
