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

from models.evencoder import EvEncoder
from models.dino import DINOv2Teacher
from dataloader.data_loader import PairedEventImageDataset
from utils.loss_utils import DistillationLoss, FeatureProjector


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


def get_distributed_dataloader(root_dir, batch_size, seq_len, num_workers, 
                                split='train', world_size=1, rank=0):
    """Get dataloader with distributed sampler."""
    from torch.utils.data import DataLoader
    
    dataset = PairedEventImageDataset(
        root_dir=root_dir,
        sequence_length=seq_len,
        image_size=(392, 518)
    )
    
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == 'train')
        )
        shuffle = None
    else:
        sampler = None
        shuffle = (split == 'train')
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return loader


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
        
        # Initialize Student (Trainable)
        if self.is_main:
            print("Initializing Student: EvEncoder...")
        self.student = EvEncoder(in_channels=args.event_voxel_bin_num, out_channels=16).to(self.device)
        
        # Initialize Projector
        self.projector = FeatureProjector(
            in_channels=16,
            out_channels=self.teacher_embed_dim
        ).to(self.device)

        # Wrap models with DDP
        if world_size > 1:
            self.student = DDP(
                self.student,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
            self.projector = DDP(
                self.projector,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )

        # Get actual model for saving (unwrap DDP if needed)
        self.student_module = self.student.module if hasattr(self.student, 'module') else self.student
        self.projector_module = self.projector.module if hasattr(self.projector, 'module') else self.projector

        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.student.parameters()) + list(self.projector.parameters()),
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
        
        # Initialize wandb
        if self.is_main and not args.no_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                resume='allow' if args.resume else None
            )
            wandb.watch(self.student, log='all', log_freq=100)

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
            teacher_tokens = self.teacher(flat_images)  # [B*T, N_patches, Embed_dim]

        # 2 Student Forward
        student_output, _ = self.student(events)  # [B, T, C_out, H_enc, W_enc]
        _, _, C_out, H_enc, W_enc = student_output.shape
        student_flat = student_output.view(B_T, C_out, H_enc, W_enc)  # [B*T, C_out, H_enc, W_enc]

        # 3Project Student Features
        student_proj = self.projector(student_flat)  # [B*T, Embed_dim, H_enc, W_enc]
        # import pdb; pdb.set_trace()
        # 4Convert Student to Token Format
        # Flatten spatial dimensions:  [B*T, Embed_dim, H_enc, W_enc] -> [B*T, Embed_dim, H_enc*W_enc]
        B_T_check, Embed_dim, H_enc, W_enc = student_proj.shape
        student_tokens = student_proj.view(B_T_check, Embed_dim, H_enc * W_enc)  # [B*T, Embed_dim, N_student_patches]
        student_tokens = student_tokens.transpose(1, 2)  # [B*T, N_student_patches, Embed_dim]
        
        N_teacher = teacher_tokens.shape[1]
        N_student = student_tokens.shape[1]
        
        if N_student != N_teacher:
            student_tokens_temp = student_tokens.transpose(1, 2)  # [B*T, Embed_dim, N_student]
            student_tokens_temp = F.interpolate(
                student_tokens_temp.unsqueeze(-1),  # [B*T, Embed_dim, N_student, 1]
                size=(N_teacher, 1),
                mode='bilinear',
                align_corners=False
            ).squeeze(-1)  # [B*T, Embed_dim, N_teacher]
            student_tokens = student_tokens_temp.transpose(1, 2)  # [B*T, N_teacher, Embed_dim]
        
        # 6Compute Loss in Token Space
        loss = self.criterion(student_feat=student_tokens, teacher_feat=teacher_tokens)
        
        return loss

    def train_epoch(self, dataloader, epoch_idx):
        """Train for one epoch."""
        self.student.train()
        self.projector.train()
        
        # #region agent log
        epoch_start_time = time.time()
        if self.is_main:
            try:
                with open('/data/fcr/.cursor/debug.log', 'a') as f:
                    json.dump({
                        'sessionId': 'debug-session',
                        'runId': 'pre-fix',
                        'hypothesisId': 'E',
                        'location': 'distill.py:243',
                        'message': 'Epoch start - set_epoch',
                        'data': {'epoch': epoch_idx, 'timestamp': time.time()},
                        'timestamp': int(time.time() * 1000)
                    }, f)
                    f.write('\n')
            except Exception as e:
                print(f"Logging error: {e}")
        # #endregion
        
        # Set epoch for distributed sampler
        set_epoch_start = time.time()
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch_idx)
        set_epoch_time = time.time() - set_epoch_start
        
        # #region agent log
        if self.is_main:
            try:
                with open('/data/fcr/.cursor/debug.log', 'a') as f:
                    json.dump({
                        'sessionId': 'debug-session',
                        'runId': 'pre-fix',
                        'hypothesisId': 'E',
                        'location': 'distill.py:252',
                        'message': 'set_epoch completed',
                        'data': {'epoch': epoch_idx, 'set_epoch_time': set_epoch_time, 'timestamp': time.time()},
                        'timestamp': int(time.time() * 1000)
                    }, f)
                    f.write('\n')
            except Exception as e:
                pass
        # #endregion
        
        epoch_loss = 0.0
        
        if self.is_main:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Train]")
        else:
            pbar = dataloader
        
        batch_times = []
        data_load_times = []
        wandb_log_times = []
        
        for batch_idx, (events, images) in enumerate(pbar):
            batch_start_time = time.time()
            
            # #region agent log
            data_load_start = time.time()
            # #endregion
            
            events = events.to(self.device)
            images = images.to(self.device)
            
            # #region agent log
            data_load_time = time.time() - data_load_start
            data_load_times.append(data_load_time)
            forward_start = time.time()
            # #endregion
            
            loss = self._process_batch(events, images)
            
            # #region agent log
            forward_time = time.time() - forward_start
            backward_start = time.time()
            # #endregion
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.projector.parameters()),
                max_norm=1.0
            )
            self.optimizer.step()
            
            # #region agent log
            backward_time = time.time() - backward_start
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            # #endregion
            
            epoch_loss += loss.item()
            
            # Update progress bar
            if self.is_main:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                # Log to wandb
                if not self.args.no_wandb and batch_idx % self.args.log_interval == 0:
                    # #region agent log
                    wandb_log_start = time.time()
                    # #endregion
                    
                    global_step = (epoch_idx - 1) * len(dataloader) + batch_idx
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.scheduler.get_last_lr()[0],
                        'train/epoch': epoch_idx,
                        'train/step': global_step
                    })
                    
                    # #region agent log
                    wandb_log_time = time.time() - wandb_log_start
                    wandb_log_times.append(wandb_log_time)
                    if batch_idx % 50 == 0:  # Log every 50 batches to avoid too much logging
                        try:
                            with open('/data/fcr/.cursor/debug.log', 'a') as f:
                                json.dump({
                                    'sessionId': 'debug-session',
                                    'runId': 'pre-fix',
                                    'hypothesisId': 'B',
                                    'location': 'distill.py:310',
                                    'message': 'Batch completed',
                                    'data': {
                                        'epoch': epoch_idx,
                                        'batch_idx': batch_idx,
                                        'batch_time': batch_time,
                                        'data_load_time': data_load_time,
                                        'forward_time': forward_time,
                                        'backward_time': backward_time,
                                        'wandb_log_time': wandb_log_time,
                                        'gpu_memory_allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0,
                                        'gpu_memory_reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2 if torch.cuda.is_available() else 0,
                                        'timestamp': time.time()
                                    },
                                    'timestamp': int(time.time() * 1000)
                                }, f)
                                f.write('\n')
                        except Exception:
                            pass
                    # #endregion
        
        # Gather losses from all processes
        # #region agent log
        gather_start = time.time()
        # #endregion
        
        if self.world_size > 1:
            epoch_loss_tensor = torch.tensor(epoch_loss).to(self.device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item() / self.world_size
        
        # #region agent log
        gather_time = time.time() - gather_start
        epoch_total_time = time.time() - epoch_start_time
        # #endregion
        
        avg_loss = epoch_loss / len(dataloader)
        
        if self.is_main:
            print(f"Epoch {epoch_idx} Avg Loss: {avg_loss:.5f}")
            
            # #region agent log
            epoch_wandb_start = time.time()
            # #endregion
            
            if not self.args.no_wandb:
                wandb.log({
                    'train/epoch_loss': avg_loss,
                    'epoch': epoch_idx
                })
            
            # #region agent log
            epoch_wandb_time = time.time() - epoch_wandb_start
            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
            avg_data_load_time = sum(data_load_times) / len(data_load_times) if data_load_times else 0
            max_batch_time = max(batch_times) if batch_times else 0
            min_batch_time = min(batch_times) if batch_times else 0
            try:
                with open('/data/fcr/.cursor/debug.log', 'a') as f:
                    json.dump({
                        'sessionId': 'debug-session',
                        'runId': 'pre-fix',
                        'hypothesisId': 'A',
                        'location': 'distill.py:360',
                        'message': 'Epoch completed',
                        'data': {
                            'epoch': epoch_idx,
                            'epoch_total_time': epoch_total_time,
                            'set_epoch_time': set_epoch_time,
                            'gather_time': gather_time,
                            'epoch_wandb_time': epoch_wandb_time,
                            'avg_batch_time': avg_batch_time,
                            'max_batch_time': max_batch_time,
                            'min_batch_time': min_batch_time,
                            'avg_data_load_time': avg_data_load_time,
                            'total_wandb_log_time': sum(wandb_log_times),
                            'num_batches': len(batch_times),
                            'gpu_memory_allocated_mb': torch.cuda.memory_allocated(self.device) / 1024**2 if torch.cuda.is_available() else 0,
                            'gpu_memory_reserved_mb': torch.cuda.memory_reserved(self.device) / 1024**2 if torch.cuda.is_available() else 0,
                            'timestamp': time.time()
                        },
                        'timestamp': int(time.time() * 1000)
                    }, f)
                    f.write('\n')
            except Exception as e:
                print(f"Logging error: {e}")
            # #endregion
        
        return avg_loss

    def validate(self, dataloader, epoch_idx):
        """Validate the model."""
        self.student.eval()
        self.projector.eval()
        
        val_loss = 0.0
        
        if self.is_main:
            pbar = tqdm(dataloader, desc="[Validation]")
        else:
            pbar = dataloader
        
        with torch.no_grad():
            for events, images in pbar:
                events = events.to(self.device)
                images = images.to(self.device)
                
                loss = self._process_batch(events, images)
                val_loss += loss.item()
                
                if self.is_main:
                    pbar.set_postfix({"Val Loss": f"{loss.item():.4f}"})
        
        # Gather losses from all processes
        if self.world_size > 1:
            val_loss_tensor = torch.tensor(val_loss).to(self.device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss = val_loss_tensor.item() / self.world_size
        
        avg_loss = val_loss / len(dataloader)
        
        if self.is_main:
            print(f"Validation Avg Loss: {avg_loss:.5f}")
            if not self.args.no_wandb:
                wandb.log({
                    'val/loss': avg_loss,
                    'epoch': epoch_idx
                })
        
        return avg_loss

    def save_checkpoint(self, path, epoch=None, loss=None):
        """Save checkpoint (only on main process)."""
        if not self.is_main:
            return
        
        # #region agent log
        checkpoint_start = time.time()
        # #endregion
        
        checkpoint = {
            'student':  self.student_module.state_dict(),
            'projector': self.projector_module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler':  self.scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'args': self.args
        }
        
        # #region agent log
        torch_save_start = time.time()
        # #endregion
        
        torch.save(checkpoint, path)
        
        # #region agent log
        torch_save_time = time.time() - torch_save_start
        wandb_save_start = time.time()
        # #endregion
        
        print(f"Checkpoint saved to {path}")
        
        # Save to wandb (non-blocking)
        wandb_save_time = 0
        if not self.args.no_wandb:
            # Use a background thread to avoid blocking training
            def save_to_wandb():
                try:
                    wandb.save(path)
                except Exception as e:
                    print(f"Warning: Failed to save checkpoint to wandb: {e}")
            
            wandb_thread = threading.Thread(target=save_to_wandb, daemon=True)
            wandb_thread.start()
            # Note: wandb_save_time is 0 because it's now non-blocking
        
        # #region agent log
        checkpoint_total_time = time.time() - checkpoint_start
        try:
            with open('/data/fcr/.cursor/debug.log', 'a') as f:
                json.dump({
                    'sessionId': 'debug-session',
                    'runId': 'pre-fix',
                    'hypothesisId': 'C',
                    'location': 'distill.py:343',
                    'message': 'Checkpoint saved',
                    'data': {
                        'epoch': epoch,
                        'path': path,
                        'checkpoint_total_time': checkpoint_total_time,
                        'torch_save_time': torch_save_time,
                        'wandb_save_time': wandb_save_time,
                        'timestamp': time.time()
                    },
                    'timestamp': int(time.time() * 1000)
                }, f)
                f.write('\n')
        except Exception as e:
            print(f"Logging error: {e}")
        # #endregion

    def load_checkpoint(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.student_module.load_state_dict(checkpoint['student'])
        self.projector_module.load_state_dict(checkpoint['projector'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint: 
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
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
    parser.add_argument("--num_workers", type=int, default=32)
    
    # Paths
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_path", type=str, default="./data/precessed_data")
    parser.add_argument("--resume", type=str, default=None)
    
    # Model
    parser.add_argument("--dino_model", type=str, default="dinov2_vitl14_reg")
    parser.add_argument("--teacher_checkpoint_path", type=str, default=None)
    parser.add_argument("--event_voxel_bin_num", type=int, default=8)
    
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
    
    # DataLoaders with distributed sampling
    train_loader = get_distributed_dataloader(
        root_dir=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        split='train',
        world_size=world_size,
        rank=rank
    )
    
    val_loader = get_distributed_dataloader(
        root_dir=args.data_path,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
        split='val',
        world_size=world_size,
        rank=rank
    )
    
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
            if not args.no_wandb:
                wandb.finish()
    
    except KeyboardInterrupt:
        if is_main_process():
            print("\nTraining interrupted by user")
            if not args.no_wandb:
                wandb.finish()
    
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()