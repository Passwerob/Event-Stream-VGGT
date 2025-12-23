RAW_DATA_PATH=/data/fcr/data/rgv_interval

export CUDA_VISIBLE_DEVICES=3,4,5,6
torchrun --nproc_per_node=4 distill.py \
    --data_path /data/fcr/code/fcr/EvEncoder/data/processed_data \
    --save_dir ./checkpoints/20251216-231614 \
    --batch_size 1 \
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name 20251216-231614 \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 2




python distill.py \
    --data_path /data/fcr/code/fcr/EvEncoder/data/processed_data \
    --save_dir ./checkpoints/20251216-231614 \
    --batch_size 1 \
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name 20251216-231614 \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 2

# 20251220
torchrun --nproc_per_node=4 distill.py \
    --data_path /data/fcr/data/rgv_interval \
    --save_dir ./checkpoints/20251220-1522 \
    --batch_size 1 \
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name 20251220-1522 \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 2

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
INSTANCE_NAME=20251220-1553

torchrun --nproc_per_node=6 \
    --master_port=29500 \
    distill.py \
    --data_path /data/fcr/data/rgv_interval \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2\
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 2


export CUDA_VISIBLE_DEVICES=3,2,5,6
INSTANCE_NAME=20251220-1846

torchrun --nproc_per_node=4 \
    --master_port=29500 \
    distill.py \
    --data_path /data/fcr/data/rgv_interval \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2\
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 2 --num_workers 64


export CUDA_VISIBLE_DEVICES=0,1
INSTANCE_NAME=20251220-2041
torchrun --nproc_per_node=2 \
    --master_port=29500 \
    distill.py \
    --data_path /data/fcr/data/rgv_interval \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2\
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 2 --num_workers 64