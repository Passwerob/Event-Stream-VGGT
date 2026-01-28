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


# Data preprocess
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_preprocessed
IMAGE_HEIGHT=392
IMAGE_WIDTH=518

cd /data/fcr/code/fcr/EvEncoder/dataloader
python preprocess_data.py \
    --input ${DATA_PATH} \
    --output ${OUTPUT_PATH} \
    --image_height ${IMAGE_HEIGHT} \
    --image_width ${IMAGE_WIDTH} --num_workers 32

# Training
export CUDA_VISIBLE_DEVICES=3,4,5,6
INSTANCE_NAME=20251222-2318
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_train
torchrun --nproc_per_node=4 \
    --master_port=29500 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 2 \
    --num_workers 2


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


## 20251222
# Data preprocess
DATA_PATH=/data/fcr/data/rgv_interval_train
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
IMAGE_HEIGHT=392
IMAGE_WIDTH=518

cd /data/fcr/code/fcr/EvEncoder/dataloader
python preprocess_data.py \
    --input ${DATA_PATH} \
    --output ${OUTPUT_PATH} \
    --image_height ${IMAGE_HEIGHT} \
    --image_width ${IMAGE_WIDTH} --num_workers 8

DATA_PATH=/data/fcr/data/rgv_interval_train
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
IMAGE_HEIGHT=392
IMAGE_WIDTH=518
cd /data/fcr/code/fcr/EvEncoder/dataloader
python checker.py \
    --input ${DATA_PATH} \
    --output ${OUTPUT_PATH} \
    --verbose

# Training
export CUDA_VISIBLE_DEVICES=0,1,2,3
INSTANCE_NAME=20251223-1600
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
torchrun --nproc_per_node=4 \
    --master_port=29502 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 2 \
    --num_workers 2


# Training
export CUDA_VISIBLE_DEVICES=0,1,2,3
INSTANCE_NAME=20251224-2100
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
torchrun --nproc_per_node=4 \
    --master_port=29502 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 16 \
    --num_workers 2


export CUDA_VISIBLE_DEVICES=0,1,2,3
INSTANCE_NAME=20251225-2323-test
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
torchrun --nproc_per_node=4 \
    --master_port=29502 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-3 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 16 \
    --num_workers 2


export CUDA_VISIBLE_DEVICES=0,1,2,3
INSTANCE_NAME=20251225-2323-test
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
torchrun --nproc_per_node=4 \
    --master_port=29502 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-3 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 16 \
    --num_workers 2

export CUDA_VISIBLE_DEVICES=4,5,6,7
INSTANCE_NAME=20251225-2323-evencoder-v2-1e-3
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
torchrun --nproc_per_node=4 \
    --master_port=29503 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-3 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 8 \
    --num_workers 2 --evencoder_type evencoder-v2


export CUDA_VISIBLE_DEVICES=1,2,3,0
INSTANCE_NAME=20251225-2355-evencoder-v2-1e-4
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
torchrun --nproc_per_node=4 \
    --master_port=29502 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 8 \
    --num_workers 2 --evencoder_type evencoder-v2



export CUDA_VISIBLE_DEVICES=1,2,3,0
INSTANCE_NAME=20251226-0400-evencoder-v2-1e-5
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
torchrun --nproc_per_node=4 \
    --master_port=29503 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-5 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 8 \
    --num_workers 2 --evencoder_type evencoder-v2

export CUDA_VISIBLE_DEVICES=4,5,6,7
INSTANCE_NAME=20251226-0405-evencoder-v2-1e-6
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_train_preprocessed
torchrun --nproc_per_node=4 \
    --master_port=29502 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-6 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 8 \
    --num_workers 2 --evencoder_type evencoder-v2



export CUDA_VISIBLE_DEVICES=0,1,2,3
INSTANCE_NAME=20260105-2205-evencoder-v2-1e-4
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_pass/raw_data
torchrun --nproc_per_node=4 \
    --master_port=29502 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-4 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 8 \
    --num_workers 2 --evencoder_type evencoder-v2 

conda activate StreamVGGT
export CUDA_VISIBLE_DEVICES=0,1,2,3
INSTANCE_NAME=20260106-2110-evencoder-v2-1e-3
DATA_PATH=/data/fcr/data/rgv_interval
OUTPUT_PATH=/data/fcr/data/rgv_interval_pass/raw_data
torchrun --nproc_per_node=4 \
    --master_port=29502 \
    distill.py \
    --data_path ${OUTPUT_PATH} \
    --save_dir ./checkpoints/${INSTANCE_NAME} \
    --batch_size 2 \
    --epochs 50 \
    --lr 1e-3 \
    --wandb_project evencoder-distillation \
    --wandb_run_name ${INSTANCE_NAME} \
    --teacher_checkpoint_path ./ckpts/dinov2_stream_vggt.pt \
    --seq_len 8 \
    --num_workers 2 --evencoder_type evencoder-v2  --save_interval 1