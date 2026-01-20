cd src/
NCCL_DEBUG=TRACE TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 accelerate launch --multi_gpu --main_process_port 26902 ./finetune.py --config-name finetune

# StreamVGGT Checkpoint path
 ~/.cache/huggingface/hub/models--lch01--StreamVGGT/snapshots/f9ba55b1955bc4f34337c51142158a9aa2862c7f/checkpoints.pth

# From image folder
python inference.py --input /share/magic_group/aigc/fcr/EventVGGT/StreamVGGT/data/test/zurich_city_02_b_49/images --output ./results/test/zurich_city_02_b_49/

# From video
python inference. py --input ./video.mp4 --output ./results/ --fps_interval 0.5

# With confidence threshold
python inference.py --input ./my_images/ --output ./results/ --conf_threshold 0.3


python inference_with_evencoder.py \
    --extractor evencoder \
    --evencoder_checkpoint /data/fcr/code/fcr/EvEncoder/checkpoints/20260106-2110-evencoder-v2-1e-3/best_evencoder.pth \
    --checkpoint /home/fcr/.cache/huggingface/hub/models--lch01--StreamVGGT/snapshots/f9ba55b1955bc4f34337c51142158a9aa2862c7f/checkpoints.pth \
    --input /data/fcr/code/event_vggt/Data_Part/StreamVGGT/src/evencoder/data/processed_data/zurich_city_02_b_49/events1  --input_type auto 

python inference_with_evencoder.py \
    --extractor evencoder-v2 \
    --evencoder_checkpoint /data/fcr/code/fcr/EvEncoder/checkpoints/20260106-2110-evencoder-v2-1e-3/best_evencoder.pth \
    --checkpoint /home/fcr/.cache/huggingface/hub/models--lch01--StreamVGGT/snapshots/f9ba55b1955bc4f34337c51142158a9aa2862c7f/checkpoints.pth \
    --input /data/fcr/code/event_vggt/Data_Part/StreamVGGT/src/evencoder/data/processed_data/zurich_city_02_b_49/events1  --input_type auto 

python inference_with_evencoder.py \
    --extractor evencoder-v2 \
    --evencoder_checkpoint /data/fcr/code/fcr/EvEncoder/checkpoints/20260106-2110-evencoder-v2-1e-3/best_evencoder.pth \
    --checkpoint /home/fcr/.cache/huggingface/hub/models--lch01--StreamVGGT/snapshots/f9ba55b1955bc4f34337c51142158a9aa2862c7f/checkpoints.pth \
    --input /data/fcr/code/event_vggt/Data_Part/StreamVGGT/src/evencoder/data/processed_data/screen-1/events  --input_type auto \
    --output ./output/screen-1/