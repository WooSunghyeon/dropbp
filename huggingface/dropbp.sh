export WANDB_ENTITY=hoochoo
learning_rate=5e-4
drop_rate=0.5

CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --custom_mode lora \
    --lr ${learning_rate} \
    --lora_r 8 \
    --train_bs 4 \
    --bf16 \
    --accumulation_steps 1 \
    --model meta-llama/Meta-Llama-3-8B \
    --seed 526 \
    --logging_steps 1 \
    --target_modules no_head \
    --metrics_enabled 0 \
    --lr_scheduler_type cosine \
    --save_dir output_models/llama3-8b/lora//dropbp_${drop_rate}_${learning_rate}\
    --drop_rate ${drop_rate}\
    --measure_time_memory \
    --throughput_path outputs/throughput.txt\
    --task instruct --dataset oasst1 --epochs 1
