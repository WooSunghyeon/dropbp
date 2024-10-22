for train_bs in {11..1}
do
        HABANA_VISIBLE_DEVICES=1 python3 ../gaudi_spawn.py --world_size 1 --master_port 29501 run_lora_clm.py \
                --deepspeed ds_config.json \
                --model_name_or_path meta-llama/Meta-Llama-3-8B \
                --dataset_name timdettmers/openassistant-guanaco \
                --bf16 True \
                --output_dir ./model_lora_llama \
                --num_train_epochs 2 \
                --max_seq_len 512 \
                --per_device_train_batch_size ${train_bs} \
                --save_strategy no \
                --learning_rate 0.0018 \
                --warmup_ratio 0.03 \
                --lr_scheduler_type "cosine" \
                --logging_steps 1 \
                --dataset_concatenation \
                --attn_softmax_bf16 True \
                --do_train \
                --use_habana \
                --use_flash_attention \
                --report_to none \
                --throughput_warmup_steps 3\
                --drop_rate 0.5\
                --measure_time_memory \
                --throughput_path /home/wshey/throughput/dropbp-0.5.txt \
done
