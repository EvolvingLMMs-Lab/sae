

torchrun --nproc_per_node="8" --nnodes="1" --node_rank="0" --master_addr="127.0.0.1" --master_port="1234" \
    src/easy_sae/launch/train.py \
    --dataset_path lmms-lab/LLaVA-OneVision-Data \
    --split train \
    --subset "CLEVR-Math(MathV360K)" \
    --image_key images \
    --run_name sae_test \
    --report_to wandb \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --bf16 \
    --target_modules model.language_model.layers.24.mlp.down_proj \
    --dataloader_num_workers 1 \
    --per_device_train_batch_size 2 \
    --logging_steps 1 \
    --save_steps 1000 \
    --output_dir checkpoints/sae_test \
    --save_total_limit 5 \
    --deepspeed ./examples/train/zero/zero2.json \
    --dead_tokens_threshold 100000