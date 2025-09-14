cd src/open-r1-multimodal

export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3
BASE_DATASET_DIR="data"
RUN_NAME="GUI-G1"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_rec.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir output/$RUN_NAME \
    --model_name_or_path BASE_MODEL_PATH \
    --dataset_name data_config/rec.yaml \
    --image_root BASE_DATASET_DIR \
    --max_prompt_length 1024 \
    --num_generations 8 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --learning_rate 1e-6 \
    --max_completion_length 128 \
    --reward_funcs iou hit box \
    --temperature 0.9 \
    --beta 0.0 \
    --use_vllm
