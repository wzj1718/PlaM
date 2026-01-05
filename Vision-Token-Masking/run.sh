
export CUDA_VISIBLE_DEVICES=0
 
for DROP in  20 
do
    echo ">>> Running drop ${DROP}..."
    python3 -m accelerate.commands.launch \
        --main_process_port 11181 \
        --num_processes=8 \
        -m lmms_eval \
        --model qwen2_5_vl \
        --model_args pretrained="Qwen2.5-VL-7B-Instruct/,use_flash_attention_2=True" \
        --tasks realworldqa \
        --batch_size 1 \
        --drop ${DROP} \
        --log_samples \
        --log_samples_suffix pro-${NAME}-drop${DROP} \
        --output_path ./logs/Qwen2.5-VL-7B-Instruct/drop${DROP}-${NAME}

    echo ">>> Finished drop ${DROP}"
    echo "--------------------------------"
done
