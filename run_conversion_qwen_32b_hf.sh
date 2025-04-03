#!/usr/bin/env bash

# Set your parameters
step_start=100  # First checkpoint step
step_end=200    # Last checkpoint step
step_interval=100  # Interval between checkpoints

# Directories
current_directory=$(dirname "$(readlink -f "$0")")
parent_directory=$(dirname "$current_directory")
current_directory=/mbz/users/fan.zhou/tron/ckpts/
megatron_model_directory=${current_directory}/qwen_32b_megatron_t8p2_cpt
merge_model_directory=${current_directory}/qwen_32b_megatron_t8p2_merge
hf_model_directory=${current_directory}/qwen_32b_megatron_t8p2_hf

# Create the necessary directories
mkdir -p ${hf_model_directory}
mkdir -p ${merge_model_directory}

# Loop over the checkpoints
for (( step=${step_start}; step<=${step_end}; step+=${step_interval} ))
do
    # Update latest_checkpointed_iteration.txt
    # echo "${step}" > ${megatron_model_directory}/latest_checkpointed_iteration.txt
    
    # Update the output directory with the current step
    output_dir=${hf_model_directory}/qwen_32b_${step}step_hf
    
    # Run checkpoint_util.py to merge the megatron model
    python tools/checkpoint_util.py \
        --target_tensor_parallel_size 1 \
        --target_pipeline_parallel_size 1 \
        --load_dir ${megatron_model_directory} \
        --save_dir ${merge_model_directory} \
        --model_type qwen \
        --true_vocab_size 152064 \
        --bf16
    
    # Run weights_conversion/megatron_to_hf.py to convert the model
    python weights_conversion/megatron_to_hf.py \
        --model qwen \
        --size 32 \
        --input_dir ${merge_model_directory} \
        --output_dir ${output_dir} \
        --vocab_file Qwen/Qwen2.5-32B \
        --num_output_shards 17

done
