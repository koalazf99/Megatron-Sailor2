#!/usr/bin/env bash

current_directory=$(dirname "$(readlink -f "$0")")
parent_directory=$(dirname "$current_directory")


###preprocess
python tools/preprocess_data.py \
	--input ${current_directory}/dataset/sample.jsonl \
	--output_prefix ${current_directory}/dataset/sample_preprocessed_data/train \
	--tokenizer_type Qwen2Tokenizer \
	--vocab_file Qwen/Qwen2.5-14B  \
	--chunk_size 32 \
	--workers 64 \
	--log_interval 20000 \
	--append_eod \
	--no_new_tokens

### Expand qwen2.5-14B to qwen2.5-20B with 16 additional layer
python ${current_directory}/tools/llama_pro.py \
    --model_name_or_path Qwen/Qwen2.5-14B \
    --output_dir ${current_directory}/qwen2.5_14b_pro_16l \
    --num_expand 16

###20B_conversion
python weights_conversion/hf_to_megatron.py qwen \
    --size=20 \
    --out ${current_directory}/model/qwen2_20b_megatron \
    --model-path ${current_directory}/model/qwen2.5_14b_pro_16l

###20B_sharding
python tools/checkpoint_util.py \
	--target_tensor_parallel_size  4 \
	--target_pipeline_parallel_size 2 \
	--load_dir ${current_directory}/model/qwen2_20b_megatron  \
	--save_dir ${current_directory}/model/qwen2_20b_megatron_t4p2  \
	--model_type qwen \
	--true_vocab_size 151936 \
	--megatron_path ${current_directory} \
	--vocab_file Qwen/Qwen2.5-14B \
	--bf16

