#!/usr/bin/env bash

current_directory=$(dirname "$(readlink -f "$0")")
parent_directory=$(dirname "$current_directory")


###preprocess
python tools/preprocess_data.py \
	--input ${current_directory}/dataset/sample.jsonl \
	--output_prefix ${current_directory}/dataset/sample_preprocessed_data/train \
	--tokenizer_type Qwen2Tokenizer \
	--vocab_file  Qwen/Qwen2-0.5B  \
	--chunk_size 32 \
	--workers 64 \
	--log_interval 20000 \
	--append_eod \
	--no_new_tokens

#######################0.5B#####################
###05B_conversion
python weights_conversion/hf_to_megatron.py qwen \
    --size=5 \
    --out ${current_directory}/model/qwen2/qwen2_05b_megatron \
    --model-path Qwen/Qwen2-0.5B

###05B_sharding for run_verify.sh
python tools/checkpoint_util.py \
	--target_tensor_parallel_size  1 \
	--target_pipeline_parallel_size 1 \
	--load_dir  ${current_directory}/model/qwen2/qwen2_05b_megatron \
	--save_dir ${current_directory}/model/qwen2/qwen2_05b_megatron_t1p1 \
	--model_type qwen \
	--true_vocab_size 151936 \
	--megatron_path ${current_directory} \
	--vocab_file Qwen/Qwen2-0.5B \
	--bf16

###05B_sharding
python tools/checkpoint_util.py \
	--target_tensor_parallel_size  2 \
	--target_pipeline_parallel_size 1 \
	--load_dir  ${current_directory}/model/qwen2/qwen2_05b_megatron \
	--save_dir ${current_directory}/model/qwen2/qwen2_05b_megatron_t2p1 \
	--model_type qwen \
	--true_vocab_size 151936 \
	--megatron_path ${current_directory} \
	--vocab_file Qwen/Qwen2-0.5B \
	--bf16
