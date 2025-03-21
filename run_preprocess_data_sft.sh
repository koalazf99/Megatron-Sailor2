#!/usr/bin/env bash

current_directory=$(dirname "$(readlink -f "$0")")
parent_directory=$(dirname "$current_directory")


python tools/preprocess_instruct_data.py \
	--input ${current_directory}/dataset/sample_sft.jsonl \
	--output_prefix ${current_directory}/dataset/sample_sft_preprocessed_data/train \
	--tokenizer_type Qwen2ChatTokenizer \
	--vocab_file Qwen/Qwen2-0.5B-Chat  \
	--chunk_size 32 \
	--workers 64 \
	--log_interval 20000 \
	--question_key=user \
	--answer_key=assistant \
	--system=system
