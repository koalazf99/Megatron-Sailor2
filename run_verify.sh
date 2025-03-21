
#!/usr/bin/env bash

current_directory=$(dirname "$(readlink -f "$0")")
parent_directory=$(dirname "$current_directory")

## qwen2-0.5B
QWEN_ARGS="--use_rms_norm --glu_activation swiglu 
--no_new_tokens --layernorm_epsilon 1e-6 --rope_theta 1e6 --use_flash_attn --bf16 --seq_length 4096"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion
--no_query_key_layer_scaling --attention_softmax_in_fp32"
DISTRIBUTED_ARGS="--nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"


CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun $DISTRIBUTED_ARGS verify_correctness.py \
	--model_name qwen \
	--model_size 5 \
	--load ${current_directory}/model/qwen2/qwen2_05b_megatron_t1p1 \
	--train_data_path ${current_directory}/dataset/sample_preprocessed_data/train_text_document \
	--tokenizer_type Qwen2Tokenizer \
	--vocab_file Qwen/Qwen2-0.5B \
	--huggingface_cache Qwen/Qwen2-0.5B \
	--huggingface_device=cuda:1 \
	$COMMON_ARGS $QWEN_ARGS


