#!/usr/bin/env bash

current_directory=$(dirname "$(readlink -f "$0")")
parent_directory=$(dirname "$current_directory")

#training
LOG_ARGS="--log_interval 1 --save_interval 1000 --eval_interval 100 --eval_iters 10"

TRAIN_ARGS="--train_iters 100000 --lr_decay_style constant --weight_decay 0.1 
--lr_warmup_iters 500 --lr 1e-4 --min_lr 5e-6 
--adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-5"

DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
# DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
# DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 2 --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 4 --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 8 --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


QWEN_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits 
--no_new_tokens --layernorm_epsilon 1e-6 --rope_theta 1e6 --use_flash_attn --bf16 --seq_length 4096"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion --no_bias_dropout_fusion  
--no_query_key_layer_scaling --attention_softmax_in_fp32"

WANDB_ARGS=""


############
MEGATRON_Trains=(
	1 ${current_directory}/dataset/sample_preprocessed_data/train_text_document
)


MEGATRON_Valids=(
	1 ${current_directory}/dataset/sample_preprocessed_data/train_text_document
)


DATA_EXTRA_PARAMS="\
    --train_data_path ${MEGATRON_Trains[*]} \
    --valid_data_path ${MEGATRON_Valids[*]} \
    --test_data_path ${MEGATRON_Valids[*]}
"


CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun $DISTRIBUTED_ARGS finetune.py \
	--tensor_model_parallel_size 2 \
	--pipeline_model_parallel_size 1 \
	--load ${current_directory}/model/qwen2/qwen2_05b_megatron_t2p1 \
	--save ${current_directory}/model/qwen2/qwen2_05b_megatron_t2p1_test  \
	--tensorboard_dir ${current_directory}/model/qwen2/qwen2_05b_megatron_t2p1_test/logs/ \
	--model_name qwen \
	--tokenizer_type Qwen2Tokenizer \
	--vocab_file Qwen/Qwen2-0.5B  \
	--micro_batch_size 4 \
	--global_batch_size 128 \
	--sequence_parallel \
    --recompute_granularity selective \
	--use_checkpoint_args \
    $COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $QWEN_ARGS $DATA_EXTRA_PARAMS $WANDB_ARGS