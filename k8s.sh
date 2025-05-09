#!/usr/bin/env bash
current_directory=/inspire/hdd/global_user/liupengfei-24025/hyzou/mmcpt/megatron-qwen
export WANDB_MODE=offline
export WANDB_DIR=/inspire/hdd/global_user/liupengfei-24025/hyzou/mmcpt/ckpts
export TZ=UTC-8

#training
EXP_NAME="mt.qwen25.32b.t8p8.cpt.mmpmax"
LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 100 --eval_iters 10"

TRAIN_ARGS="--train_iters 100000 --lr_decay_style constant --weight_decay 0.01 
--lr_warmup_iters 500 --lr 1e-4 --min_lr 5e-6 
--adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-5"

DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 8 --node_rank $PET_NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

QWEN_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits 
--no_new_tokens --layernorm_epsilon 1e-6 --rope_theta 1e6 --use_flash_attn --bf16 --seq_length 4096"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion --no_bias_dropout_fusion  
--no_query_key_layer_scaling --attention_softmax_in_fp32 --intra_doc"

WANDB_ARGS="--wandb_logger --wandb_project test
--wandb_entity xuefengli0301
--wandb_id $EXP_NAME
--wandb_api_key 1badc41f0d258400b42ad079d39f9d58376dabf0
--tensorboard_dir /inspire/hdd/global_user/liupengfei-24025/hyzou/mmcpt/ckpts"

#####debug mode#####
MEGATRON_Trains=()
for i in {0..99}
do
    num_str=$(printf "%01d" $i)
    MEGATRON_Trains+=("1 /inspire/hdd/global_user/liupengfei-24025/hyzou/mmcpt/dataset/megatron_dataset/megamath_data/megamath-pro-max/data_${num_str}_text_document")
done

for item in "${MEGATRON_Trains[@]}"; do
    echo "$item"
done

# MEGATRON_Trains=(
#  1 ${current_directory}/dataset/sample_preprocessed_data/train_text_document
# )

MEGATRON_Valids=(
 1 /inspire/hdd/global_user/liupengfei-24025/hyzou/mmcpt/megatron-qwen/dataset/qa/qa_226_text_document
)


DATA_EXTRA_PARAMS="\
    --train_data_path ${MEGATRON_Trains[*]} \
    --valid_data_path ${MEGATRON_Valids[*]} \
    --test_data_path ${MEGATRON_Valids[*]}
"

###first run
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun $DISTRIBUTED_ARGS ${current_directory}/finetune.py \
 --tensor_model_parallel_size 8 \
 --pipeline_model_parallel_size 8 \
 --load /inspire/hdd/global_user/liupengfei-24025/hyzou/mmcpt/ckpts/Megatron/megatron-qwen-32b-t8p8  \
 --save /inspire/hdd/global_user/liupengfei-24025/hyzou/mmcpt/ckpts/Megatron/cpt/$EXP_NAME  \
 --tensorboard_dir /inspire/hdd/global_user/liupengfei-24025/hyzou/mmcpt/ckpts/Megatron/cpt/$EXP_NAME/logs/ \
 --model_name qwen \
 --tokenizer_type Qwen2Tokenizer \
 --vocab_file /inspire/hdd/global_user/liupengfei-24025/hyzou/wiles/model/huggingface/Qwen/Qwen2.5-32B  \
 --micro_batch_size 1 \
 --global_batch_size 1024 \
 --sequence_parallel \
    --recompute_granularity selective \
 --use_checkpoint_args \
    $COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $QWEN_ARGS $DATA_EXTRA_PARAMS $WANDB_ARGS
