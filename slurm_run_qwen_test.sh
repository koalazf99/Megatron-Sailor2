#!/bin/bash
#SBATCH --job-name=megatron-slm-math
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=50
#SBATCH --mem=0
#SBATCH --gres=gpu:8
#SBATCH --output=/mbz/users/fan.zhou/tron/Megatron-Sailor2/slurm/slurm_2nodes_%j.log
#SBATCH --error=/mbz/users/fan.zhou/tron/Megatron-Sailor2/slurm/slurm_2nodes_%j.log
#SBATCH --exclude=g42-h100-instance-[065,066,078,084,085,070,135,139,210,124,125,127,095,096,129]

#SBATCH --partition=higherprio

# test node usability, mainly NCCL test
# test 1: g42-h100-instance-[020-022,036-040,042-046,132,137,145]
# test 2: g42-h100-instance-[020-022,036-040,042-046,089,132,137]
# test 3: g42-h100-instance-[036-040,042-046,089,132,137,185,195,205]
# test 4: g42-h100-instance-[020-022,036-040,042-046,089,132,137]
# test 5: g42-h100-instance-[131-132,137-138,166,193,195,205]

eval "$(conda shell.bash hook)"
conda deactivate

export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=8
NNODES=16

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export WANDB_API_KEY=99bf1d8cdfb7226ae980b4cf8e12d840f05cdb24
export WANDB_ENTITY=fanzhou
export WANDB_PROJECT=megamath


# megatron-slm config
# current_directory=$(dirname "$(readlink -f "$0")")
# parent_directory=$(dirname "$current_directory")
current_directory=/mbz/users/fan.zhou/tron/Megatron-Sailor2

echo $current_directory

# current_directory=$(dirname "$(readlink -f "$0")")
parent_directory=$(dirname "$current_directory")

# ###zili args
# LOG_ARGS="--log_interval 1 --save_interval 2500 --eval_interval 500 --eval_iters 10"

#training
LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 100 --eval_iters 10"

TRAIN_ARGS="--train_iters 2500 --lr_decay_style constant --weight_decay 0.01 
--lr_warmup_iters 0 --lr 5e-6 --min_lr 5e-6 
--adam_beta1 0.9 --adam_beta2 0.95 --adam_eps 1e-5"
TRAIN_ARGS+=" --override_opt_param_scheduler"

# DISTRIBUTED_ARGS="--nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
# DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8000"
# DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 2 --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 4 --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# DISTRIBUTED_ARGS="--nproc_per_node 8 --nnodes 8 --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --rdzv_id $RANDOM
    --rdzv_backend c10d
    --rdzv_endpoint $head_node_ip:29500
)

QWEN_ARGS="--use_rms_norm --glu_activation swiglu --no_tie_embed_logits 
--no_new_tokens --layernorm_epsilon 1e-6 --rope_theta 1e6 --use_flash_attn --bf16 --seq_length 4096"
COMMON_ARGS="--hidden_dropout 0.0 --attention_dropout 0.0 --no_bias_gelu_fusion --no_bias_dropout_fusion  
--no_query_key_layer_scaling --attention_softmax_in_fp32 --intra_doc"

WANDB_ARGS="--wandb_logger --wandb_project megamath
--wandb_entity fanzhou --wandb_id megamath_32b_t8p2_final
--wandb_api_key 99bf1d8cdfb7226ae980b4cf8e12d840f05cdb24"
# WANDB_ARGS=""


#####training mode stage 1#####

MEGATRON_Trains=()
for i in {0..148}
do
    MEGATRON_Trains+=("1 /mbz/users/fan.zhou/tron/Megatron-Sailor2/dataset/web_pro/web_pro_${i}_text_document")
done


MEGATRON_Valids=(
	1 /mbz/users/fan.zhou/tron/Megatron-Sailor2/dataset/web_pro/web_pro_149_text_document
)

#####training mode stage 2#####

# MEGATRON_Trains=()
# for i in {251..499}
# do
#     MEGATRON_Trains+=("1 /mbz/users/fan.zhou/megatron-slm/sailor2-megatron/chunk_${i}_normalized_text_document")
# done
# MEGATRON_Valids=(
# 	1 /mbz/users/fan.zhou/megatron-slm/sailor2-megatron/chunk_500_normalized_text_document
# )


# #####debug mode#####
# MEGATRON_Trains=(
# 	1 ${current_directory}/dataset/web_pro/train_text_document
# )

# MEGATRON_Valids=(
# 	1 ${current_directory}/dataset/sample_preprocessed_data/train_text_document
# )


DATA_EXTRA_PARAMS="\
    --train_data_path ${MEGATRON_Trains[*]} \
    --valid_data_path ${MEGATRON_Valids[*]} \
    --test_data_path ${MEGATRON_Valids[*]}
"

# ###first run
# CUDA_DEVICE_MAX_CONNECTIONS=1 \
# torchrun $DISTRIBUTED_ARGS finetune.py \
# 	--tensor_model_parallel_size 4 \
# 	--pipeline_model_parallel_size 2 \
# 	--load ${current_directory}/model/qwen2_20b_megatron_t4p2  \
# 	--save ${current_directory}/model/qwen2_20b_megatron_t4p2_test  \
# 	--tensorboard_dir ${current_directory}/model/qwen2_20b_megatron_t4p2_test/logs/ \
# 	--model_name qwen \
# 	--tokenizer_type Qwen2Tokenizer \
# 	--vocab_file Qwen/Qwen2.5-14B  \
# 	--micro_batch_size 1 \
# 	--global_batch_size 1024 \
# 	--sequence_parallel \
#     --recompute_granularity selective \
# 	--use_checkpoint_args \
#     $COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $QWEN_ARGS $DATA_EXTRA_PARAMS $WANDB_ARGS


###resume run
srun --container-name=stanford_2 --container-mounts="/mbz:/mnt" \
    --container-image=/mbz/users/fan.zhou/tron/Megatron-Sailor2/stanford_2.sqsh \
    --chdir=/mbz/users/fan.zhou/tron/Megatron-Sailor2 \
    torchrun ${DISTRIBUTED_ARGS[@]} /mbz/users/fan.zhou/tron/Megatron-Sailor2/finetune.py \
    --tensor_model_parallel_size 8 \
	--pipeline_model_parallel_size 2 \
	--load /mbz/users/fan.zhou/tron/ckpts/qwen_32b_megatron_t8p2  \
	--save /mbz/users/fan.zhou/tron/ckpts/qwen_32b_megatron_t8p2_cpt  \
	--tensorboard_dir /mbz/users/fan.zhou/tron/ckpts/qwen_32b_megatron_t8p2_cpt/logs/ \
	--model_name qwen \
	--tokenizer_type Qwen2Tokenizer \
	--vocab_file Qwen/Qwen2.5-32B  \
	--micro_batch_size 2 \
	--global_batch_size 1024 \
    --sequence_parallel \
    --recompute_granularity selective \
	--use_checkpoint_args \
    $COMMON_ARGS $LOG_ARGS $TRAIN_ARGS $QWEN_ARGS $DATA_EXTRA_PARAMS $WANDB_ARGS
