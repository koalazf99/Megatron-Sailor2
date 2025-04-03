export BASE_IMAGE=docker://nvcr.io#nvidia/pytorch:23.07-py3
enroot import -o ngc_torch_23_07.sqsh  $BASE_IMAGE
enroot create --name stanford_2 ngc_torch_23_07.sqsh
enroot start -w --mount /mbz:/mnt stanford_2

cd ~/tron/Megatron-Sailor2 # enter to the repo folder
conda deactivate # make sure you use /usr/bin/python!
/usr/local/bin/pip install -r requirements.txt
exit
enroot export -o stanford_2.sqsh stanford_2

# then you can use the stanford_2.sqsh to run the following commands
enroot create --name stanford_2 stanford_2.sqsh
enroot start -w --mount /mbz:/mnt stanford_2


# some running preparation to shard the model, and to tokenize the dataset
# ====
# 32B conversion
# if conda keeps activated, use /usr/bin/python for absolute path
enroot start -w --mount /mbz:/mnt stanford_2
python weights_conversion/hf_to_megatron.py qwen \
    --size 3 \
    --out ~/tron/ckpts/qwen_3b_megatron \
    --model-path ~/tron/ckpts/Qwen2.5-3B # your hf model path
# we use tp=8, pp=2, to shard the model
python tools/checkpoint_util.py \
	--target_tensor_parallel_size  4 \
	--target_pipeline_parallel_size 1 \
	--load_dir  ~/tron/ckpts/qwen_3b_megatron \
	--save_dir ~/tron/ckpts/qwen_3b_megatron_t4p1 \
	--model_type qwen \
	--true_vocab_size 151936 \
	--megatron_path ~/tron/Megatron-Sailor2 \
	--vocab_file Qwen/Qwen2.5-3B \
	--bf16

# ===
# preprocess
enroot start -w --mount /mbz:/mnt stanford_2
python tools/preprocess_data_parquet.py \
	--input /mbz/users/fan.zhou/tron/Megatron-Sailor2/jsonl_dataset/web_pro \
	--output_prefix /mbz/users/fan.zhou/tron/Megatron-Sailor2/dataset/web_pro/web_pro \
	--tokenizer_type Qwen2Tokenizer \
	--vocab_file Qwen/Qwen2.5-32B  \
	--chunk_size 32 \
	--workers 64 \
	--log_interval 20000 \
	--append_eod \
	--no_new_tokens
