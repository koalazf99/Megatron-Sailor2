export BASE_IMAGE=docker://nvcr.io#nvidia/pytorch:23.07-py3
enroot import -o ngc_torch_23_07.sqsh  $BASE_IMAGE
enroot create --name qwen_ngc_23_07 ngc_torch_23_07.sqsh
enroot start -w --mount /mbz:/mnt qwen_ngc_23_07
cd ~/tron/Megatron-Sailor2
conda deactivate
/usr/local/bin/pip install -r requirements.txt
exit

enroot export -o stanford_2.sqsh qwen_ngc_23_07
enroot create --name stanford_2 stanford_2.sqsh
enroot start -w --mount /mbz:/mnt stanford_2


# use old image
# ====

enroot create --name slm_ngc_23_07_copy_old ngc_torch_23_07_2.sqsh
enroot start -w --mount /mbz:/mnt slm_ngc_23_07_copy_old
cd ~/tron/Megatron-Sailor2


# 32B conversion
# if conda keeps activated, use /usr/bin/python for absolute path
python weights_conversion/hf_to_megatron.py qwen \
    --size 32 \
    --out ~/tron/ckpts/qwen_32b_megatron \
    --model-path ~/tron/ckpts/Qwen2.5-32B

python tools/checkpoint_util.py \
	--target_tensor_parallel_size  8 \
	--target_pipeline_parallel_size 2 \
	--load_dir  ~/tron/ckpts/qwen_32b_megatron \
	--save_dir ~/tron/ckpts/qwen_32b_megatron_t8p2 \
	--model_type qwen \
	--true_vocab_size 152064 \
	--megatron_path ~/tron/Megatron-Sailor2 \
	--vocab_file Qwen/Qwen2.5-32B \
	--bf16


###preprocess
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
# ===

export BASE_IMAGE=docker://nvcr.io#nvidia/pytorch:23.07-py3
enroot import -o ngc_torch_23_07.sqsh $BASE_IMAGE
enroot create --name slm_ngc_23_07_copy ngc_torch_23_07.sqsh
enroot start -w --mount /mbz:/mnt slm_ngc_23_07_copy
cd ~/tron/Megatron-Sailor2
conda deactivate
bash run_setup.sh
pip install -r requirements.txt
pip install sentencepiece
exit
enroot export -o slm_ngc_23_07_copy.sqsh slm_ngc_23_07_copy



enroot create --name slm_ngc_23_07_copy slm_ngc_23_07_copy.sqsh
enroot start -w --mount /mbz:/mnt slm_ngc_23_07_copy
enroot export -o slm_ngc_23_07_copy.sqsh slm_ngc_23_07_copy
