This codebase is based on [Megatron-LLM](https://github.com/epfLLM/Megatron-LLM/) with adaption for Qwen2.5 and Sailor2.

# Quickstart for training Sailor2-20B. 

1. Apply the running pod

Run with the common nvcr docker
```
sudo docker run --gpus all -it --rm -v /path/to/Megatron-Sailor2/:/mpt/Megatron-Sailor2 nvcr.io/nvidia/pytorch:23.07-py3 --shm-size 512g
```


Note: “if you use Torch multiprocessing for multi-threaded data loaders, the default shared memory segment size that the container runs with may not be enough. Therefore, you should increase the shared memory size by issuing … "
We set the shm-size to be 128gb since model sharding takes large shared memory.


Enter the repository:
```
cd /mpt/Megatron-Sailor2/
```

Install the additional dependencies not included in the `nvcr` image
```
pip install -r requirements.txt
```


2. Configure your huggingface token and wandb token in `run_setup.sh` then run

```
bash run_setup.sh
```

3. Install the ```megatron/data/helpers``` binary:

```
cd megatron/data/
make
cd ../../
```

4. Run data preprocess and model convert

```
bash run_sailor2.sh
```

5. Training model

```
bash run_train_sailor2_20b.sh
```

Adjust `DISTRIBUTED_ARGS` in the running bash for more GPUs (default=8).



### SFT tuning

```
bash run_preprocess_data_sft.sh
bash run_train_qwen2_05b_sft.sh
```

### Acknowledgement
Please refer to [Megatron-LLM-Documentation](https://epfllm.github.io/Megatron-LLM/guide/getting_started.html#setup) for more details.