# Megatron-Sailor2: Training Multilingual LLMs for South-East Asia Languages

[![Homepage](https://img.shields.io/badge/🏠-Homepage-3C47EB.svg)](https://sea-sailor.github.io/blog/sailor2/) &nbsp;&nbsp; [![Technical Report](https://img.shields.io/badge/arXiv-Report-b31b1b.svg)](https://arxiv.org/pdf/2502.12982) &nbsp;&nbsp; [![HuggingFace](https://img.shields.io/badge/🤗-Model&Demo-E87948.svg)](https://huggingface.co/collections/sail/sailor2-language-models-674d7c9e6b4dbbd9a869906b) &nbsp;&nbsp;



This repository contains the code for training Sailor2, a powerful and inclusive open language models for South-East Asia.


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



## SFT tuning

```
bash run_preprocess_data_sft.sh
bash run_train_qwen2_05b_sft.sh
```

## Acknowledgement
Please refer to [Megatron-LLM-Documentation](https://epfllm.github.io/Megatron-LLM/guide/getting_started.html#setup) for more details.


## Citation

  ```
@article{sailor2report,
  title  = {Sailor2: Sailing in South-East Asia with Inclusive Multilingual LLM},
  author = {Longxu Dou and Qian Liu and Fan Zhou and Changyu Chen and Zili Wang and Ziqi Jin and Zichen Liu and Tongyao Zhu and Cunxiao Du and Penghui Yang and Haonan Wang and Jiaheng Liu and Yongchi Zhao and Xiachong Feng and Xin Mao and Man Tsung Yeung and Kunat Pipatanakul and Fajri Koto and Min Si Thu and Hynek Kydl{\'\i}{\v{c}}ek and Zeyi Liu and Qunshu Lin and Sittipong Sripaisarnmongkol and Kridtaphad Sae-Khow and Nirattisai Thongchim and Taechawat Konkaew and Narong Borijindargoon and Anh Dao and Matichon Maneegard and Phakphum Artkaew and Zheng-Xin Yong and Quan Nguyen and Wannaphong Phatthiyaphaibun and Hoang H. Tran and Mike Zhang and Shiqi Chen and Tianyu Pang and Chao Du and Xinyi Wan and Wei Lu and Min Lin},
  journal={arXiv preprint arXiv:2502.12982},
  year   = {2025}
}
  ```

  ```
@software{epfmgtrn,
  author       = {Alejandro Hernández Cano  and
                  Matteo Pagliardini  and
                  Andreas Köpf  and
                  Kyle Matoba  and
                  Amirkeivan Mohtashami  and
                  Xingyao Wang  and
                  Olivia Simin Fan  and
                  Axel Marmet  and
                  Deniz Bayazit  and
                  Igor Krawczuk  and
                  Zeming Chen  and
                  Francesco Salvi  and
                  Antoine Bosselut  and
                  Martin Jaggi},
  title        = {epfLLM Megatron-LLM},
  year         = 2023,
  url          = {https://github.com/epfLLM/Megatron-LLM}
}
  ```