"""
Convert weights from models in other formats (primairly huggingface) to megatron checkpoints.

This script supports converting Falcon, LLaMa and LLaMa 2 weights to megatron checkpoints.
Depending on the model to convert, the inputs might differ.
- Falcon:
    Weights are automatically retrieved from the official implementation hosted in huggingface.
    Thus, the `--cache-dir` argument is optional, if specified it should point to
    the huggingface cache directory where the huggingface Falcon weights will be stored.
    You will need to specify the `--size` argument to determine which version to download
    (i.e. Falcon 7B or 40B).
- LLaMa, LLaMa 2 and CodeLlama:
    Converting llama weights can be done either fetching the weights hosted
    in huggingface (recommended as it is the easier method) or directly from the
    weights provided by Meta.
    - From Meta weights (only available for LLaMa and LLaMa 2):
        You will need to specify the `--cache-dir` to the directory where the
        llama weights are stored.
        This will by default have the form `xB` (e.g. 7B or 70B) for llama v1,
        or `llama-2-xb` (e.g. llama-2-7b) for llama v2.
    - From huggingface weights:
        If `--cache-dir` is not specified or the directory specified does not
        contain the format expected from Meta weights, the converter will automatically
        retrieve the weights from huggingface, in which case the `--cache-dir` will
        have the same semantics as with Falcon.

        Note that to download llama v2 weights from huggingface, you will need to
        login using `huggingface-cli login` with a huggingface account which has been
        granted access to the `meta-llama/Llama-2-7b-hf` model.
        

In all cases, the megatron checkpoint will be stored in the `--out` argument.
If a huggingface is specified, the intermediate weights (i.e. the huggingface weights)
stored therein will not be removed when the conversion succeeds.
"""

import re
import sys
import shutil
import os
from pathlib import Path
from typing import Optional
from argparse import ArgumentParser, Namespace

import torch
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer

from utils.permute_qkv import permute_qkv, permute_qkv_bias
from utils.merge_llama import merge_llama


llama_s2layer = {7: 32, 13: 40, 30: 60, 34: 48, 65: 80, 70: 80}
llama_s2heads = {7: 32, 13: 40, 30: 52, 34: 64, 65: 64, 70: 64}
llama_s2dense = {7: 11008, 13: 13824, 30: 17920, 34: 22016, 65: 22016,
                 70: 28672}  # should be (2/3)*4*d, but it isn't exaclty that
llama_s2hidden = {7: 4096, 13: 5120, 30: 6656, 34: 8192, 65: 8192, 70: 8192}


qwen_s2layer = {5: 24, 24: 48, 15: 28, 7: 28, 8: 32, 14: 48, 18: 64, 20: 64, 32: 64, 3: 36} #num_hidden_layers
qwen_s2heads = {5: 14, 24: 14, 15: 12, 7: 28, 8: 28, 14: 40, 18: 40, 20: 40, 32: 40, 3: 16} #num_attention_heads
qwen_s2kvheads = {5: 2, 24: 2, 15: 2, 7: 4, 8: 4, 14: 8, 18: 8, 20: 8, 32: 8, 3: 2} #num_key_value_heads
qwen_s2dense = {5: 4864, 24: 4864, 15: 8960, 7: 18944, 8: 18944, 14: 13824, 18: 13824, 20: 13824, 32: 27648, 3: 11008} #intermediate_size
qwen_s2hidden = {5: 896, 24: 896, 15: 1536, 7: 3584, 8: 3584, 14: 5120, 18: 5120, 20: 5120, 32: 5120, 3: 2048} #hidden_size
qwen_s2rope = {5: 1e6, 24: 1e6, 15: 1e6, 7: 1e6, 8: 1e6, 14: 1e6, 18: 1e6, 20: 1e6, 32: 1e6, 3: 1e6} #rope_theta
qwen_s2vocab = {5: 151936, 24: 151936, 15: 151936, 7: 152064, 8: 152064, 14: 152064, 18: 152064, 20: 152064, 32: 152064, 3: 151936} #padded_vocab_size

def falcon_to_megatron(weights: dict, size: int) -> dict:
    def permute(qkv_w):
        return permute_qkv(qkv_w, dim, n_heads, n_heads_kv)

    embedding = {}
    transformer = {}
    if size == 7:
        n_layer = 32
        dim = 4544
        n_heads = 71
        n_heads_kv = 1
    else:
        n_layer = 60
        dim = 8192
        n_heads = 128
        n_heads_kv = 8

    # weights independent of layers (i.e. token embeddings and layernorms
    assert torch.allclose(weights["lm_head.weight"],
                          weights["transformer.word_embeddings.weight"])
    embedding["word_embeddings.weight"] = weights["transformer.word_embeddings.weight"]
    transformer["final_layernorm.weight"] = weights["transformer.ln_f.weight"]
    transformer["final_layernorm.bias"] = weights["transformer.ln_f.bias"]

    # copy weights for each transformer layer
    for layer in trange(n_layer, desc="Converting weights"):
        prefix1 = f"layers.{layer}"
        prefix2 = f"transformer.h.{layer}"
        # mlp
        transformer[f"{prefix1}.mlp.dense_h_to_4h.weight"] = \
            weights[f"{prefix2}.mlp.dense_h_to_4h.weight"]
        transformer[f"{prefix1}.mlp.dense_4h_to_h.weight"] = \
            weights[f"{prefix2}.mlp.dense_4h_to_h.weight"]
        # qkv weights
        transformer[f"{prefix1}.attention.query_key_value.weight"] = \
            permute(weights[f"{prefix2}.self_attention.query_key_value.weight"])
        # dense
        transformer[f"{prefix1}.attention.dense.weight"] = \
            weights[f"{prefix2}.self_attention.dense.weight"]
        # falcon7 and falcon40 differ in the input layernorms
        if size == 7:
            transformer[f"{prefix1}.input_layernorm.weight"] = \
                weights[f"{prefix2}.input_layernorm.weight"]
            transformer[f"{prefix1}.input_layernorm.bias"] = \
                weights[f"{prefix2}.input_layernorm.bias"]
        else:
            transformer[f"{prefix1}.input_layernorm.weight"] = \
                weights[f"{prefix2}.ln_attn.weight"]
            transformer[f"{prefix1}.mlp_layernorm.weight"] = \
                weights[f"{prefix2}.ln_mlp.weight"]
            transformer[f"{prefix1}.input_layernorm.bias"] = \
                weights[f"{prefix2}.ln_attn.bias"]
            transformer[f"{prefix1}.mlp_layernorm.bias"] = \
                weights[f"{prefix2}.ln_mlp.bias"]
    return {"embedding": embedding, "transformer": transformer}


def llama_to_megatron(weights: dict, size: int, source: str = "meta",
                      version: int = 1) -> dict:
    def permute(qkv_w):
        if source == "hf":
            return permute_qkv(qkv_w, hidden, n_heads, n_kv_heads)
        return qkv_w

    def rearrange_qkv(wq, wk, wv):
        wq = torch.split(wq, n_hidden_per_head, dim=0)
        wk = torch.split(wk, n_hidden_per_head, dim=0)
        wv = torch.split(wv, n_hidden_per_head, dim=0)
        assert len(wq) == n_heads
        assert len(wk) == n_kv_heads
        assert len(wv) == n_kv_heads
        n_qs_per_kv = n_heads//n_kv_heads
        w_qkv = []
        for i in range(n_kv_heads):
            w_qkv += [wq[i*n_qs_per_kv + j] for j in range(n_qs_per_kv)]
            w_qkv += [wk[i], wv[i]]
        return permute(torch.concat(w_qkv))

    # config
    n_layer = llama_s2layer[size]
    hidden = llama_s2hidden[size]
    n_heads = llama_s2heads[size]
    n_hidden_per_head = hidden//n_heads
    n_kv_heads = n_heads if version == 1 or size <= 13 else 8

    # weights independent of layers
    embedding = {"word_embeddings.weight": weights["tok_embeddings.weight"]}
    transformer = {"final_layernorm.weight": weights["norm.weight"]}
    lm_head = weights["output.weight"]

    # get all the other weights
    for layer in trange(n_layer, desc="Converting weights"):
        prefix = f"layers.{layer}"
        # identical weights
        transformer[f"{prefix}.attention.dense.weight"] = \
            weights[f"{prefix}.attention.wo.weight"]
        transformer[f"{prefix}.post_attention_layernorm.weight"] = \
            weights[f"{prefix}.ffn_norm.weight"]
        transformer[f"{prefix}.input_layernorm.weight"] = \
            weights[f"{prefix}.attention_norm.weight"]
        transformer[f"{prefix}.mlp.dense_4h_to_h.weight"] = \
            weights[f"{prefix}.feed_forward.w2.weight"]
        # concatenate up, gate mlp weights
        transformer[f"{prefix}.mlp.dense_h_to_4h.weight"] = torch.concat([
            weights[f"{prefix}.feed_forward.w3.weight"],
            weights[f"{prefix}.feed_forward.w1.weight"]
        ])
        # finally, qkv requires serious manipulation to get right
        transformer[f"{prefix}.attention.query_key_value.weight"] = rearrange_qkv(
            weights[f"{prefix}.attention.wq.weight"],
            weights[f"{prefix}.attention.wk.weight"],
            weights[f"{prefix}.attention.wv.weight"]
        )

        # release references to original weights (free mem)
        del weights[f"{prefix}.feed_forward.w3.weight"]
        del weights[f"{prefix}.feed_forward.w1.weight"]
        del weights[f"{prefix}.attention.wq.weight"]
        del weights[f"{prefix}.attention.wk.weight"]
        del weights[f"{prefix}.attention.wv.weight"]

    return {"embedding": embedding, "transformer": transformer,
            "lm_head": lm_head}


def mistral_to_megatron(weights: dict, size: int) -> dict:
    assert size == 7
    def permute(qkv_w):
        # by default, we pull mistrals weights from huggingface
        return permute_qkv(qkv_w, hidden, n_heads, n_kv_heads)
        # return qkv_w

    def rearrange_qkv(wq, wk, wv):
        wq = torch.split(wq, n_hidden_per_head, dim=0)
        wk = torch.split(wk, n_hidden_per_head, dim=0)
        wv = torch.split(wv, n_hidden_per_head, dim=0)
        assert len(wq) == n_heads
        assert len(wk) == n_kv_heads
        assert len(wv) == n_kv_heads
        n_qs_per_kv = n_heads//n_kv_heads
        w_qkv = []
        for i in range(n_kv_heads):
            w_qkv += [wq[i*n_qs_per_kv + j] for j in range(n_qs_per_kv)]
            w_qkv += [wk[i], wv[i]]
        return permute(torch.concat(w_qkv))

    # config
    if size == 7:
        n_layer = 32
        hidden = 4096
        n_heads = 32
        n_kv_heads = 8
    n_hidden_per_head = hidden // n_heads

    print('embedding size', weights["model.embed_tokens.weight"].size())

    # weights independent of layers
    embedding = {"word_embeddings.weight": weights["model.embed_tokens.weight"]}
    transformer = {"final_layernorm.weight": weights["model.norm.weight"]}
    lm_head = weights["lm_head.weight"]

    # get all the other weights
    for layer in trange(n_layer, desc="Converting weights"):
        prefix = f"layers.{layer}"
        hf_prefix = f"model.{prefix}"
        # identical weights
        transformer[f"{prefix}.attention.dense.weight"] = \
            weights[f"{hf_prefix}.self_attn.o_proj.weight"]
        transformer[f"{prefix}.post_attention_layernorm.weight"] = \
            weights[f"{hf_prefix}.post_attention_layernorm.weight"]
        transformer[f"{prefix}.input_layernorm.weight"] = \
            weights[f"{hf_prefix}.input_layernorm.weight"]
        transformer[f"{prefix}.mlp.dense_4h_to_h.weight"] = \
            weights[f"{hf_prefix}.mlp.down_proj.weight"]
        # concatenate up, gate mlp weights
        transformer[f"{prefix}.mlp.dense_h_to_4h.weight"] = torch.concat([
            weights[f"{hf_prefix}.mlp.up_proj.weight"],  # w3
            weights[f"{hf_prefix}.mlp.gate_proj.weight"]  # w1
        ])
        # finally, qkv requires serious manipulation to get right (probably same as llama-2)
        transformer[f"{prefix}.attention.query_key_value.weight"] = rearrange_qkv(
            weights[f"{hf_prefix}.self_attn.q_proj.weight"],
            weights[f"{hf_prefix}.self_attn.k_proj.weight"],
            weights[f"{hf_prefix}.self_attn.v_proj.weight"]
        )

        # release references to original weights (free mem)
        del weights[f"{hf_prefix}.mlp.up_proj.weight"]
        del weights[f"{hf_prefix}.mlp.gate_proj.weight"]
        del weights[f"{hf_prefix}.self_attn.q_proj.weight"]
        del weights[f"{hf_prefix}.self_attn.k_proj.weight"]
        del weights[f"{hf_prefix}.self_attn.v_proj.weight"]

    return {"embedding": embedding, "transformer": transformer,
            "lm_head": lm_head}


def qwen_to_megatron(weights: dict, size: int) -> dict:
    def permute(qkv_w, hidden, n_heads, n_kv_heads):
        return permute_qkv(qkv_w, hidden, n_heads, n_kv_heads)

    def rearrange_qkv(wq, wk, wv, n_kv_heads, n_heads, n_hidden_per_head):
        wq = torch.split(wq, n_hidden_per_head, dim=0) 
        wk = torch.split(wk, n_hidden_per_head, dim=0) 
        wv = torch.split(wv, n_hidden_per_head, dim=0) 

        assert len(wq) == n_heads, f"Expected {n_heads} heads, but got {len(wq)}"
        assert len(wk) == n_kv_heads, f"Expected {n_kv_heads} kv heads, but got {len(wk)}"
        assert len(wv) == n_kv_heads, f"Expected {n_kv_heads} kv heads, but got {len(wv)}"
        
        n_qs_per_kv = n_heads//n_kv_heads
        w_qkv = []
        for i in range(n_kv_heads):
            w_qkv += [wq[i*n_qs_per_kv + j] for j in range(n_qs_per_kv)]
            w_qkv += [wk[i], wv[i]]

        return permute(torch.concat(w_qkv), n_hidden_per_head * n_heads, n_heads, n_kv_heads) 

    def rearrange_qkv_bias(bq, bk, bv, n_kv_heads, n_heads, n_hidden_per_head):
        bq = torch.split(bq, n_hidden_per_head)
        bk = torch.split(bk, n_hidden_per_head)
        bv = torch.split(bv, n_hidden_per_head)

        assert len(bq) == n_heads, f"Expected {n_heads} heads, but got {len(bq)}"
        assert len(bk) == n_kv_heads, f"Expected {n_kv_heads} kv heads, but got {len(bk)}"
        assert len(bv) == n_kv_heads, f"Expected {n_kv_heads} kv heads, but got {len(bv)}"
        
        n_qs_per_kv = n_heads // n_kv_heads
        b_qkv = []
        for i in range(n_kv_heads):
            b_qkv += [bq[i * n_qs_per_kv + j] for j in range(n_qs_per_kv)]
            b_qkv += [bk[i], bv[i]]

        return permute_qkv_bias(torch.cat(b_qkv, dim=0), n_hidden_per_head * n_heads, n_heads, n_kv_heads)

    n_layer = qwen_s2layer[size]
    hidden = qwen_s2hidden[size]
    n_heads = qwen_s2heads[size]
    n_kv_heads = qwen_s2kvheads[size]

    n_hidden_per_head = hidden // n_heads # 64 = 1024 / 16

    print('embedding size', weights["model.embed_tokens.weight"].size())

    embedding = {"word_embeddings.weight": weights["model.embed_tokens.weight"]}
    transformer = {"final_layernorm.weight": weights["model.norm.weight"]}
    lm_head = weights["lm_head.weight"]

    # get all the other weights
    for layer in trange(n_layer, desc="Converting weights"):
        prefix = f"layers.{layer}"
        hf_prefix = f"model.{prefix}"
        # identical weights
        transformer[f"{prefix}.attention.dense.weight"] = \
            weights[f"{hf_prefix}.self_attn.o_proj.weight"]
        transformer[f"{prefix}.post_attention_layernorm.weight"] = \
            weights[f"{hf_prefix}.post_attention_layernorm.weight"]
        transformer[f"{prefix}.input_layernorm.weight"] = \
            weights[f"{hf_prefix}.input_layernorm.weight"]
        transformer[f"{prefix}.mlp.dense_4h_to_h.weight"] = \
            weights[f"{hf_prefix}.mlp.down_proj.weight"]
        # concatenate up, gate mlp weights
        transformer[f"{prefix}.mlp.dense_h_to_4h.weight"] = torch.concat([
            weights[f"{hf_prefix}.mlp.up_proj.weight"],  # w3
            weights[f"{hf_prefix}.mlp.gate_proj.weight"]  # w1
        ])
        # finally, qkv requires serious manipulation to get right (probably same as llama-2)
        transformer[f"{prefix}.attention.query_key_value.weight"] = rearrange_qkv(
            weights[f"{hf_prefix}.self_attn.q_proj.weight"],
            weights[f"{hf_prefix}.self_attn.k_proj.weight"],
            weights[f"{hf_prefix}.self_attn.v_proj.weight"],
            n_kv_heads,
            n_heads,
            n_hidden_per_head,
        )

        # release references to original weights (free mem)
        del weights[f"{hf_prefix}.mlp.up_proj.weight"]
        del weights[f"{hf_prefix}.mlp.gate_proj.weight"]
        del weights[f"{hf_prefix}.self_attn.q_proj.weight"]
        del weights[f"{hf_prefix}.self_attn.k_proj.weight"]
        del weights[f"{hf_prefix}.self_attn.v_proj.weight"]

        transformer[f"{prefix}.attention.query_key_value.bias"] = rearrange_qkv_bias(
            weights[f"{hf_prefix}.self_attn.q_proj.bias"],
            weights[f"{hf_prefix}.self_attn.k_proj.bias"],
            weights[f"{hf_prefix}.self_attn.v_proj.bias"],
            n_kv_heads,
            n_heads,
            n_hidden_per_head,
        )
        
        del weights[f"{hf_prefix}.self_attn.q_proj.bias"]
        del weights[f"{hf_prefix}.self_attn.k_proj.bias"]
        del weights[f"{hf_prefix}.self_attn.v_proj.bias"]


    return {"embedding": embedding, "transformer": transformer,
            "lm_head": lm_head}


def main(model_name: str = "falcon", size: int = 7, out: Optional[Path] = None,
         cache_dir: Optional[Path] = None, model_path: Optional[str] = None, not_tie: bool=True):
    output_path = str(out)
    if out is None:
        out = Path(f"falcon{size}b_megatron.pt").absolute()
    if os.path.exists(out):
        shutil.rmtree(out)

    # get weights from or specified directory
    if model_name == "falcon":
        print("Fetching weights from huggingface")
        if model_path is None:
            model_path = f"tiiuae/falcon-{size}b",
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     trust_remote_code=True,
                                                     cache_dir=cache_dir)
        hf_weights = model.state_dict()
    elif model_name == "mistral":
        if model_path is None:
            print("Fetching weights from huggingface")
            model_path = "mistralai/Mistral-7B-v0.1"
        else:
            print("Fetching weights from disk")

        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    trust_remote_code=True,
                                                    cache_dir=cache_dir)
        hf_weights = model.state_dict()
    elif model_name == "qwen":
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                    trust_remote_code=True,
                                                    cache_dir=cache_dir)
        hf_weights = model.state_dict()
    else:
        print("Getting llama...")
        version = 2 if "2" in model_name else 1
        hf_weights, llama_source = merge_llama(size, version, root_dir=cache_dir,
                                               model_path=model_path)


    # convert state dict to be megatron-compatible
    if model_name == "falcon":
        megatron_weights = falcon_to_megatron(hf_weights, size)
    elif model_name == "mistral":
        megatron_weights = mistral_to_megatron(hf_weights, size)
    elif model_name == "qwen":
        megatron_weights = qwen_to_megatron(hf_weights, size)
    elif model_name == "gemma":
        megatron_weights = gemma_to_megatron(hf_weights, size)
    else:
        megatron_weights = llama_to_megatron(hf_weights, size, llama_source,
                                             version=1 if model_name == "llama" else 2)

    # set args
    dtype = megatron_weights["embedding"]["word_embeddings.weight"].dtype
    if model_name == "falcon":
        if size == 7:
            args = {"num_layers": 32, "hidden_size": 4544,
                    "num_attention_heads": 71, "num_attention_heads_kv": 1}
        else:
            args = {"num_layers": 60, "hidden_size": 8192,
                    "num_attention_heads": 128, "num_attention_heads_kv": 8,
                    "parallel_layernorm": True}
        args.update({"tokenizer_type": "FalconTokenizer", "use_flash_attn": True,
                     "hidden_dropout": 0.0,
                     "parallel_attn": True, "max_position_embeddings": 2048,
                     "seq_length": 2048})
    elif model_name == "mistral":
        assert size == 7
        # mistral-7b mostly uses the same args as llama-7b
        # https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json
        args = {
            "num_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_attention_heads_kv": 8,  # except this - GroupedAttention
            "ffn_hidden_size": 14336,  # except this
            "parallel_attn": False,
            "make_vocab_size_divisible_by": 128,
            "glu_activation": "swiglu",  # == silu
            "padded_vocab_size": 44864,
            "use_rms_norm": True,
            "tie_embed_logits": False,
            "tokenizer_type": "HFTokenizer",
            "max_position_embeddings": 32768,
            "seq_length": 32768,
            "layernorm_epsilon": 1e-5,
            "rope_theta": 10000.0,
            "sliding_window_size": 4096,
        }
    elif model_name == "qwen":
        args = {
            "num_layers": qwen_s2layer[size], #config
            "hidden_size": qwen_s2hidden[size], #config
            "num_attention_heads": qwen_s2heads[size], #config
            "num_attention_heads_kv": qwen_s2kvheads[size],  # config
            "ffn_hidden_size": qwen_s2dense[size],  # config
            "parallel_attn": False, #no 
            "make_vocab_size_divisible_by": 128, #no
            "glu_activation": "swiglu",  # hf document
            "padded_vocab_size": qwen_s2vocab[size], #config
            "use_rms_norm": True, # hf document
            "tie_embed_logits": True if size in [3, 5, 15] else False, #config
            "tokenizer_type": "Qwen2ChatTokenizer" if '_sft_' in output_path else "Qwen2Tokenizer", #no
            "max_position_embeddings": 32768, #config
            "seq_length": 32768, #no 
            "layernorm_epsilon": 1e-6, #config
            "rope_theta": qwen_s2rope[size], #config
            "sliding_window_size": 32768, #config
        }
    else:  # llama1, llama2, codellama
        args = {"num_layers": llama_s2layer[size],
                "hidden_size": llama_s2hidden[size],
                "num_attention_heads": llama_s2heads[size],
                "ffn_hidden_size": llama_s2dense[size],
                "parallel_attn": False,
                "make_vocab_size_divisible_by": 128,
                "glu_activation": "swiglu",
                "padded_vocab_size": 32000,
                "use_rms_norm": True,
                "tie_embed_logits": False,
                "tokenizer_type": "HFTokenizer"}
        if model_name == "llama":
            args.update({"max_position_embeddings": 2048, "seq_length": 2048,
                         "layernorm_epsilon": 1e-6})
        elif model_name == "llama2":
            args.update({"max_position_embeddings": 4096, "seq_length": 4096,
                         "layernorm_epsilon": 1e-5})
            if size >= 34:
                args.update({"num_attention_heads_kv": 8})
        elif model_name == "codellama":
            args.update({"max_position_embeddings": 16384, "seq_length": 16384,
                         "layernorm_epsilon": 1e-5, "rope_theta": 1e6})
            if size >= 34:
                args.update({"num_attention_heads_kv": 8})
            if size < 34 and not re.match(r"CodeLlama-\d+b-Python", cache_dir):
                args.update({"padded_vocab_size": 32016})
        else:
            sys.exit(f"Model name has to be llama, llama2 or codellama, not {model_name}.")


    args.update({
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "iteration": "release",
        "bias_gelu_fusion": False,
        "bias_droput_fusion": False,
        "position_embedding_type": "rotary"
    })

    # save converted weights in specified out
    (out/"release"/"mp_rank_00").mkdir(parents=True)
    with open(out/"latest_checkpointed_iteration.txt", "w+") as f:
        f.write("release")
    final_dict = {"iteration": "release", "model": {"language_model": megatron_weights},
                  "checkpoint_version": 3.0, "args": Namespace(**args)}
    torch.save(final_dict, out/"release"/"mp_rank_00"/"model_optim_rng.pt")
    print("Saved weights in", out)

    if model_name in {"llama", "llama2"} and llama_source == "hf":
        tokenizer = None
        if model_path is not None:
            try:
                tokenizer = LlamaTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
            except OSError:
                warnings.warn(f"Model path {model_path} does not have a "
                              "tokenizer, using default tokenizer instead")
        if tokenizer is None:
            if model_name == "llama2":
                name = "meta-llama/Llama-2-7b-hf"
            else:
                name = "decapoda-research/llama-7b-hf"
            tokenizer = LlamaTokenizer.from_pretrained(name, cache_dir=cache_dir)

        token_path = out/"tokenizer.model"
        vocab_file = tokenizer.vocab_file
        shutil.copy(vocab_file, token_path)
        print("Saved tokenizer.model in", token_path)
    elif model_name == "mistral":
        tokenizer = None
        if model_path is not None:
            try:
                tokenizer = LlamaTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
            except OSError:
                warnings.warn(f"Model path {model_path} does not have a "
                              "tokenizer, using default tokenizer instead")
        if tokenizer is None:
            print('reading mistralai/Mistral-7B-v0.1 tokenizer')
            tokenizer = LlamaTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", cache_dir=cache_dir)
        
        token_path = out/"tokenizer.model"
        vocab_file = tokenizer.vocab_file
        shutil.copy(vocab_file, token_path)
        print("Saved tokenizer.model in", token_path)
    elif model_name == "qwen":
        print('reading qwen-7B tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        token_path = out/"tokenizer.model"
    elif model_name == "gemma":
        print('reading Gemma tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        token_path = out/"tokenizer.model"

    print("Done")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert Huggingface llama or falcon weights to "
                                        "megatron-compatible weights")
    parser.add_argument("model", choices={"falcon", "llama", "llama2", "codellama", "mistral", "qwen", "gemma"})
    parser.add_argument("--size", default=7, choices={2, 3, 5, 24, 15, 18, 4, 7, 8, 9, 13, 14, 18, 20, 30, 32, 34, 40, 65, 70}, type=int,
                        help="The size of the model")
    parser.add_argument("--out", type=Path,
                        help="Directory to store the megatron weights (as checkpoint)")
    parser.add_argument("--model-path",
                        help="Sets model_name_or_path when fetching weights from huggingface")
    parser.add_argument("--cache-dir", type=Path,
                        help=("Directory to use as cache for the huggingface "
                              "weights, or in case of the llama model, the path "
                              "of the weights privided Meta"))
    args = parser.parse_args()

    # small arg verification
    if args.model == "falcon":
        assert args.size in {7, 40}
    elif args.model == "llama":
        assert args.size in {7, 13, 30, 65}
    elif args.model == "codellama":
        assert args.size in {7, 13, 34}
    elif args.model == "mistral":
        assert args.size in {7}
    elif args.model == "qwen":
        assert args.size in {3, 5, 24, 15, 18, 4, 7, 8, 14, 18, 20, 32}  
    elif args.model == "gemma":
        assert args.size in {2, 7, 9}  
    else:
        assert args.size in {7, 13, 70}

    main(args.model, args.size, args.out, args.cache_dir, args.model_path)