# https://github.com/ollmer/mmlu/tree/master
import argparse
import json
import os
import time
import sys
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig
from transformers import Trainer, TrainingArguments

from transformers.modeling_utils import load_sharded_checkpoint

from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention, GPTNeoMLP, GPTNeoForCausalLM
from transformers.models.gptj.modeling_gptj import GPTJAttention, GPTJMLP, GPTJForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaForCausalLM

from SwapLinear import swapUniformQ

def quantize_model(model): 
    if isinstance(model, LlamaForCausalLM):
        for name, m in model.model.named_modules():
            if isinstance(m, LlamaMLP):
                print(name)
                m.gate_proj = swapUniformQ(m.gate_proj)
                m.up_proj = swapUniformQ(m.up_proj)
                m.down_proj = swapUniformQ(m.down_proj)
            elif isinstance(m, LlamaAttention):
                print(name)
                m.q_proj = swapUniformQ(m.q_proj)
                m.k_proj = swapUniformQ(m.k_proj)
                m.v_proj = swapUniformQ(m.v_proj)
                m.o_proj = swapUniformQ(m.o_proj)
    else:
        raise NotImplementedError
    return model


def main(args):
    if args.model is None:
        print('Original model is empty!')
        raise NotImplementedError
    else:
        fp_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16, 
                cache_dir=args.cache_dir,
                device_map = 'balanced',
                use_safetensors=False,
                )
        model = deepcopy(fp_model)

        tokenizer = LlamaTokenizer.from_pretrained(
                args.model,
                cache_dir=args.cache_dir,
                )

        if args.path is None:
            print('Quantized model is empty!')
            raise NotImplementedError
        else:
            with torch.no_grad():
                model = quantize_model(model) 
            print('Before loading the model!')
            print(args.path + '/' + args.model)
            load_sharded_checkpoint(model, args.path + '/' + args.model, strict=False)
            print('After loading the model!')

    fp_model.eval()
    model.eval()

    with torch.no_grad():
        if 'llama' in args.model.lower():
            for idx in range(fp_model.config.num_hidden_layers):
                fp_model.model.layers[idx].self_attn.q_proj.weight.copy_(model.model.layers[idx].self_attn.q_proj.quantized_weight.data)
                fp_model.model.layers[idx].self_attn.k_proj.weight.copy_(model.model.layers[idx].self_attn.k_proj.quantized_weight.data)
                fp_model.model.layers[idx].self_attn.v_proj.weight.copy_(model.model.layers[idx].self_attn.v_proj.quantized_weight.data)
                fp_model.model.layers[idx].self_attn.o_proj.weight.copy_(model.model.layers[idx].self_attn.o_proj.quantized_weight.data)
                fp_model.model.layers[idx].mlp.up_proj.weight.copy_(model.model.layers[idx].mlp.up_proj.quantized_weight.data)
                fp_model.model.layers[idx].mlp.down_proj.weight.copy_(model.model.layers[idx].mlp.down_proj.quantized_weight.data)
                fp_model.model.layers[idx].mlp.gate_proj.weight.copy_(model.model.layers[idx].mlp.gate_proj.quantized_weight.data)
                print(f"Complete the {idx+1}-th layer!")
        else:
            raise NotImplementedError
    
        training_args = TrainingArguments(
            output_dir=args.output_dir,
        )

        trainer = Trainer(
            model=fp_model,
            args=training_args,
            tokenizer=tokenizer,
        )

        os.makedirs(args.output_dir + '/' + args.model, exist_ok=True)
        trainer.save_model(args.output_dir + '/' + args.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", "-o", type=str, default='./')
    args = parser.parse_args()
    main(args)
