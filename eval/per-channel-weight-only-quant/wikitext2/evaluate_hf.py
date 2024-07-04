# https://github.com/IST-DASLab/gptq/blob/main/datautils.py
import argparse
import json
import os
import time
import sys
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoConfig


def get_wikitext2(tokenizer, nsamples=128, seed=0, seqlen=2048):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def main(args):
    if args.model is None:
        print('model is empty!')
        raise NotImplementedError
    else:
        model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16, 
                device_map = 'balanced',
                use_safetensors=False,
                )

        tokenizer = LlamaTokenizer.from_pretrained(
                args.model,
                )

    model.eval()
    print('tokenizer.model_max_length: ', tokenizer.model_max_length)
    if tokenizer.model_max_length > 2048:
        tokenizer.model_max_length = 2048
        print('tokenizer.model_max_length: ', tokenizer.model_max_length)
    print()

    _, testenc = get_wikitext2(tokenizer)
    seq_len = testenc.input_ids.size(1)
    print('testenc.input_ids.size(): ', testenc.input_ids.size())

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, 2048)):
        end_loc = min(begin_loc + tokenizer.model_max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = testenc.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    print('ppl: ', ppl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str)
    args = parser.parse_args()
    main(args)
