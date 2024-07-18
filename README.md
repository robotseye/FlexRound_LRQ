# FlexRound (ICML 2023) & LRQ 

Code and models for papers: (i) [**FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization**](https://arxiv.org/pdf/2306.00317), and (ii) [**LRQ: Optimizing Post-Training Quantization for Large Language Models by Learning Low-Rank Weight-Scaling Matrices**](https://arxiv.org/pdf/2407.11534)

(The current code can be applied to only Llama and Llama 2 models)

## Quantized Llama 2 Models by LRQ

| Model | W4A16 | W3A16 | W4A8 |
| ----- | ---- | ---- | ---- |
| Llama-2-7b | [Llama-2-7b-hf-LRQ-w4a16](https://huggingface.co/onliwad101/Llama-2-7b-hf-LRQ-w4a16) | [Llama-2-7b-hf-LRQ-w3a16](https://huggingface.co/onliwad101/Llama-2-7b-hf-LRQ-w3a16) | [Llama-2-7b-hf-LRQ-w4a8](https://huggingface.co/onliwad101/Llama-2-7b-hf-LRQ-w4a8) |
| Llama-2-13b | [Llama-2-13b-hf-LRQ-w4a16](https://huggingface.co/onliwad101/Llama-2-13b-hf-LRQ-w4a16) | [Llama-2-13b-hf-LRQ-w3a16](https://huggingface.co/onliwad101/Llama-2-13b-hf-LRQ-w3a16) | [Llama-2-13b-hf-LRQ-w4a8](https://huggingface.co/onliwad101/Llama-2-13b-hf-LRQ-w4a8) |
| Llama-2-70b | [Llama-2-70b-hf-LRQ-w4a16](https://huggingface.co/onliwad101/Llama-2-70b-hf-LRQ-w4a16) | [Llama-2-70b-hf-LRQ-w3a16](https://huggingface.co/onliwad101/Llama-2-70b-hf-LRQ-w3a16) | [Llama-2-70b-hf-LRQ-w4a8](https://huggingface.co/onliwad101/Llama-2-70b-hf-LRQ-w4a8) |

Quantized Llama 2 models by FlexRound will be uploaded soon.


## How to quantize Llama 2 models

### 0) Setup

```
pip install -r requirement.txt
```

### 1) FlexRound
```
cd scripts/FlexRound
```
and run one of the bash files depending on the desired model and bits.

For example, if you want the quantized Llama 2 7B model to W4A16 by FlexRound, then
```
run Llama-2-7b-hf-FlexRound-w4a16.sh
```

### 2) LRQ
```
cd scripts/LRQ
```
and run one of the bash files depending on the desired model and bits.

For example, if you want the quantized Llama 2 7B model to W4A16 by LRQ, then
```
run Llama-2-7b-hf-LRQ-w4a16.sh
```

### 3) Transformation

As the quantized model by FlexRound or LRQ possesses custom linear layers, we transform custom linear layers into nn.Linear.

For example, you quantized the Llama 2 7B model and save it to path/to/quantized_model, then
```
cd utils
python transform.py --model meta-llama/Llama-2-7b --path path/to/quantized_model --output_dir path/to/output_dir
```


## How to evaluate quantized Llama 2 models

### 1) W4A16 or W3A16

#### (1) WikiText2

```
cd eval/per-channel-weight-only-quant/wikitext2
bash run.sh
```

#### (2) Commonsense reasoning tasks

```
cd eval/per-channel-weight-only-quant/lm-evaluation-harness
bash run.sh
```

### 2) W4A8

#### (1) MMLU

```
cd eval/per-channel-weight-per-token-activation-quant/mmlu
```
After downloading the test set as described in README.md,
```
bash run.sh
```

#### (2) Commonsense reasoning tasks

```
cd val/per-channel-weight-per-token-activation-quant/lm-evaluation-harness
bash run.sh
```


## Citation

    @misc{lee2023flexroundlearnableroundingbased,
      title={FlexRound: Learnable Rounding based on Element-wise Division for Post-Training Quantization}, 
      author={Jung Hyun Lee and Jeonghoon Kim and Se Jung Kwon and Dongsoo Lee},
      year={2023},
      eprint={2306.00317},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2306.00317}, 
    }

    @misc{lee2024lrqoptimizingposttrainingquantization,
      title={LRQ: Optimizing Post-Training Quantization for Large Language Models by Learning Low-Rank Weight-Scaling Matrices}, 
      author={Jung Hyun Lee and Jeonghoon Kim and June Yong Yang and Se Jung Kwon and Eunho Yang and Kang Min Yoo and Dongsoo Lee},
      year={2024},
      eprint={2407.11534},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.11534}, 
    }

