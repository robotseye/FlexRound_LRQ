### LRQ W4A16KV16
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-7b-hf-LRQ-w4a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-13b-hf-LRQ-w4a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0,1 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-70b-hf-LRQ-w4a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache

### LRQ W3A16KV16
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-7b-hf-LRQ-w3a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-13b-hf-LRQ-w3a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0,1 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-70b-hf-LRQ-w3a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache

### FlexRound W4A16KV16
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-7b-hf-FlexRound-w4a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-13b-hf-FlexRound-w4a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0,1 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-70b-hf-FlexRound-w4a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache

### FlexRound W3A16KV16
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-7b-hf-FlexRound-w3a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-13b-hf-FlexRound-w3a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0,1 python main.py     --model hf-causal-experimental     --model_args pretrained=onliwad101/Llama-2-70b-hf-FlexRound-w3a16     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache

