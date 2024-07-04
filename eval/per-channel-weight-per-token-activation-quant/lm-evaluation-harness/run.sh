### LRQ W4A8KV8
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=Llama-2-7b-hf-LRQ-w4a8     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=Llama-2-13b-hf-LRQ-w4a8     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0,1 python main.py     --model hf-causal-experimental     --model_args pretrained=Llama-2-70b-hf-LRQ-w4a8     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache

### FlexRound W4A8KV8
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=Llama-2-7b-hf-FlexRound-w4a8     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0 python main.py     --model hf-causal-experimental     --model_args pretrained=Llama-2-13b-hf-FlexRound-w4a8     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache
CUDA_VISIBLE_DEVICES=0,1 python main.py     --model hf-causal-experimental     --model_args pretrained=Llama-2-70b-hf-FlexRound-w4a8     --tasks openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq    --num_fewshot 0    --no_cache

