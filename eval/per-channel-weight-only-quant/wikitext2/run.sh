### LRQ W4A16KV16
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model Llama-2-7b-hf-LRQ-w4a16
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model Llama-2-13b-hf-LRQ-w4a16
CUDA_VISIBLE_DEVICES=0,1 python evaluate_hf.py --model Llama-2-70b-hf-LRQ-w4a16

### LRQ W3A16KV16
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model Llama-2-7b-hf-LRQ-w3a16
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model Llama-2-13b-hf-LRQ-w3a16
CUDA_VISIBLE_DEVICES=0,1 python evaluate_hf.py --model Llama-2-70b-hf-LRQ-w3a16

### FlexRound W4A16KV16
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model Llama-2-7b-hf-FlexRound-w4a16
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model Llama-2-13b-hf-FlexRound-w4a16
CUDA_VISIBLE_DEVICES=0,1 python evaluate_hf.py --model Llama-2-70b-hf-FlexRound-w4a16

### FlexRound W3A16KV16
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model Llama-2-7b-hf-FlexRound-w3a16
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model Llama-2-13b-hf-FlexRound-w3a16
CUDA_VISIBLE_DEVICES=0,1 python evaluate_hf.py --model Llama-2-70b-hf-FlexRound-w3a16


