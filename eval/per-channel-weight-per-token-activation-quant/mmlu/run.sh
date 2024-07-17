### LRQ W4A8KV8
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model onliwad101/Llama-2-7b-hf-LRQ-w4a8 --per_token_act_quant --kv_output_quant
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model onliwad101/Llama-2-13b-hf-LRQ-w4a8 --per_token_act_quant --kv_output_quant
CUDA_VISIBLE_DEVICES=0,1 python evaluate_hf.py --model onliwad101/Llama-2-70b-hf-LRQ-w4a8 --per_token_act_quant --kv_output_quant

### FlexRound W4A8KV8
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model onliwad101/Llama-2-7b-hf-FlexRound-w4a8 --per_token_act_quant --kv_output_quant
CUDA_VISIBLE_DEVICES=0 python evaluate_hf.py --model onliwad101/Llama-2-13b-hf-FlexRound-w4a8 --per_token_act_quant --kv_output_quant
CUDA_VISIBLE_DEVICES=0,1 python evaluate_hf.py --model onliwad101/Llama-2-70b-hf-FlexRound-w4a8 --per_token_act_quant --kv_output_quant

