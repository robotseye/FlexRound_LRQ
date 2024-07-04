batch_size_inference=1
batch_size_quant=1
cache_dir=$YOUR_CACHE_PATH
channel_wise=True
dataset=c4
do_train=False
epoch=0
mode='lrq'
iters_w=5000
model_name_or_path=meta-llama/Llama-2-7b-hf
n_bits_w=4
num_samples=512
output_dir=$YOUR_OUTPUT_PATH
quantization_dataset=train
symmetric=False
clipping=False
transformer_block_size=1
w_lr=9e-4

CUDA_VISIBLE_DEVICES=0 python run_clm.py  \
--model_name_or_path $model_name_or_path  \
--do_train $do_train  --do_eval  \
--dataset_name $dataset \
--per_device_train_batch_size $batch_size_quant \
--per_device_eval_batch_size $batch_size_inference \
--num_train_epochs $epoch  \
--output_dir $output_dir \
--n_bits_w $n_bits_w \
--num_samples $num_samples \
--iters_w $iters_w \
--keep_cpu \
--overwrite_cache \
--channel_wise $channel_wise \
--symmetric $symmetric \
--clipping $clipping \
--w_lr $w_lr \
--mode $mode \
--overwrite_output_dir \
--transformer_block_size $transformer_block_size \
--quantization_dataset $quantization_dataset \
--cache_dir $cache_dir \
--recon_fp16 \
