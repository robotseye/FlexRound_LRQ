## For the sample command:

    CUDA_VISIBLE_DEVICES=0 python run_clm.py  \
    --model_name_or_path $model_name_or_path  \
    --do_train $do_train  --do_eval  \
    --dataset_name $dataset \
    --per_device_train_batch_size $batch_size_quant \
    --per_device_eval_batch_size $batch_size_inference \
    --learning_rate $lr   \
    --num_train_epochs $epoch  \
    --output_dir $output_dir \
    --n_bits_w $n_bits_w \
    --num_samples $num_samples \
    --iters_w $iters_w \
    --weight $weight\
    --keep_cpu \
    --wwq \
    --b_start $b_start --b_end $b_end\
    --warmup $warmup \
    --init_wmode $init_wmode \
    --order $order \
    --prob $prob --input_prob $input_prob \
    --overwrite_cache \
    --channel_wise $channel_wise \
    --symmetric $symmetric \
    --w_lr $w_lr \
    --flexround $flexround \
    --embedding_8bit $embedding_8bit \
    --overwrite_output_dir \
    --transformer_block_size $transformer_block_size \
    --quantization_dataset $quantization_dataset \
    --cache_dir $cache_dir 


## Configurations:

### Meta-llama Command
    b_end=2
    b_start=20
    batch_size_inference=1
    batch_size_quant=1
    cache_dir=$YOUR_CACHE_PATH
    channel_wise=True
    dataset=c4
    do_train=False
    embedding_8bit=False
    epoch=10
    flexround=True
    init_amode='mse'
    init_wmode='mse'
    input_prob=0.5
    iters_w=5000
    lr=4e-5
    model_name_or_path=meta-llama/Llama-2-7b-hf
    n_bits_a=8
    n_bits_w=8
    num_samples=512
    order='together'
    output_dir=$YOUR_OUTPUT_PATH
    prob=0.5
    quantization_dataset=train
    symmetric=False
    transformer_block_size=1
    w_lr=1e-3
    warmup=0.2
    weight=0.01
    weight_decay=0.05
