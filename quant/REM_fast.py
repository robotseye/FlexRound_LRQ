import torch
import torch.nn as nn
from tqdm import tqdm
from quant.loss import LossFunction
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from quant.UniformQuantizationLinear import swapUniformQ, UniformAffineQuantizer
from quant.cached_loader_fast import cachedDataset_fast, DataSaverHook, DataCacheWrapper
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
class REM_fast():
    def __init__(self, 
                    model, fp_model, data_loader,
                    tokenizer = None,
                    transformer_block_size:int = 1, 
                    n_bits:int = 4, 
                    group_alpha:int = -1,
                    batch_size:int = 16, 
                    iters:int = 10000, 
                    weight:float = 0.01,
                    opt_mode: str = 'mse', 
                    warmup: float = 0.0, 
                    p: float = 2.0, 
                    input_prob: float = 0.5, 
                    num_samples: int = 1024, 
                    w_lr: float = 4e-5, 
                    a_lr: float = 4e-5, 
                    fp16: bool = False,
                    b_range: tuple = (20, 2),
                    channel_wise: bool = True,
                    flexround:bool = False,
                    uniformQuantization:bool = True,
                    symmetric:bool = False,
                    collate_fn = None,
                    split_qkv=False,
                    tp_size=1,
                ):
        self.n_bits = n_bits
        self.group_alpha = group_alpha
        self.modelName =  (model.config._name_or_path).lower()
        self.fp_model = fp_model
        self.model = model
        self.data_loader = data_loader
        self.tokenizer = tokenizer
        self.batch_size = max(batch_size, 2)
        self.iters = iters
        self.weight = weight
        self.opt_mode = opt_mode
        self.input_prob = input_prob
        self.num_samples = num_samples
        self.fp16 = fp16
        self.w_lr = w_lr
        self.a_lr = a_lr
        self.b_range = b_range
        self.warmup = warmup
        self.p = p
        self.channel_wise = channel_wise
        self.flexround = flexround
        self.uniformQuantization = uniformQuantization
        self.symmetric = symmetric
        self.data_saver_fp = DataSaverHook(store_input=True, store_output=True, stop_forward=True)
        self.data_saver_q = DataSaverHook(store_input=False, store_output=True, stop_forward=True)
        self.collate_fn = collate_fn
        self.split_qkv = split_qkv
        self.tp_size = tp_size

        self.attention_mask = None
        if 'llama' in self.modelName:
            for step, batch in enumerate(self.data_loader):
                self.position_ids = torch.arange(0, batch['attention_mask'].size()[1], dtype=torch.long, device='cuda:0').unsqueeze(0).view(-1, batch['attention_mask'].size()[1])
                self.attention_mask = self.fp_model.model._prepare_decoder_attention_mask(batch['attention_mask'], (max(batch['attention_mask'].size()[0], 2), batch['attention_mask'].size()[1]), self.fp_model.model.embed_tokens(batch['input_ids']), 0).to('cuda:0')
                break

    def quantization(self):
        if 'llama' in self.modelName:
            blockUnits = self.model.model.layers
            blockUnits_fp = self.fp_model.model.layers
        else:
            raise NotImplementedError

        fp_block = blockUnits_fp[0]           
        wrapped_block = DataCacheWrapper(fp_block)
        self.fp_model.model.layers[0] = wrapped_block
        cached_data = cachedDataset_fast(self.fp_model, self.data_loader, self.input_prob, self.num_samples, self.data_saver_fp, self.data_saver_q, self.tokenizer)
        self.fp_model.model.layers[0] = wrapped_block.module

        for idx in tqdm(range(self.model.config.num_hidden_layers)):
            print('='*60)
            print(f'    Layer {idx} Optimization Start')
            print('='*60)
            print(f'\n1. Full-precision model Activation Caching')

            if idx > 0 :
                fp_block = blockUnits_fp[idx]           
                wrapped_block = DataCacheWrapper(fp_block)
                cached_data.fp_data_caching(wrapped_block, self.input_prob, self.num_samples, self.data_saver_fp, self.data_saver_q, self.tokenizer)

            cached_dataloader = DataLoader(
                cached_data,
                shuffle=True,
                batch_size = max(self.data_loader.batch_size, 2)
                )

            print(f'\n2. Make independent Block')
            if 'llama' in self.modelName:
                independentBlock = LlamaDecoderLayer(self.model.config)
                independentBlock.hidden_size = blockUnits[idx].hidden_size
                independentBlock.self_attn = blockUnits[idx].self_attn.to('cuda:0')
                independentBlock.mlp = blockUnits[idx].mlp.to('cuda:0')
                independentBlock.input_layernorm = blockUnits[idx].input_layernorm.to('cuda:0')
                independentBlock.post_attention_layernorm = blockUnits[idx].post_attention_layernorm.to('cuda:0')
            else:
                raise NotImplementedError
            
            print(f'\n3. Quantize independent Block')
            independentBlock = self.quantizeBlock(independentBlock)

            print(f'\n4. Block Minimization Reconstruction Error')
            independentBlock = self.blockReconstruction(independentBlock.to('cuda').float(), cached_dataloader)
            torch.cuda.empty_cache()

            print(f'\n5. Dequantize Block')
            self.dequantizeBlock(independentBlock.half(), idx)
            
            if 'llama' in self.modelName:
                blockUnits[idx].self_attn.k_proj = independentBlock.self_attn.k_proj
                blockUnits[idx].self_attn.v_proj = independentBlock.self_attn.v_proj
                blockUnits[idx].self_attn.q_proj = independentBlock.self_attn.q_proj
                blockUnits[idx].self_attn.o_proj = independentBlock.self_attn.o_proj
                blockUnits[idx].mlp.gate_proj = independentBlock.mlp.gate_proj
                blockUnits[idx].mlp.down_proj = independentBlock.mlp.down_proj
                blockUnits[idx].mlp.up_proj = independentBlock.mlp.up_proj
            else:
                raise NotImplementedError
            del independentBlock
            torch.cuda.empty_cache()

            print(f'\n6. Quantized model Activation caching')
            if idx < self.model.config.num_hidden_layers-1:
                q_block = blockUnits[idx]           
                wrapped_block = DataCacheWrapper(q_block)
                cached_data.q_data_caching(wrapped_block, self.input_prob, self.num_samples, self.data_saver_fp, self.data_saver_q, self.tokenizer)
            else:
                for i in range(self.num_samples):
                    cached_data.cached_fp_input[i] = cached_data.cached_fp_input[i].cpu()
                    cached_data.cached_fp_output[i] = cached_data.cached_fp_output[i].cpu()
                    cached_data.cached_q_input[i] = cached_data.cached_q_input[i].cpu()
                torch.cuda.empty_cache()
            
            print('-'*60)

    def dequantizeBlock(self, block, idx):
        if self.uniformQuantization:
            if 'llama' in self.modelName:
                block.self_attn.k_proj.quantized_weight = nn.Parameter(block.self_attn.k_proj.weight_quantizer(block.self_attn.k_proj.org_weight).clone().detach(), requires_grad=False)
                delattr(block.self_attn.k_proj, 'org_weight')
                for i in range(5):
                    delattr(block.self_attn.k_proj.weight_quantizer, 'delta' + str(i+1))
                delattr(block.self_attn.k_proj, 'weight_quantizer')
                block.self_attn.v_proj.quantized_weight = nn.Parameter(block.self_attn.v_proj.weight_quantizer(block.self_attn.v_proj.org_weight).clone().detach(), requires_grad=False)
                delattr(block.self_attn.v_proj, 'org_weight')
                for i in range(5):
                    delattr(block.self_attn.v_proj.weight_quantizer, 'delta' + str(i+1))
                delattr(block.self_attn.v_proj, 'weight_quantizer')
                block.self_attn.q_proj.quantized_weight = nn.Parameter(block.self_attn.q_proj.weight_quantizer(block.self_attn.q_proj.org_weight).clone().detach(), requires_grad=False)
                delattr(block.self_attn.q_proj, 'org_weight')
                for i in range(5):
                    delattr(block.self_attn.q_proj.weight_quantizer, 'delta' + str(i+1))
                delattr(block.self_attn.q_proj, 'weight_quantizer')
                block.self_attn.o_proj.quantized_weight = nn.Parameter(block.self_attn.o_proj.weight_quantizer(block.self_attn.o_proj.org_weight).clone().detach(), requires_grad=False)
                delattr(block.self_attn.o_proj, 'org_weight')
                for i in range(5):
                    delattr(block.self_attn.o_proj.weight_quantizer, 'delta' + str(i+1))
                delattr(block.self_attn.o_proj, 'weight_quantizer')
                block.mlp.gate_proj.quantized_weight = nn.Parameter(block.mlp.gate_proj.weight_quantizer(block.mlp.gate_proj.org_weight).clone().detach(), requires_grad=False)
                delattr(block.mlp.gate_proj, 'org_weight')
                for i in range(5):
                    delattr(block.mlp.gate_proj.weight_quantizer, 'delta' + str(i+1))
                delattr(block.mlp.gate_proj, 'weight_quantizer')
                block.mlp.down_proj.quantized_weight = nn.Parameter(block.mlp.down_proj.weight_quantizer(block.mlp.down_proj.org_weight).clone().detach(), requires_grad=False)
                delattr(block.mlp.down_proj, 'org_weight')
                for i in range(5):
                    delattr(block.mlp.down_proj.weight_quantizer, 'delta' + str(i+1))
                delattr(block.mlp.down_proj, 'weight_quantizer')
                block.mlp.up_proj.quantized_weight = nn.Parameter(block.mlp.up_proj.weight_quantizer(block.mlp.up_proj.org_weight).clone().detach(), requires_grad=False)
                delattr(block.mlp.up_proj, 'org_weight')
                for i in range(5):
                    delattr(block.mlp.up_proj.weight_quantizer, 'delta' + str(i+1))
                delattr(block.mlp.up_proj, 'weight_quantizer')
            else:
                raise NotImplementedError

    def quantizeBlock(self, block):
        if self.uniformQuantization:
            swapFunc = swapUniformQ
        else:
            swapFunc = swapBCQ

        wq_params = {'n_bits':  self.n_bits, 'channel_wise':  self.channel_wise, 
            'flexround' : self.flexround, 'symmetric': self.symmetric, 'num_alpha': self.group_alpha} 

        if 'llama' in self.modelName:
            block.self_attn.k_proj = swapFunc(block.self_attn.k_proj, **wq_params)
            block.self_attn.v_proj = swapFunc(block.self_attn.v_proj, **wq_params)
            block.self_attn.q_proj = swapFunc(block.self_attn.q_proj, **wq_params)
            block.self_attn.o_proj = swapFunc(block.self_attn.o_proj, **wq_params)
            block.mlp.gate_proj = swapFunc(block.mlp.gate_proj, **wq_params)
            block.mlp.down_proj = swapFunc(block.mlp.down_proj, **wq_params)
            block.mlp.up_proj = swapFunc(block.mlp.up_proj, **wq_params)
        else:
            raise NotImplementedError

        return block

    def blockReconstruction(self, block_q:nn.Module, cached_dataloader):
        device = 'cuda:0'
        w_para = []
        w_opt = None
        scheduler = None
        
        print('    => uniformQuantization')
        if self.flexround:
            print('    => with FlexRound')
            for name, module in block_q.named_modules():
                if isinstance(module, UniformAffineQuantizer):
                    print(f'Weight: {name}')
                    w_para += [getattr(module, 'delta' + str(idx+1)) for idx in range(5) if getattr(module, 'delta' + str(idx+1)) is not None]
        else: 
            print('with AdaRound')
            if 'llama' in self.modelName:
                w_para += [block_q.self_attn.k_proj.weight_quantizer.alpha]
                w_para += [block_q.self_attn.v_proj.weight_quantizer.alpha]
                w_para += [block_q.self_attn.q_proj.weight_quantizer.alpha]
                w_para += [block_q.self_attn.o_proj.weight_quantizer.alpha]
                w_para += [block_q.mlp.gate_proj.weight_quantizer.alpha]
                w_para += [block_q.mlp.down_proj.weight_quantizer.alpha]
                w_para += [block_q.mlp.up_proj.weight_quantizer.alpha]
            else:
                raise NotImplementedError

        all_params = [ {'params': w_para, 'lr': self.w_lr}]
        optimizer = torch.optim.Adam(all_params)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.iters, eta_min=0.)

        print('w_lr: ', self.w_lr, ' / a_lr: ', self.a_lr)

        loss_mode = 'relaxation'
        rec_loss = self.opt_mode
        loss_func = LossFunction(block_q, round_loss=loss_mode, weight=self.weight, max_count=self.iters, rec_loss=rec_loss,
                                b_range=self.b_range, decay_start=0, warmup=self.warmup)

        epochs = int(self.iters/len(cached_dataloader))
        remainder = self.iters - len(cached_dataloader) * epochs
        
        scaler = torch.cuda.amp.GradScaler()
        for epoch in tqdm(range(epochs+1)):
            for step, batch in enumerate(cached_dataloader):
                if epoch == epochs and step == remainder:
                    break

                cur_inp, cur_sym = batch[0].squeeze(1).to(device), batch[-1].squeeze(1).to(device)
                cur_out = batch[1].to(device)

                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled = self.fp16):
                    if 'llama' in self.modelName:
                        out_quant = block_q(hidden_states=cur_inp.float(), attention_mask=self.attention_mask, position_ids=self.position_ids)
                    else:
                        out_quant = block_q(hidden_states=cur_inp.float())
                    cur_out = cur_out.squeeze(1)
                    err = loss_func(out_quant, cur_out)

                scaler.scale(err).backward(retain_graph=True)

                scaler.step(optimizer)
                if scheduler:
                    scheduler.step()

                scaler.update()
        
        del optimizer
        torch.cuda.empty_cache()

        if not self.flexround:
            if 'llama' in self.modelName:
                block_q.self_attn.k_proj.weight_quantizer.soft_targets = False
                block_q.self_attn.v_proj.weight_quantizer.soft_targets = False
                block_q.self_attn.q_proj.weight_quantizer.soft_targets = False
                block_q.self_attn.o_proj.weight_quantizer.soft_targets = False
                block_q.mlp.gate_proj.weight_quantizer.soft_targets = False
                block_q.mlp.down_proj.weight_quantizer.soft_targets = False
                block_q.mlp.up_proj.weight_quantizer.soft_targets = False
            else:
                raise NotImplementedError

        return block_q