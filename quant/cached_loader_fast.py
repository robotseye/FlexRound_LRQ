from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn


class cachedDataset_fast(Dataset):
    def __init__(self, fp_model, cali_data, input_prob=None, num_samples:int=1024, data_saver_fp=None, data_saver_q=None, tokenizer=None):
        super().__init__()
        self.device0 = 'cuda:0'
        self.device1 = 'cuda:0'

        self.batch_size = cali_data.batch_size

        self.cached_fp_input  = []
        self.cached_fp_output = []
        self.cached_fp_other  = []

        print(f'Initial Activation Caching')
        fp_model.model.embed_tokens = fp_model.model.embed_tokens.to(self.device1)
        fp_model.model.layers[0] = fp_model.model.layers[0].to(self.device1)
        get_inp_out_fp = GetLayerInpOut_fp(fp_block=fp_model, device0=self.device0, device1=self.device1, input_prob=input_prob, data_saver_fp=data_saver_fp)
        with torch.no_grad():
            for step, batch in enumerate(tqdm(cali_data)):
                if step >= num_samples / cali_data.batch_size:
                    break
                
                if input_prob:
                    cur_out, cur_input_fp, cur_other = get_inp_out_fp(batch, is_first_block=True)
                    self.cached_fp_input.append(cur_input_fp.to(self.device0))
                    self.cached_fp_output.append(cur_out.to(self.device0))
                    self.cached_fp_other.append(cur_other)
                else:
                    cur_out = get_inp_out_fp(batch, is_first_block=True).to(self.device0)
                    self.cached_fp_output.append(cur_out.to(self.device0))
        
        self.cached_q_input = self.cached_fp_input

        fp_model.model.embed_tokens = fp_model.model.embed_tokens.to('cpu')
        fp_model.model.layers[0] = fp_model.model.layers[0].to('cpu')
        del get_inp_out_fp
        torch.cuda.empty_cache()

    def fp_data_caching(self, fp_block, input_prob=None, num_samples:int=1024, data_saver_fp=None, data_saver_q=None, tokenizer=None):
        self.cached_fp_input  = self.cached_fp_output
        self.cached_fp_output = []

        fp_block = fp_block.to(self.device1)
        get_inp_out_fp = GetLayerInpOut_fp(fp_block=fp_block, device0=self.device0, device1=self.device1, input_prob=input_prob, data_saver_fp=data_saver_fp)

        cali_data = self.cached_fp_input
        other_data = self.cached_fp_other
        self.cached_fp_other = []
        with torch.no_grad():
            for step, batch in enumerate( tqdm( zip(cali_data, other_data) ) ):
                if step >= num_samples / self.batch_size:
                    break

                if input_prob:
                    cur_out, _, cur_other = get_inp_out_fp(batch)
                    self.cached_fp_output.append(cur_out.to(self.device0))
                    self.cached_fp_other.append(cur_other)
                else:
                    cur_out = get_inp_out_fp(batch)
                    self.cached_fp_output.append(cur_out.to(self.device0))
        
        if not input_prob:
            del self.cached_fp_input
 
        get_inp_out_fp.fp_block.cpu()
        del get_inp_out_fp
        torch.cuda.empty_cache()
    
    def q_data_caching(self, q_block, input_prob=None, num_samples:int=1024, data_saver_fp=None, data_saver_q=None, tokenizer=None):
        self.cached_q_output = []

        q_block = q_block.to(self.device1)
        get_inp_out_q = GetLayerInpOut_q(q_block, device0=self.device0, device1=self.device1, data_saver_q=data_saver_q)
        
        cali_data = self.cached_q_input
        other_data = self.cached_fp_other
        with torch.no_grad():
            for step, batch in enumerate( tqdm( zip(cali_data, other_data) ) ):
                if step >= num_samples / self.batch_size:
                    break

                cur_output_q = get_inp_out_q(batch)

                self.cached_q_output.append(cur_output_q.to(self.device0))
        self.cached_q_input = self.cached_q_output

        get_inp_out_q.q_block.cpu()
        del self.cached_q_output
        del get_inp_out_q
        torch.cuda.empty_cache()

    def __getitem__(self, index):
        return self.cached_q_input[index], self.cached_fp_output[index], self.cached_fp_input[index] 

    def __len__(self):
        return len(self.cached_q_input)


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


class DataCacheWrapper(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.inp_data   = None
        self.out_data   = None
        self.other_data = None

    def forward(self, inp, **other):
        self.inp_data = inp
        
        output = self.module(inp, **other)
        
        self.out_data = output
        self.other_data = other
        raise StopForwardException


class GetLayerInpOut_fp:
    def __init__(self, fp_block, device0, device1, input_prob: bool = False, data_saver_fp= None):
        self.fp_block = fp_block
        self.device0 = device0
        self.device1 = device1
        self.data_saver_fp = data_saver_fp
        self.input_prob = input_prob

    def __call__(self, model_input, is_first_block=False):
        if is_first_block:
            model_input = {k: v.to(device = self.device1, non_blocking=True) if hasattr(v, 'to') else v for k, v in model_input.items()}
        else:
            model_input = list(model_input)
            model_input[0] = model_input[0].to(self.device1)

        with torch.no_grad():
            try:
                if is_first_block:
                    _ = self.fp_block(**model_input)
                else:
                    _ = self.fp_block(model_input[0], **model_input[1])
            except StopForwardException:
                pass
        
        if is_first_block:
            out_data   = self.fp_block.model.layers[0].out_data
            inp_data   = self.fp_block.model.layers[0].inp_data
            other_data = self.fp_block.model.layers[0].other_data
        else:
            out_data   = self.fp_block.out_data
            inp_data   = self.fp_block.inp_data
            other_data = self.fp_block.other_data
        
        if 'tuple' in str(type(out_data)):
            out_data = out_data[0]

        if self.input_prob:
            return (
                out_data.detach().to(self.device0),
                inp_data.detach().to(self.device0),
                other_data,
            )

        return out_data.detach().to(self.device0)


class GetLayerInpOut_q:
    def __init__(self, q_block, device0, device1, data_saver_q= None):
        self.q_block = q_block
        self.device0 = device0
        self.device1 = device1
        self.data_saver_q = data_saver_q

    def __call__(self, model_input):
        model_input = list(model_input)
        model_input[0] = model_input[0].to(self.device1)

        with torch.no_grad():
            try:
                _ = self.q_block(model_input[0], **model_input[1])
            except StopForwardException:
                pass

        if  'tuple' in str(type(self.q_block.out_data)):
            self.q_block.out_data = self.q_block.out_data[0]

        return self.q_block.out_data.detach().to(self.device0)
