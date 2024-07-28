import os

from transformers import AutoTokenizer # huggingface tokenizers
import torch
import numpy as np


class Dataloader:
    def __init__(self, data_dir: str, block_size: int, batch_size: int, device: str, **kwargs) -> None:        
        self.data_dir = data_dir
        self.B = batch_size
        self.T = block_size
        
        torch.manual_seed(32)
        
        self.encoding = AutoTokenizer.from_pretrained(os.path.join(data_dir, "dardja_tokenizer"))
        self.vocab_size = self.encoding.vocab_size
        self.encode = lambda s: self.encoding.encode(s)
        self.decode = lambda s: self.encoding.decode(s)
        
        self.curr_pos = {'train': 0, 'val': 0}
        self.device = device
    
    #TODO add padding when there aren't enough tokens to cover the full block size?
    def get_batch(self, split: str):
        data = np.memmap(os.path.join(self.data_dir, f'{split}.bin'), np.uint16, mode='r')
        if self.curr_pos[split] > len(data) - self.B * self.T:
            self.curr_pos[split] = 0
        #TODO permute things here, we shouldn't get the same batch compositions every time
        pos = self.curr_pos[split]
        x = torch.reshape(torch.from_numpy(data[pos: pos + (self.B * self.T)].astype(np.int64)), [self.B, self.T])
        y = torch.reshape(torch.from_numpy(data[pos + 1: pos + (self.B * self.T) + 1].astype(np.int64)), [self.B, self.T]) # shifted to the right by 1 position
        
        x, y = x.to(self.device), y.to(self.device)
        
        self.curr_pos[split] += self.B * self.T + 1
        
        return x, y
    