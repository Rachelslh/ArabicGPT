import tiktoken
import torch

#TODO parallelize this into multiple num workers, add batch permutation, add shards

class Dataloader:
    def __init__(self, path: str, block_size: int, batch_size: int) -> None:
        self.B = batch_size
        self.T = block_size
        
        with open(path, 'r') as f:
            self.raw_data = f.read()
            
        torch.manual_seed(32)
        
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoding.n_vocab
        self.encode = lambda s: self.encoding.encode(s)
        self.decode = lambda s: self.encoding.decode(s)
        self.data = torch.tensor(self.encode(self.raw_data))
        
        self.num_samples = len(self.data) // self.T
        self.curr_pos = 0
    
    def get_batch(self, ):
        if self.curr_pos > len(self.data) - self.B * self.T:
            self.curr_pos = 0
            
        x = torch.split(self.data[self.curr_pos: self.curr_pos + (self.B * self.T)], [self.B, self.T])
        y = torch.split(self.data[self.curr_pos + 1: self.curr_pos + (self.B * self.T) + 1], [self.B, self.T]) # shifted to the right by 1 position
        
        self.curr_pos = self.B * self.T + 1
        
        return x, y
    