import tiktoken
import torch

#TODO Implement batch retrieval, encoder and decoder using lambda functions

class TokenDataset:
    def __init__(self, path: str, block_size: int) -> None:
        with open(path, 'r') as f:
            self.raw_data = f.read()
            
        torch.manual_seed(32)
        
        '''
        Keeping this character-based code, in case i decide to roll back to chars instead of words/subwords.
        
        #logger.debug(f'length of dataset in characters: {len(self.raw_data)}')
        #self.chars = sorted(list(set(self.data)))
        #self.vocab_size = len(self.chars)
        #logger.debug(f"vocabulary: [{''.join(self.chars)}], size: {len(self.chars)}")
        
        #ctoi = {c: i for i, c in enumerate(self.chars)}
        #self.encode = lambda s: [ctoi[c] for c in s]
    
        #itoc = {i: c for i, c in enumerate(self.chars)}
        #self.decode = lambda ints: ''.join([itoc[i] for i in ints])
        '''
        
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoding.n_vocab
        self.data = torch.tensor(self.encoding.encode(self.raw_data))
        
        self.block_size = block_size
        self.max_num_samples = len(self.data) // self.block_size
        
    
    def get_batch(self, ):
        index = index * self.block_size + 1
        if index > len(self.data)-self.block_size:
            index = torch.randint(high=len(self.data)-self.block_size, size=(1,)).item
        x = self.data[index: index + self.block_size]
        y = self.data[index+1: index + self.block_size + 1]
        
        return x, y
    