import os
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer # huggingface tokenizers

from model import GPTConfig, GPT


config = OmegaConf.load("config/config.yaml")

model_config = GPTConfig(**config.model)
model = GPT(model_config)
model.load_from_checkpoint(config.inference.path)

encoding = AutoTokenizer.from_pretrained("data/dardja_tokenizer")
encode = lambda s: encoding.encode(s)
decode = lambda s: encoding.decode(s)
    
enc = [encode('وحد نهار')]
seq = model.generate(torch.as_tensor(enc, device='mps'), 100, 2, 'mps')

print(decode(seq[0].tolist()))