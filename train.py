import time

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import Dataloader
from model import GPTConfig, GPT


config = OmegaConf.load("config/config.yaml")

block_size = config.model.block_size
batch_size = config.dataloader.batch_size
iterations = config.trainer.steps * config.trainer.epochs
device ='mps'

dataloader = Dataloader(**config.dataloader, block_size=block_size, device=device)

model_config = GPTConfig(vocab_size=dataloader.vocab_size, **config.model)
model = GPT(model_config)

#TODO add logging into wandb, add checkpointing

optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.init_lr)
model.to(device)
optimizer.zero_grad()

for step in range(iterations):
    t0 = time.time()
    
    x, y = dataloader.get_batch(split='train')
    logits, loss = model(x, y, device=device)
    loss.backward()
    optimizer.step()
    
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = dataloader.B * dataloader.T
    tokens_per_sec = tokens_processed / dt

    print(f"step {step:4d} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")