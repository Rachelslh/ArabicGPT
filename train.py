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

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    device = torch.device("mps")
    
dataloader = Dataloader(**config.dataloader, block_size=block_size, device=device)

model_config = GPTConfig(vocab_size=dataloader.vocab_size, **config.model)
model = GPT(model_config)

#TODO add logging into wandb, add checkpointing

optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.init_lr)
optimizer.zero_grad()

model.to(device)

loss_per_step = {'train': [], 'val': []}


for step in range(iterations):
    t0 = time.time() 
    
    x, y = dataloader.get_batch(split='train')
    logits, loss = model(x, y, device=device)
    loss_per_step['train'].append(loss.item())
    
    loss.backward()
    optimizer.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = dataloader.B * dataloader.T
    tokens_per_sec = tokens_processed / dt # throughput

    print(f"step {step:4d} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
epochs_array = np.arange(1, iterations + 1)
# Plot and label the training and validation loss values
plt.plot(epochs_array, loss_per_step['train'], label='Training Loss')
#plt.plot(epochs_array, loss_per_step['val'][1:], label='Validation Loss') # Avoiding the sanity check val step here
 
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
 
plt.legend(loc='best')

plt.savefig('assets/loss.jpg')

plt.show()