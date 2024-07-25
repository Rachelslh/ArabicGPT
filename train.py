import os
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
val_steps = config.trainer.val_steps
ckpt_dir = config.trainer.checkpoint_dir

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

model_config = GPTConfig(**config.model)
model = GPT(model_config)

#TODO add logging into wandb, add checkpointing

optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.init_lr)
optimizer.zero_grad()

model.to(device)

loss_per_step = {'train': [], 'val': []}

best_val_loss = float('inf')
for step in range(iterations):
    t0 = time.time() 
    
    x, y = dataloader.get_batch(split='train')
    logits, loss = model(x, y, device=device)
    loss_per_step['train'].append(loss.item())
    
    loss.backward()
    optimizer.step()
    
    # Evaluate on validation data every n iterations
    if step > 0 and step % 20 == 0:
        losses = torch.zeros(val_steps)
        for val_step in range(val_steps):
            x, y = dataloader.get_batch(split='val')
            _, loss = model(x, y, device=device)
            losses[val_step] = loss.item()
        loss_per_step['val'].append(losses.mean().item())
        
        # Checkpointing best model
        if step > 0 and loss < best_val_loss:
            best_val_loss = loss_per_step['val'][-1]
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_config.__dict__,
                'iter_num': step,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {ckpt_dir}")
            torch.save(checkpoint, os.path.join(ckpt_dir, 'ckpt.pt'))
    
    # Flush the gradients before next step
    optimizer.zero_grad(set_to_none=True)
    
    # Measure throughput
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = dataloader.B * dataloader.T
    tokens_per_sec = tokens_processed / dt # throughput
    print(f"step {step:4d} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    
# Simple plot training vs validation loss curves
epochs_array = np.arange(1, iterations + 1)
# Plot and label the training and validation loss values
plt.plot(epochs_array, loss_per_step['train'], label='Training Loss')
plt.plot(epochs_array, loss_per_step['val'], label='Validation Loss')
 
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
 
plt.legend(loc='best')

plt.savefig('assets/loss.jpg')

plt.show()