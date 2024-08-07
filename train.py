import os
import time

from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import numpy as np
import torch

from dataset import Dataloader
from model import GPTConfig, GPT


def evaluate():
    model.eval()
    losses = torch.zeros(val_steps)
    for val_step in range(val_steps):
        x, y = dataloader.get_batch(split='val')
        _, loss = model(x, y, device=device)
        losses[val_step] = loss.item()
    loss = losses.mean().item()
    model.train()
    
    return loss


def get_grad_norm():
    # L2 norm
    norm = 0
    for p in model.parameters():
        try:
            norm += torch.linalg.norm(p.grad.detach().data).item()**2
        except:
            pass

    #parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    #if len(parameters) == 0:
    #    total_norm = 0.0
    #else:
    #    device = parameters[0].grad.device
    #    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2.0).item()
        
    return norm**0.5
 
#TODO add wandb

config = OmegaConf.load("config/config.yaml")

block_size = config.model.block_size
batch_size = config.dataloader.batch_size
iterations = config.trainer.steps * config.trainer.epochs
gradient_accumulation_steps = config.trainer.gradient_accumulation_steps
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
x, y = dataloader.get_batch(split='train')

best_val_loss = float('inf')
norm = []
for step in range(iterations):    
    # Evaluate on validation data every n iterations
    if step > 0 and step % 20 == 0:
        val_loss = evaluate()
        loss_per_step['val'].append(val_loss)
        
        # Checkpointing best model
        if step > 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
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
            
    t0 = time.time() 
    micro_losses = torch.zeros(gradient_accumulation_steps)
    for micro_step in range(gradient_accumulation_steps):
        logits, loss = model(x, y, device=device)
        loss /= gradient_accumulation_steps
        micro_losses[micro_step] = loss
        loss.backward()  
        norm.append(get_grad_norm())
        
        x, y = dataloader.get_batch(split='train')
        
    optimizer.step()
    loss_per_step['train'].append(micro_losses.mean().item())
        
    # Flush the gradients before next step
    optimizer.zero_grad(set_to_none=True)
    
    # Measure throughput
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    # estimate throughput
    tokens_per_sec = model.estimate_mfu(dataloader.B * gradient_accumulation_steps, dt)
    print(f"step {step:4d} | loss: {loss.item():.6f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
    
    
# Simple plot training vs validation loss curves
epochs_array = np.arange(1, iterations + 1)
# Plot and label the training and validation loss values
plt.plot(epochs_array, loss_per_step['train'], label='Training Loss')

val_epochs_array = np.arange(1, len(loss_per_step['val']) + 1) * 20
y_inter = interp.interp1d(val_epochs_array, loss_per_step['val'])
y_ = y_inter(np.linspace(20, val_epochs_array[-1], iterations))
plt.plot(epochs_array, y_, label='Validation Loss')
 
plt.title('Training and Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
 
plt.legend(loc='best')

plt.savefig('assets/loss.jpg')

plt.show()

plt.plot(norm)
plt.xlabel('Steps')
plt.ylabel('Gradient Norm')
 
plt.legend(loc='best')

plt.savefig('assets/gradient_norm.jpg')

plt.show()
