
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import TokenDataset


config = OmegaConf.load("config/config.yaml")

block_size = config.model.block_size
train_dataset = TokenDataset(**config.data.val, block_size=block_size)
val_dataset = TokenDataset(**config.data.val, block_size=block_size)



def training_step(self, batch, batch_idx):
    x, y = batch
    logits, loss = self.forward(x, y)
    self.training_step_outputs.append(loss.item())
    self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    return loss

def on_train_epoch_end(self) -> None:
    self.loss['train'].append(np.mean(self.training_step_outputs))
    self.training_step_outputs.clear()
    return super().on_train_epoch_end()

def validation_step(self, batch, batch_idx):
    x, y = batch
    _, loss = self.forward(x, y)
    self.validation_step_outputs.append(loss.item())
    self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
    return loss
    
def on_validation_epoch_end(self) -> None:
    self.loss['val'].append(np.mean(self.validation_step_outputs))
    self.validation_step_outputs.clear()
    return super().on_validation_epoch_end()

def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer