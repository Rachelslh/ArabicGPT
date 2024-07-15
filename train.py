
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import TokenDataset


config = OmegaConf.load("config/config.yaml")

block_size = config.model.block_size
train_dataset = TokenDataset(**config.data.val, block_size=block_size)
val_dataset = TokenDataset(**config.data.val, block_size=block_size)
