from typing import Union
import math

import torch
import torch.nn as nn
from torch.nn.functional import softmax, scaled_dot_product_attention

from dataclasses import dataclass

torch.manual_seed(32)

#TODO Parallelize attention, keep both versions though
RESIDUAL_PROJECTION_LAYER_NAME = 'proj_layer'


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    embedding_dim: int = 768
    n_layers: int = 12
    heads: int = 12
    head_size: int = 64
    block_size: int = 64
    
    
class GPT(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        
        # Model layers
        self.transformer = nn.ModuleDict(dict(
            embedding_table = nn.Embedding(config.vocab_size, config.embedding_dim),
            positional_encodings_table = nn.Embedding(config.block_size, config.embedding_dim),
            blocks = nn.ModuleList(TransformerBlock(config.heads, config.head_size, config.block_size, config.embedding_dim) for _ in range(config.n_layers))
        ))
        
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size)
        
        # weight sharing scheme as in GPT2, i.e. same weights are in embedding layer and the LM head (last linear layer)
        self.transformer.embedding_table.weight = self.lm_head.weight
        
        self.loss_func = nn.CrossEntropyLoss()
        self.loss = {'train': [], 'val': []}
        
        self.apply(self._init_weights)
                
    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # torch normally init this with a uniform
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in module.named_parameters():
            if pn.endswith(F'{RESIDUAL_PROJECTION_LAYER_NAME}.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layers))
        
    def forward(self, tokens, targets=None, device=None):
        B, T = tokens.shape
        assert T <= self.block_size, "The sequence is longer than the permitted block size"
        # tokens is of shape [B, T], targets shape [B, T]
        emb_input = self.transformer.embedding_table(tokens) # [B, T, emb_d]
        # Add positional encoding
        pos_emb = self.transformer.positional_encodings_table((torch.arange(T, device=device))) # [T, emb_d]
        emb_input += pos_emb
        for layer in self.transformer.blocks:
            out = layer(emb_input)
        
        if targets is not None:
            logits = self.lm_head(out) # [B, T, C=num_tokens]
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T,)
            # Apply cross-entropy loss
            loss = self.loss_func(logits, targets)
        else:
            # Return only last time position
            logits = self.lm_head(out[:, [-1], :]) # [B, -1, C=num_tokens]
            loss = None
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, sequence, max_new_tokens, top_k, device):
        self.to(device)
        self.eval()
        for _ in range(max_new_tokens):
            pre_seq = sequence if sequence.size(1) <= self.block_size else sequence[:, -self.block_size:]
            logits, _ = self(pre_seq)
            logits = logits[:, -1, :]
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float('-inf')
            probs = softmax(logits, -1)
            next_token = torch.multinomial(probs, num_samples=1)
            sequence = torch.cat((sequence, next_token), dim=1)
            
        return sequence


class TransformerBlock(nn.Module):
    def __init__(self, heads: int, head_size: int, block_size: int, emb_d: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attention_block = MultiHeadAttention(heads, head_size, block_size, emb_d)
        self.feed_forward = FeedForwardNetwork(emb_d, emb_d*4)
        self.layer_norm1 = nn.LayerNorm(emb_d, bias=False)
        self.layer_norm2 = nn.LayerNorm(emb_d, bias=False)
        
    def forward(self, inputs):
        att_output = inputs + self.attention_block(self.layer_norm1(inputs))
        out = att_output + self.feed_forward(self.layer_norm2(att_output))
        
        return out
    
    
#TODO Parallelize this
class ScaledSelfAttentionHead(nn.Module):
    def __init__(self, head_size: int, block_size: int, emb_d: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.head_size = head_size
        self.block_size = block_size
        
        self.register_buffer('tril', torch.tril(torch.ones((block_size, block_size))))
        self.key = nn.Linear(emb_d, self.head_size)
        self.query = nn.Linear(emb_d, self.head_size)
        self.value = nn.Linear(emb_d, self.head_size)
    
    def forward(self, inputs):
        B, T, C = inputs.shape
        # B, T, emb_d = inputs.shape
        k = self.key(inputs)   # [B, T, head_size]
        q = self.query(inputs) # [B, T, head_size]
        v = self.value(inputs) # [B, T, head_size]
        
        #weights = q @ k.transpose(-2, -1) * self.head_size**-0.5 # [B, T, head_size] * [B, T, head_Size] -> [B, T, head_size] * [B, head_Size, T] = [B, T, T]
        #weights = weights.masked_fill((self.tril[:T, :T] == 0), float('-inf'))
        #weights = softmax(weights, dim=1)
        #input_w_past_attention = weights @ v # [B, T, T] * [B, T, head_size] = [B, T, head_size]
        
        # Using Flash attention here for some optimization instead of the 4 lines commented-out above
        input_w_past_attention = scaled_dot_product_attention(q, k, v, is_causal=True)
        return input_w_past_attention
    
    
class MultiHeadAttention(nn.Module): 
    def __init__(self, heads: int, head_size: int, block_size: int, emb_d: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.head_size = head_size
        self.block_Size = block_size
        
        self.self_attention_blocks = nn.ModuleList([ScaledSelfAttentionHead(head_size, block_size, emb_d) for _ in range(heads)])
        setattr(self, RESIDUAL_PROJECTION_LAYER_NAME, nn.Linear(heads*head_size, emb_d)) # output projection
        
    def forward(self, inputs):
        outputs = [attention(inputs) for attention in self.self_attention_blocks]
        out = torch.cat(outputs, -1)
        return self.proj_layer(out)
        
        
class FeedForwardNetwork(nn.Module): 
    def __init__(self, features_in: int, features_out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.linear_block = nn.Sequential(
            nn.Linear(features_in, features_out),
            nn.ReLU(),
        )
        setattr(self, RESIDUAL_PROJECTION_LAYER_NAME, nn.Linear(features_out, features_in))
        
    def forward(self, inputs):
        out = self.linear_block(inputs)
        return self.proj_layer(out)
        