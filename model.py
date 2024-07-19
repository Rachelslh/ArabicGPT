import torch
import torch.nn as nn
from torch.nn.functional import softmax

from dataclasses import dataclass

torch.manual_seed(32)

#TODO Convert this into from-scratch code, using just plain pytorch, without lightning framework
 
@dataclass
class GPTConfig:
    embedding_dim: 768
    n_layers: 12
    heads: 12
    head_size: 64
    block_size: 64
    
    
class GPT(nn.Module):
    def __init__(self, num_tokens: int, embedding_dim: int, block_size: int, n_layers: int, heads: int, head_size: int, **kwargs) -> None:
        super().__init__()
        self.block_size = block_size
        
        # Model layers
        self.transformer = nn.ModuleDict(dict(
            embedding_table = nn.Embedding(num_tokens, embedding_dim),
            positional_encodings_table = nn.Embedding(block_size, embedding_dim),
            blocks = nn.ModuleList(TransformerBlock(heads, head_size, block_size, embedding_dim) for _ in range(n_layers))
        ))
        
        self.lm_head = nn.Linear(embedding_dim, num_tokens)
        
        self.loss_func = nn.CrossEntropyLoss()
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.loss = {'train': [], 'val': []}
        
    def forward(self, tokens, targets=None):
        B, T = tokens.shape
        assert T <= self.block_size, "The sequence is longer than the permitted block size"
        # tokens is of shape [B, T], targets shape [B, T]
        emb_input = self.transformer.embedding_table(tokens) # [B, T, emb_d]
        # Add positional encoding
        pos_emb = self.transformer.positional_encodings_table((torch.arange(T, device=self.device))) # [T, emb_d]
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
        
        weights = q @ k.transpose(-2, -1) * self.head_size**-0.5 # [B, T, head_size] * [B, T, head_Size] -> [B, T, head_size] * [B, head_Size, T] = [B, T, T]
        
        weights = weights.masked_fill((self.tril[:T, :T] == 0), float('-inf'))
        weights = softmax(weights, dim=1)
        input_w_past_attention = weights @ v # [B, T, T] * [B, T, head_size] = [B, T, head_size]
        
        return input_w_past_attention
    
    
class MultiHeadAttention(nn.Module): 
    def __init__(self, heads: int, head_size: int, block_size: int, emb_d: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = heads
        self.head_size = head_size
        self.block_Size = block_size
        
        self.self_attention_blocks = nn.ModuleList([ScaledSelfAttentionHead(head_size, block_size, emb_d) for _ in range(heads)])
        self.ln_layer = nn.Linear(heads*head_size, emb_d)
        
    def forward(self, inputs):
        outputs = [attention(inputs) for attention in self.self_attention_blocks]
        out = torch.cat(outputs, -1)
        return self.ln_layer(out)
        
        
class FeedForwardNetwork(nn.Module): 
    def __init__(self, features_in: int, features_out: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.linear_block = nn.Sequential(
            nn.Linear(features_in, features_out),
            nn.ReLU(),
            nn.Linear(features_out, features_in)
        )
        
    def forward(self, inputs):
        return self.linear_block(inputs)
        