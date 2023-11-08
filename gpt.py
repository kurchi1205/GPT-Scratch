import torch.nn as nn
import torch
import torch.nn.functional as F

class FeedFoward(nn.Module):
    def __init__(self, embedding_dim):
        super(FeedFoward, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 4*embedding_dim)
        self.fc2 = nn.Linear(4*embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class Head(nn.Module):
    def __init__(self, embedding_dim, head_size):
        super(Head, self).__init__()
        self.key = nn.Linear(embedding_dim, head_size)
        self.query = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        weights = query @ key.transpose(-2,-1) * key.shape[-1]**-0.5
        mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1)
        weights = weights.masked_fill(mask==0, float('-inf'))
        weights = F.softmax(weights, dim=1)
        weights = self.dropout(weights)
        value = self.value(x)
        weights = weights @ value.T
        return weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_head, d_model) -> None:
        super().__init__()
        head_size = d_model // num_head
        self.heads = nn.ModuleList([Head(embedding_dim, head_size) for _ in range(num_head)])
        self.proj = nn.Linear(d_model, embedding_dim)

    def forward(self, x):
        heads = torch.cat([head(x) for head in self.heads], dim=-1)
        projection = self.proj(heads)
        return projection
    
class Block(nn.Module):
    def __init__(self, embedding_dim, d_model, num_heads):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedFoward(embedding_dim) 
        self.sa = MultiHeadAttention(embedding_dim, num_heads, d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size, d_model, num_heads, num_layers):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, d_model, num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x, targets=None):
        x = self.embedding(x)
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
