import torch.nn as nn

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



class Block(nn.Module):
    def __init__(self, embedding_dim, head_size, num_heads):
        super(Block, self).__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        self.ff = FeedFoward(embedding_dim) 


class GPTModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, block_size):
        super(GPTModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = Blocks()