import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import time
from torch.cuda.amp import custom_fwd, custom_bwd

BLOCK_SIZE = 500

class FeedFoward(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 4*embedding_dim)
        self.fc2 = nn.Linear(4*embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class flash_atten(torch.autograd.Function):
    @staticmethod
    # @custom_fwd(cast_inputs=torch.float16)
    def forward(Q, K, V, mask): 
        BC = BLOCK_SIZE
        BR = min(BLOCK_SIZE, Q.shape[-1])
        O = torch.zeros_like(Q, requires_grad=True).to(Q.device)
        l = torch.zeros(Q.shape[:-1])[...,None]
        m = torch.ones(Q.shape[:-1])[...,None] * -1e4
        l = l.to(Q.device)
        m = m.to(Q.device)
        Q_BLOCKS = torch.split(Q, BR, dim=1)
        K_BLOCKS = torch.split(K, BC, dim=1)
        V_BLOCKS = torch.split(V, BC, dim=1)
        O_BLOCKS = list(torch.split(O, BR, dim=1))
        l_blocks = list(torch.split(l, BR, dim=1))
        m_blocks = list(torch.split(m, BR, dim=1))

        mask_BLOCKS = list(torch.split(mask, BC, dim=1))
        for j in range(len(K_BLOCKS)):
            kj = K_BLOCKS[j]
            vj = V_BLOCKS[j]
            maskj = mask_BLOCKS[j]
            maskij = list(torch.split(maskj, BR, dim=0))
            for i in range(len(Q_BLOCKS)):
                qi = Q_BLOCKS[i]
                oi = O_BLOCKS[i]
                li = l_blocks[i]
                mi = m_blocks[i]
                mask_fill = maskij[i] 
                qi_scaled = qi / Q.shape[-1]**-0.5
                sij = qi_scaled @ kj.transpose(-2, -1)
                maskij_temp = torch.unsqueeze(mask_fill, dim=0)
                sij = sij.masked_fill(maskij_temp==0, float('-inf'))
                mij, _ = torch.max(sij, -1, keepdims=True)
                pij = torch.exp(sij - mij)
                lij = torch.sum(pij, -1, keepdims=True)
                mi_new = torch.maximum(mi, mij)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(mij - mi_new) * lij
                P_ij_Vj = pij @ vj
                O_BLOCKS[i] = (li/li_new) * torch.exp(mi - mi_new) * oi + (torch.exp(mij - mi_new) / li_new) * P_ij_Vj
                l_blocks[i] = li_new
                m_blocks[i] = mi_new
                del qi, oi, li, mi, mask_fill, maskij_temp, qi_scaled, sij, mij, pij
        O = torch.cat(O_BLOCKS, dim=1)
        # ctx.save_for_backward(Q, K, V, mask, O, l, m) 
        return O
    
    @staticmethod
    # @custom_bwd
    def backward(ctx, dO):
        Q, K, V, mask, O, l, m = ctx.saved_tensors
        BC = BLOCK_SIZE
        BR = min(BLOCK_SIZE, Q.shape[-1])
        TR = math.ceil(Q.shape[1]/BR)
        TC = math.ceil(Q.shape[1]/BC)
        Q_BLOCKS = torch.split(Q, TR, dim=1)
        K_BLOCKS = torch.split(K, TC, dim=1)
        V_BLOCKS = torch.split(V, TC, dim=1)
        O_BLOCKS = torch.split(O, TR, dim=1)
        dO_BLOCKS = torch.split(dO, TR, dim=1)
        l_blocks = list(torch.split(l, TR, dim=1))
        m_blocks = list(torch.split(m, TR, dim=1))
        dQ = torch.zeros_like(Q, requires_grad=True).to(device='cuda')
        dK = torch.zeros_like(K, requires_grad=True).to(device='cuda')
        dV = torch.zeros_like(V, requires_grad=True).to(device='cuda')
        
        dQ_BLOCKS = list(torch.split(dQ, TR, dim=1))
        dK_BLOCKS = list(torch.split(dK, TC, dim=1))
        dV_BLOCKS = list(torch.split(dV, TC, dim=1))
        mask_BLOCKS = list(torch.split(mask, TC, dim=1))
        for j in range(TC):
            kj = K_BLOCKS[j]
            vj = V_BLOCKS[j]
            dKj = torch.zeros_like(dK_BLOCKS[j], requires_grad=True).to(device='cuda')
            dVj = torch.zeros_like(dV_BLOCKS[j], requires_grad=True).to(device='cuda')
            maskj = mask_BLOCKS[j]
            maskij = list(torch.split(maskj, TR, dim=0))
            for i in range(TR):
                qi = Q_BLOCKS[i]
                oi = O_BLOCKS[i]
                dOi = dO_BLOCKS[i]
                dQi = dQ_BLOCKS[i]
                li = l_blocks[i]
                mi = m_blocks[i]
                mask_fill = maskij[i]
                scale = 1 / Q.shape[-1]**-0.5
                qi_scaled = qi / Q.shape[-1]**-0.5
                sij = qi_scaled @ kj.transpose(-2, -1)
                maskij_temp = torch.unsqueeze(mask_fill, dim=0)
                sij = sij.masked_fill(maskij_temp==0, float('-inf'))
                pij = torch.exp(sij - mi)/li
                dVj = dVj + (pij.transpose(-2, -1) @ dOi)
                dpij = dOi @ vj.transpose(-2, -1)
                Di = torch.sum(dOi * oi, dim=-1, keepdims=True)
                dsij = pij * (dpij - Di)
                dQ_BLOCKS[i] = dQi + (dsij @ kj)
                dKj = dKj + scale * (dsij.transpose(-2, -1) @ qi)
            dK_BLOCKS[j] = dKj
            dV_BLOCKS[j] = dVj
        dQ = torch.cat(dQ_BLOCKS, dim=1)
        dK = torch.cat(dK_BLOCKS, dim=1)
        dV = torch.cat(dV_BLOCKS, dim=1)
        return dQ, dK, dV, None
                

class Head(nn.Module):
    def __init__(self, embedding_dim, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size)
        self.query = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, use_flash_attention=False):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        mask = torch.tril(torch.ones(x.size(1), x.size(1))).to(query.device)

        if not use_flash_attention:
            weights = query @ key.transpose(-2,-1) * key.shape[-1]**-0.5
            weights = weights.masked_fill(mask==0, float('-1e4'))
            weights = F.softmax(weights, dim=-1)
            weights = self.dropout(weights)
            weights = weights @ value
            return weights
        # fl_atten = flash_atten.apply
        weights = flash_atten.forward(query, key, value, mask)
        return weights


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_head, d_model) -> None:
        super().__init__()
        head_size = d_model // num_head
        self.heads = nn.ModuleList([Head(embedding_dim, head_size) for _ in range(num_head)])
        self.proj = nn.Linear(d_model, embedding_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        heads = torch.cat([head(x, False) for head in self.heads], dim=-1)
        projection = self.proj(heads)
        projection = self.dropout(projection)
        return projection
    
class Block(nn.Module):
    def __init__(self, embedding_dim, d_model, num_heads):
        super().__init__()
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
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(block_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, d_model, num_heads) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        x = self.embedding(x)
        pos = torch.arange(x.size(1), device=x.device)
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
