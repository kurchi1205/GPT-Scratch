import torch
import torch.nn.functional as F
import time
from einops import rearrange

BLOCK_SIZE = 500
def normal_attention_causal(Q, K, V):
    Q_LEN = Q.shape[1]
    K_LEN = K.shape[1]
    mask = torch.tril(torch.ones(Q_LEN, Q_LEN)).to(Q.device)
    weights = Q @ K.transpose(-2,-1) * K.shape[-1]**-0.5
    weights = weights.masked_fill(mask==0, float('-inf'))
    weights = F.softmax(weights, dim=-1)
    # weights = self.dropout(weights)
    weights = weights @ V
    return weights

def flash_attention_causal(Q, K, V, mask):
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
            mij, _ = torch.max(sij, -1, keepdim=True)
            pij = torch.exp(sij - mij)
            lij = torch.sum(pij, -1, keepdim=True)
            mi_new = torch.maximum(mi, mij)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij - mi_new) * lij
            expr = li * torch.exp(mi - mi_new) * oi / li_new
            O_BLOCKS[i] = (li * torch.exp(mi - mi_new) * oi / li_new) +  (torch.exp(mij - mi_new) * pij / li_new) @ vj
            l_blocks[i] = li_new
            m_blocks[i] = mi_new
    O = torch.cat(O_BLOCKS, dim=1)
    return O

if __name__ == "__main__":
    Q = torch.randn(32, 500, 64, requires_grad=True).to(device='cuda')
    K = torch.randn(32, 500, 64, requires_grad=True).to(device='cuda')
    V = torch.randn(32, 500, 64, requires_grad=True).to(device='cuda')
    mask = torch.tril(torch.ones(500, 500)).to(device='cuda')
    
    start1 = time.time_ns()
    normal_attention_causal(Q, K, V)
    end1 = time.time_ns()
    t1 = (end1 - start1) / 1000000
    print(f'{t1}ms')

    start1 = time.time_ns()
    flash_attention_causal(Q, K, V, mask)
    end1 = time.time_ns()
    t2 = (end1 - start1) / 1000000
    print(f'{t2}ms')