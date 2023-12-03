import torch
import torch.nn.functional as F
import time
from einops import rearrange
import numpy as np

BLOCK_SIZE = 64
def normal_attention_causal(Q, K, V, mask):
    # Q_LEN = Q.shape[1]
    # K_LEN = K.shape[1]
    # mask = torch.tril(torch.ones(Q_LEN, Q_LEN)).to(Q.device)
    weights = Q @ K.transpose(-2,-1) * K.shape[-1]**-0.5
    # weights = weights.masked_fill(mask==0, float('-inf'))
    weights = F.softmax(weights, dim=-1)
    # weights = self.dropout(weights)
    weights = weights @ V
    return weights

def flash_attention_causal(Q, K, V, mask):
    BC = BLOCK_SIZE
    BR = min(BLOCK_SIZE, Q.shape[-1])
    # print(BC, BR)
    O = torch.zeros_like(Q, requires_grad=True).to(Q.device)
    l = torch.zeros(Q.shape[:-1])[...,None]
    m = torch.ones(Q.shape[:-1])[...,None] * -1e10
    l = l.to(Q.device)
    m = m.to(Q.device)
    Q_BLOCKS = torch.split(Q, BR, dim=1)
    K_BLOCKS = torch.split(K, BC, dim=1)
    V_BLOCKS = torch.split(V, BC, dim=1)
    O_BLOCKS = list(torch.split(O, BR, dim=1))
    l_blocks = list(torch.split(l, BR, dim=1))
    m_blocks = list(torch.split(m, BR, dim=1))
    Tr = len(Q_BLOCKS)
    Tc = len(K_BLOCKS)
    mask_BLOCKS = list(torch.split(mask, BC, dim=1))
    for j in range(Tc):
        kj = K_BLOCKS[j]
        vj = V_BLOCKS[j]
        # maskj = mask_BLOCKS[j]
        # maskij = list(torch.split(maskj, BR, dim=0))
        for i in range(Tr):
            qi = Q_BLOCKS[i]
            oi = O_BLOCKS[i]
            li = l_blocks[i]
            mi = m_blocks[i]
            # mask_fill = maskij[i] 
            # qi_scaled = qi / 
            scale = 1 / np.sqrt(Q.shape[-1])
            Qi_scaled  = qi * scale
            sij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, kj)
            # maskij_temp = torch.unsqueeze(mask_fill, dim=0)
            # sij = sij.masked_fill(maskij_temp==0, float('-inf'))
            # del maskij_temp
            # del mask_fill
            mij, _ = torch.max(sij, -1, keepdims=True)
            pij = torch.exp(sij - mij)
            lij = torch.sum(pij, -1, keepdims=True) + 1e-10
            P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', pij, vj)
            
            mi_new = torch.maximum(mi, mij)
            li_new = torch.exp(mi - mi_new) * li + torch.exp(mij - mi_new) * lij
            # expr = li * torch.exp(mi - mi_new) * oi / li_new
            O_BLOCKS[i] = (li/li_new) * torch.exp(mi - mi_new) * oi + (torch.exp(mij - mi_new) / li_new) * P_ij_Vj
            l_blocks[i] = li_new
            m_blocks[i] = mi_new
    O = torch.cat(O_BLOCKS, dim=1)
    l = torch.cat(l_blocks, dim=1)
    m = torch.cat(m_blocks, dim=1)
    return O

if __name__ == "__main__":
    Q = torch.randn(1, 500, 64, requires_grad=True).to(device='cuda')
    K = torch.randn(1, 500, 64, requires_grad=True).to(device='cuda')
    V = torch.randn(1, 500, 64, requires_grad=True).to(device='cuda')
    mask = torch.tril(torch.ones(500, 500)).to(device='cuda')
    mask.requires_grad = True
    for i in range(10):
        start1 = time.time_ns()
        out1 = normal_attention_causal(Q, K, V, mask)
        end1 = time.time_ns()
        t1 = (end1 - start1) / 1000000
        print(f'{t1}ms')

        start2 = time.time_ns()
        out2 = flash_attention_causal(Q, K, V, mask)
        end2 = time.time_ns()
        t2 = (end2 - start2) / 1000000
        print(f'{t2}ms')
        
        print(torch.allclose(out1, out2, atol=1e-5))