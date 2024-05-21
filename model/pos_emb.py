import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, max_length, dim):
        super(PositionalEncoding, self).__init__()
        position_encoding = torch.tensor([[pos / math.pow(10000, 2.0 * (j // 2) / dim) for j in range(dim)] for pos in range(dim+1)]).float()
        position_encoding[:, 0::2] = torch.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = torch.cos(position_encoding[:, 1::2])
        self.position_encoding = nn.Embedding(max_length + 1, dim)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False).float()
    
    def forward(self, x):
        N, L, D = x.shape
        x_pos_emb_size = torch.arange(L).expand((N, L)).to(self.position_encoding.weight.device)
        x = x + self.position_encoding(x_pos_emb_size)
        return x