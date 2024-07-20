import torch
from torch import nn
import numpy as np

# What does Nin do? 
# Network in network - allow model to learn a linear combination of features at each spatial location
# In this model it is used to adjust number of channels
class Nin(nn.Module):

    def __init__(self, in_dim, out_dim, scale=1e-10):
        super(Nin, self).__init__()

        n = (in_dim + out_dim) / 2
        limit = np.sqrt(3 * scale / n)
        self.W = nn.Parameter(torch.zeros((in_dim, out_dim), dtype=torch.float32).uniform_(-limit, limit))
        self.b = nn.Parameter(torch.zeros((1, out_dim, 1, 1), dtype=torch.float32))

    def forward(self, x):
        return torch.einsum('bchw, co->bowh', x, self.W) + self.b