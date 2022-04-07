import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

class DALLE2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
