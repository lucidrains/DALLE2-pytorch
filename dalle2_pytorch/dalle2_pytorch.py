import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# use x-clip

from x_clip import CLIP

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# for controlling freezing of CLIP

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

# diffusion prior

class DiffusionPrior(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# decoder

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# main class

class DALLE2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
