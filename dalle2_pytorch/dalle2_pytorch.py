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

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# for controlling freezing of CLIP

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

# diffusion prior

class DiffusionPrior(nn.Module):
    def __init__(
        self,
        *,
        clip
    ):
        super().__init__()
        assert isinstance(clip, CLIP)
        freeze_model_and_make_eval_(clip)

    def forward(
        self,
        *,
        text,
        image = None
    ):
        return image_embed

# decoder

class Decoder(nn.Module):
    def __init__(
        self,
        *,
        clip,
        prior
    ):
        super().__init__()
        assert isinstance(clip, CLIP)
        assert isinstance(prior, DiffusionPrior)
        freeze_model_and_make_eval_(clip)

    def forward(
        self,
        *,
        image,
        image_embed,
        text_embed = None  # in paper, text embedding was optional for conditioning decoder
    ):
        return image

# main class

class DALLE2(nn.Module):
    def __init__(
        self,
        *,
        clip,
        prior,
        decoder
    ):
        super().__init__()
        assert isinstance(clip), CLIP
        assert isinstance(prior), DiffusionPrior
        assert isinstance(decoder), Decoder

    @torch.no_grad()
    def forward(
        self,
        *,
        text
    ):
        return image
