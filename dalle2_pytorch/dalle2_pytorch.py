import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from einops_exts import rearrange_many

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

def FeedForward(dim, mult = 4):
    inner_dim = int(mult * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None):
        n, device = x.shape[1], x.device

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = rearrange_many(qkv, 'b n (h d) -> b h n d')

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j')
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        causal_mask = torch.ones((n, n), dtype = torch.bool, device = device).triu(1)
        sim = sim.masked_fill(causal_mask, max_neg_value)

        sim = sim - sim.amax(dim = -1, keepdim = True)
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_out = False
    ):
        super().__init__()
        # todo - bring in rotary embeddings or alibi

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options

    def forward(
        self,
        x,
        mask = None    # we will need a mask here, due to variable length of the text encodings - also offer dalle1 strategy with padding token embeddings
    ):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

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
        decoder,
        tokenizer = None
    ):
        super().__init__()
        assert isinstance(clip), CLIP
        assert isinstance(prior), DiffusionPrior
        assert isinstance(decoder), Decoder
        self.tokenizer = tokenizer

    @torch.no_grad()
    def forward(
        self,
        *,
        text
    ):
        return image
