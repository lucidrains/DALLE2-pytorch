import math
from tqdm import tqdm
from inspect import isfunction
from functools import partial, wraps
from contextlib import contextmanager
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom

from kornia.filters import gaussian_blur2d
import kornia.augmentation as K

from dalle2_pytorch.tokenizer import tokenizer
from dalle2_pytorch.vqgan_vae import NullVQGanVAE, VQGanVAE

from resize_right import resize

# rotary embeddings

from rotary_embedding_torch import RotaryEmbedding

# use x-clip

from x_clip import CLIP
from coca_pytorch import CoCa

# constants

NAT = 1. / math.log(2.)

# helper functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def maybe(fn):
    @wraps(fn)
    def inner(x):
        if not exists(x):
            return x
        return fn(x)
    return inner

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, length = 1):
    if isinstance(val, list):
        val = tuple(val)

    return val if isinstance(val, tuple) else ((val,) * length)

def module_device(module):
    return next(module.parameters()).device

@contextmanager
def null_context(*args, **kwargs):
    yield

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

def pad_tuple_to_length(t, length, fillvalue = None):
    remain_length = length - len(t)
    if remain_length <= 0:
        return t
    return (*t, *((fillvalue,) * remain_length))

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

# tensor helpers

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

def l2norm(t):
    return F.normalize(t, dim = -1)

def resize_image_to(image, target_image_size):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    scale_factors = target_image_size / orig_image_size
    return resize(image, scale_factors = scale_factors)

# image normalization functions
# ddpms expect images to be in the range of -1 to 1
# but CLIP may otherwise

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

# clip related adapters

EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings', 'text_mask'])
EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])

class BaseClipAdapter(nn.Module):
    def __init__(self, clip, **kwargs):
        super().__init__()
        self.clip = clip
        self.overrides = kwargs

    @property
    def dim_latent(self):
        raise NotImplementedError

    @property
    def image_size(self):
        raise NotImplementedError

    @property
    def image_channels(self):
        raise NotImplementedError

    @property
    def max_text_len(self):
        raise NotImplementedError

    def embed_text(self, text):
        raise NotImplementedError

    def embed_image(self, image):
        raise NotImplementedError

class XClipAdapter(BaseClipAdapter):
    @property
    def dim_latent(self):
        return self.clip.dim_latent

    @property
    def image_size(self):
        return self.clip.image_size

    @property
    def image_channels(self):
        return self.clip.image_channels

    @property
    def max_text_len(self):
        return self.clip.text_seq_len

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]
        text_mask = text != 0
        encoder_output = self.clip.text_transformer(text)
        text_cls, text_encodings = encoder_output[:, 0], encoder_output[:, 1:]
        text_embed = self.clip.to_text_latent(text_cls)
        return EmbeddedText(l2norm(text_embed), text_encodings, text_mask)

    @torch.no_grad()
    def embed_image(self, image):
        image = resize_image_to(image, self.image_size)
        encoder_output = self.clip.visual_transformer(image)
        image_cls, image_encodings = encoder_output[:, 0], encoder_output[:, 1:]
        image_embed = self.clip.to_visual_latent(image_cls)
        return EmbeddedImage(l2norm(image_embed), image_encodings)

class CoCaAdapter(BaseClipAdapter):
    @property
    def dim_latent(self):
        return self.clip.dim

    @property
    def image_size(self):
        assert 'image_size' in self.overrides
        return self.overrides['image_size']

    @property
    def image_channels(self):
        assert 'image_channels' in self.overrides
        return self.overrides['image_channels']

    @property
    def max_text_len(self):
        assert 'max_text_len' in self.overrides
        return self.overrides['max_text_len']

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]
        text_mask = text != 0
        text_embed, text_encodings = self.clip.embed_text(text)
        return EmbeddedText(text_embed, text_encodings, text_mask)

    @torch.no_grad()
    def embed_image(self, image):
        image = resize_image_to(image, self.image_size)
        image_embed, image_encodings = self.clip.embed_image(image)
        return EmbeddedImage(image_embed, image_encodings)

class OpenAIClipAdapter(BaseClipAdapter):
    def __init__(
        self,
        name = 'ViT-B/32'
    ):
        import clip
        openai_clip, preprocess = clip.load(name)
        super().__init__(openai_clip)

        text_attention_final = self.find_layer('ln_final')
        self.handle = text_attention_final.register_forward_hook(self._hook)
        self.clip_normalize = preprocess.transforms[-1]
        self.cleared = False

    def find_layer(self,  layer):
        modules = dict([*self.clip.named_modules()])
        return modules.get(layer, None)

    def clear(self):
        if self.cleared:
            return

        self.handle()

    def _hook(self, _, inputs, outputs):
        self.text_encodings = outputs

    @property
    def dim_latent(self):
        return 512

    @property
    def image_size(self):
        return self.clip.visual.input_resolution

    @property
    def image_channels(self):
        return 3

    @property
    def max_text_len(self):
        return self.clip.context_length

    @torch.no_grad()
    def embed_text(self, text):
        text = text[..., :self.max_text_len]
        text_mask = text != 0
        assert not self.cleared

        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float(), text_mask)

    @torch.no_grad()
    def embed_image(self, image):
        assert not self.cleared
        image = resize_image_to(image, self.image_size)
        image = self.clip_normalize(image)
        image_embed = self.clip.encode_image(image)
        return EmbeddedImage(l2norm(image_embed.float()), None)

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# gaussian diffusion helper functions

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def meanflat(x):
    return x.mean(dim = tuple(range(1, len(x.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(((2.0 / math.pi) ** 0.5) * (x + 0.044715 * (x ** 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres = 0.999):
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus)
    log_one_minus_cdf_min = log(1. - cdf_min)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(x < -thres,
        log_cdf_plus,
        torch.where(x > thres,
            log_one_minus_cdf_min,
            log(cdf_delta)))

    return log_probs

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start**2, beta_end**2, timesteps, dtype = torch.float64) ** 2


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype = torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class BaseGaussianDiffusion(nn.Module):
    def __init__(self, *, beta_schedule, timesteps, loss_type):
        super().__init__()

        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "quadratic":
            betas = quadratic_beta_schedule(timesteps)
        elif beta_schedule == "jsd":
            betas = 1.0 / torch.linspace(timesteps, 1, timesteps)
        elif beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError()

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        self.loss_type = loss_type
        self.loss_fn = loss_fn

        # register buffer helper function to cast double back to float

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

# diffusion prior

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# mlp

class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        expansion_factor = 2.,
        depth = 2,
        norm = False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.SiLU(),
            norm_fn()
        )]

        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                norm_fn()
            ))

        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())

# relative positional bias for causal transformer

class RelPosBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# feedforward

class SwiGLU(nn.Module):
    """ used successfully in https://arxiv.org/abs/2204.0231 """
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.silu(gate)

def FeedForward(
    dim,
    mult = 4,
    dropout = 0.,
    post_activation_norm = False
):
    """ post-activation norm https://arxiv.org/abs/2110.09456 """

    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        rotary_emb = None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.rotary_emb = rotary_emb

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # rotary embeddings

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 d', b = b)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class CausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True
    ):
        super().__init__()
        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(
        self,
        x,
        mask = None    # we will need a mask here, due to variable length of the text encodings - also offer dalle1 strategy with padding token embeddings
    ):
        n, device = x.shape[1], x.device

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_bias = attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)

class DiffusionPriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps = None,
        num_time_embeds = 1,
        num_image_embeds = 1,
        num_text_embeds = 1,
        **kwargs
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.num_image_embeds = num_image_embeds
        self.num_text_embeds = num_text_embeds

        self.to_text_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_text_embeds) if num_text_embeds > 1 else nn.Identity(),
            Rearrange('b (n d) -> b n d', n = num_text_embeds)
        )

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        self.to_image_embeds = nn.Sequential(
            nn.Linear(dim, dim * num_image_embeds) if num_image_embeds > 1 else nn.Identity(),
            Rearrange('b (n d) -> b n d', n = num_image_embeds)
        )

        self.learned_query = nn.Parameter(torch.randn(dim))
        self.causal_transformer = CausalTransformer(dim = dim, **kwargs)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        diffusion_timesteps,
        *,
        text_embed,
        text_encodings = None,
        mask = None,
        cond_drop_prob = 0.
    ):
        batch, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

        num_time_embeds, num_image_embeds, num_text_embeds = self.num_time_embeds, self.num_image_embeds, self.num_text_embeds

        # in section 2.2, last paragraph
        # "... consisting of encoded text, CLIP text embedding, diffusion timestep embedding, noised CLIP image embedding, final embedding for prediction"

        text_embed = self.to_text_embeds(text_embed)
        image_embed = self.to_image_embeds(image_embed)

        # make text encodings optional
        # although the paper seems to suggest it is present <--

        if not exists(text_encodings):
            text_encodings = torch.empty((batch, 0, dim), device = device, dtype = dtype)

        if not exists(mask):
            mask = torch.ones((batch, text_encodings.shape[-2]), device = device, dtype = torch.bool)

        # classifier free guidance

        keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
        keep_mask = rearrange(keep_mask, 'b -> b 1')

        mask &= keep_mask

        # whether text embedding is masked or not depends on the classifier free guidance conditional masking

        keep_mask = repeat(keep_mask, 'b 1 -> b n', n = num_text_embeds)
        mask = torch.cat((mask, keep_mask), dim = 1)

        # whether text embedding is used for conditioning depends on whether text encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right

        if exists(mask):
            attend_padding = 1 + num_time_embeds + num_image_embeds # 1 for learned queries + number of image embeds + time embeds
            mask = F.pad(mask, (0, attend_padding), value = True) # extend mask for text embedding, noised image embedding, time step embedding, and learned query

        time_embed = self.to_time_embeds(diffusion_timesteps)

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

        tokens = torch.cat((
            text_encodings,
            text_embed,
            time_embed,
            image_embed,
            learned_queries
        ), dim = -2)

        # attend

        tokens = self.causal_transformer(tokens, mask = mask)

        # get learned query, which should predict the image embedding (per DDPM timestep)

        pred_image_embed = tokens[..., -1, :]

        return pred_image_embed

class DiffusionPrior(BaseGaussianDiffusion):
    def __init__(
        self,
        net,
        *,
        clip = None,
        image_embed_dim = None,
        image_size = None,
        image_channels = 3,
        timesteps = 1000,
        cond_drop_prob = 0.,
        loss_type = "l2",
        predict_x_start = True,
        beta_schedule = "cosine",
        condition_on_text_encodings = True, # the paper suggests this is needed, but you can turn it off for your CLIP preprocessed text embed -> image embed training
        sampling_clamp_l2norm = False,
        training_clamp_l2norm = False,
        init_image_embed_l2norm = False,
        image_embed_scale = None,           # this is for scaling the l2-normed image embedding, so it is more suitable for gaussian diffusion, as outlined by Katherine (@crowsonkb) https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        clip_adapter_overrides = dict()
    ):
        super().__init__(
            beta_schedule = beta_schedule,
            timesteps = timesteps,
            loss_type = loss_type
        )

        if exists(clip):
            assert image_channels == clip.image_channels, f'channels of image ({image_channels}) should be equal to the channels that CLIP accepts ({clip.image_channels})'

            if isinstance(clip, CLIP):
                clip = XClipAdapter(clip, **clip_adapter_overrides)
            elif isinstance(clip, CoCa):
                clip = CoCaAdapter(clip, **clip_adapter_overrides)

            assert isinstance(clip, BaseClipAdapter)
            freeze_model_and_make_eval_(clip)
            self.clip = clip
        else:
            assert exists(image_embed_dim), 'latent dimension must be given, if training prior network without CLIP given'
            self.clip = None

        self.net = net
        self.image_embed_dim = default(image_embed_dim, lambda: clip.dim_latent)
        self.channels = default(image_channels, lambda: clip.image_channels)

        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.
        self.condition_on_text_encodings = condition_on_text_encodings

        # in paper, they do not predict the noise, but predict x0 directly for image embedding, claiming empirically better results. I'll just offer both.
        self.predict_x_start = predict_x_start

        # @crowsonkb 's suggestion - https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        self.image_embed_scale = default(image_embed_scale, self.image_embed_dim ** 0.5)

        # whether to force an l2norm, similar to clipping denoised, when sampling
        self.sampling_clamp_l2norm = sampling_clamp_l2norm
        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

    def p_mean_variance(self, x, t, text_cond, clip_denoised = False, cond_scale = 1.):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = self.net.forward_with_cond_scale(x, t, cond_scale = cond_scale, **text_cond)

        if self.predict_x_start:
            x_recon = pred
            # not 100% sure of this above line - for any spectators, let me know in the github issues (or through a pull request) if you know how to correctly do this
            # i'll be rereading https://arxiv.org/abs/2111.14822, where i think a similar approach is taken
        else:
            x_recon = self.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised and not self.predict_x_start:
            x_recon.clamp_(-1., 1.)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_recon = l2norm(x_recon) * self.image_embed_scale

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, clip_denoised = True, repeat_noise = False, cond_scale = 1.):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, text_cond = text_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, text_cond, cond_scale = 1.):
        device = self.betas.device

        b = shape[0]
        image_embed = torch.randn(shape, device=device)

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            times = torch.full((b,), i, device = device, dtype = torch.long)
            image_embed = self.p_sample(image_embed, times, text_cond = text_cond, cond_scale = cond_scale)

        return image_embed

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.q_sample(x_start = image_embed, t = times, noise = noise)

        pred = self.net(
            image_embed_noisy,
            times,
            cond_drop_prob = self.cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = l2norm(pred) * self.image_embed_scale

        target = noise if not self.predict_x_start else image_embed

        loss = self.loss_fn(pred, target)
        return loss

    @torch.no_grad()
    @eval_decorator
    def sample_batch_size(self, batch_size, text_cond, cond_scale = 1.):
        device = self.betas.device
        shape = (batch_size, self.image_embed_dim)

        img = torch.randn(shape, device = device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device = device, dtype = torch.long), text_cond = text_cond, cond_scale = cond_scale)
        return img

    @torch.no_grad()
    @eval_decorator
    def sample(self, text, num_samples_per_batch = 2, cond_scale = 1.):
        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, 'b ... -> (b r) ...', r = num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        text_embed, text_encodings, text_mask = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings, 'mask': text_mask}

        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond = text_cond, cond_scale = cond_scale)

        # retrieve original unscaled image embed

        image_embeds /= self.image_embed_scale

        text_embeds = text_cond['text_embed']

        text_embeds = rearrange(text_embeds, '(b r) d -> b r d', r = num_samples_per_batch)
        image_embeds = rearrange(image_embeds, '(b r) d -> b r d', r = num_samples_per_batch)

        text_image_sims = einsum('b r d, b r d -> b r', l2norm(text_embeds), l2norm(image_embeds))
        top_sim_indices = text_image_sims.topk(k = 1).indices

        top_sim_indices = repeat(top_sim_indices, 'b 1 -> b 1 d', d = image_embed_dim)

        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, 'b 1 d -> b d')

    def forward(
        self,
        text = None,
        image = None,
        text_embed = None,      # allow for training on preprocessed CLIP text and image embeddings
        image_embed = None,
        text_encodings = None,  # as well as CLIP text encodings
        text_mask = None,       # text mask <- may eventually opt for the learned padding tokens technique from DALL-E1 to reduce complexity
        *args,
        **kwargs
    ):
        assert exists(text) ^ exists(text_embed), 'either text or text embedding must be supplied'
        assert exists(image) ^ exists(image_embed), 'either text or text embedding must be supplied'
        assert not (self.condition_on_text_encodings and (not exists(text_encodings) and not exists(text))), 'text encodings must be present if you specified you wish to condition on it on initialization'

        if exists(image):
            image_embed, _ = self.clip.embed_image(image)

        # calculate text conditionings, based on what is passed in

        if exists(text):
            text_embed, text_encodings, text_mask = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings, 'mask': text_mask}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = torch.randint(0, self.num_timesteps, (batch,), device = device, dtype = torch.long)

        # scale image embed (Katherine)

        image_embed *= self.image_embed_scale

        # calculate forward loss

        return self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)

# decoder

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1)

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
        self.project = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.project(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        groups = 8
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            self.cross_attn = EinopsToAndFrom(
                'b c h w',
                'b (h w) c',
                CrossAttention(
                    dim = dim_out,
                    context_dim = cond_dim
                )
            )

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond = None, time_emb = None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context = cond) + h

        h = self.block2(h)
        return h + self.res_conv(x)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        norm_context = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

    def forward(self, x, context, mask = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b h 1 d', h = self.heads,  b = b)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GridAttention(nn.Module):
    def __init__(self, *args, window_size = 8, **kwargs):
        super().__init__()
        self.window_size = window_size
        self.attn = Attention(*args, **kwargs)

    def forward(self, x):
        h, w = x.shape[-2:]
        wsz = self.window_size
        x = rearrange(x, 'b c (w1 h) (w2 w) -> (b h w) (w1 w2) c', w1 = wsz, w2 = wsz)
        out = self.attn(x)
        out = rearrange(out, '(b h w) (w1 w2) c -> b c (w1 h) (w2 w)', w1 = wsz, w2 = wsz, h = h // wsz, w = w // wsz)
        return out

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        dim_head = 32,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm = ChanLayerNorm(dim)

        self.nonlin = nn.GELU()
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1, bias = False),
            ChanLayerNorm(dim)
        )

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]

        fmap = self.norm(fmap)
        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = rearrange_many((q, k, v), 'b (h c) x y -> (b h) (x y) c', h = h)

        q = q.softmax(dim = -1)
        k = k.softmax(dim = -2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h = h, x = x, y = y)

        out = self.nonlin(out)
        return self.to_out(out)

class CrossEmbedLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        kernel_sizes,
        dim_out = None,
        stride = 2
    ):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(nn.Conv2d(dim_in, dim_scale, kernel, stride = stride, padding = (kernel - stride) // 2))

    def forward(self, x):
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        return torch.cat(fmaps, dim = 1)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_embed_dim = None,
        text_embed_dim = None,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        channels_out = None,
        attn_dim_head = 32,
        attn_heads = 16,
        lowres_cond = False, # for cascading diffusion - https://cascaded-diffusion.github.io/
        sparse_attn = False,
        attend_at_middle = True, # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        cond_on_text_encodings = False,
        max_text_len = 256,
        cond_on_image_embeds = False,
        init_dim = None,
        init_conv_kernel_size = 7,
        resnet_groups = 8,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        **kwargs
    ):
        super().__init__()
        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        del self._locals['self']
        del self._locals['__class__']

        # for eventual cascading diffusion

        self.lowres_cond = lowres_cond

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        init_channels = channels if not lowres_cond else channels * 2 # in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        init_dim = default(init_dim, dim // 3 * 2)

        self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time, image embeddings, and optional text encoding

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4

        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.GELU()
        )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        self.image_to_cond = nn.Sequential(
            nn.Linear(image_embed_dim, cond_dim * num_image_tokens),
            Rearrange('b (n d) -> b n d', n = num_image_tokens)
        ) if cond_on_image_embeds and image_embed_dim != cond_dim else nn.Identity()

        self.norm_cond = nn.LayerNorm(cond_dim)
        self.norm_mid_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text_encodings:
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text_encodings is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)

        # finer control over whether to condition on image embeddings and text encodings
        # so one can have the latter unets in the cascading DDPMs only focus on super-resoluting

        self.cond_on_text_encodings = cond_on_text_encodings
        self.cond_on_image_embeds = cond_on_image_embeds

        # for classifier free guidance

        self.null_image_embed = nn.Parameter(torch.randn(1, num_image_tokens, cond_dim))

        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))

        # attention related params

        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)

        # resnet block klass

        resnet_groups = cast_tuple(resnet_groups, len(in_out))

        assert len(resnet_groups) == len(in_out)

        # downsample klass

        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), groups) in enumerate(zip(in_out, resnet_groups)):
            is_first = ind == 0
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if not is_first else None

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_cond_dim = time_cond_dim, groups = groups),
                Residual(LinearAttention(dim_out, **attn_kwargs)) if sparse_attn else nn.Identity(),
                ResnetBlock(dim_out, dim_out, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups),
                downsample_klass(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c', Residual(Attention(mid_dim, **attn_kwargs))) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])

        for ind, ((dim_in, dim_out), groups) in enumerate(zip(reversed(in_out[1:]), reversed(resnet_groups))):
            is_last = ind >= (num_resolutions - 2)
            layer_cond_dim = cond_dim if not is_last else None

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups),
                Residual(LinearAttention(dim_in, **attn_kwargs)) if sparse_attn else nn.Identity(),
                ResnetBlock(dim_in, dim_in, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups),
                Upsample(dim_in)
            ]))

        self.final_conv = nn.Sequential(
            ResnetBlock(dim, dim, groups = resnet_groups[0]),
            nn.Conv2d(dim, self.channels_out, 1)
        )

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        channels,
        channels_out,
        cond_on_image_embeds,
        cond_on_text_encodings
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_image_embeds == self.cond_on_image_embeds and \
            cond_on_text_encodings == self.cond_on_text_encodings and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            channels = channels,
            channels_out = channels_out,
            cond_on_image_embeds = cond_on_image_embeds,
            cond_on_text_encodings = cond_on_text_encodings
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, text_cond_drop_prob = 1., image_cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        *,
        image_embed,
        lowres_cond_img = None,
        text_encodings = None,
        text_mask = None,
        image_cond_drop_prob = 0.,
        text_cond_drop_prob = 0.,
        blur_sigma = None,
        blur_kernel_size = None
    ):
        batch_size, device = x.shape[0], x.device

        # add low resolution conditioning, if present

        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # initial convolution

        x = self.init_conv(x)

        # time conditioning

        time_hiddens = self.to_time_hiddens(time)

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # conditional dropout

        image_keep_mask = prob_mask_like((batch_size,), 1 - image_cond_drop_prob, device = device)
        text_keep_mask = prob_mask_like((batch_size,), 1 - text_cond_drop_prob, device = device)

        image_keep_mask, text_keep_mask = rearrange_many((image_keep_mask, text_keep_mask), 'b -> b 1 1')

        # mask out image embedding depending on condition dropout
        # for classifier free guidance

        image_tokens = None

        if self.cond_on_image_embeds:
            image_tokens = self.image_to_cond(image_embed)
            null_image_embed = self.null_image_embed.to(image_tokens.dtype) # for some reason pytorch AMP not working

            image_tokens = torch.where(
                image_keep_mask,
                image_tokens,
                null_image_embed
            )

        # take care of text encodings (optional)

        text_tokens = None

        if exists(text_encodings) and self.cond_on_text_encodings:
            text_tokens = self.text_to_cond(text_encodings)
            text_tokens = text_tokens[:, :self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value = False)

                text_mask = rearrange(text_mask, 'b n -> b n 1')
                text_keep_mask = text_mask & text_keep_mask

            null_text_embed = self.null_text_embed.to(text_tokens.dtype) # for some reason pytorch AMP not working

            text_tokens = torch.where(
                text_keep_mask,
                text_tokens,
                null_text_embed
            )

        # main conditioning tokens (c)

        c = time_tokens

        if exists(image_tokens):
            c = torch.cat((c, image_tokens), dim = -2)

        # text and image conditioning tokens (mid_c)
        # to save on compute, only do cross attention based conditioning on the inner most layers of the Unet

        mid_c = c if not exists(text_tokens) else torch.cat((c, text_tokens), dim = -2)

        # normalize conditioning tokens

        c = self.norm_cond(c)
        mid_c = self.norm_mid_cond(mid_c)

        # go through the layers of the unet, down and up

        hiddens = []

        for block1, sparse_attn, block2, downsample in self.downs:
            x = block1(x, c, t)
            x = sparse_attn(x)
            x = block2(x, c, t)
            hiddens.append(x)
            x = downsample(x)

        x = self.mid_block1(x, mid_c, t)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, mid_c, t)

        for block1, sparse_attn, block2, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = block1(x, c, t)
            x = sparse_attn(x)
            x = block2(x, c, t)
            x = upsample(x)

        return self.final_conv(x)

class LowresConditioner(nn.Module):
    def __init__(
        self,
        downsample_first = True,
        blur_sigma = 0.1,
        blur_kernel_size = 3,
    ):
        super().__init__()
        self.downsample_first = downsample_first
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size

    def forward(
        self,
        cond_fmap,
        *,
        target_image_size,
        downsample_image_size = None,
        blur_sigma = None,
        blur_kernel_size = None
    ):
        if self.training and self.downsample_first and exists(downsample_image_size):
            cond_fmap = resize_image_to(cond_fmap, downsample_image_size)

        if self.training:
            # when training, blur the low resolution conditional image
            blur_sigma = default(blur_sigma, self.blur_sigma)
            blur_kernel_size = default(blur_kernel_size, self.blur_kernel_size)
            cond_fmap = gaussian_blur2d(cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2))

        cond_fmap = resize_image_to(cond_fmap, target_image_size)

        return cond_fmap

class Decoder(BaseGaussianDiffusion):
    def __init__(
        self,
        unet,
        *,
        clip = None,
        image_size = None,
        channels = 3,
        vae = tuple(),
        timesteps = 1000,
        image_cond_drop_prob = 0.1,
        text_cond_drop_prob = 0.5,
        loss_type = 'l2',
        beta_schedule = 'cosine',
        predict_x_start = False,
        predict_x_start_for_latent_diffusion = False,
        image_sizes = None,                         # for cascading ddpm, image size at each stage
        random_crop_sizes = None,                   # whether to random crop the image at that stage in the cascade (super resoluting convolutions at the end may be able to generalize on smaller crops)
        lowres_downsample_first = True,             # cascading ddpm - resizes to lower resolution, then to next conditional resolution + blur
        blur_sigma = 0.1,                           # cascading ddpm - blur sigma
        blur_kernel_size = 3,                       # cascading ddpm - blur kernel size
        condition_on_text_encodings = False,        # the paper suggested that this didn't do much in the decoder, but i'm allowing the option for experimentation
        clip_denoised = True,
        clip_x_start = True,
        clip_adapter_overrides = dict(),
        learned_variance = True,
        vb_loss_weight = 0.001,
        unconditional = False,
        auto_normalize_img = True,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        use_dynamic_thres = False,                  # from the Imagen paper
        dynamic_thres_percentile = 0.9
    ):
        super().__init__(
            beta_schedule = beta_schedule,
            timesteps = timesteps,
            loss_type = loss_type
        )

        self.unconditional = unconditional

        # text conditioning

        assert not (condition_on_text_encodings and unconditional), 'unconditional decoder image generation cannot be set to True if conditioning on text is present'
        self.condition_on_text_encodings = condition_on_text_encodings

        # clip

        self.clip = None
        if exists(clip):
            assert not unconditional, 'clip must not be given if doing unconditional image training'
            assert channels == clip.image_channels, f'channels of image ({channels}) should be equal to the channels that CLIP accepts ({clip.image_channels})'

            if isinstance(clip, CLIP):
                clip = XClipAdapter(clip, **clip_adapter_overrides)
            elif isinstance(clip, CoCa):
                clip = CoCaAdapter(clip, **clip_adapter_overrides)

            freeze_model_and_make_eval_(clip)
            assert isinstance(clip, BaseClipAdapter)

            self.clip = clip

        # determine image size, with image_size and image_sizes taking precedence

        if exists(image_size) or exists(image_sizes):
            assert exists(image_size) ^ exists(image_sizes), 'only one of image_size or image_sizes must be given'
            image_size = default(image_size, lambda: image_sizes[-1])
        elif exists(clip):
            image_size = clip.image_size
        else:
            raise Error('either image_size, image_sizes, or clip must be given to decoder')

        # channels

        self.channels = channels

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unet)
        vaes = pad_tuple_to_length(cast_tuple(vae), len(unets), fillvalue = NullVQGanVAE(channels = self.channels))

        # whether to use learned variance, defaults to True for the first unet in the cascade, as in paper

        learned_variance = pad_tuple_to_length(cast_tuple(learned_variance), len(unets), fillvalue = False)
        self.learned_variance = learned_variance
        self.vb_loss_weight = vb_loss_weight

        # construct unets and vaes

        self.unets = nn.ModuleList([])
        self.vaes = nn.ModuleList([])

        for ind, (one_unet, one_vae, one_unet_learned_var) in enumerate(zip(unets, vaes, learned_variance)):
            assert isinstance(one_unet, Unet)
            assert isinstance(one_vae, (VQGanVAE, NullVQGanVAE))

            is_first = ind == 0
            latent_dim = one_vae.encoded_dim if exists(one_vae) else None

            unet_channels = default(latent_dim, self.channels)
            unet_channels_out = unet_channels * (1 if not one_unet_learned_var else 2)

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                cond_on_image_embeds = is_first and not unconditional,
                cond_on_text_encodings = one_unet.cond_on_text_encodings and not unconditional,
                channels = unet_channels,
                channels_out = unet_channels_out
            )

            self.unets.append(one_unet)
            self.vaes.append(one_vae.copy_for_eval())

        # unet image sizes

        image_sizes = default(image_sizes, (image_size,))
        image_sizes = tuple(sorted(set(image_sizes)))

        assert len(self.unets) == len(image_sizes), f'you did not supply the correct number of u-nets ({len(self.unets)}) for resolutions {image_sizes}'
        self.image_sizes = image_sizes
        self.sample_channels = cast_tuple(self.channels, len(image_sizes))

        # random crop sizes (for super-resoluting unets at the end of cascade?)

        self.random_crop_sizes = cast_tuple(random_crop_sizes, len(image_sizes))

        # predict x0 config

        self.predict_x_start = cast_tuple(predict_x_start, len(unets)) if not predict_x_start_for_latent_diffusion else tuple(map(lambda t: isinstance(t, VQGanVAE), self.vaes))

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (len(self.unets) - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.to_lowres_cond = LowresConditioner(
            downsample_first = lowres_downsample_first,
            blur_sigma = blur_sigma,
            blur_kernel_size = blur_kernel_size,
        )

        # classifier free guidance

        self.image_cond_drop_prob = image_cond_drop_prob
        self.text_cond_drop_prob = text_cond_drop_prob
        self.can_classifier_guidance = image_cond_drop_prob > 0. or text_cond_drop_prob > 0.

        # whether to clip when sampling

        self.clip_denoised = clip_denoised
        self.clip_x_start = clip_x_start

        # dynamic thresholding settings, if clipping denoised during sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1
        return self.unets[index]

    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.get_unet(unet_number)

        self.cuda()

        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.cuda()

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    def p_mean_variance(self, unet, x, t, image_embed, text_encodings = None, text_mask = None, lowres_cond_img = None, clip_denoised = True, predict_x_start = False, learned_variance = False, cond_scale = 1., model_output = None):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'the decoder was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, t, image_embed = image_embed, text_encodings = text_encodings, text_mask = text_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img))

        if learned_variance:
            pred, var_interp_frac_unnormalized = pred.chunk(2, dim = 1)

        if predict_x_start:
            x_recon = pred
        else:
            x_recon = self.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised:
            # s is the threshold amount
            # static thresholding would just be s = 1
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if learned_variance:
            # if learned variance, posterio variance and posterior log variance are predicted by the network
            # by an interpolation of the max and min log beta values
            # eq 15 - https://arxiv.org/abs/2102.09672
            min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract(torch.log(self.betas), t, x.shape)
            var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

            posterior_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
            posterior_variance = posterior_log_variance.exp()

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, unet, x, t, image_embed, text_encodings = None, text_mask = None, cond_scale = 1., lowres_cond_img = None, predict_x_start = False, learned_variance = False, clip_denoised = True, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x, t = t, image_embed = image_embed, text_encodings = text_encodings, text_mask = text_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, clip_denoised = clip_denoised, predict_x_start = predict_x_start, learned_variance = learned_variance)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, unet, shape, image_embed, predict_x_start = False, learned_variance = False, clip_denoised = True, lowres_cond_img = None, text_encodings = None, text_mask = None, cond_scale = 1, is_latent_diffusion = False):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device = device)

        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample(
                unet,
                img,
                torch.full((b,), i, device = device, dtype = torch.long),
                image_embed = image_embed,
                text_encodings = text_encodings,
                text_mask = text_mask,
                cond_scale = cond_scale,
                lowres_cond_img = lowres_cond_img,
                predict_x_start = predict_x_start,
                learned_variance = learned_variance,
                clip_denoised = clip_denoised
            )

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    def p_losses(self, unet, x_start, times, *, image_embed, lowres_cond_img = None, text_encodings = None, text_mask = None, predict_x_start = False, noise = None, learned_variance = False, clip_denoised = False, is_latent_diffusion = False):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        if not is_latent_diffusion:
            x_start = self.normalize_img(x_start)
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t

        x_noisy = self.q_sample(x_start = x_start, t = times, noise = noise)

        model_output = unet(
            x_noisy,
            times,
            image_embed = image_embed,
            text_encodings = text_encodings,
            text_mask = text_mask,
            lowres_cond_img = lowres_cond_img,
            image_cond_drop_prob = self.image_cond_drop_prob,
            text_cond_drop_prob = self.text_cond_drop_prob,
        )

        if learned_variance:
            pred, _ = model_output.chunk(2, dim = 1)
        else:
            pred = model_output

        target = noise if not predict_x_start else x_start

        loss = self.loss_fn(pred, target)

        if not learned_variance:
            # return simple loss if not using learned variance
            return loss

        # most of the code below is transcribed from
        # https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
        # the Improved DDPM paper then further modified it so that the mean is detached (shown a couple lines before), and weighted to be smaller than the l1 or l2 "simple" loss
        # it is questionable whether this is really needed, looking at some of the figures in the paper, but may as well stay faithful to their implementation

        # if learning the variance, also include the extra weight kl loss

        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start = x_start, x_t = x_noisy, t = times)
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x_noisy, t = times, image_embed = image_embed, clip_denoised = clip_denoised, learned_variance = True, model_output = model_output)

        # kl loss with detached model predicted mean, for stability reasons as in paper

        detached_model_mean = model_mean.detach()

        kl = normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = meanflat(kl) * NAT

        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means = detached_model_mean, log_scales = 0.5 * model_log_variance)
        decoder_nll = meanflat(decoder_nll) * NAT

        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        vb_losses = torch.where(times == 0, decoder_nll, kl)

        # weight the vb loss smaller, for stability, as in the paper (recommended 0.001)

        vb_loss = vb_losses.mean() * self.vb_loss_weight

        return loss + vb_loss

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        image_embed = None,
        text = None,
        text_mask = None,
        text_encodings = None,
        batch_size = 1,
        cond_scale = 1.,
        stop_at_unet_number = None
    ):
        assert self.unconditional or exists(image_embed), 'image embed must be present on sampling from decoder unless if trained unconditionally'

        if not self.unconditional:
            batch_size = image_embed.shape[0]

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip)
            _, text_encodings, text_mask = self.clip.embed_text(text)

        assert not (self.condition_on_text_encodings and not exists(text_encodings)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text_encodings and exists(text_encodings)), 'decoder specified not to be conditioned on text, yet it is presented'

        img = None
        is_cuda = next(self.parameters()).is_cuda

        for unet_number, unet, vae, channel, image_size, predict_x_start, learned_variance in tqdm(zip(range(1, len(self.unets) + 1), self.unets, self.vaes, self.sample_channels, self.image_sizes, self.predict_x_start, self.learned_variance)):

            context = self.one_unet_in_gpu(unet = unet) if is_cuda else null_context()

            with context:
                lowres_cond_img = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_cond_img = self.to_lowres_cond(img, target_image_size = image_size)

                is_latent_diffusion = isinstance(vae, VQGanVAE)
                image_size = vae.get_encoded_fmap_size(image_size)
                shape = (batch_size, vae.encoded_dim, image_size, image_size)

                lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    image_embed = image_embed,
                    text_encodings = text_encodings,
                    text_mask = text_mask,
                    cond_scale = cond_scale,
                    predict_x_start = predict_x_start,
                    learned_variance = learned_variance,
                    clip_denoised = not is_latent_diffusion,
                    lowres_cond_img = lowres_cond_img,
                    is_latent_diffusion = is_latent_diffusion
                )

                img = vae.decode(img)

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        return img

    def forward(
        self,
        image,
        text = None,
        image_embed = None,
        text_encodings = None,
        text_mask = None,
        unet_number = None
    ):
        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1

        unet = self.get_unet(unet_number)

        vae                 = self.vaes[unet_index]
        target_image_size   = self.image_sizes[unet_index]
        predict_x_start     = self.predict_x_start[unet_index]
        random_crop_size    = self.random_crop_sizes[unet_index]
        learned_variance    = self.learned_variance[unet_index]
        b, c, h, w, device, = *image.shape, image.device

        check_shape(image, 'b c h w', c = self.channels)
        assert h >= target_image_size and w >= target_image_size

        times = torch.randint(0, self.num_timesteps, (b,), device = device, dtype = torch.long)

        if not exists(image_embed) and not self.unconditional:
            assert exists(self.clip), 'if you want to derive CLIP image embeddings automatically, you must supply `clip` to the decoder on init'
            image_embed, _ = self.clip.embed_image(image)

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip), 'if you are passing in raw text, you need to supply `clip` to the decoder'
            _, text_encodings, text_mask = self.clip.embed_text(text)

        assert not (self.condition_on_text_encodings and not exists(text_encodings)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text_encodings and exists(text_encodings)), 'decoder specified not to be conditioned on text, yet it is presented'

        lowres_cond_img = self.to_lowres_cond(image, target_image_size = target_image_size, downsample_image_size = self.image_sizes[unet_index - 1]) if unet_number > 1 else None
        image = resize_image_to(image, target_image_size)

        if exists(random_crop_size):
            aug = K.RandomCrop((random_crop_size, random_crop_size), p = 1.)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            image = aug(image)
            lowres_cond_img = aug(lowres_cond_img, params = aug._params)

        is_latent_diffusion = not isinstance(vae, NullVQGanVAE)

        vae.eval()
        with torch.no_grad():
            image = vae.encode(image)
            lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

        return self.p_losses(unet, image, times, image_embed = image_embed, text_encodings = text_encodings, text_mask = text_mask, lowres_cond_img = lowres_cond_img, predict_x_start = predict_x_start, learned_variance = learned_variance, is_latent_diffusion = is_latent_diffusion)

# main class

class DALLE2(nn.Module):
    def __init__(
        self,
        *,
        prior,
        decoder,
        prior_num_samples = 2
    ):
        super().__init__()
        assert isinstance(prior, DiffusionPrior)
        assert isinstance(decoder, Decoder)
        self.prior = prior
        self.decoder = decoder

        self.prior_num_samples = prior_num_samples
        self.decoder_need_text_cond = self.decoder.condition_on_text_encodings

        self.to_pil = T.ToPILImage()

    @torch.no_grad()
    @eval_decorator
    def forward(
        self,
        text,
        cond_scale = 1.,
        prior_cond_scale = 1.,
        return_pil_images = False
    ):
        device = module_device(self)
        one_text = isinstance(text, str) or (not is_list_str(text) and text.shape[0] == 1)

        if isinstance(text, str) or is_list_str(text):
            text = [text] if not isinstance(text, (list, tuple)) else text
            text = tokenizer.tokenize(text).to(device)

        image_embed = self.prior.sample(text, num_samples_per_batch = self.prior_num_samples, cond_scale = prior_cond_scale)

        text_cond = text if self.decoder_need_text_cond else None
        images = self.decoder.sample(image_embed, text = text_cond, cond_scale = cond_scale)

        if return_pil_images:
            images = list(map(self.to_pil, images.unbind(dim = 0)))

        if one_text:
            return images[0]

        return images
