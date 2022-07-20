import math
import random
from tqdm.auto import tqdm
from functools import partial, wraps
from contextlib import contextmanager
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat, reduce
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

def first(arr, d = None):
    if len(arr) == 0:
        return d
    return arr[0]

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(val, length = None, validate = True):
    if isinstance(val, list):
        val = tuple(val)

    out = val if isinstance(val, tuple) else ((val,) * default(length, 1))

    if exists(length) and validate:
        assert len(out) == length

    return out

def module_device(module):
    if isinstance(module, nn.Identity):
        return 'cpu' # It doesn't matter
    return next(module.parameters()).device

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if exists(m.bias):
        nn.init.zeros_(m.bias)

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

def resize_image_to(
    image,
    target_image_size,
    clamp_range = None,
    nearest = False,
    **kwargs
):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    if not nearest:
        scale_factors = target_image_size / orig_image_size
        out = resize(image, scale_factors = scale_factors, **kwargs)
    else:
        out = F.interpolate(image, target_image_size, mode = 'nearest')

    if exists(clamp_range):
        out = out.clamp(*clamp_range)

    return out

# image normalization functions
# ddpms expect images to be in the range of -1 to 1
# but CLIP may otherwise

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

# clip related adapters

EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings'])
EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])

class BaseClipAdapter(nn.Module):
    def __init__(self, clip, **kwargs):
        super().__init__()
        self.clip = clip
        self.overrides = kwargs

    def validate_and_resize_image(self, image):
        image_size = image.shape[-1]
        assert image_size >= self.image_size, f'you are passing in an image of size {image_size} but CLIP requires the image size to be at least {self.image_size}'
        return resize_image_to(image, self.image_size)

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
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        return EmbeddedText(l2norm(text_embed), text_encodings)

    @torch.no_grad()
    def embed_image(self, image):
        image = self.validate_and_resize_image(image)
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
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        return EmbeddedText(text_embed, text_encodings)

    @torch.no_grad()
    def embed_image(self, image):
        image = self.validate_and_resize_image(image)
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
        self.eos_id = 49407 # for handling 0 being also '!'

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

        is_eos_id = (text == self.eos_id)
        text_mask_excluding_eos = is_eos_id.cumsum(dim = -1) == 0
        text_mask = F.pad(text_mask_excluding_eos, (1, -1), value = True)
        assert not self.cleared

        text_embed = self.clip.encode_text(text)
        text_encodings = self.text_encodings
        text_encodings = text_encodings.masked_fill(~text_mask[..., None], 0.)
        del self.text_encodings
        return EmbeddedText(l2norm(text_embed.float()), text_encodings.float())

    @torch.no_grad()
    def embed_image(self, image):
        assert not self.cleared
        image = self.validate_and_resize_image(image)
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

def meanflat(x):
    return x.mean(dim = tuple(range(1, len(x.shape))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(((2.0 / math.pi) ** 0.5) * (x + 0.044715 * (x ** 3))))

def discretized_gaussian_log_likelihood(x, *, means, log_scales, thres = 0.999):
    assert x.shape == means.shape == log_scales.shape

    # attempting to correct nan gradients when learned variance is turned on
    # in the setting of deepspeed fp16
    eps = 1e-12 if x.dtype == torch.float32 else 1e-3

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = log(cdf_plus, eps = eps)
    log_one_minus_cdf_min = log(1. - cdf_min, eps = eps)
    cdf_delta = cdf_plus - cdf_min

    log_probs = torch.where(x < -thres,
        log_cdf_plus,
        torch.where(x > thres,
            log_one_minus_cdf_min,
            log(cdf_delta, eps = eps)))

    return log_probs

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / first(alphas_cumprod)
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
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype = torch.float64) ** 2


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps, dtype = torch.float64)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


class NoiseScheduler(nn.Module):
    def __init__(self, *, beta_schedule, timesteps, loss_type, p2_loss_weight_gamma = 0., p2_loss_weight_k = 1):
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

        # p2 loss reweighting

        self.has_p2_loss_reweighting = p2_loss_weight_gamma > 0.
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def sample_random_times(self, batch):
        return torch.randint(0, self.num_timesteps, (batch,), device = self.betas.device, dtype = torch.long)

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

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def p2_reweigh_loss(self, loss, times):
        if not self.has_p2_loss_reweighting:
            return loss
        return loss * extract(self.p2_loss_weight, times, loss.shape)

# diffusion prior

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5, stable = False):
        super().__init__()
        self.eps = eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim = -1, keepdim = True).detach()

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + self.eps).rsqrt() * self.g

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5, stable = False):
        super().__init__()
        self.eps = eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim = 1, keepdim = True).detach()

        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + self.eps).rsqrt() * self.g

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
        rotary_emb = None,
        pb_relax_alpha = 128
    ):
        super().__init__()
        self.pb_relax_alpha = pb_relax_alpha
        self.scale = dim_head ** -0.5 * (pb_relax_alpha ** -1)

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
        sim = sim * self.pb_relax_alpha

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
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
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
        max_text_len = 256,
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

        # dalle1 learned padding strategy

        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, dim))

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
    
        mask = torch.any(text_encodings != 0., dim = -1)

        # replace any padding in the text encodings with learned padding tokens unique across position

        text_encodings = text_encodings[:, :self.max_text_len]
        mask = mask[:, :self.max_text_len]

        text_len = text_encodings.shape[-2]
        remainder = self.max_text_len - text_len

        if remainder > 0:
            text_encodings = F.pad(text_encodings, (0, 0, 0, remainder), value = 0.)
            mask = F.pad(mask, (0, remainder), value = False)

        null_text_embeds = self.null_text_embed.to(text_encodings.dtype)

        text_encodings = torch.where(
            rearrange(mask, 'b n -> b n 1').clone(),
            text_encodings,
            null_text_embeds
        )

        # classifier free guidance

        keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
        keep_mask = rearrange(keep_mask, 'b -> b 1')

        mask &= keep_mask

        # whether text embedding is masked or not depends on the classifier free guidance conditional masking

        keep_mask = repeat(keep_mask, 'b 1 -> b n', n = num_text_embeds)
        mask = torch.cat((mask, keep_mask), dim = 1)

        # whether text embedding is used for conditioning depends on whether text encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right

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

        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)

        pred_image_embed = tokens[..., -1, :]

        return pred_image_embed

class DiffusionPrior(nn.Module):
    def __init__(
        self,
        net,
        *,
        clip = None,
        image_embed_dim = None,
        image_size = None,
        image_channels = 3,
        timesteps = 1000,
        sample_timesteps = None,
        cond_drop_prob = 0.,
        loss_type = "l2",
        predict_x_start = True,
        beta_schedule = "cosine",
        condition_on_text_encodings = True,  # the paper suggests this is needed, but you can turn it off for your CLIP preprocessed text embed -> image embed training
        sampling_clamp_l2norm = False,       # whether to l2norm clamp the image embed at each denoising iteration (analogous to -1 to 1 clipping for usual DDPMs)
        sampling_final_clamp_l2norm = False, # whether to l2norm the final image embedding output (this is also done for images in ddpm)
        training_clamp_l2norm = False,
        init_image_embed_l2norm = False,
        image_embed_scale = None,            # this is for scaling the l2-normed image embedding, so it is more suitable for gaussian diffusion, as outlined by Katherine (@crowsonkb) https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132
        clip_adapter_overrides = dict()
    ):
        super().__init__()

        self.sample_timesteps = sample_timesteps

        self.noise_scheduler = NoiseScheduler(
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
        self.sampling_final_clamp_l2norm = sampling_final_clamp_l2norm

        self.training_clamp_l2norm = training_clamp_l2norm
        self.init_image_embed_l2norm = init_image_embed_l2norm

        # device tracker

        self.register_buffer('_dummy', torch.tensor([True]), persistent = False)

    @property
    def device(self):
        return self._dummy.device

    def l2norm_clamp_embed(self, image_embed):
        return l2norm(image_embed) * self.image_embed_scale

    def p_mean_variance(self, x, t, text_cond, clip_denoised = False, cond_scale = 1.):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'the model was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = self.net.forward_with_cond_scale(x, t, cond_scale = cond_scale, **text_cond)

        if self.predict_x_start:
            x_recon = pred
        else:
            x_recon = self.noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised and not self.predict_x_start:
            x_recon.clamp_(-1., 1.)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_recon = l2norm(x_recon) * self.image_embed_scale

        model_mean, posterior_variance, posterior_log_variance = self.noise_scheduler.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, clip_denoised = True, cond_scale = 1.):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, text_cond = text_cond, clip_denoised = clip_denoised, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale = 1.):
        batch, device = shape[0], self.device
        image_embed = torch.randn(shape, device = device)

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step', total=self.noise_scheduler.num_timesteps):
            times = torch.full((batch,), i, device = device, dtype = torch.long)
            image_embed = self.p_sample(image_embed, times, text_cond = text_cond, cond_scale = cond_scale)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop_ddim(self, shape, text_cond, *, timesteps, eta = 1., cond_scale = 1.):
        batch, device, alphas, total_timesteps = shape[0], self.device, self.noise_scheduler.alphas_cumprod_prev, self.noise_scheduler.num_timesteps

        times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        image_embed = torch.randn(shape, device = device)

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = alphas[time]
            alpha_next = alphas[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            pred = self.net.forward_with_cond_scale(image_embed, time_cond, cond_scale = cond_scale, **text_cond)

            if self.predict_x_start:
                x_start = pred
                pred_noise = self.noise_scheduler.predict_noise_from_start(image_embed, t = time_cond, x0 = pred)
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(image_embed, t = time_cond, noise = pred)
                pred_noise = pred

            if not self.predict_x_start:
                x_start.clamp_(-1., 1.)

            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(image_embed) if time_next > 0 else 0.

            image_embed = x_start * alpha_next.sqrt() + \
                          c1 * noise + \
                          c2 * pred_noise

        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

    @torch.no_grad()
    def p_sample_loop(self, *args, timesteps = None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps

        if not is_ddim:
            return self.p_sample_loop_ddpm(*args, **kwargs)

        return self.p_sample_loop_ddim(*args, **kwargs, timesteps = timesteps)

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.noise_scheduler.q_sample(x_start = image_embed, t = times, noise = noise)

        pred = self.net(
            image_embed_noisy,
            times,
            cond_drop_prob = self.cond_drop_prob,
            **text_cond
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        target = noise if not self.predict_x_start else image_embed

        loss = self.noise_scheduler.loss_fn(pred, target)
        return loss

    @torch.no_grad()
    @eval_decorator
    def sample_batch_size(self, batch_size, text_cond, cond_scale = 1.):
        device = self.betas.device
        shape = (batch_size, self.image_embed_dim)

        img = torch.randn(shape, device = device)

        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc = 'sampling loop time step', total = self.noise_scheduler.num_timesteps):
            img = self.p_sample(img, torch.full((batch_size,), i, device = device, dtype = torch.long), text_cond = text_cond, cond_scale = cond_scale)
        return img

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        text,
        num_samples_per_batch = 2,
        cond_scale = 1.,
        timesteps = None
    ):
        timesteps = default(timesteps, self.sample_timesteps)

        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, 'b ... -> (b r) ...', r = num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond = text_cond, cond_scale = cond_scale, timesteps = timesteps)

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
            text_embed, text_encodings = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            assert exists(text_encodings), 'text encodings must be present for diffusion prior if specified'
            text_cond = {**text_cond, 'text_encodings': text_encodings}

        # timestep conditioning from ddpm

        batch, device = image_embed.shape[0], image_embed.device
        times = self.noise_scheduler.sample_random_times(batch)

        # scale image embed (Katherine)

        image_embed *= self.image_embed_scale

        # calculate forward loss

        return self.p_losses(image_embed, times, text_cond = text_cond, *args, **kwargs)

# decoder

def NearestUpsample(dim, dim_out = None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

class PixelShuffleUpsample(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """
    def __init__(self, dim, dim_out = None):
        super().__init__()
        dim_out = default(dim_out, dim)
        conv = nn.Conv2d(dim, dim_out * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            nn.PixelShuffle(2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, 'o ... -> (o 4) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        return self.net(x)

def Downsample(dim, *, dim_out = None):
    dim_out = default(dim_out, dim)
    return nn.Conv2d(dim, dim_out, 4, 2, 1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        dtype, device = x.dtype, x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = device, dtype = dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim = -1).type(dtype)

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

    def forward(self, x, time_emb = None, cond = None):

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
        norm_context = False,
        pb_relax_alpha = 32 ** 2
    ):
        super().__init__()
        self.pb_relax_alpha = pb_relax_alpha
        self.scale = dim_head ** -0.5 * (pb_relax_alpha ** -1)
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
        sim = sim * self.pb_relax_alpha

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

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
        self_attn = False,
        attn_dim_head = 32,
        attn_heads = 16,
        lowres_cond = False,             # for cascading diffusion - https://cascaded-diffusion.github.io/
        lowres_noise_cond = False,       # for conditioning on low resolution noising, based on Imagen
        sparse_attn = False,
        attend_at_middle = True,         # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        cond_on_text_encodings = False,
        max_text_len = 256,
        cond_on_image_embeds = False,
        add_image_embeds_to_time = True, # alerted by @mhh0318 to a phrase in the paper - "Specifically, we modify the architecture described in Nichol et al. (2021) by projecting and adding CLIP embeddings to the existing timestep embedding"
        init_dim = None,
        init_conv_kernel_size = 7,
        resnet_groups = 8,
        num_resnet_blocks = 2,
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        memory_efficient = False,
        scale_skip_connection = False,
        pixel_shuffle_upsample = True,
        final_conv_kernel_size = 1,
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
        init_dim = default(init_dim, dim)

        self.init_conv = CrossEmbedLayer(init_channels, dim_out = init_dim, kernel_sizes = init_cross_embed_kernel_sizes, stride = 1) if init_cross_embed else nn.Conv2d(init_channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        num_stages = len(in_out)

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

        self.image_to_tokens = nn.Sequential(
            nn.Linear(image_embed_dim, cond_dim * num_image_tokens),
            Rearrange('b (n d) -> b n d', n = num_image_tokens)
        ) if cond_on_image_embeds and image_embed_dim != cond_dim else nn.Identity()

        self.to_image_hiddens = nn.Sequential(
            nn.Linear(image_embed_dim, time_cond_dim),
            nn.GELU()
        ) if cond_on_image_embeds and add_image_embeds_to_time else None

        self.norm_cond = nn.LayerNorm(cond_dim)
        self.norm_mid_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None
        self.text_embed_dim = None

        if cond_on_text_encodings:
            assert exists(text_embed_dim), 'text_embed_dim must be given to the unet if cond_on_text_encodings is True'
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)
            self.text_embed_dim = text_embed_dim

        # low resolution noise conditiong, based on Imagen's upsampler training technique

        self.lowres_noise_cond = lowres_noise_cond

        self.to_lowres_noise_cond = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.GELU(),
            nn.Linear(time_cond_dim, time_cond_dim)
        ) if lowres_noise_cond else None

        # finer control over whether to condition on image embeddings and text encodings
        # so one can have the latter unets in the cascading DDPMs only focus on super-resoluting

        self.cond_on_text_encodings = cond_on_text_encodings
        self.cond_on_image_embeds = cond_on_image_embeds

        # for classifier free guidance

        self.null_image_embed = nn.Parameter(torch.randn(1, num_image_tokens, cond_dim))
        self.null_image_hiddens = nn.Parameter(torch.randn(1, time_cond_dim))

        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))

        # whether to scale skip connection, adopted in Imagen

        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # attention related params

        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)

        self_attn = cast_tuple(self_attn, num_stages)

        create_self_attn = lambda dim: EinopsToAndFrom('b c h w', 'b (h w) c', Residual(Attention(dim, **attn_kwargs)))

        # resnet block klass

        resnet_groups = cast_tuple(resnet_groups, num_stages)
        top_level_resnet_group = first(resnet_groups)

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_stages)

        # downsample klass

        downsample_klass = Downsample
        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # upsample klass

        upsample_klass = NearestUpsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # give memory efficient unet an initial resnet block

        self.init_resnet_block = ResnetBlock(init_dim, init_dim, time_cond_dim = time_cond_dim, groups = top_level_resnet_group) if memory_efficient else None

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        skip_connect_dims = [] # keeping track of skip connection dimensions

        for ind, ((dim_in, dim_out), groups, layer_num_resnet_blocks, layer_self_attn) in enumerate(zip(in_out, resnet_groups, num_resnet_blocks, self_attn)):
            is_first = ind == 0
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if not is_first else None

            dim_layer = dim_out if memory_efficient else dim_in
            skip_connect_dims.append(dim_layer)

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_layer)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_layer, **attn_kwargs))

            self.downs.append(nn.ModuleList([
                downsample_klass(dim_in, dim_out = dim_out) if memory_efficient else None,
                ResnetBlock(dim_layer, dim_layer, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(dim_layer, dim_layer, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups) for _ in range(layer_num_resnet_blocks)]),
                attention,
                downsample_klass(dim_layer, dim_out = dim_out) if not is_last and not memory_efficient else nn.Conv2d(dim_layer, dim_out, 1)
            ]))

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])
        self.mid_attn = create_self_attn(mid_dim)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim, groups = resnet_groups[-1])

        for ind, ((dim_in, dim_out), groups, layer_num_resnet_blocks, layer_self_attn) in enumerate(zip(reversed(in_out), reversed(resnet_groups), reversed(num_resnet_blocks), reversed(self_attn))):
            is_last = ind >= (len(in_out) - 1)
            layer_cond_dim = cond_dim if not is_last else None

            skip_connect_dim = skip_connect_dims.pop()

            attention = nn.Identity()
            if layer_self_attn:
                attention = create_self_attn(dim_out)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_out, **attn_kwargs))

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups),
                nn.ModuleList([ResnetBlock(dim_out + skip_connect_dim, dim_out, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim, groups = groups)  for _ in range(layer_num_resnet_blocks)]),
                attention,
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else nn.Identity()
            ]))

        self.final_resnet_block = ResnetBlock(dim * 2, dim, time_cond_dim = time_cond_dim, groups = top_level_resnet_group)

        out_dim_in = dim + (channels if lowres_cond else 0)

        self.to_out = nn.Conv2d(out_dim_in, self.channels_out, kernel_size = final_conv_kernel_size, padding = final_conv_kernel_size // 2)

        zero_init_(self.to_out) # since both OpenAI and @crowsonkb are doing it

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        lowres_noise_cond,
        channels,
        channels_out,
        cond_on_image_embeds,
        cond_on_text_encodings,
    ):
        if lowres_cond == self.lowres_cond and \
            channels == self.channels and \
            cond_on_image_embeds == self.cond_on_image_embeds and \
            cond_on_text_encodings == self.cond_on_text_encodings and \
            cond_on_lowres_noise == self.cond_on_lowres_noise and \
            channels_out == self.channels_out:
            return self

        updated_kwargs = dict(
            lowres_cond = lowres_cond,
            channels = channels,
            channels_out = channels_out,
            cond_on_image_embeds = cond_on_image_embeds,
            cond_on_text_encodings = cond_on_text_encodings,
            lowres_noise_cond = lowres_noise_cond
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
        lowres_noise_level = None,
        text_encodings = None,
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
        r = x.clone() # final residual

        # time conditioning

        time = time.type_as(x)
        time_hiddens = self.to_time_hiddens(time)

        time_tokens = self.to_time_tokens(time_hiddens)
        t = self.to_time_cond(time_hiddens)

        # low res noise conditioning (similar to time above)

        if exists(lowres_noise_level):
            assert exists(self.to_lowres_noise_cond), 'lowres_noise_cond must be set to True on instantiation of the unet in order to conditiong on lowres noise'
            lowres_noise_level = lowres_noise_level.type_as(x)
            t = t + self.to_lowres_noise_cond(lowres_noise_level)

        # conditional dropout

        image_keep_mask = prob_mask_like((batch_size,), 1 - image_cond_drop_prob, device = device)
        text_keep_mask = prob_mask_like((batch_size,), 1 - text_cond_drop_prob, device = device)

        text_keep_mask = rearrange(text_keep_mask, 'b -> b 1 1')

        # image embedding to be summed to time embedding
        # discovered by @mhh0318 in the paper

        if exists(image_embed) and exists(self.to_image_hiddens):
            image_hiddens = self.to_image_hiddens(image_embed)
            image_keep_mask_hidden = rearrange(image_keep_mask, 'b -> b 1')
            null_image_hiddens = self.null_image_hiddens.to(image_hiddens.dtype)

            image_hiddens = torch.where(
                image_keep_mask_hidden,
                image_hiddens,
                null_image_hiddens
            )

            t = t + image_hiddens

        # mask out image embedding depending on condition dropout
        # for classifier free guidance

        image_tokens = None

        if self.cond_on_image_embeds:
            image_keep_mask_embed = rearrange(image_keep_mask, 'b -> b 1 1')
            image_tokens = self.image_to_tokens(image_embed)
            null_image_embed = self.null_image_embed.to(image_tokens.dtype) # for some reason pytorch AMP not working

            image_tokens = torch.where(
                image_keep_mask_embed,
                image_tokens,
                null_image_embed
            )

        # take care of text encodings (optional)

        text_tokens = None

        if exists(text_encodings) and self.cond_on_text_encodings:
            assert text_encodings.shape[0] == batch_size, f'the text encodings being passed into the unet does not have the proper batch size - text encoding shape {text_encodings.shape} - required batch size is {batch_size}'
            assert self.text_embed_dim == text_encodings.shape[-1], f'the text encodings you are passing in have a dimension of {text_encodings.shape[-1]}, but the unet was created with text_embed_dim of {self.text_embed_dim}.'

            text_mask = torch.any(text_encodings != 0., dim = -1)

            text_tokens = self.text_to_cond(text_encodings)

            text_tokens = text_tokens[:, :self.max_text_len]
            text_mask = text_mask[:, :self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
                text_mask = F.pad(text_mask, (0, remainder), value = False)

            text_mask = rearrange(text_mask, 'b n -> b n 1')

            assert text_mask.shape[0] == text_keep_mask.shape[0], f'text_mask has shape of {text_mask.shape} while text_keep_mask has shape {text_keep_mask.shape}. text encoding is of shape {text_encodings.shape}'
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

        # initial resnet block

        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)

        # go through the layers of the unet, down and up

        hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn, post_downsample in self.downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t, c)
                hiddens.append(x)

            x = attn(x)
            hiddens.append(x.contiguous())

            if exists(post_downsample):
                x = post_downsample(x)

        x = self.mid_block1(x, t, mid_c)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, t, mid_c)

        connect_skip = lambda fmap: torch.cat((fmap, hiddens.pop() * self.skip_connect_scale), dim = 1)

        for init_block, resnet_blocks, attn, upsample in self.ups:
            x = connect_skip(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = connect_skip(x)
                x = resnet_block(x, t, c)

            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_resnet_block(x, t)

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        return self.to_out(x)

class LowresConditioner(nn.Module):
    def __init__(
        self,
        downsample_first = True,
        use_blur = True,
        blur_prob = 0.5,
        blur_sigma = 0.6,
        blur_kernel_size = 3,
        use_noise = False,
        input_image_range = None,
        normalize_img_fn = identity,
        unnormalize_img_fn = identity
    ):
        super().__init__()
        self.downsample_first = downsample_first
        self.input_image_range = input_image_range

        self.use_blur = use_blur
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size

        self.use_noise = use_noise
        self.normalize_img = normalize_img_fn
        self.unnormalize_img = unnormalize_img_fn
        self.noise_scheduler = NoiseScheduler(beta_schedule = 'linear', timesteps = 1000, loss_type = 'l2') if use_noise else None

    def noise_image(self, cond_fmap, noise_levels = None):
        assert exists(self.noise_scheduler)

        batch = cond_fmap.shape[0]
        cond_fmap = self.normalize_img(cond_fmap)

        random_noise_levels = default(noise_levels, lambda: self.noise_scheduler.sample_random_times(batch))
        cond_fmap = self.noise_scheduler.q_sample(cond_fmap, t = random_noise_levels, noise = torch.randn_like(cond_fmap))

        cond_fmap = self.unnormalize_img(cond_fmap)
        return cond_fmap, random_noise_levels

    def forward(
        self,
        cond_fmap,
        *,
        target_image_size,
        downsample_image_size = None,
        should_blur = True,
        blur_sigma = None,
        blur_kernel_size = None
    ):
        if self.downsample_first and exists(downsample_image_size):
            cond_fmap = resize_image_to(cond_fmap, downsample_image_size, clamp_range = self.input_image_range, nearest = True)

        # blur is only applied 50% of the time
        # section 3.1 in https://arxiv.org/abs/2106.15282

        if self.use_blur and should_blur and random.random() < self.blur_prob:

            # when training, blur the low resolution conditional image

            blur_sigma = default(blur_sigma, self.blur_sigma)
            blur_kernel_size = default(blur_kernel_size, self.blur_kernel_size)

            # allow for drawing a random sigma between lo and hi float values

            if isinstance(blur_sigma, tuple):
                blur_sigma = tuple(map(float, blur_sigma))
                blur_sigma = random.uniform(*blur_sigma)

            # allow for drawing a random kernel size between lo and hi int values

            if isinstance(blur_kernel_size, tuple):
                blur_kernel_size = tuple(map(int, blur_kernel_size))
                kernel_size_lo, kernel_size_hi = blur_kernel_size
                blur_kernel_size = random.randrange(kernel_size_lo, kernel_size_hi + 1)

            cond_fmap = gaussian_blur2d(cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2))

        # resize to target image size

        cond_fmap = resize_image_to(cond_fmap, target_image_size, clamp_range = self.input_image_range, nearest = True)

        # noise conditioning, as done in Imagen
        # as a replacement for the BSR noising, and potentially replace blurring for first stage too

        random_noise_levels = None

        if self.use_noise:
            cond_fmap, random_noise_levels = self.noise_image(cond_fmap)

        # return conditioning feature map, as well as the augmentation noise levels

        return cond_fmap, random_noise_levels

class Decoder(nn.Module):
    def __init__(
        self,
        unet,
        *,
        clip = None,
        image_size = None,
        channels = 3,
        vae = tuple(),
        timesteps = 1000,
        sample_timesteps = None,
        image_cond_drop_prob = 0.1,
        text_cond_drop_prob = 0.5,
        loss_type = 'l2',
        beta_schedule = None,
        predict_x_start = False,
        predict_x_start_for_latent_diffusion = False,
        image_sizes = None,                         # for cascading ddpm, image size at each stage
        random_crop_sizes = None,                   # whether to random crop the image at that stage in the cascade (super resoluting convolutions at the end may be able to generalize on smaller crops)
        use_noise_for_lowres_cond = False,          # whether to use Imagen-like noising for low resolution conditioning  
        use_blur_for_lowres_cond = True,            # whether to use the blur conditioning used in the original cascading ddpm paper, as well as DALL-E2
        lowres_downsample_first = True,             # cascading ddpm - resizes to lower resolution, then to next conditional resolution + blur
        blur_prob = 0.5,                            # cascading ddpm - when training, the gaussian blur is only applied 50% of the time
        blur_sigma = 0.6,                           # cascading ddpm - blur sigma
        blur_kernel_size = 3,                       # cascading ddpm - blur kernel size
        lowres_noise_sample_level = 0.2,            # in imagen paper, they use a 0.2 noise level at sample time for low resolution conditioning
        clip_denoised = True,
        clip_x_start = True,
        clip_adapter_overrides = dict(),
        learned_variance = True,
        learned_variance_constrain_frac = False,
        vb_loss_weight = 0.001,
        unconditional = False,                      # set to True for generating images without conditioning
        auto_normalize_img = True,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        use_dynamic_thres = False,                  # from the Imagen paper
        dynamic_thres_percentile = 0.95,
        p2_loss_weight_gamma = 0.,                  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.                      # can be set to 0. for deterministic sampling afaict
    ):
        super().__init__()

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


        # normalize and unnormalize image functions

        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity

        # verify conditioning method

        unets = cast_tuple(unet)
        num_unets = len(unets)
        self.num_unets = num_unets

        self.unconditional = unconditional

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        vaes = pad_tuple_to_length(cast_tuple(vae), len(unets), fillvalue = NullVQGanVAE(channels = self.channels))

        # whether to use learned variance, defaults to True for the first unet in the cascade, as in paper

        learned_variance = pad_tuple_to_length(cast_tuple(learned_variance), len(unets), fillvalue = False)
        self.learned_variance = learned_variance
        self.learned_variance_constrain_frac = learned_variance_constrain_frac # whether to constrain the output of the network (the interpolation fraction) from 0 to 1
        self.vb_loss_weight = vb_loss_weight

        # default and validate conditioning parameters

        use_noise_for_lowres_cond = cast_tuple(use_noise_for_lowres_cond, num_unets - 1, validate = False)
        use_blur_for_lowres_cond = cast_tuple(use_blur_for_lowres_cond, num_unets - 1, validate = False)

        if len(use_noise_for_lowres_cond) < num_unets:
            use_noise_for_lowres_cond = (False, *use_noise_for_lowres_cond)

        if len(use_blur_for_lowres_cond) < num_unets:
            use_blur_for_lowres_cond = (False, *use_blur_for_lowres_cond)

        assert not use_noise_for_lowres_cond[0], 'first unet will never need low res noise conditioning'
        assert not use_blur_for_lowres_cond[0], 'first unet will never need low res blur conditioning'

        assert num_unets == 1 or all((use_noise or use_blur) for use_noise, use_blur in zip(use_noise_for_lowres_cond[1:], use_blur_for_lowres_cond[1:]))

        # construct unets and vaes

        self.unets = nn.ModuleList([])
        self.vaes = nn.ModuleList([])

        for ind, (one_unet, one_vae, one_unet_learned_var, lowres_noise_cond) in enumerate(zip(unets, vaes, learned_variance, use_noise_for_lowres_cond)):
            assert isinstance(one_unet, Unet)
            assert isinstance(one_vae, (VQGanVAE, NullVQGanVAE))

            is_first = ind == 0
            latent_dim = one_vae.encoded_dim if exists(one_vae) else None

            unet_channels = default(latent_dim, self.channels)
            unet_channels_out = unet_channels * (1 if not one_unet_learned_var else 2)

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                lowres_noise_cond = lowres_noise_cond,
                cond_on_image_embeds = not unconditional and is_first,
                cond_on_text_encodings = not unconditional and one_unet.cond_on_text_encodings,
                channels = unet_channels,
                channels_out = unet_channels_out
            )

            self.unets.append(one_unet)
            self.vaes.append(one_vae.copy_for_eval())

        # sampling timesteps, defaults to non-ddim with full timesteps sampling

        self.sample_timesteps = cast_tuple(sample_timesteps, num_unets)
        self.ddim_sampling_eta = ddim_sampling_eta

        # create noise schedulers per unet

        if not exists(beta_schedule):
            beta_schedule = ('cosine', *(('cosine',) * max(num_unets - 2, 0)), *(('linear',) * int(num_unets > 1)))

        beta_schedule = cast_tuple(beta_schedule, num_unets)
        p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        self.noise_schedulers = nn.ModuleList([])

        for ind, (unet_beta_schedule, unet_p2_loss_weight_gamma, sample_timesteps) in enumerate(zip(beta_schedule, p2_loss_weight_gamma, self.sample_timesteps)):
            assert not exists(sample_timesteps) or sample_timesteps <= timesteps, f'sampling timesteps {sample_timesteps} must be less than or equal to the number of training timesteps {timesteps} for unet {ind + 1}'

            noise_scheduler = NoiseScheduler(
                beta_schedule = unet_beta_schedule,
                timesteps = timesteps,
                loss_type = loss_type,
                p2_loss_weight_gamma = unet_p2_loss_weight_gamma,
                p2_loss_weight_k = p2_loss_weight_k
            )

            self.noise_schedulers.append(noise_scheduler)

        # unet image sizes

        image_sizes = default(image_sizes, (image_size,))
        image_sizes = tuple(sorted(set(image_sizes)))

        assert self.num_unets == len(image_sizes), f'you did not supply the correct number of u-nets ({self.num_unets}) for resolutions {image_sizes}'
        self.image_sizes = image_sizes
        self.sample_channels = cast_tuple(self.channels, len(image_sizes))

        # random crop sizes (for super-resoluting unets at the end of cascade?)

        self.random_crop_sizes = cast_tuple(random_crop_sizes, len(image_sizes))
        assert not exists(self.random_crop_sizes[0]), 'you would not need to randomly crop the image for the base unet'

        # predict x0 config

        self.predict_x_start = cast_tuple(predict_x_start, len(unets)) if not predict_x_start_for_latent_diffusion else tuple(map(lambda t: isinstance(t, VQGanVAE), self.vaes))

        # input image range

        self.input_image_range = (-1. if not auto_normalize_img else 0., 1.)

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.lowres_conds = nn.ModuleList([])

        for unet_index, use_noise, use_blur in zip(range(num_unets), use_noise_for_lowres_cond, use_blur_for_lowres_cond):
            if unet_index == 0:
                self.lowres_conds.append(None)
                continue

            lowres_cond = LowresConditioner(
                downsample_first = lowres_downsample_first,
                use_blur = use_blur,
                use_noise = use_noise,
                blur_prob = blur_prob,
                blur_sigma = blur_sigma,
                blur_kernel_size = blur_kernel_size,
                input_image_range = self.input_image_range,
                normalize_img_fn = self.normalize_img,
                unnormalize_img_fn = self.unnormalize_img
            )

            self.lowres_conds.append(lowres_cond)

        self.lowres_noise_sample_level = lowres_noise_sample_level

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

        # device tracker

        self.register_buffer('_dummy', torch.Tensor([True]), persistent = False)

    @property
    def device(self):
        return self._dummy.device

    @property
    def condition_on_text_encodings(self):
        return any([unet.cond_on_text_encodings for unet in self.unets if isinstance(unet, Unet)])

    def get_unet(self, unet_number):
        assert 0 < unet_number <= self.num_unets
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

    def dynamic_threshold(self, x):
        """ proposed in https://arxiv.org/abs/2205.11487 as an improved clamping in the setting of classifier free guidance """
        
        # s is the threshold amount
        # static thresholding would just be s = 1
        s = 1.
        if self.use_dynamic_thres:
            s = torch.quantile(
                rearrange(x, 'b ... -> b (...)').abs(),
                self.dynamic_thres_percentile,
                dim = -1
            )

            s.clamp_(min = 1.)
            s = s.view(-1, *((1,) * (x.ndim - 1)))

        # clip by threshold, depending on whether static or dynamic
        x = x.clamp(-s, s) / s
        return x

    def p_mean_variance(self, unet, x, t, image_embed, noise_scheduler, text_encodings = None, lowres_cond_img = None, clip_denoised = True, predict_x_start = False, learned_variance = False, cond_scale = 1., model_output = None, lowres_noise_level = None):
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'the decoder was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, t, image_embed = image_embed, text_encodings = text_encodings, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_level = lowres_noise_level))

        if learned_variance:
            pred, var_interp_frac_unnormalized = pred.chunk(2, dim = 1)

        if predict_x_start:
            x_recon = pred
        else:
            x_recon = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised:
            x_recon = self.dynamic_threshold(x_recon)

        model_mean, posterior_variance, posterior_log_variance = noise_scheduler.q_posterior(x_start=x_recon, x_t=x, t=t)

        if learned_variance:
            # if learned variance, posterio variance and posterior log variance are predicted by the network
            # by an interpolation of the max and min log beta values
            # eq 15 - https://arxiv.org/abs/2102.09672
            min_log = extract(noise_scheduler.posterior_log_variance_clipped, t, x.shape)
            max_log = extract(torch.log(noise_scheduler.betas), t, x.shape)
            var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

            if self.learned_variance_constrain_frac:
                var_interp_frac = var_interp_frac.sigmoid()

            posterior_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log
            posterior_variance = posterior_log_variance.exp()

        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, unet, x, t, image_embed, noise_scheduler, text_encodings = None, cond_scale = 1., lowres_cond_img = None, predict_x_start = False, learned_variance = False, clip_denoised = True, lowres_noise_level = None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x, t = t, image_embed = image_embed, text_encodings = text_encodings, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, clip_denoised = clip_denoised, predict_x_start = predict_x_start, noise_scheduler = noise_scheduler, learned_variance = learned_variance, lowres_noise_level = lowres_noise_level)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop_ddpm(
        self,
        unet,
        shape,
        image_embed,
        noise_scheduler,
        predict_x_start = False,
        learned_variance = False,
        clip_denoised = True,
        lowres_cond_img = None,
        text_encodings = None,
        cond_scale = 1,
        is_latent_diffusion = False,
        lowres_noise_level = None,
        inpaint_image = None,
        inpaint_mask = None
    ):
        device = self.device

        b = shape[0]
        img = torch.randn(shape, device = device)

        if exists(inpaint_image):
            inpaint_image = self.normalize_img(inpaint_image)
            inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest = True)
            inpaint_mask = rearrange(inpaint_mask, 'b h w -> b 1 h w').float()
            inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest = True)
            inpaint_mask = inpaint_mask.bool()

        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        for i in tqdm(reversed(range(0, noise_scheduler.num_timesteps)), desc = 'sampling loop time step', total = noise_scheduler.num_timesteps):
            times = torch.full((b,), i, device = device, dtype = torch.long)

            if exists(inpaint_image):
                # following the repaint paper
                # https://arxiv.org/abs/2201.09865
                noised_inpaint_image = noise_scheduler.q_sample(inpaint_image, t = times)
                img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

            img = self.p_sample(
                unet,
                img,
                times,
                image_embed = image_embed,
                text_encodings = text_encodings,
                cond_scale = cond_scale,
                lowres_cond_img = lowres_cond_img,
                lowres_noise_level = lowres_noise_level,
                predict_x_start = predict_x_start,
                noise_scheduler = noise_scheduler,
                learned_variance = learned_variance,
                clip_denoised = clip_denoised
            )

        if exists(inpaint_image):
            img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img

    @torch.no_grad()
    def p_sample_loop_ddim(
        self,
        unet,
        shape,
        image_embed,
        noise_scheduler,
        timesteps,
        eta = 1.,
        predict_x_start = False,
        learned_variance = False,
        clip_denoised = True,
        lowres_cond_img = None,
        text_encodings = None,
        cond_scale = 1,
        is_latent_diffusion = False,
        lowres_noise_level = None,
        inpaint_image = None,
        inpaint_mask = None
    ):
        batch, device, total_timesteps, alphas, eta = shape[0], self.device, noise_scheduler.num_timesteps, noise_scheduler.alphas_cumprod_prev, self.ddim_sampling_eta

        times = torch.linspace(0., total_timesteps, steps = timesteps + 2)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if exists(inpaint_image):
            inpaint_image = self.normalize_img(inpaint_image)
            inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest = True)
            inpaint_mask = rearrange(inpaint_mask, 'b h w -> b 1 h w').float()
            inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest = True)
            inpaint_mask = inpaint_mask.bool()

        img = torch.randn(shape, device = device)

        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = alphas[time]
            alpha_next = alphas[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            if exists(inpaint_image):
                # following the repaint paper
                # https://arxiv.org/abs/2201.09865
                noised_inpaint_image = noise_scheduler.q_sample(inpaint_image, t = time_cond)
                img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

            pred = unet.forward_with_cond_scale(img, time_cond, image_embed = image_embed, text_encodings = text_encodings, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_level = lowres_noise_level)

            if learned_variance:
                pred, _ = pred.chunk(2, dim = 1)

            if predict_x_start:
                x_start = pred
                pred_noise = noise_scheduler.predict_noise_from_start(img, t = time_cond, x0 = pred)
            else:
                x_start = noise_scheduler.predict_start_from_noise(img, t = time_cond, noise = pred)
                pred_noise = pred

            if clip_denoised:
                x_start = self.dynamic_threshold(x_start)

            c1 = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
            noise = torch.randn_like(img) if time_next > 0 else 0.

            img = x_start * alpha_next.sqrt() + \
                  c1 * noise + \
                  c2 * pred_noise

        if exists(inpaint_image):
            img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        img = self.unnormalize_img(img)
        return img

    @torch.no_grad()
    def p_sample_loop(self, *args, noise_scheduler, timesteps = None, **kwargs):
        num_timesteps = noise_scheduler.num_timesteps

        timesteps = default(timesteps, num_timesteps)
        assert timesteps <= num_timesteps
        is_ddim = timesteps < num_timesteps

        if not is_ddim:
            return self.p_sample_loop_ddpm(*args, noise_scheduler = noise_scheduler, **kwargs)

        return self.p_sample_loop_ddim(*args, noise_scheduler = noise_scheduler, timesteps = timesteps, **kwargs)

    def p_losses(self, unet, x_start, times, *, image_embed, noise_scheduler, lowres_cond_img = None, text_encodings = None, predict_x_start = False, noise = None, learned_variance = False, clip_denoised = False, is_latent_diffusion = False, lowres_noise_level = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        if not is_latent_diffusion:
            x_start = self.normalize_img(x_start)
            lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t

        x_noisy = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)

        model_output = unet(
            x_noisy,
            times,
            image_embed = image_embed,
            text_encodings = text_encodings,
            lowres_cond_img = lowres_cond_img,
            lowres_noise_level = lowres_noise_level,
            image_cond_drop_prob = self.image_cond_drop_prob,
            text_cond_drop_prob = self.text_cond_drop_prob,
        )

        if learned_variance:
            pred, _ = model_output.chunk(2, dim = 1)
        else:
            pred = model_output

        target = noise if not predict_x_start else x_start

        loss = noise_scheduler.loss_fn(pred, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = noise_scheduler.p2_reweigh_loss(loss, times)

        loss = loss.mean()

        if not learned_variance:
            # return simple loss if not using learned variance
            return loss

        # most of the code below is transcribed from
        # https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
        # the Improved DDPM paper then further modified it so that the mean is detached (shown a couple lines before), and weighted to be smaller than the l1 or l2 "simple" loss
        # it is questionable whether this is really needed, looking at some of the figures in the paper, but may as well stay faithful to their implementation

        # if learning the variance, also include the extra weight kl loss

        true_mean, _, true_log_variance_clipped = noise_scheduler.q_posterior(x_start = x_start, x_t = x_noisy, t = times)
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x_noisy, t = times, image_embed = image_embed, noise_scheduler = noise_scheduler, clip_denoised = clip_denoised, learned_variance = True, model_output = model_output)

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
        image = None,
        image_embed = None,
        text = None,
        text_encodings = None,
        batch_size = 1,
        cond_scale = 1.,
        start_at_unet_number = 1,
        stop_at_unet_number = None,
        distributed = False,
        inpaint_image = None,
        inpaint_mask = None
    ):
        assert self.unconditional or exists(image_embed), 'image embed must be present on sampling from decoder unless if trained unconditionally'

        if not self.unconditional:
            batch_size = image_embed.shape[0]

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip)
            _, text_encodings = self.clip.embed_text(text)

        assert not (self.condition_on_text_encodings and not exists(text_encodings)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text_encodings and exists(text_encodings)), 'decoder specified not to be conditioned on text, yet it is presented'

        assert not (exists(inpaint_image) ^ exists(inpaint_mask)), 'inpaint_image and inpaint_mask (boolean mask of [batch, height, width]) must be both given for inpainting'

        img = None
        if start_at_unet_number > 1:
            # Then we are not generating the first image and one must have been passed in
            assert exists(image), 'image must be passed in if starting at unet number > 1'
            assert image.shape[0] == batch_size, 'image must have batch size of {} if starting at unet number > 1'.format(batch_size)
            prev_unet_output_size = self.image_sizes[start_at_unet_number - 2]
            img = resize_image_to(image, prev_unet_output_size, nearest = True)
        is_cuda = next(self.parameters()).is_cuda

        num_unets = self.num_unets
        cond_scale = cast_tuple(cond_scale, num_unets)

        for unet_number, unet, vae, channel, image_size, predict_x_start, learned_variance, noise_scheduler, lowres_cond, sample_timesteps, unet_cond_scale in tqdm(zip(range(1, num_unets + 1), self.unets, self.vaes, self.sample_channels, self.image_sizes, self.predict_x_start, self.learned_variance, self.noise_schedulers, self.lowres_conds, self.sample_timesteps, cond_scale)):
            if unet_number < start_at_unet_number:
                continue  # It's the easiest way to do it

            context = self.one_unet_in_gpu(unet = unet) if is_cuda else null_context()

            with context:
                # prepare low resolution conditioning for upsamplers

                lowres_cond_img = lowres_noise_level = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_cond_img = resize_image_to(img, target_image_size = image_size, clamp_range = self.input_image_range, nearest = True)

                    if lowres_cond.use_noise:
                        lowres_noise_level = torch.full((batch_size,), int(self.lowres_noise_sample_level * 1000), dtype = torch.long, device = self.device)
                        lowres_cond_img, _ = lowres_cond.noise_image(lowres_cond_img, lowres_noise_level)

                # latent diffusion

                is_latent_diffusion = isinstance(vae, VQGanVAE)
                image_size = vae.get_encoded_fmap_size(image_size)
                shape = (batch_size, vae.encoded_dim, image_size, image_size)

                lowres_cond_img = maybe(vae.encode)(lowres_cond_img)

                # denoising loop for image

                img = self.p_sample_loop(
                    unet,
                    shape,
                    image_embed = image_embed,
                    text_encodings = text_encodings,
                    cond_scale = unet_cond_scale,
                    predict_x_start = predict_x_start,
                    learned_variance = learned_variance,
                    clip_denoised = not is_latent_diffusion,
                    lowres_cond_img = lowres_cond_img,
                    lowres_noise_level = lowres_noise_level,
                    is_latent_diffusion = is_latent_diffusion,
                    noise_scheduler = noise_scheduler,
                    timesteps = sample_timesteps,
                    inpaint_image = inpaint_image,
                    inpaint_mask = inpaint_mask
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
        unet_number = None,
        return_lowres_cond_image = False # whether to return the low resolution conditioning images, for debugging upsampler purposes
    ):
        assert not (self.num_unets > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {self.num_unets}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1

        unet = self.get_unet(unet_number)

        vae                 = self.vaes[unet_index]
        noise_scheduler     = self.noise_schedulers[unet_index]
        lowres_conditioner  = self.lowres_conds[unet_index]
        target_image_size   = self.image_sizes[unet_index]
        predict_x_start     = self.predict_x_start[unet_index]
        random_crop_size    = self.random_crop_sizes[unet_index]
        learned_variance    = self.learned_variance[unet_index]
        b, c, h, w, device, = *image.shape, image.device

        check_shape(image, 'b c h w', c = self.channels)
        assert h >= target_image_size and w >= target_image_size

        times = torch.randint(0, noise_scheduler.num_timesteps, (b,), device = device, dtype = torch.long)

        if not exists(image_embed) and not self.unconditional:
            assert exists(self.clip), 'if you want to derive CLIP image embeddings automatically, you must supply `clip` to the decoder on init'
            image_embed, _ = self.clip.embed_image(image)

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip), 'if you are passing in raw text, you need to supply `clip` to the decoder'
            _, text_encodings = self.clip.embed_text(text)

        assert not (self.condition_on_text_encodings and not exists(text_encodings)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text_encodings and exists(text_encodings)), 'decoder specified not to be conditioned on text, yet it is presented'

        lowres_cond_img, lowres_noise_level = lowres_conditioner(image, target_image_size = target_image_size, downsample_image_size = self.image_sizes[unet_index - 1]) if exists(lowres_conditioner) else (None, None)
        image = resize_image_to(image, target_image_size, nearest = True)

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

        losses = self.p_losses(unet, image, times, image_embed = image_embed, text_encodings = text_encodings, lowres_cond_img = lowres_cond_img, predict_x_start = predict_x_start, learned_variance = learned_variance, is_latent_diffusion = is_latent_diffusion, noise_scheduler = noise_scheduler, lowres_noise_level = lowres_noise_level)

        if not return_lowres_cond_image:
            return losses

        return losses, lowres_cond_img

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
            return first(images)

        return images
