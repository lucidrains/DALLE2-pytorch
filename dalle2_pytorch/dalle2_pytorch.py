import math
from tqdm import tqdm
from inspect import isfunction
from functools import partial
from contextlib import contextmanager
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom

from kornia.filters import gaussian_blur2d

from dalle2_pytorch.tokenizer import tokenizer
from dalle2_pytorch.vqgan_vae import NullVQGanVAE, VQGanVAE

# use x-clip

from x_clip import CLIP

# helper functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

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

def l2norm(t):
    return F.normalize(t, dim = -1)

def resize_image_to(t, image_size, mode = 'bilinear'): # take a look at https://github.com/assafshocher/ResizeRight
    shape = cast_tuple(image_size, 2)
    orig_image_size = t.shape[-2:]

    if orig_image_size == shape:
        return t

    return F.interpolate(t, size = shape, mode = mode, align_corners = False)

# image normalization functions
# ddpms expect images to be in the range of -1 to 1
# but CLIP may otherwise

def normalize_img(img):
    return img * 2 - 1

def unnormalize_img(normed_img):
    return (normed_img + 1) * 0.5

# clip related adapters

EmbeddedText = namedtuple('EmbedTextReturn', ['text_embed', 'text_encodings', 'text_mask'])
EmbeddedImage = namedtuple('EmbedImageReturn', ['image_embed', 'image_encodings'])

class BaseClipAdapter(nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.clip = clip

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
        return EmbeddedText(text_embed.float(), text_encodings.float(), text_mask)

    @torch.no_grad()
    def embed_image(self, image):
        assert not self.cleared
        image = resize_image_to(image, self.image_size)
        image = self.clip_normalize(unnormalize_img(image))
        image_embed = self.clip.encode_image(image)
        return EmbeddedImage(image_embed.float(), None)

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

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start**2, beta_end**2, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(-6, 6, timesteps)
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

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

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

def FeedForward(dim, mult = 4, dropout = 0., post_activation_norm = False):
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
        causal = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.causal = causal
        self.norm = LayerNorm(dim)
        self.post_norm = LayerNorm(dim)     # sandwich norm from Coqview paper + Normformer
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 d', b = b)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q = q * self.scale

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

        sim = sim - sim.amax(dim = -1, keepdim = True)
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.post_norm(out)

class CausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_out = False,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True
    ):
        super().__init__()
        self.rel_pos_bias = RelPosBias(heads = heads)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
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
        l2norm_output = False,  # whether to restrict image embedding output with l2norm at the end (may make it easier to learn?)
        **kwargs
    ):
        super().__init__()
        self.time_embeddings = nn.Embedding(num_timesteps, dim) if exists(num_timesteps) else nn.Sequential(Rearrange('b -> b 1'), MLP(1, dim)) # also offer a continuous version of timestep embeddings, with a 2 layer MLP
        self.learned_query = nn.Parameter(torch.randn(dim))
        self.causal_transformer = CausalTransformer(dim = dim, **kwargs)
        self.l2norm_output = l2norm_output

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
        cond_drop_prob = 0.2
    ):
        batch, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype

        # in section 2.2, last paragraph
        # "... consisting of encoded text, CLIP text embedding, diffusion timestep embedding, noised CLIP image embedding, final embedding for prediction"

        text_embed, image_embed = rearrange_many((text_embed, image_embed), 'b d -> b 1 d')

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

        mask = torch.cat((mask, keep_mask), dim = 1)

        # whether text embedding is used for conditioning depends on whether text encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right

        if exists(mask):
            mask = F.pad(mask, (0, 2), value = True) # extend mask for text embedding, noised image embedding, time step embedding, and learned query

        time_embed = self.time_embeddings(diffusion_timesteps)
        time_embed = rearrange(time_embed, 'b d -> b 1 d')

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

        tokens = torch.cat((
            text_encodings,
            text_embed,
            time_embed,
            learned_queries
        ), dim = -2)

        # attend

        tokens = self.causal_transformer(tokens, mask = mask)

        # get learned query, which should predict the image embedding (per DDPM timestep)

        pred_image_embed = tokens[..., -1, :]

        output_fn = l2norm if self.l2norm_output else identity
        return output_fn(pred_image_embed)

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
        cond_drop_prob = 0.2,
        loss_type = "l1",
        predict_x_start = True,
        beta_schedule = "cosine",
        condition_on_text_encodings = True, # the paper suggests this is needed, but you can turn it off for your CLIP preprocessed text embed -> image embed training
        sampling_clamp_l2norm = False
    ):
        super().__init__(
            beta_schedule = beta_schedule,
            timesteps = timesteps,
            loss_type = loss_type
        )

        if exists(clip):
            if isinstance(clip, CLIP):
                clip = XClipAdapter(clip)

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
        self.condition_on_text_encodings = condition_on_text_encodings

        self.predict_x_start = predict_x_start
        # in paper, they do not predict the noise, but predict x0 directly for image embedding, claiming empirically better results. I'll just offer both.

        # whether to force an l2norm, similar to clipping denoised, when sampling
        self.sampling_clamp_l2norm = sampling_clamp_l2norm

    def p_mean_variance(self, x, t, text_cond, clip_denoised: bool):
        pred = self.net(x, t, **text_cond)

        if self.predict_x_start:
            x_recon = pred
            # not 100% sure of this above line - for any spectators, let me know in the github issues (or through a pull request) if you know how to correctly do this
            # i'll be rereading https://arxiv.org/abs/2111.14822, where i think a similar approach is taken
        else:
            x_recon = self.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised and not self.predict_x_start:
            x_recon.clamp_(-1., 1.)

        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_recon = l2norm(x_recon)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, text_cond = None, clip_denoised = True, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, text_cond = text_cond, clip_denoised = clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, text_cond):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device = device, dtype = torch.long), text_cond = text_cond)
        return img

    def p_losses(self, image_embed, times, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.q_sample(x_start = image_embed, t = times, noise = noise)

        pred = self.net(
            image_embed_noisy,
            times,
            cond_drop_prob = self.cond_drop_prob,
            **text_cond
        )

        target = noise if not self.predict_x_start else image_embed

        loss = self.loss_fn(pred, target)
        return loss

    @torch.no_grad()
    @eval_decorator
    def sample(self, text, num_samples_per_batch = 2):
        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, 'b ... -> (b r) ...', r = num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        text_embed, text_encodings, text_mask = self.clip.embed_text(text)

        text_cond = dict(text_embed = text_embed)

        if self.condition_on_text_encodings:
            text_cond = {**text_cond, 'text_encodings': text_encodings, 'mask': text_mask}

        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond = text_cond)
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

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim = None,
        time_cond_dim = None,
        mult = 2,
        norm = True
    ):
        super().__init__()
        need_projection = dim != dim_out

        self.cross_attn = None

        if exists(cond_dim):
            self.cross_attn = EinopsToAndFrom(
                'b c h w',
                'b (h w) c',
                CrossAttention(
                    dim = dim,
                    context_dim = cond_dim
                )
            )

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_cond_dim, dim)
            )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        inner_dim = int(dim_out * mult)
        self.net = nn.Sequential(
            ChanLayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, inner_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(inner_dim, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if need_projection else nn.Identity()

    def forward(self, x, cond = None, time = None):
        h = self.ds_conv(x)

        if exists(time) and exists(self.time_mlp):
            t = self.time_mlp(time)
            h = rearrange(t, 'b c -> b c 1 1') + h

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context = cond) + h

        h = self.net(h)

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
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

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

        sim = sim - sim.amax(dim = -1, keepdim = True)
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
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = False)

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

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_embed_dim,
        text_embed_dim = None,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_dim_head = 32,
        attn_heads = 16,
        lowres_cond = False, # for cascading diffusion - https://cascaded-diffusion.github.io/
        sparse_attn = False,
        attend_at_middle = True, # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        cond_on_text_encodings = False,
        max_text_len = 256,
        cond_on_image_embeds = False,
        init_dim = None,
        init_conv_kernel_size = 7
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

        init_channels = channels if not lowres_cond else channels * 2 # in cascading diffusion, one concats the low resolution image, blurred, for conditioning the higher resolution synthesis
        init_dim = default(init_dim, dim // 2)

        assert (init_conv_kernel_size % 2) == 1
        self.init_conv = nn.Conv2d(init_channels, init_dim, init_conv_kernel_size, padding = init_conv_kernel_size // 2)

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
        ) if image_embed_dim != cond_dim else nn.Identity()

        # text encoding conditioning (optional)

        self.text_to_cond = None

        if cond_on_text_encodings:
            self.text_to_cond = nn.LazyLinear(cond_dim) if not exists(text_embed_dim) else nn.Linear(text_embed_dim, cond_dim)

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

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_first = ind == 0
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if not is_first else None

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_cond_dim = time_cond_dim, norm = ind != 0),
                Residual(LinearAttention(dim_out, **attn_kwargs)) if sparse_attn else nn.Identity(),
                ConvNextBlock(dim_out, dim_out, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]

        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim)
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c', Residual(Attention(mid_dim, **attn_kwargs))) if attend_at_middle else None
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, cond_dim = cond_dim, time_cond_dim = time_cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 2)
            layer_cond_dim = cond_dim if not is_last else None

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim),
                Residual(LinearAttention(dim_in, **attn_kwargs)) if sparse_attn else nn.Identity(),
                ConvNextBlock(dim_in, dim_in, cond_dim = layer_cond_dim, time_cond_dim = time_cond_dim),
                Upsample(dim_in)
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        channels,
        cond_on_image_embeds
    ):
        if lowres_cond == self.lowres_cond and channels == self.channels and cond_on_image_embeds == self.cond_on_image_embeds:
            return self

        updated_kwargs = {'lowres_cond': lowres_cond, 'channels': channels, 'cond_on_image_embeds': cond_on_image_embeds}
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

            image_tokens = torch.where(
                image_keep_mask,
                image_tokens,
                self.null_image_embed
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

            text_tokens = torch.where(
                text_keep_mask,
                text_tokens,
                self.null_text_embed
            )

        # main conditioning tokens (c)

        c = time_tokens

        if exists(image_tokens):
            c = torch.cat((c, image_tokens), dim = -2)

        # text and image conditioning tokens (mid_c)
        # to save on compute, only do cross attention based conditioning on the inner most layers of the Unet

        mid_c = c if not exists(text_tokens) else torch.cat((c, text_tokens), dim = -2)

        # go through the layers of the unet, down and up

        hiddens = []

        for convnext, sparse_attn, convnext2, downsample in self.downs:
            x = convnext(x, c, t)
            x = sparse_attn(x)
            x = convnext2(x, c, t)
            hiddens.append(x)
            x = downsample(x)

        x = self.mid_block1(x, mid_c, t)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, mid_c, t)

        for convnext, sparse_attn, convnext2, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = convnext(x, c, t)
            x = sparse_attn(x)
            x = convnext2(x, c, t)
            x = upsample(x)

        return self.final_conv(x)

class LowresConditioner(nn.Module):
    def __init__(
        self,
        cond_upsample_mode = 'bilinear',
        downsample_first = True,
        blur_sigma = 0.1,
        blur_kernel_size = 3,
    ):
        super().__init__()
        self.cond_upsample_mode = cond_upsample_mode
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
        target_image_size = cast_tuple(target_image_size, 2)

        if self.training and self.downsample_first and exists(downsample_image_size):
            cond_fmap = resize_image_to(cond_fmap, downsample_image_size, mode = self.cond_upsample_mode)

        if self.training:
            # when training, blur the low resolution conditional image
            blur_sigma = default(blur_sigma, self.blur_sigma)
            blur_kernel_size = default(blur_kernel_size, self.blur_kernel_size)
            cond_fmap = gaussian_blur2d(cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2))

        cond_fmap = resize_image_to(cond_fmap, target_image_size, mode = self.cond_upsample_mode)

        return cond_fmap

class Decoder(BaseGaussianDiffusion):
    def __init__(
        self,
        unet,
        *,
        clip,
        vae = tuple(),
        timesteps = 1000,
        image_cond_drop_prob = 0.1,
        text_cond_drop_prob = 0.5,
        loss_type = 'l1',
        beta_schedule = 'cosine',
        predict_x_start = False,
        predict_x_start_for_latent_diffusion = False,
        image_sizes = None,                         # for cascading ddpm, image size at each stage
        lowres_cond_upsample_mode = 'bilinear',     # cascading ddpm - low resolution upsample mode
        lowres_downsample_first = True,             # cascading ddpm - resizes to lower resolution, then to next conditional resolution + blur
        blur_sigma = 0.1,                           # cascading ddpm - blur sigma
        blur_kernel_size = 3,                       # cascading ddpm - blur kernel size
        condition_on_text_encodings = False,        # the paper suggested that this didn't do much in the decoder, but i'm allowing the option for experimentation
        clip_denoised = True,
        clip_x_start = True
    ):
        super().__init__(
            beta_schedule = beta_schedule,
            timesteps = timesteps,
            loss_type = loss_type
        )

        if isinstance(clip, CLIP):
            clip = XClipAdapter(clip)

        freeze_model_and_make_eval_(clip)
        assert isinstance(clip, BaseClipAdapter)

        self.clip = clip
        self.clip_image_size = clip.image_size
        self.channels = clip.image_channels

        self.condition_on_text_encodings = condition_on_text_encodings

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution image produced by previous unet

        unets = cast_tuple(unet)
        vaes = pad_tuple_to_length(cast_tuple(vae), len(unets), fillvalue = NullVQGanVAE(channels = self.channels))

        self.unets = nn.ModuleList([])
        self.vaes = nn.ModuleList([])

        for ind, (one_unet, one_vae) in enumerate(zip(unets, vaes)):
            assert isinstance(one_unet, Unet)
            assert isinstance(one_vae, (VQGanVAE, NullVQGanVAE))

            is_first = ind == 0
            latent_dim = one_vae.encoded_dim if exists(one_vae) else None

            unet_channels = default(latent_dim, self.channels)

            one_unet = one_unet.cast_model_parameters(
                lowres_cond = not is_first,
                cond_on_image_embeds = is_first,
                channels = unet_channels
            )

            self.unets.append(one_unet)
            self.vaes.append(one_vae.copy_for_eval())

        # unet image sizes

        image_sizes = default(image_sizes, (clip.image_size,))
        image_sizes = tuple(sorted(set(image_sizes)))

        assert len(self.unets) == len(image_sizes), f'you did not supply the correct number of u-nets ({len(self.unets)}) for resolutions {image_sizes}'
        self.image_sizes = image_sizes
        self.sample_channels = cast_tuple(self.channels, len(image_sizes))

        # predict x0 config

        self.predict_x_start = cast_tuple(predict_x_start, len(unets)) if not predict_x_start_for_latent_diffusion else tuple(map(lambda t: isinstance(t, VQGanVAE), self.vaes))

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (len(self.unets) - 1))), 'the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True'

        self.to_lowres_cond = LowresConditioner(
            cond_upsample_mode = lowres_cond_upsample_mode,
            downsample_first = lowres_downsample_first,
            blur_sigma = blur_sigma,
            blur_kernel_size = blur_kernel_size,
        )

        # classifier free guidance

        self.image_cond_drop_prob = image_cond_drop_prob
        self.text_cond_drop_prob = text_cond_drop_prob

        # whether to clip when sampling

        self.clip_denoised = clip_denoised
        self.clip_x_start = clip_x_start

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
        self.unets.cpu()

        unet.cuda()
        yield
        unet.cpu()


    @torch.no_grad()
    def get_image_embed(self, image):
        image_embed, _ = self.clip.embed_image(image)
        return image_embed

    def p_mean_variance(self, unet, x, t, image_embed, text_encodings = None, text_mask = None, lowres_cond_img = None, clip_denoised = True, predict_x_start = False, cond_scale = 1.):
        pred = unet.forward_with_cond_scale(x, t, image_embed = image_embed, text_encodings = text_encodings, text_mask = text_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img)

        if predict_x_start:
            x_recon = pred
        else:
            x_recon = self.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, unet, x, t, image_embed, text_encodings = None, text_mask = None, cond_scale = 1., lowres_cond_img = None, predict_x_start = False, clip_denoised = True, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x, t = t, image_embed = image_embed, text_encodings = text_encodings, text_mask = text_mask, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, clip_denoised = clip_denoised, predict_x_start = predict_x_start)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, unet, shape, image_embed, predict_x_start = False, clip_denoised = True, lowres_cond_img = None, text_encodings = None, text_mask = None, cond_scale = 1):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device = device)

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
                clip_denoised = clip_denoised
            )

        return img

    def p_losses(self, unet, x_start, times, *, image_embed, lowres_cond_img = None, text_encodings = None, text_mask = None, predict_x_start = False, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start = x_start, t = times, noise = noise)

        pred = unet(
            x_noisy,
            times,
            image_embed = image_embed,
            text_encodings = text_encodings,
            text_mask = text_mask,
            lowres_cond_img = lowres_cond_img,
            image_cond_drop_prob = self.image_cond_drop_prob,
            text_cond_drop_prob = self.text_cond_drop_prob,
        )

        target = noise if not predict_x_start else x_start

        loss = self.loss_fn(pred, target)
        return loss

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        image_embed,
        text = None,
        cond_scale = 1.,
        stop_at_unet_number = None
    ):
        batch_size = image_embed.shape[0]

        text_encodings = text_mask = None
        if exists(text):
            _, text_encodings, text_mask = self.clip.embed_text(text)

        assert not (self.condition_on_text_encodings and not exists(text_encodings)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text_encodings and exists(text_encodings)), 'decoder specified not to be conditioned on text, yet it is presented'

        img = None

        for unet_number, unet, vae, channel, image_size, predict_x_start in tqdm(zip(range(1, len(self.unets) + 1), self.unets, self.vaes, self.sample_channels, self.image_sizes, self.predict_x_start)):

            context = self.one_unet_in_gpu(unet = unet) if image_embed.is_cuda else null_context()

            with context:
                lowres_cond_img = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_cond_img = self.to_lowres_cond(img, target_image_size = image_size)

                is_latent_diffusion = isinstance(vae, VQGanVAE)
                image_size = vae.get_encoded_fmap_size(image_size)
                shape = (batch_size, vae.encoded_dim, image_size, image_size)

                if exists(lowres_cond_img):
                    lowres_cond_img = vae.encode(lowres_cond_img)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    image_embed = image_embed,
                    text_encodings = text_encodings,
                    text_mask = text_mask,
                    cond_scale = cond_scale,
                    predict_x_start = predict_x_start,
                    clip_denoised = not is_latent_diffusion,
                    lowres_cond_img = lowres_cond_img
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
        unet_number = None
    ):
        assert not (len(self.unets) > 1 and not exists(unet_number)), f'you must specify which unet you want trained, from a range of 1 to {len(self.unets)}, if you are training cascading DDPM (multiple unets)'
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1

        unet = self.get_unet(unet_number)

        target_image_size = self.image_sizes[unet_index]
        vae = self.vaes[unet_index]
        predict_x_start = self.predict_x_start[unet_index]

        b, c, h, w, device, = *image.shape, image.device

        check_shape(image, 'b c h w', c = self.channels)
        assert h >= target_image_size and w >= target_image_size

        times = torch.randint(0, self.num_timesteps, (b,), device = device, dtype = torch.long)

        if not exists(image_embed):
            image_embed, _ = self.clip.embed_image(image)

        text_encodings = text_mask = None
        if exists(text) and not exists(text_encodings):
            _, text_encodings, text_mask = self.clip.embed_text(text)

        assert not (self.condition_on_text_encodings and not exists(text_encodings)), 'text or text encodings must be passed into decoder if specified'
        assert not (not self.condition_on_text_encodings and exists(text_encodings)), 'decoder specified not to be conditioned on text, yet it is presented'

        lowres_cond_img = self.to_lowres_cond(image, target_image_size = target_image_size, downsample_image_size = self.image_sizes[unet_index - 1]) if unet_number > 1 else None
        image = resize_image_to(image, target_image_size)

        vae.eval()
        with torch.no_grad():
            image = vae.encode(image)

            if exists(lowres_cond_img):
                lowres_cond_img = vae.encode(lowres_cond_img)

        return self.p_losses(unet, image, times, image_embed = image_embed, text_encodings = text_encodings, text_mask = text_mask, lowres_cond_img = lowres_cond_img, predict_x_start = predict_x_start)

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
        return_pil_images = False
    ):
        device = next(self.parameters()).device
        one_text = isinstance(text, str) or (not is_list_str(text) and text.shape[0] == 1)

        if isinstance(text, str) or is_list_str(text):
            text = [text] if not isinstance(text, (list, tuple)) else text
            text = tokenizer.tokenize(text).to(device)

        image_embed = self.prior.sample(text, num_samples_per_batch = self.prior_num_samples)

        text_cond = text if self.decoder_need_text_cond else None
        images = self.decoder.sample(image_embed, text = text_cond, cond_scale = cond_scale)

        if return_pil_images:
            images = list(map(self.to_pil, images.unbind(dim = 0)))

        if one_text:
            return images[0]

        return images
