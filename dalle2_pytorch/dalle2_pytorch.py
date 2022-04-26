import math
from tqdm import tqdm
from inspect import isfunction
from functools import partial
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape
from einops_exts.torch import EinopsToAndFrom

from kornia.filters import gaussian_blur2d

from dalle2_pytorch.tokenizer import tokenizer
from dalle2_pytorch.vqgan_vae import NullVQGanVAE, VQGanVAE
from dalle2_pytorch.attention import QueryAttnUpsample

# use x-clip

from x_clip import CLIP

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

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

    return F.interpolate(t, size = shape, mode = mode)

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
        **kwargs
    ):
        super().__init__()
        self.time_embeddings = nn.Embedding(num_timesteps, dim) if exists(num_timesteps) else nn.Sequential(Rearrange('b -> b 1'), MLP(1, dim)) # also offer a continuous version of timestep embeddings, with a 2 layer MLP
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
        text_encodings,
        text_embed,
        mask=None,
        cond_drop_prob=0.1
    ):
        """
        Complete one forward pass through the diffusion prior network.

        If the dictionary passed from the DiffusionPrior wrapper has the 
        key:value pair text_encodings:None, then we will assume
        we are doing "embedding only" training.

        In the case of embedding only training we will drop the text
        encoding from the tokens entirely and train on the remaining
        tokens mentioned in section 2.2 of the DALLE2 paper.
        """

        # grab some contextual information
        batch, device = image_embed.shape[0], image_embed.device

        # gather tokens for casual transformer
        text_embed, image_embed = rearrange_many(
            (text_embed, image_embed), 'b d -> b 1 d')
        time_embed = self.time_embeddings(diffusion_timesteps)
        time_embed = rearrange(time_embed, 'b d -> b 1 d')
        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b=batch)

        # check for embedding only training
        if text_encodings is not None:
            text_enc_len = text_encodings.shape[-2]

            # text conditioning can be passed during training for CFG; as described in the paper

            if exists(mask):
                not_all_masked_out = mask.any(dim=-1)
                mask = torch.cat(
                    (mask, rearrange(not_all_masked_out, 'b -> b 1')), dim=1)
                # extend mask for text embedding, noised image embedding, etc.
                mask = F.pad(mask, (0, 2), value=True)

            if not exists(mask):
                mask = torch.ones((batch, text_enc_len),
                                  device=device, dtype=torch.bool)

            # classifier free guidance
            cond_prob_mask = prob_mask_like(
                (batch,), cond_drop_prob, device=device)
            mask &= rearrange(cond_prob_mask, 'b -> b 1')

            # actually combine everything as described originally
            tokens = torch.cat((
                text_encodings,
                text_embed,
                time_embed,
                learned_queries
            ), dim=-2)

        # TODO: someone should double check this block of code below
        else:
            # during embedding only, just drop text_encodings entirely
            tokens = torch.cat((
                text_embed,
                time_embed,
                learned_queries
            ), dim=-2)

            # make a mask of all one's since we don't do CFG
            mask = torch.ones(
                (batch, tokens.shape[1]), dtype=torch.bool, device=device)

        tokens = self.causal_transformer(tokens, mask=mask)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -1, :]

        return pred_image_embed

class DiffusionPrior(nn.Module):
    def __init__(
        self,
        net,
        *,
        clip,
        timesteps = 1000,
        cond_drop_prob = 0.2,
        loss_type = "l1",
        predict_x_start = True,
        beta_schedule = "cosine",
    ):
        super().__init__()

        # add ability for clip to be None (aka training on embeddings only)
        if clip is not None:
            assert isinstance(clip, CLIP)
            freeze_model_and_make_eval_(clip)
            
            self.clip = clip
            self.channels = clip.image_channels
            self.image_size = clip.image_size
            self.image_embed_dim = clip.dim_latent
            # conditional dropping only happens with text_cond
        
        self.cond_drop_prob = cond_drop_prob

        # store diffusion prior network
        self.net = net

        self.predict_x_start = predict_x_start
        # in paper, they do not predict the noise, but predict x0 directly for image embedding, claiming empirically better results. I'll just offer both.

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
        self.loss_type = loss_type

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

    @torch.no_grad()
    def get_image_embed(self, image):
        image_encoding = self.clip.visual_transformer(image)
        image_cls = image_encoding[:, 0]
        image_embed = self.clip.to_visual_latent(image_cls)
        return l2norm(image_embed)

    @torch.no_grad()
    def get_text_cond(self, text):
        text_encodings = self.clip.text_transformer(text)
        text_cls, text_encodings = text_encodings[:, 0], text_encodings[:, 1:]
        text_embed = self.clip.to_text_latent(text_cls)
        text_embed = l2norm(text_embed)
        return dict(text_encodings = text_encodings, text_embed = text_embed, mask = text != 0)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

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

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, image_embed, t, text_cond, noise = None):
        noise = default(noise, lambda: torch.randn_like(image_embed))

        image_embed_noisy = self.q_sample(x_start = image_embed, t = t, noise = noise)

        # pass noisy image to diffusion network
        x_recon = self.net(
            image_embed_noisy,
            t,
            cond_drop_prob = self.cond_drop_prob,
            **text_cond
        )

        to_predict = noise if not self.predict_x_start else image_embed

        if self.loss_type == 'l1':
            loss = F.l1_loss(to_predict, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(to_predict, x_recon)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(to_predict, x_recon)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    @eval_decorator
    def sample(self, text, num_samples_per_batch = 2):
        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, 'b ... -> (b r) ...', r = num_samples_per_batch)

        batch_size = text.shape[0]
        image_embed_dim = self.image_embed_dim

        text_cond = self.get_text_cond(text)

        image_embeds = self.p_sample_loop((batch_size, image_embed_dim), text_cond = text_cond)
        text_embeds = text_cond['text_embed']

        text_embeds = rearrange(text_embeds, '(b r) d -> b r d', r = num_samples_per_batch)
        image_embeds = rearrange(image_embeds, '(b r) d -> b r d', r = num_samples_per_batch)

        text_image_sims = einsum('b r d, b r d -> b r', l2norm(text_embeds), l2norm(image_embeds))
        top_sim_indices = text_image_sims.topk(k = 1).indices

        top_sim_indices = repeat(top_sim_indices, 'b 1 -> b 1 d', d = image_embed_dim)

        top_image_embeds = image_embeds.gather(1, top_sim_indices)
        return rearrange(top_image_embeds, 'b 1 d -> b d')

    def forward(self, text, image, text_embed=None, image_embed=None, *args, **kwargs):
        # check if we are doing embedding-only training
        if text is not None:
            b, device, img_size, = image.shape[0], image.device, self.image_size
            check_shape(image, 'b c h w', h=img_size,
                        w=img_size, c=self.channels)
            text_cond = self.get_text_cond(text)
        else:
            b, device = image_embed.shape[0], image_embed.device
            text_cond = {
                'text_encodings': None,
                'text_embed': text_embed,
                'mask': None
            }

        # TODO: might wanna refactor to avoid a self-assignment
        image_embed = self.get_image_embed(
            image) if image is not None else image_embed

        times = torch.randint(0, self.num_timesteps, (b,),
                              device=device, dtype=torch.long)

        loss = self.p_losses(image_embed, times,
                             text_cond=text_cond, *args, **kwargs)

        return loss

# decoder

def Upsample(dim):
    return QueryAttnUpsample(dim)

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

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        inner_dim = int(dim_out * mult)
        self.net = nn.Sequential(
            ChanLayerNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, inner_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(inner_dim, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if need_projection else nn.Identity()

    def forward(self, x, cond = None):
        h = self.ds_conv(x)

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

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_embed_dim,
        cond_dim = None,
        num_image_tokens = 4,
        num_time_tokens = 2,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        attn_dim_head = 32,
        attn_heads = 8,
        lowres_cond = False, # for cascading diffusion - https://cascaded-diffusion.github.io/
        sparse_attn = False,
        sparse_attn_window = 8,  # window size for sparse attention
        attend_at_middle = True, # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        cond_on_text_encodings = False,
        cond_on_image_embeds = False,
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

        dims = [init_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time, image embeddings, and optional text encoding

        cond_dim = default(cond_dim, dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        self.image_to_cond = nn.Sequential(
            nn.Linear(image_embed_dim, cond_dim * num_image_tokens),
            Rearrange('b (n d) -> b n d', n = num_image_tokens)
        ) if image_embed_dim != cond_dim else nn.Identity()

        self.text_to_cond = nn.LazyLinear(cond_dim)

        # finer control over whether to condition on image embeddings and text encodings
        # so one can have the latter unets in the cascading DDPMs only focus on super-resoluting

        self.cond_on_text_encodings = cond_on_text_encodings
        self.cond_on_image_embeds = cond_on_image_embeds

        # for classifier free guidance

        self.null_image_embed = nn.Parameter(torch.randn(1, num_image_tokens, cond_dim))
        self.null_text_embed = nn.Parameter(torch.randn(1, 1, cond_dim))

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
                ConvNextBlock(dim_in, dim_out, norm = ind != 0),
                Residual(GridAttention(dim_out, window_size = sparse_attn_window, **attn_kwargs)) if sparse_attn else nn.Identity(),
                ConvNextBlock(dim_out, dim_out, cond_dim = layer_cond_dim),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]

        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, cond_dim = cond_dim)
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c', Residual(Attention(mid_dim, **attn_kwargs))) if attend_at_middle else None
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, cond_dim = cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 2)
            layer_cond_dim = cond_dim if not is_last else None

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, cond_dim = layer_cond_dim),
                Residual(GridAttention(dim_in, window_size = sparse_attn_window, **attn_kwargs)) if sparse_attn else nn.Identity(),
                ConvNextBlock(dim_in, dim_in, cond_dim = layer_cond_dim),
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
        channels
    ):
        if lowres_cond == self.lowres_cond and channels == self.channels:
            return self

        updated_kwargs = {**self._locals, 'lowres_cond': lowres_cond, 'channels': channels}
        return self.__class__(**updated_kwargs)

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
        x,
        time,
        *,
        image_embed,
        lowres_cond_img = None,
        text_encodings = None,
        cond_drop_prob = 0.,
        blur_sigma = None,
        blur_kernel_size = None
    ):
        batch_size, device = x.shape[0], x.device

        # add low resolution conditioning, if present

        assert not (self.lowres_cond and not exists(lowres_cond_img)), 'low resolution conditioning image must be present'

        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim = 1)

        # time conditioning

        time_tokens = self.time_mlp(time)

        # conditional dropout

        cond_prob_mask = prob_mask_like((batch_size,), cond_drop_prob, device = device)
        cond_prob_mask = rearrange(cond_prob_mask, 'b -> b 1 1')

        # mask out image embedding depending on condition dropout
        # for classifier free guidance

        image_tokens = None

        if self.cond_on_image_embeds:
            image_tokens = self.image_to_cond(image_embed)

            image_tokens = torch.where(
                cond_prob_mask,
                image_tokens,
                self.null_image_embed
            )

        # take care of text encodings (optional)

        text_tokens = None

        if exists(text_encodings) and self.cond_on_text_encodings:
            text_tokens = self.text_to_cond(text_encodings)
            text_tokens = torch.where(
                cond_prob_mask,
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
            x = convnext(x, c)
            x = sparse_attn(x)
            x = convnext2(x, c)
            hiddens.append(x)
            x = downsample(x)

        x = self.mid_block1(x, mid_c)

        if exists(self.mid_attn):
            x = self.mid_attn(x)

        x = self.mid_block2(x, mid_c)

        for convnext, sparse_attn, convnext2, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = convnext(x, c)
            x = sparse_attn(x)
            x = convnext2(x, c)
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
            cond_fmap = resize_image_to(cond_fmap, target_image_size, mode = self.cond_upsample_mode)

        if self.training:
            # when training, blur the low resolution conditional image
            blur_sigma = default(blur_sigma, self.blur_sigma)
            blur_kernel_size = default(blur_kernel_size, self.blur_kernel_size)
            cond_fmap = gaussian_blur2d(cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2))

        cond_fmap = resize_image_to(cond_fmap, target_image_size, mode = self.cond_upsample_mode)

        return cond_fmap

class Decoder(nn.Module):
    def __init__(
        self,
        unet,
        *,
        clip,
        vae = None,
        timesteps = 1000,
        cond_drop_prob = 0.2,
        loss_type = 'l1',
        beta_schedule = 'cosine',
        predict_x_start = False,
        predict_x_start_for_latent_diffusion = False,
        image_sizes = None,                         # for cascading ddpm, image size at each stage
        lowres_cond_upsample_mode = 'bilinear',     # cascading ddpm - low resolution upsample mode
        lowres_downsample_first = True,             # cascading ddpm - resizes to lower resolution, then to next conditional resolution + blur
        blur_sigma = 0.1,                           # cascading ddpm - blur sigma
        blur_kernel_size = 3,                       # cascading ddpm - blur kernel size
    ):
        super().__init__()
        assert isinstance(clip, CLIP)
        freeze_model_and_make_eval_(clip)
        self.clip = clip
        self.clip_image_size = clip.image_size
        self.channels = clip.image_channels

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

        self.cond_drop_prob = cond_drop_prob

        # noise schedule

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
        self.loss_type = loss_type

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
    def get_text_encodings(self, text):
        text_encodings = self.clip.text_transformer(text)
        return text_encodings[:, 1:]

    @torch.no_grad()
    def get_image_embed(self, image):
        image = resize_image_to(image, self.clip_image_size)
        image_encoding = self.clip.visual_transformer(image)
        image_cls = image_encoding[:, 0]
        image_embed = self.clip.to_visual_latent(image_cls)
        return l2norm(image_embed)

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, unet, x, t, image_embed, text_encodings = None, lowres_cond_img = None, clip_denoised = True, predict_x_start = False, cond_scale = 1.):
        pred = unet.forward_with_cond_scale(x, t, image_embed = image_embed, text_encodings = text_encodings, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img)

        if predict_x_start:
            x_recon = pred
        else:
            x_recon = self.predict_start_from_noise(x, t = t, noise = pred)

        if clip_denoised and not predict_x_start:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, unet, x, t, image_embed, text_encodings = None, cond_scale = 1., lowres_cond_img = None, predict_x_start = False, clip_denoised = True, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x, t = t, image_embed = image_embed, text_encodings = text_encodings, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, clip_denoised = clip_denoised, predict_x_start = predict_x_start)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, unet, shape, image_embed, predict_x_start = False, lowres_cond_img = None, text_encodings = None, cond_scale = 1):
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
                cond_scale = cond_scale,
                lowres_cond_img = lowres_cond_img,
                predict_x_start = predict_x_start
            )

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, unet, x_start, t, *, image_embed, lowres_cond_img = None, text_encodings = None, predict_x_start = False, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start = x_start, t = t, noise = noise)

        x_recon = unet(
            x_noisy,
            t,
            image_embed = image_embed,
            text_encodings = text_encodings,
            lowres_cond_img = lowres_cond_img,
            cond_drop_prob = self.cond_drop_prob
        )

        target = noise if not predict_x_start else x_start

        if self.loss_type == 'l1':
            loss = F.l1_loss(target, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(target, x_recon)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(target, x_recon)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    @eval_decorator
    def sample(self, image_embed, text = None, cond_scale = 1.):
        batch_size = image_embed.shape[0]

        text_encodings = self.get_text_encodings(text) if exists(text) else None

        img = None

        for unet, vae, channel, image_size, predict_x_start in tqdm(zip(self.unets, self.vaes, self.sample_channels, self.image_sizes, self.predict_x_start)):
            with self.one_unet_in_gpu(unet = unet):
                lowres_cond_img = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_cond_img = self.to_lowres_cond(img, target_image_size = image_size)

                image_size = vae.get_encoded_fmap_size(image_size)
                shape = (batch_size, vae.encoded_dim, image_size, image_size)

                if exists(lowres_cond_img):
                    lowres_cond_img = vae.encode(lowres_cond_img)

                img = self.p_sample_loop(
                    unet,
                    shape,
                    image_embed = image_embed,
                    text_encodings = text_encodings,
                    cond_scale = cond_scale,
                    predict_x_start = predict_x_start,
                    lowres_cond_img = lowres_cond_img
                )

                img = vae.decode(img)

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
            image_embed = self.get_image_embed(image)

        text_encodings = self.get_text_encodings(text) if exists(text) and not exists(text_encodings) else None

        lowres_cond_img = self.to_lowres_cond(image, target_image_size = target_image_size, downsample_image_size = self.image_sizes[unet_index - 1]) if unet_number > 1 else None
        image = resize_image_to(image, target_image_size)

        vae.eval()
        with torch.no_grad():
            image = vae.encode(image)

            if exists(lowres_cond_img):
                lowres_cond_img = vae.encode(lowres_cond_img)

        return self.p_losses(unet, image, times, image_embed = image_embed, text_encodings = text_encodings, lowres_cond_img = lowres_cond_img, predict_x_start = predict_x_start)

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

    @torch.no_grad()
    @eval_decorator
    def forward(
        self,
        text,
        cond_scale = 1.
    ):
        device = next(self.parameters()).device
        one_text = isinstance(text, str) or (not is_list_str(text) and text.shape[0] == 1)

        if isinstance(text, str) or is_list_str(text):
            text = [text] if not isinstance(text, (list, tuple)) else text
            text = tokenizer.tokenize(text).to(device)

        image_embed = self.prior.sample(text, num_samples_per_batch = self.prior_num_samples)
        images = self.decoder.sample(image_embed, cond_scale = cond_scale)

        if one_text:
            return images[0]

        return images
