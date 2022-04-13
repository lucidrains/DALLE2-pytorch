import tqdm
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange
from einops_exts import rearrange_many, repeat_many

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
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# diffusion prior

class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        squared_sum = (x ** 2).sum(dim = -1, keepdim = True)
        inv_norm = torch.rsqrt(squared_sum + self.eps)
        return x * inv_norm * self.gamma * self.scale

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(mult * dim)
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(heads, 2, dim_head))
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask = None):
        b, n, device = x.shape[:2], x.device

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = rearrange_many(qkv, 'b n (h d) -> b h n d')

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'h d -> b h 1 d', b = b)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j')
        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
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
        norm_out = False,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        # todo - bring in rotary embeddings or alibi

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options

    def forward(
        self,
        x,
        mask = None    # we will need a mask here, due to variable length of the text encodings - also offer dalle1 strategy with padding token embeddings
    ):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.norm(x)

class DiffusionPriorNetwork(nn.Module):
    def __init__(
        self,
        dim,
        num_timesteps = 1000,
        **kwargs
    ):
        super().__init__()
        self.time_embeddings = nn.Embedding(num_timesteps, dim)  # also offer a continuous version of timestep embeddings, with a 2 layer MLP
        self.learned_query = nn.Parameter(torch.randn(dim))
        self.causal_transformer = Transformer(**kwargs)

    def forward_with_cond_scale(
        self,
        x,
        *,
        cond_scale = 1.,
        **kwargs
    ):
        if cond_scale == 1:
            return self.forward(x, **kwargs)

        logits = self.forward(x, **kwargs)
        null_logits = self.forward(x, cond_prob_drop = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        image_embed,
        *,
        diffusion_timesteps,
        text_encodings,
        text_embed,
        mask = None,
        cond_drop_prob = 0.2
    ):
        batch, text_enc_len, device = image_embed.shape[0], text_encodings.shape[-2], image_embed.device

        # in section 2.2, last paragraph
        # "... consisting of encoded text, CLIP text embedding, diffusion timestep embedding, noised CLIP image embedding, final embedding for prediction"

        text_embed, image_embed = rearrange_many((text_embed, image_embed), 'b d -> b 1 d')

        if exists(mask):
            mask = F.pad(mask, (0, 4), value = True) # extend mask for text embedding, noised image embedding, time step embedding, and learned query

        time_embed = self.time_embeddings(diffusion_timesteps)

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

        tokens = torch.cat((
            text_encodings,
            text_embed,
            time_embed,
            learned_queries
        ), dim = -2)

        # mask if it doesn't exist

        if not exists(mask):
            mask = torch.ones((batch, text_enc_len), device = device, dtype = torch.bool)

        # classifier free guidance

        cond_prob_mask = prob_mask_like((batch_size,), cond_prob_drop, device = device)
        mask &= rearrange(cond_prob_mask, 'b -> b 1')

        # attend

        tokens = self.causal_transformer(tokens, mask = mask)

        # get learned query, which should predict the image embedding (per DDPM timestep)

        pred_image_embed = tokens[..., -1, :]

        return pred_image_embed

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
        mult = 2,
        norm = True
    ):
        super().__init__()
        need_projection = dim != dim_out

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(cond_dim, dim)
        ) if exists(cond_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        inner_dim = int(dim_out * mult)
        self.net = nn.Sequential(
            RMSNorm(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, inner_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(inner_dim, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if need_projection else nn.Identity()

    def forward(self, x, cond = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(cond)
            condition = self.mlp(cond)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        image_embed_dim,
        time_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = default(time_dim, dim)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.null_image_embed = nn.Parameter(torch.randn(image_embed_dim))

        cond_dim = time_dim + image_embed_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, cond_dim = cond_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, cond_dim = cond_dim),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block = ConvNextBlock(mid_dim, mid_dim, cond_dim = cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, cond_dim = cond_dim),
                ConvNextBlock(dim_in, dim_in, cond_dim = cond_dim),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward_with_cond_scale(
        self,
        x,
        *,
        cond_scale = 1.,
        **kwargs
    ):
        if cond_scale == 1:
            return self.forward(x, **kwargs)

        logits = self.forward(x, **kwargs)
        null_logits = self.forward(x, cond_prob_drop = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        *,
        image_embed,
        time,
        text_encodings = None,
        cond_prob_drop = 0.
    ):
        t = self.time_mlp(time)

        cond_prob_mask = prob_mask_like((batch_size,), cond_prob_drop, device = device)

        # mask out image embedding depending on condition dropout
        # for classifier free guidance

        image_embed = torch.where(
            rearrange(cond_prob_mask, 'b -> b 1'),
            image_embed,
            rearrange(self.null_image_embed, 'd -> 1 d')
        )

        cond = torch.cat((t, image_embed), dim = -1)

        hiddens = []

        for convnext, convnext2, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            hiddens.append(x)
            x = downsample(x)

        x = self.mid_block(x, t)

        for convnext, convnext2, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = upsample(x)

        return self.final_conv(x)

class Decoder(nn.Module):
    def __init__(
        self,
        net,
        *,
        clip,
        timesteps = 1000,
        cond_prob_drop = 0.2,
        loss_type = 'l1'
    ):
        super().__init__()
        assert isinstance(clip, CLIP)
        freeze_model_and_make_eval_(clip)

        self.net = net
        self.channels = clip.image_channels
        self.image_size = clip.image_size
        self.cond_prob_drop = cond_prob_drop

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (0, 1), value = 1.)

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

    def get_image_embed(self, image):
        image_encoding = self.clip.visual_transformer(image)
        image_cls = image_encoding[:, 0]
        image_embed = self.clip.to_visual_latent(image_cls)
        return image_embed

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

    def p_mean_variance(self, x, t, image_embed, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t = t, noise = self.net(x, t, image_embed = image_embed))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, image_embed, clip_denoised = True, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, image_embed):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device = device, dtype = torch.long), image_embed = image_embed)
        return img

    @torch.no_grad()
    def sample(self, image_embed):
        batch_size = image_embed.shape[0]
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), image_embed = image_embed)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, image_embed, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start = x_start, t = t, noise = noise)

        x_recon = self.net(
            x_noisy,
            t,
            image_embed = image_embed,
            cond_prob_drop = self.cond_prob_drop
        )

        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, image, *args, **kwargs):
        b, device, img_size, = image.shape[0], image.device, self.image_size
        check_shape(image, 'b c h w', h = img_size, w = img_size, c = self.channels)

        times = torch.randint(0, self.num_timesteps, (b,), device = device, dtype = torch.long)
        image_embed = self.get_image_embed(image)

        loss = self.p_losses(x, times, image_embed = image_embed, *args, **kwargs)
        return loss

# main class

class DALLE2(nn.Module):
    def __init__(
        self,
        *,
        prior,
        decoder,
        tokenizer = None
    ):
        super().__init__()
        assert isinstance(prior), DiffusionPrior
        assert isinstance(decoder), Decoder
        self.tokenizer = tokenizer

    @torch.no_grad()
    def forward(
        self,
        *,
        text
    ):
        if isinstance(text, str):
            assert exists(self.tokenizer), 'tokenizer must be passed in if you were to pass in the text as a string'
            text = self.tokenizer.encode(text)

        image_embed = prior.sample(text, num_samples_per_batch = 2)
        images = decoder.sample(image_embed)
        return images
