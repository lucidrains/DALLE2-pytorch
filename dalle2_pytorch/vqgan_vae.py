import copy
import math
from math import sqrt
from functools import partial, wraps

from vector_quantize_pytorch import VectorQuantize as VQ

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import torchvision

from einops import rearrange, reduce, repeat
from einops_exts import rearrange_many
from einops.layers.torch import Rearrange

# constants

MList = nn.ModuleList

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# decorators

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, string_input):
    return string_input.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# tensor helper functions

def log(t, eps = 1e-10):
    return torch.log(t + eps)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs = output, inputs = images,
                           grad_outputs = torch.ones(output.size(), device = images.device),
                           create_graph = True, retain_graph = True, only_inputs = True)[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def l2norm(t):
    return F.normalize(t, dim = -1)

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(0.1)

def stable_softmax(t, dim = -1, alpha = 32 ** 2):
    t = t / alpha
    t = t - torch.amax(t, dim = dim, keepdim = True).detach()
    return (t * alpha).softmax(dim = dim)

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# vqgan vae

class LayerNormChan(nn.Module):
    def __init__(
        self,
        dim,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma

# discriminator

class Discriminator(nn.Module):
    def __init__(
        self,
        dims,
        channels = 3,
        groups = 16,
        init_kernel_size = 5
    ):
        super().__init__()
        dim_pairs = zip(dims[:-1], dims[1:])

        self.layers = MList([nn.Sequential(nn.Conv2d(channels, dims[0], init_kernel_size, padding = init_kernel_size // 2), leaky_relu())])

        for dim_in, dim_out in dim_pairs:
            self.layers.append(nn.Sequential(
                nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1),
                nn.GroupNorm(groups, dim_out),
                leaky_relu()
            ))

        dim = dims[-1]
        self.to_logits = nn.Sequential( # return 5 x 5, for PatchGAN-esque training
            nn.Conv2d(dim, dim, 1),
            leaky_relu(),
            nn.Conv2d(dim, 1, 4)
        )

    def forward(self, x):
        for net in self.layers:
            x = net(x)

        return self.to_logits(x)

# positional encoding

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(self, *, dim, heads, layers = 2):
        super().__init__()
        self.net = MList([])
        self.net.append(nn.Sequential(nn.Linear(2, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, x):
        n, device = x.shape[-1], x.device
        fmap_size = int(sqrt(n))

        if not exists(self.rel_pos):
            pos = torch.arange(fmap_size, device = device)
            grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
            grid = rearrange(grid, 'c i j -> (i j) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')
            rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)
            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.float()

        for layer in self.net:
            rel_pos = layer(rel_pos)

        bias = rearrange(rel_pos, 'i j h -> h i j')
        return x + bias

# resnet encoder / decoder

class ResnetEncDec(nn.Module):
    def __init__(
        self,
        dim,
        *,
        channels = 3,
        layers = 4,
        layer_mults = None,
        num_resnet_blocks = 1,
        resnet_groups = 16,
        first_conv_kernel_size = 5,
        use_attn = True,
        attn_dim_head = 64,
        attn_heads = 8,
        attn_dropout = 0.,
    ):
        super().__init__()
        assert dim % resnet_groups == 0, f'dimension {dim} must be divisible by {resnet_groups} (groups for the groupnorm)'

        self.layers = layers

        self.encoders = MList([])
        self.decoders = MList([])

        layer_mults = default(layer_mults, list(map(lambda t: 2 ** t, range(layers))))
        assert len(layer_mults) == layers, 'layer multipliers must be equal to designated number of layers'

        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.encoded_dim = dims[-1]

        dim_pairs = zip(dims[:-1], dims[1:])

        append = lambda arr, t: arr.append(t)
        prepend = lambda arr, t: arr.insert(0, t)

        if not isinstance(num_resnet_blocks, tuple):
            num_resnet_blocks = (*((0,) * (layers - 1)), num_resnet_blocks)

        if not isinstance(use_attn, tuple):
            use_attn = (*((False,) * (layers - 1)), use_attn)

        assert len(num_resnet_blocks) == layers, 'number of resnet blocks config must be equal to number of layers'
        assert len(use_attn) == layers

        for layer_index, (dim_in, dim_out), layer_num_resnet_blocks, layer_use_attn in zip(range(layers), dim_pairs, num_resnet_blocks, use_attn):
            append(self.encoders, nn.Sequential(nn.Conv2d(dim_in, dim_out, 4, stride = 2, padding = 1), leaky_relu()))
            prepend(self.decoders, nn.Sequential(nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1), leaky_relu()))

            if layer_use_attn:
                prepend(self.decoders, VQGanAttention(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, dropout = attn_dropout))

            for _ in range(layer_num_resnet_blocks):
                append(self.encoders, ResBlock(dim_out, groups = resnet_groups))
                prepend(self.decoders, GLUResBlock(dim_out, groups = resnet_groups))

            if layer_use_attn:
                append(self.encoders, VQGanAttention(dim = dim_out, heads = attn_heads, dim_head = attn_dim_head, dropout = attn_dropout))

        prepend(self.encoders, nn.Conv2d(channels, dim, first_conv_kernel_size, padding = first_conv_kernel_size // 2))
        append(self.decoders, nn.Conv2d(dim, channels, 1))

    def get_encoded_fmap_size(self, image_size):
        return image_size // (2 ** self.layers)

    @property
    def last_dec_layer(self):
        return self.decoders[-1].weight

    def encode(self, x):
        for enc in self.encoders:
            x = enc(x)
        return x

    def decode(self, x):
        for dec in self.decoders:
            x = dec(x)
        return x

class GLUResBlock(nn.Module):
    def __init__(self, chan, groups = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            nn.GLU(dim = 1),
            nn.GroupNorm(groups, chan),
            nn.Conv2d(chan, chan * 2, 3, padding = 1),
            nn.GLU(dim = 1),
            nn.GroupNorm(groups, chan),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class ResBlock(nn.Module):
    def __init__(self, chan, groups = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GroupNorm(groups, chan),
            leaky_relu(),
            nn.Conv2d(chan, chan, 3, padding = 1),
            nn.GroupNorm(groups, chan),
            leaky_relu(),
            nn.Conv2d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

# vqgan attention layer

class VQGanAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.dropout = nn.Dropout(dropout)
        self.pre_norm = LayerNormChan(dim)

        self.cpb = ContinuousPositionBias(dim = dim // 4, heads = heads)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1, bias = False)

    def forward(self, x):
        h = self.heads
        height, width, residual = *x.shape[-2:], x.clone()

        x = self.pre_norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = 1)

        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = h), (q, k, v))

        sim = einsum('b h c i, b h c j -> b h i j', q, k) * self.scale

        sim = self.cpb(sim)

        attn = stable_softmax(sim, dim = -1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h c j -> b h c i', attn, v)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x = height, y = width)
        out = self.to_out(out)

        return out + residual

# ViT encoder / decoder

class RearrangeImage(nn.Module):
    def forward(self, x):
        n = x.shape[1]
        w = h = int(sqrt(n))
        return rearrange(x, 'b (h w) ... -> b h w ...', h = h, w = w)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 32
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

def FeedForward(dim, mult = 4):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult, bias = False),
        nn.GELU(),
        nn.Linear(dim * mult, dim, bias = False)
    )

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViTEncDec(nn.Module):
    def __init__(
        self,
        dim,
        channels = 3,
        layers = 4,
        patch_size = 8,
        dim_head = 32,
        heads = 8,
        ff_mult = 4
    ):
        super().__init__()
        self.encoded_dim = dim
        self.patch_size = patch_size

        input_dim = channels * (patch_size ** 2)

        self.encoder = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(input_dim, dim),
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers
            ),
            RearrangeImage(),
            Rearrange('b h w c -> b c h w')
        )

        self.decoder = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),
            Transformer(
                dim = dim,
                dim_head = dim_head,
                heads = heads,
                ff_mult = ff_mult,
                layers = layers
            ),
            nn.Sequential(
                nn.Linear(dim, dim * 4, bias = False),
                nn.Tanh(),
                nn.Linear(dim * 4, input_dim, bias = False),
            ),
            RearrangeImage(),
            Rearrange('b h w (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size)
        )

    def get_encoded_fmap_size(self, image_size):
        return image_size // self.patch_size

    @property
    def last_dec_layer(self):
        return self.decoder[-3][-1].weight

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

# main vqgan-vae classes

class NullVQGanVAE(nn.Module):
    def __init__(
        self,
        *,
        channels
    ):
        super().__init__()
        self.encoded_dim = channels
        self.layers = 0

    def get_encoded_fmap_size(self, size):
        return size

    def copy_for_eval(self):
        return self

    def encode(self, x):
        return x

    def decode(self, x):
        return x

class VQGanVAE(nn.Module):
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels = 3,
        layers = 4,
        l2_recon_loss = False,
        use_hinge_loss = True,
        vgg = None,
        vq_codebook_dim = 256,
        vq_codebook_size = 512,
        vq_decay = 0.8,
        vq_commitment_weight = 1.,
        vq_kmeans_init = True,
        vq_use_cosine_sim = True,
        use_vgg_and_gan = True,
        vae_type = 'resnet',
        discr_layers = 4,
        **kwargs
    ):
        super().__init__()
        vq_kwargs, kwargs = groupby_prefix_and_trim('vq_', kwargs)
        encdec_kwargs, kwargs = groupby_prefix_and_trim('encdec_', kwargs)

        self.image_size = image_size
        self.channels = channels
        self.codebook_size = vq_codebook_size

        if vae_type == 'resnet':
            enc_dec_klass = ResnetEncDec
        elif vae_type == 'vit':
            enc_dec_klass = ViTEncDec
        else:
            raise ValueError(f'{vae_type} not valid')

        self.enc_dec = enc_dec_klass(
            dim = dim,
            channels = channels,
            layers = layers,
            **encdec_kwargs
        )

        self.vq = VQ(
            dim = self.enc_dec.encoded_dim,
            codebook_dim = vq_codebook_dim,
            codebook_size = vq_codebook_size,
            decay = vq_decay,
            commitment_weight = vq_commitment_weight,
            accept_image_fmap = True,
            kmeans_init = vq_kmeans_init,
            use_cosine_sim = vq_use_cosine_sim,
            **vq_kwargs
        )

        # reconstruction loss

        self.recon_loss_fn = F.mse_loss if l2_recon_loss else F.l1_loss

        # turn off GAN and perceptual loss if grayscale

        self.vgg = None
        self.discr = None
        self.use_vgg_and_gan = use_vgg_and_gan

        if not use_vgg_and_gan:
            return

        # preceptual loss

        if exists(vgg):
            self.vgg = vgg
        else:
            self.vgg = torchvision.models.vgg16(pretrained = True)
            self.vgg.classifier = nn.Sequential(*self.vgg.classifier[:-2])

        # gan related losses

        layer_mults = list(map(lambda t: 2 ** t, range(discr_layers)))
        layer_dims = [dim * mult for mult in layer_mults]
        dims = (dim, *layer_dims)

        self.discr = Discriminator(dims = dims, channels = channels)

        self.discr_loss = hinge_discr_loss if use_hinge_loss else bce_discr_loss
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    @property
    def encoded_dim(self):
        return self.enc_dec.encoded_dim

    def get_encoded_fmap_size(self, image_size):
        return self.enc_dec.get_encoded_fmap_size(image_size)

    def copy_for_eval(self):
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    @remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    @remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    @property
    def codebook(self):
        return self.vq.codebook

    def encode(self, fmap):
        fmap = self.enc_dec.encode(fmap)
        return fmap

    def decode(self, fmap, return_indices_and_loss = False):
        fmap, indices, commit_loss = self.vq(fmap)

        fmap = self.enc_dec.decode(fmap)

        if not return_indices_and_loss:
            return fmap

        return fmap, indices, commit_loss

    def forward(
        self,
        img,
        return_loss = False,
        return_discr_loss = False,
        return_recons = False,
        add_gradient_penalty = True
    ):
        batch, channels, height, width, device = *img.shape, img.device
        assert height == self.image_size and width == self.image_size, 'height and width of input image must be equal to {self.image_size}'
        assert channels == self.channels, 'number of channels on image or sketch is not equal to the channels set on this VQGanVAE'

        fmap = self.encode(img)

        fmap, indices, commit_loss = self.decode(fmap, return_indices_and_loss = True)

        if not return_loss and not return_discr_loss:
            return fmap

        assert return_loss ^ return_discr_loss, 'you should either return autoencoder loss or discriminator loss, but not both'

        # whether to return discriminator loss

        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            fmap.detach_()
            img.requires_grad_()

            fmap_discr_logits, img_discr_logits = map(self.discr, (fmap, img))

            discr_loss = self.discr_loss(fmap_discr_logits, img_discr_logits)

            if add_gradient_penalty:
                gp = gradient_penalty(img, img_discr_logits)
                loss = discr_loss + gp

            if return_recons:
                return loss, fmap

            return loss

        # reconstruction loss

        recon_loss = self.recon_loss_fn(fmap, img)

        # early return if training on grayscale

        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, fmap

            return recon_loss

        # perceptual loss

        img_vgg_input = img
        fmap_vgg_input = fmap

        if img.shape[1] == 1:
            # handle grayscale for vgg
            img_vgg_input, fmap_vgg_input = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (img_vgg_input, fmap_vgg_input))

        img_vgg_feats = self.vgg(img_vgg_input)
        recon_vgg_feats = self.vgg(fmap_vgg_input)
        perceptual_loss = F.mse_loss(img_vgg_feats, recon_vgg_feats)

        # generator loss

        gen_loss = self.gen_loss(self.discr(fmap))

        # calculate adaptive weight

        last_dec_layer = self.enc_dec.last_dec_layer

        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max = 1e4)

        # combine losses

        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss

        if return_recons:
            return loss, fmap

        return loss
