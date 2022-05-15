import time
import copy
from math import ceil
from functools import partial
from collections.abc import Iterable

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from dalle2_pytorch.dalle2_pytorch import Decoder, DiffusionPrior
from dalle2_pytorch.optimizer import get_optimizer

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

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

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# gradient accumulation functions

def split_iterable(it, split_size):
    accum = []
    for ind in range(ceil(len(it) / split_size)):
        start_index = ind * split_size
        accum.append(it[start_index: (start_index + split_size)])
    return accum

def split(t, split_size = None):
    if not exists(split_size):
        return t

    if isinstance(t, torch.Tensor):
        return t.split(split_size, dim = 0)

    if isinstance(t, Iterable):
        return split_iterable(t, split_size)

    return TypeError

def find_first(cond, arr):
    for el in arr:
        if cond(el):
            return el
    return None

def split_args_and_kwargs(*args, split_size = None, **kwargs):
    all_args = (*args, *kwargs.values())
    len_all_args = len(all_args)
    first_tensor = find_first(lambda t: isinstance(t, torch.Tensor), all_args)
    assert exists(first_tensor)

    batch_size = len(first_tensor)
    split_size = default(split_size, batch_size)
    num_chunks = ceil(batch_size / split_size)

    dict_len = len(kwargs)
    dict_keys = kwargs.keys()
    split_kwargs_index = len_all_args - dict_len

    split_all_args = [split(arg, split_size = split_size) if exists(arg) and isinstance(arg, (torch.Tensor, Iterable)) else ((arg,) * num_chunks) for arg in all_args]
    chunk_sizes = tuple(map(len, split_all_args[0]))

    for (chunk_size, *chunked_all_args) in tuple(zip(chunk_sizes, *split_all_args)):
        chunked_args, chunked_kwargs_values = chunked_all_args[:split_kwargs_index], chunked_all_args[split_kwargs_index:]
        chunked_kwargs = dict(tuple(zip(dict_keys, chunked_kwargs_values)))
        chunk_size_frac = chunk_size / batch_size
        yield chunk_size_frac, (chunked_args, chunked_kwargs)

# print helpers

def print_ribbon(s, symbol = '=', repeat = 40):
    flank = symbol * repeat
    return f'{flank} {s} {flank}'

# saving and loading functions

# for diffusion prior

def load_diffusion_model(dprior_path, device):
    dprior_path = Path(dprior_path)
    assert dprior_path.exists(), 'Dprior model file does not exist'
    loaded_obj = torch.load(str(dprior_path), map_location='cpu')

    # Get hyperparameters of loaded model
    dpn_config = loaded_obj['hparams']['diffusion_prior_network']
    dp_config = loaded_obj['hparams']['diffusion_prior']
    image_embed_dim = loaded_obj['image_embed_dim']['image_embed_dim']

    # Create DiffusionPriorNetwork and DiffusionPrior with loaded hyperparameters

    # DiffusionPriorNetwork
    prior_network = DiffusionPriorNetwork( dim = image_embed_dim, **dpn_config).to(device)

    # DiffusionPrior with text embeddings and image embeddings pre-computed
    diffusion_prior = DiffusionPrior(net = prior_network, **dp_config, image_embed_dim = image_embed_dim).to(device)

    # Load state dict from saved model
    diffusion_prior.load_state_dict(loaded_obj['model'])

    return diffusion_prior, loaded_obj

def save_diffusion_model(save_path, model, optimizer, scaler, config, image_embed_dim):
    # Saving State Dict
    print_ribbon('Saving checkpoint')

    state_dict = dict(model=model.state_dict(),
                      optimizer=optimizer.state_dict(),
                      scaler=scaler.state_dict(),
                      hparams = config,
                      image_embed_dim = {"image_embed_dim":image_embed_dim})
    torch.save(state_dict, save_path+'/'+str(time.time())+'_saved_model.pth')

# exponential moving average wrapper

class EMA(nn.Module):
    def __init__(
        self,
        model,
        beta = 0.9999,
        update_after_step = 1000,
        update_every = 10,
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model
        self.ema_model = copy.deepcopy(model)

        self.update_after_step = update_after_step # only start EMA after this step number, starting at 0
        self.update_every = update_every

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0.]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def update(self):
        self.step += 1

        if self.step <= self.update_after_step or (self.step % self.update_every) != 0:
            return

        if not self.initted:
            self.ema_model.state_dict(self.online_model.state_dict())
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    def update_moving_average(self, ma_model, current_model):
        def calculate_ema(beta, old, new):
            if not exists(old):
                return new
            return old * beta + (1 - beta) * new

        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = calculate_ema(self.beta, old_weight, up_weight)

        for current_buffer, ma_buffer in zip(current_model.buffers(), ma_model.buffers()):
            new_buffer_value = calculate_ema(self.beta, ma_buffer, current_buffer)
            ma_buffer.copy_(new_buffer_value)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

# diffusion prior trainer

class DiffusionPriorTrainer(nn.Module):
    def __init__(
        self,
        diffusion_prior,
        use_ema = True,
        lr = 3e-4,
        wd = 1e-2,
        eps = 1e-6,
        max_grad_norm = None,
        amp = False,
        **kwargs
    ):
        super().__init__()
        assert isinstance(diffusion_prior, DiffusionPrior)
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)

        self.diffusion_prior = diffusion_prior

        # exponential moving average

        self.use_ema = use_ema
        if self.use_ema:
            self.ema_diffusion_prior = EMA(diffusion_prior, **ema_kwargs)

        # optimizer and mixed precision stuff

        self.amp = amp

        self.scaler = GradScaler(enabled = amp)

        self.optimizer = get_optimizer(
            diffusion_prior.parameters(),
            lr = lr,
            wd = wd,
            eps = eps,
            **kwargs
        )

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        self.register_buffer('step', torch.tensor([0.]))

    def update(self):
        if exists(self.max_grad_norm):
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.diffusion_prior.parameters(), self.max_grad_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        if self.use_ema:
            self.ema_diffusion_prior.update()

        self.step += 1

    @torch.inference_mode()
    def p_sample_loop(self, *args, **kwargs):
        return self.ema_diffusion_prior.ema_model.p_sample_loop(*args, **kwargs)

    @torch.inference_mode()
    def sample(self, *args, **kwargs):
        return self.ema_diffusion_prior.ema_model.sample(*args, **kwargs)

    @torch.inference_mode()
    def sample_batch_size(self, *args, **kwargs):
        return self.ema_diffusion_prior.ema_model.sample_batch_size(*args, **kwargs)

    def forward(
        self,
        *args,
        max_batch_size = None,
        **kwargs
    ):
        total_loss = 0.

        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            with autocast(enabled = self.amp):
                loss = self.diffusion_prior(*chunked_args, **chunked_kwargs)
                loss = loss * chunk_size_frac

            total_loss += loss.item()
            self.scaler.scale(loss).backward()

        return total_loss

# decoder trainer

class DecoderTrainer(nn.Module):
    def __init__(
        self,
        decoder,
        use_ema = True,
        lr = 2e-5,
        wd = 1e-2,
        eps = 1e-8,
        max_grad_norm = None,
        amp = False,
        **kwargs
    ):
        super().__init__()
        assert isinstance(decoder, Decoder)
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)

        self.decoder = decoder
        self.num_unets = len(self.decoder.unets)

        self.use_ema = use_ema

        if use_ema:
            has_lazy_linear = any([type(module) == nn.LazyLinear for module in decoder.modules()])
            assert not has_lazy_linear, 'you must set the text_embed_dim on your u-nets if you plan on doing automatic exponential moving average'

        self.ema_unets = nn.ModuleList([])

        self.amp = amp

        # be able to finely customize learning rate, weight decay
        # per unet

        lr, wd, eps = map(partial(cast_tuple, length = self.num_unets), (lr, wd, eps))

        for ind, (unet, unet_lr, unet_wd, unet_eps) in enumerate(zip(self.decoder.unets, lr, wd, eps)):
            optimizer = get_optimizer(
                unet.parameters(),
                lr = unet_lr,
                wd = unet_wd,
                eps = unet_eps,
                **kwargs
            )

            setattr(self, f'optim{ind}', optimizer) # cannot use pytorch ModuleList for some reason with optimizers

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

            scaler = GradScaler(enabled = amp)
            setattr(self, f'scaler{ind}', scaler)

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        self.register_buffer('step', torch.tensor([0.]))

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def scale(self, loss, *, unet_number):
        assert 1 <= unet_number <= self.num_unets
        index = unet_number - 1
        scaler = getattr(self, f'scaler{index}')
        return scaler.scale(loss)

    def update(self, unet_number):
        assert 1 <= unet_number <= self.num_unets
        index = unet_number - 1
        unet = self.decoder.unets[index]

        optimizer = getattr(self, f'optim{index}')
        scaler = getattr(self, f'scaler{index}')

        if exists(self.max_grad_norm):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(unet.parameters(), self.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if self.use_ema:
            ema_unet = self.ema_unets[index]
            ema_unet.update()

        self.step += 1

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        if self.use_ema:
            trainable_unets = self.decoder.unets
            self.decoder.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = self.decoder.sample(*args, **kwargs)

        if self.use_ema:
            self.decoder.unets = trainable_unets             # restore original training unets

        # cast the ema_model unets back to original device
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        return output

    def forward(
        self,
        *args,
        unet_number,
        max_batch_size = None,
        **kwargs
    ):
        total_loss = 0.

        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            with autocast(enabled = self.amp):
                loss = self.decoder(*chunked_args, unet_number = unet_number, **chunked_kwargs)
                loss = loss * chunk_size_frac

            total_loss += loss.item()
            self.scale(loss, unet_number = unet_number).backward()

        return total_loss
