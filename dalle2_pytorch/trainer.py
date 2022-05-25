import time
import copy
from pathlib import Path
from math import ceil
from functools import partial, wraps
from collections.abc import Iterable

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from dalle2_pytorch.dalle2_pytorch import Decoder, DiffusionPrior
from dalle2_pytorch.optimizer import get_optimizer

import numpy as np

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

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def get_pkg_version():
    from pkg_resources import get_distribution
    return get_distribution('dalle2_pytorch').version

# decorators

def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        device = kwargs.pop('_device', next(model.parameters()).device)
        cast_device = kwargs.pop('_cast_device', True)

        kwargs_keys = kwargs.keys()
        all_args = (*args, *kwargs.values())
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args))

        if cast_device:
            all_args = tuple(map(lambda t: t.to(device) if exists(t) and isinstance(t, torch.Tensor) else t, all_args))

        args, kwargs_values = all_args[:split_kwargs_index], all_args[split_kwargs_index:]
        kwargs = dict(tuple(zip(kwargs_keys, kwargs_values)))

        out = fn(model, *args, **kwargs)
        return out
    return inner

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

        self.update_every = update_every
        self.update_after_step = update_after_step  // update_every # only start EMA after this step number, starting at 0

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def copy_params_from_model_to_ema(self):
        self.ema_model.state_dict(self.online_model.state_dict())

    def update(self):
        self.step += 1

        if (self.step % self.update_every) != 0:
            return

        if self.step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted:
            self.copy_params_from_model_to_ema()
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

def prior_sample_in_chunks(fn):
    @wraps(fn)
    def inner(self, *args, max_batch_size = None, **kwargs):
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)

        outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs)]
        return torch.cat(outputs, dim = 0)
    return inner

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
        group_wd_params = True,
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
            group_wd_params = group_wd_params,
            **kwargs
        )

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        self.register_buffer('step', torch.tensor([0]))

    def save(self, path, overwrite = True, **kwargs):
        path = Path(path)
        assert not (path.exists() and not overwrite)
        path.parent.mkdir(parents = True, exist_ok = True)

        save_obj = dict(
            scaler = self.scaler.state_dict(),
            optimizer = self.optimizer.state_dict(),
            model = self.diffusion_prior.state_dict(),
            version = get_pkg_version(),
            step = self.step.item(),
            **kwargs
        )

        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_diffusion_prior.state_dict()}

        torch.save(save_obj, str(path))

    def load(self, path, only_model = False, strict = True):
        path = Path(path)
        assert path.exists()

        loaded_obj = torch.load(str(path))

        if get_pkg_version() != loaded_obj['version']:
            print(f'loading saved diffusion prior at version {loaded_obj["version"]} but current package version is at {get_pkg_version()}')

        self.diffusion_prior.load_state_dict(loaded_obj['model'], strict = strict)
        self.step.copy_(torch.ones_like(self.step) * loaded_obj['step'])

        if only_model:
            return loaded_obj

        self.scaler.load_state_dict(loaded_obj['scaler'])
        self.optimizer.load_state_dict(loaded_obj['optimizer'])

        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_diffusion_prior.load_state_dict(loaded_obj['ema'], strict = strict)

        return loaded_obj

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

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def p_sample_loop(self, *args, **kwargs):
        return self.ema_diffusion_prior.ema_model.p_sample_loop(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def sample(self, *args, **kwargs):
        return self.ema_diffusion_prior.ema_model.sample(*args, **kwargs)

    @torch.no_grad()
    def sample_batch_size(self, *args, **kwargs):
        return self.ema_diffusion_prior.ema_model.sample_batch_size(*args, **kwargs)

    @cast_torch_tensor
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

            if self.training:
                self.scaler.scale(loss).backward()

        return total_loss

# decoder trainer

def decoder_sample_in_chunks(fn):
    @wraps(fn)
    def inner(self, *args, max_batch_size = None, **kwargs):
        if not exists(max_batch_size):
            return fn(self, *args, **kwargs)

        if self.decoder.unconditional:
            batch_size = kwargs.get('batch_size')
            batch_sizes = num_to_groups(batch_size, max_batch_size)
            outputs = [fn(self, *args, **{**kwargs, 'batch_size': sub_batch_size}) for sub_batch_size in batch_sizes]
        else:
            outputs = [fn(self, *chunked_args, **chunked_kwargs) for _, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs)]

        return torch.cat(outputs, dim = 0)
    return inner

class DecoderTrainer(nn.Module):
    def __init__(
        self,
        decoder,
        use_ema = True,
        lr = 1e-4,
        wd = 1e-2,
        eps = 1e-8,
        max_grad_norm = 0.5,
        amp = False,
        group_wd_params = True,
        **kwargs
    ):
        super().__init__()
        assert isinstance(decoder, Decoder)
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)

        self.decoder = decoder
        self.num_unets = len(self.decoder.unets)

        self.use_ema = use_ema
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
                group_wd_params = group_wd_params,
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

    def save(self, path, overwrite = True, **kwargs):
        path = Path(path)
        assert not (path.exists() and not overwrite)
        path.parent.mkdir(parents = True, exist_ok = True)

        save_obj = dict(
            model = self.decoder.state_dict(),
            version = get_pkg_version(),
            step = self.step.item(),
            **kwargs
        )

        for ind in range(0, self.num_unets):
            scaler_key = f'scaler{ind}'
            optimizer_key = f'scaler{ind}'
            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)
            save_obj = {**save_obj, scaler_key: scaler.state_dict(), optimizer_key: optimizer.state_dict()}

        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_unets.state_dict()}

        torch.save(save_obj, str(path))

    def load(self, path, only_model = False, strict = True):
        path = Path(path)
        assert path.exists()

        loaded_obj = torch.load(str(path))

        if get_pkg_version() != loaded_obj['version']:
            print(f'loading saved decoder at version {loaded_obj["version"]}, but current package version is {get_pkg_version()}')

        self.decoder.load_state_dict(loaded_obj['model'], strict = strict)
        self.step.copy_(torch.ones_like(self.step) * loaded_obj['step'])

        if only_model:
            return loaded_obj

        for ind in range(0, self.num_unets):
            scaler_key = f'scaler{ind}'
            optimizer_key = f'scaler{ind}'
            scaler = getattr(self, scaler_key)
            optimizer = getattr(self, optimizer_key)

            scaler.load_state_dict(loaded_obj[scaler_key])
            optimizer.load_state_dict(loaded_obj[optimizer_key])

        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_unets.load_state_dict(loaded_obj['ema'], strict = strict)

        return loaded_obj

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def scale(self, loss, *, unet_number):
        assert 1 <= unet_number <= self.num_unets
        index = unet_number - 1
        scaler = getattr(self, f'scaler{index}')
        return scaler.scale(loss)

    def update(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert exists(unet_number) and 1 <= unet_number <= self.num_unets
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
    @cast_torch_tensor
    @decoder_sample_in_chunks
    def sample(self, *args, **kwargs):
        if kwargs.pop('use_non_ema', False) or not self.use_ema:
            return self.decoder.sample(*args, **kwargs)

        trainable_unets = self.decoder.unets
        self.decoder.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = self.decoder.sample(*args, **kwargs)

        self.decoder.unets = trainable_unets             # restore original training unets

        # cast the ema_model unets back to original device
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        return output

    @cast_torch_tensor
    def forward(
        self,
        *args,
        unet_number = None,
        max_batch_size = None,
        **kwargs
    ):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        total_loss = 0.

        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            with autocast(enabled = self.amp):
                loss = self.decoder(*chunked_args, unet_number = unet_number, **chunked_kwargs)
                loss = loss * chunk_size_frac

            total_loss += loss.item()

            if self.training:
                self.scale(loss, unet_number = unet_number).backward()

        return total_loss
