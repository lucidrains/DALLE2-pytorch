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
from dalle2_pytorch.version import __version__
from packaging import version

from accelerate import Accelerator

import numpy as np

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

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

def clamp(value, min_value = None, max_value = None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value

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
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    def __init__(
        self,
        model,
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 2/3,
        min_value = 0.0,
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model
        self.ema_model = copy.deepcopy(model)

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def copy_params_from_model_to_ema(self):
        for ma_param, current_param in zip(list(self.ema_model.parameters()), list(self.online_model.parameters())):
            ma_param.data.copy_(current_param.data)

        for ma_buffer, current_buffer in zip(list(self.ema_model.buffers()), list(self.online_model.buffers())):
            ma_buffer.data.copy_(current_buffer.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value = 0)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch <= 0:
            return 0.

        return clamp(value, min_value = self.min_value, max_value = self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for current_params, ma_params in zip(list(current_model.parameters()), list(ma_model.parameters())):
            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for current_buffer, ma_buffer in zip(list(current_model.buffers()), list(ma_model.buffers())):
            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

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
        device = None,
        accelerator = None,
        **kwargs
    ):
        super().__init__()
        assert isinstance(diffusion_prior, DiffusionPrior)
        assert not exists(accelerator) or isinstance(accelerator, Accelerator)
        assert exists(accelerator) or exists(device), "You must supply some method of obtaining a device."
        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)

        # assign some helpful member vars
        self.accelerator = accelerator
        self.device = accelerator.device if exists(accelerator) else device
        self.text_conditioned = diffusion_prior.condition_on_text_encodings

        # save model

        self.diffusion_prior = diffusion_prior

        # optimizer and mixed precision stuff

        self.amp = amp

        self.scaler = GradScaler(enabled = amp)

        self.optim_kwargs = dict(lr=lr, wd=wd, eps=eps, group_wd_params=group_wd_params)

        self.optimizer = get_optimizer(
            self.diffusion_prior.parameters(),
            **self.optim_kwargs,
            **kwargs
        )

        # distribute the model if using HFA
        if exists(self.accelerator):
            self.diffusion_prior, self.optimizer = self.accelerator.prepare(self.diffusion_prior, self.optimizer)

        # exponential moving average stuff

        self.use_ema = use_ema

        if self.use_ema:
            self.ema_diffusion_prior = EMA(self.unwrap_model(self.diffusion_prior), **ema_kwargs)

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        # track steps internally

        self.register_buffer('step', torch.tensor([0]))

    # accelerator wrappers

    def print(self, msg):
        if exists(self.accelerator):
            self.accelerator.print(msg)
        else:
            print(msg)

    def unwrap_model(self, model):
        if exists(self.accelerator):
            return self.accelerator.unwrap_model(model)
        else:
            return model

    def wait_for_everyone(self):
        if exists(self.accelerator):
            self.accelerator.wait_for_everyone()

    def is_main_process(self):
        if exists(self.accelerator):
            return self.accelerator.is_main_process
        else:
            return True

    def clip_grad_norm_(self, *args):
        if exists(self.accelerator):
            return self.accelerator.clip_grad_norm_(*args)
        else:
            return torch.nn.utils.clip_grad_norm_(*args)

    def backprop(self, x):
        if exists(self.accelerator):
            self.accelerator.backward(x)
        else:
            try:
                x.backward()
            except Exception as e:
                self.print(f"Caught error in backprop call: {e}")

    # utility

    def save(self, path, overwrite = True, **kwargs):
        # ensure we sync gradients before continuing
        self.wait_for_everyone()

        # only save on the main process
        if self.is_main_process():
            self.print(f"Saving checkpoint at step: {self.step.item()}")
            path = Path(path)
            assert not (path.exists() and not overwrite)
            path.parent.mkdir(parents = True, exist_ok = True)

            save_obj = dict(
                scaler = self.scaler.state_dict(),
                optimizer = self.optimizer.state_dict(),
                model = self.unwrap_model(self.diffusion_prior).state_dict(), # unwrap the model from distribution if applicable
                version = version.parse(__version__),
                step = self.step.item(),
                **kwargs
            )

            if self.use_ema:
                save_obj = {
                    **save_obj,
                    'ema': self.ema_diffusion_prior.state_dict(),
                    'ema_model': self.ema_diffusion_prior.ema_model.state_dict() # save the ema model specifically for easy ema-only reload
                }

            torch.save(save_obj, str(path))

    def load(self, path, overwrite_lr = True, strict = True):
        """
        Load a checkpoint of a diffusion prior trainer.

        Will load the entire trainer, including the optimizer and EMA.

        Params:
            - path (str): a path to the DiffusionPriorTrainer checkpoint file
            - overwrite_lr (bool): wether or not to overwrite the stored LR with the LR specified in the new trainer
            - strict (bool): kwarg for `torch.nn.Module.load_state_dict`, will force an exact checkpoint match

        Returns:
            loaded_obj (dict): The loaded checkpoint dictionary
        """

        # all processes need to load checkpoint. no restriction here
        path = Path(path)
        assert path.exists()

        loaded_obj = torch.load(str(path), map_location=self.device)

        if version.parse(__version__) != loaded_obj['version']:
            print(f'loading saved diffusion prior at version {loaded_obj["version"]} but current package version is at {__version__}')

        # unwrap the model when loading from checkpoint
        self.unwrap_model(self.diffusion_prior).load_state_dict(loaded_obj['model'], strict = strict)
        self.step.copy_(torch.ones_like(self.step) * loaded_obj['step'])

        self.scaler.load_state_dict(loaded_obj['scaler'])
        self.optimizer.load_state_dict(loaded_obj['optimizer'])

        if overwrite_lr:
            new_lr = self.optim_kwargs["lr"]

            self.print(f"Overriding LR to be {new_lr}")

            for group in self.optimizer.param_groups:
                group["lr"] = new_lr

        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_diffusion_prior.load_state_dict(loaded_obj['ema'], strict = strict)
            # below not be necessary, but I had a suspicion that this wasn't being loaded correctly
            self.ema_diffusion_prior.ema_model.load_state_dict(loaded_obj["ema_model"])

        # sync and inform
        self.wait_for_everyone()
        self.print(f"Loaded model")

        return loaded_obj

    # model functionality

    def update(self):
        # only continue with updates until all ranks finish
        self.wait_for_everyone()

        if exists(self.max_grad_norm):
            self.scaler.unscale_(self.optimizer)
            # utilize HFA clipping where applicable
            self.clip_grad_norm_(self.diffusion_prior.parameters(), self.max_grad_norm)

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
        if self.use_ema:
            return self.ema_diffusion_prior.ema_model.p_sample_loop(*args, **kwargs)
        else:
            return self.diffusion_prior.p_sample_loop(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def sample(self, *args, **kwargs):
        if self.use_ema:
            return self.ema_diffusion_prior.ema_model.sample(*args, **kwargs)
        else:
            return self.diffusion_prior.sample(*args, **kwargs)

    @torch.no_grad()
    def sample_batch_size(self, *args, **kwargs):
        if self.use_ema:
            return self.ema_diffusion_prior.ema_model.sample_batch_size(*args, **kwargs)
        else:
            return self.diffusion_prior.sample_batch_size(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_text(self, *args, **kwargs):
        return self.unwrap_model(self.diffusion_prior).clip.embed_text(*args, **kwargs)

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

            # backprop with accelerate if applicable

            if self.training:
                self.backprop(self.scaler.scale(loss))

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
        accelerator = None,
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

        self.accelerator = default(accelerator, Accelerator)

        self.num_unets = len(decoder.unets)

        self.use_ema = use_ema
        self.ema_unets = nn.ModuleList([])

        self.amp = amp

        # be able to finely customize learning rate, weight decay
        # per unet

        lr, wd, eps = map(partial(cast_tuple, length = self.num_unets), (lr, wd, eps))

        optimizers = []

        for unet, unet_lr, unet_wd, unet_eps in zip(decoder.unets, lr, wd, eps):
            optimizer = get_optimizer(
                unet.parameters(),
                lr = unet_lr,
                wd = unet_wd,
                eps = unet_eps,
                group_wd_params = group_wd_params,
                **kwargs
            )

            optimizers.append(optimizer)

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        self.register_buffer('step', torch.tensor([0.]))
        results = list(self.accelerator.prepare(decoder, *optimizers))
        self.decoder = results.pop(0)
        for opt_ind in range(len(optimizers)):
            setattr(self, f'optim{opt_ind}', results.pop(0))

    def save(self, path, overwrite = True, **kwargs):
        path = Path(path)
        assert not (path.exists() and not overwrite)
        path.parent.mkdir(parents = True, exist_ok = True)

        save_obj = dict(
            model = self.accelerator.unwrap_model(self.decoder).state_dict(),
            version = __version__,
            step = self.step.item(),
            **kwargs
        )

        for ind in range(0, self.num_unets):
            optimizer_key = f'optim{ind}'
            optimizer = getattr(self, optimizer_key)
            save_obj = {**save_obj, optimizer_key: self.accelerator.unwrap_model(optimizer).state_dict()}

        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_unets.state_dict()}

        self.accelerator.save(save_obj, str(path))

    def load(self, path, only_model = False, strict = True):
        path = Path(path)
        assert path.exists()

        loaded_obj = torch.load(str(path), map_location = 'cpu')

        if version.parse(__version__) != version.parse(loaded_obj['version']):
            self.accelerator.print(f'loading saved decoder at version {loaded_obj["version"]}, but current package version is {__version__}')

        self.accelerator.unwrap_model(self.decoder).load_state_dict(loaded_obj['model'], strict = strict)
        self.step.copy_(torch.ones_like(self.step) * loaded_obj['step'])

        if only_model:
            return loaded_obj

        for ind in range(0, self.num_unets):
            optimizer_key = f'optim{ind}'
            optimizer = getattr(self, optimizer_key)

            self.accelerator.unwrap_model(optimizer).load_state_dict(loaded_obj[optimizer_key])

        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_unets.load_state_dict(loaded_obj['ema'], strict = strict)

        return loaded_obj

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def update(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert exists(unet_number) and 1 <= unet_number <= self.num_unets
        index = unet_number - 1

        optimizer = getattr(self, f'optim{index}')

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)  # Automatically unscales gradients
        optimizer.step()
        optimizer.zero_grad()

        if self.use_ema:
            ema_unet = self.ema_unets[index]
            ema_unet.update()

        self.step += 1

    @torch.no_grad()
    @cast_torch_tensor
    @decoder_sample_in_chunks
    def sample(self, *args, **kwargs):
        distributed = self.accelerator.num_processes > 1
        base_decoder = self.accelerator.unwrap_model(self.decoder)
        if kwargs.pop('use_non_ema', False) or not self.use_ema:
            return base_decoder.sample(*args, **kwargs, distributed = distributed)

        trainable_unets = self.accelerator.unwrap_model(self.decoder).unets
        base_decoder.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = base_decoder.sample(*args, **kwargs, distributed = distributed)

        base_decoder.unets = trainable_unets             # restore original training unets

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
            # with autocast(enabled = self.amp):
            with self.accelerator.autocast():
                loss = self.decoder(*chunked_args, unet_number = unet_number, **chunked_kwargs)
                loss = loss * chunk_size_frac

            total_loss += loss.item()

            if self.training:
                self.accelerator.backward(loss)

        return total_loss
