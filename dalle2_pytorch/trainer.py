import time
import copy
from pathlib import Path
from math import ceil
from functools import partial, wraps
from contextlib import nullcontext
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from dalle2_pytorch.dalle2_pytorch import Decoder, DiffusionPrior
from dalle2_pytorch.optimizer import get_optimizer
from dalle2_pytorch.version import __version__
from packaging import version

import pytorch_warmup as warmup

from ema_pytorch import EMA

from accelerate import Accelerator, DistributedType

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

# decorators

def cast_torch_tensor(fn):
    @wraps(fn)
    def inner(model, *args, **kwargs):
        device = kwargs.pop('_device', next(model.parameters()).device)
        cast_device = kwargs.pop('_cast_device', True)
        cast_deepspeed_precision = kwargs.pop('_cast_deepspeed_precision', True)

        kwargs_keys = kwargs.keys()
        all_args = (*args, *kwargs.values())
        split_kwargs_index = len(all_args) - len(kwargs_keys)
        all_args = tuple(map(lambda t: torch.from_numpy(t) if exists(t) and isinstance(t, np.ndarray) else t, all_args))

        if cast_device:
            all_args = tuple(map(lambda t: t.to(device) if exists(t) and isinstance(t, torch.Tensor) else t, all_args))

        if cast_deepspeed_precision:
            try:
                accelerator = model.accelerator
                if accelerator is not None and accelerator.distributed_type == DistributedType.DEEPSPEED:
                    cast_type_map = {
                        "fp16": torch.half,
                        "bf16": torch.bfloat16,
                        "no": torch.float
                    }
                    precision_type = cast_type_map[accelerator.mixed_precision]
                    all_args = tuple(map(lambda t: t.to(precision_type) if exists(t) and isinstance(t, torch.Tensor) else t, all_args))
            except AttributeError:
                # Then this model doesn't have an accelerator
                pass

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
        accelerator = None,
        use_ema = True,
        lr = 3e-4,
        wd = 1e-2,
        eps = 1e-6,
        max_grad_norm = None,
        group_wd_params = True,
        warmup_steps = None,
        cosine_decay_max_steps = None,
        **kwargs
    ):
        super().__init__()
        assert isinstance(diffusion_prior, DiffusionPrior)

        ema_kwargs, kwargs = groupby_prefix_and_trim('ema_', kwargs)
        accelerator_kwargs, kwargs = groupby_prefix_and_trim('accelerator_', kwargs)

        if not exists(accelerator):
            accelerator = Accelerator(**accelerator_kwargs)

        # assign some helpful member vars

        self.accelerator = accelerator
        self.text_conditioned = diffusion_prior.condition_on_text_encodings

        # setting the device

        self.device = accelerator.device
        diffusion_prior.to(self.device)

        # save model

        self.diffusion_prior = diffusion_prior

        # mixed precision checks

        if (
            exists(self.accelerator) 
            and self.accelerator.distributed_type == DistributedType.DEEPSPEED 
            and self.diffusion_prior.clip is not None
            ):
            # Then we need to make sure clip is using the correct precision or else deepspeed will error
            cast_type_map = {
                "fp16": torch.half,
                "bf16": torch.bfloat16,
                "no": torch.float
            }
            precision_type = cast_type_map[accelerator.mixed_precision]
            assert precision_type == torch.float, "DeepSpeed currently only supports float32 precision when using on the fly embedding generation from clip"
            self.diffusion_prior.clip.to(precision_type)

        # optimizer stuff

        self.optim_kwargs = dict(lr=lr, wd=wd, eps=eps, group_wd_params=group_wd_params)

        self.optimizer = get_optimizer(
            self.diffusion_prior.parameters(),
            **self.optim_kwargs,
            **kwargs
        )

        if exists(cosine_decay_max_steps):
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max = cosine_decay_max_steps)
        else:
            self.scheduler = LambdaLR(self.optimizer, lr_lambda = lambda _: 1.0)
        
        self.warmup_scheduler = warmup.LinearWarmup(self.optimizer, warmup_period = warmup_steps) if exists(warmup_steps) else None

        # distribute the model if using HFA

        self.diffusion_prior, self.optimizer, self.scheduler = self.accelerator.prepare(self.diffusion_prior, self.optimizer, self.scheduler)

        # exponential moving average stuff

        self.use_ema = use_ema

        if self.use_ema:
            self.ema_diffusion_prior = EMA(self.accelerator.unwrap_model(self.diffusion_prior), **ema_kwargs)

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        # track steps internally

        self.register_buffer('step', torch.tensor([0], device = self.device))

    # utility

    def save(self, path, overwrite = True, **kwargs):

        # only save on the main process
        if self.accelerator.is_main_process:
            print(f"Saving checkpoint at step: {self.step.item()}")
            path = Path(path)
            assert not (path.exists() and not overwrite)
            path.parent.mkdir(parents = True, exist_ok = True)

            # FIXME: LambdaLR can't be saved due to pickling issues
            save_obj = dict(
                optimizer = self.optimizer.state_dict(),
                scheduler = self.scheduler.state_dict(),
                warmup_scheduler = self.warmup_scheduler,
                model = self.accelerator.unwrap_model(self.diffusion_prior).state_dict(),
                version = version.parse(__version__),
                step = self.step,
                **kwargs
            )

            if self.use_ema:
                save_obj = {
                    **save_obj,
                    'ema': self.ema_diffusion_prior.state_dict(),
                    'ema_model': self.ema_diffusion_prior.ema_model.state_dict() # save the ema model specifically for easy ema-only reload
                }

            torch.save(save_obj, str(path))

    def load(self, path_or_state, overwrite_lr = True, strict = True):
        """
        Load a checkpoint of a diffusion prior trainer.

        Will load the entire trainer, including the optimizer and EMA.

        Params:
            - path_or_state (str | torch): a path to the DiffusionPriorTrainer checkpoint file
            - overwrite_lr (bool): wether or not to overwrite the stored LR with the LR specified in the new trainer
            - strict (bool): kwarg for `torch.nn.Module.load_state_dict`, will force an exact checkpoint match

        Returns:
            loaded_obj (dict): The loaded checkpoint dictionary
        """

        # all processes need to load checkpoint. no restriction here
        if isinstance(path_or_state, str):
            path = Path(path_or_state)
            assert path.exists()
            loaded_obj = torch.load(str(path), map_location=self.device)

        elif isinstance(path_or_state, dict):
            loaded_obj = path_or_state

        if version.parse(__version__) != loaded_obj['version']:
            print(f'loading saved diffusion prior at version {loaded_obj["version"]} but current package version is at {__version__}')

        # unwrap the model when loading from checkpoint
        self.accelerator.unwrap_model(self.diffusion_prior).load_state_dict(loaded_obj['model'], strict = strict)
        self.step.copy_(torch.ones_like(self.step, device=self.device) * loaded_obj['step'].to(self.device))

        self.optimizer.load_state_dict(loaded_obj['optimizer'])
        self.scheduler.load_state_dict(loaded_obj['scheduler'])

        # set warmupstep
        if exists(self.warmup_scheduler):
            self.warmup_scheduler.last_step = self.step.item()

        # ensure new lr is used if different from old one
        if overwrite_lr:
            new_lr = self.optim_kwargs["lr"]

            for group in self.optimizer.param_groups:
                group["lr"] = new_lr if group["lr"] > 0.0 else 0.0

        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_diffusion_prior.load_state_dict(loaded_obj['ema'], strict = strict)
            # below might not be necessary, but I had a suspicion that this wasn't being loaded correctly
            self.ema_diffusion_prior.ema_model.load_state_dict(loaded_obj["ema_model"])

        return loaded_obj

    # model functionality

    def update(self):

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.diffusion_prior.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        self.optimizer.zero_grad()

        # accelerator will ocassionally skip optimizer steps in a "dynamic loss scaling strategy"
        if not self.accelerator.optimizer_step_was_skipped:
            sched_context = self.warmup_scheduler.dampening if exists(self.warmup_scheduler) else nullcontext
            with sched_context():
                self.scheduler.step()

        if self.use_ema:
            self.ema_diffusion_prior.update()

        self.step += 1

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def p_sample_loop(self, *args, **kwargs):
        model = self.ema_diffusion_prior.ema_model if self.use_ema else self.diffusion_prior
        return model.p_sample_loop(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def sample(self, *args, **kwargs):
        model = self.ema_diffusion_prior.ema_model if self.use_ema else self.diffusion_prior
        return model.sample(*args, **kwargs)

    @torch.no_grad()
    def sample_batch_size(self, *args, **kwargs):
        model = self.ema_diffusion_prior.ema_model if self.use_ema else self.diffusion_prior
        return model.sample_batch_size(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_text(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.diffusion_prior).clip.embed_text(*args, **kwargs)

    @cast_torch_tensor
    def forward(
        self,
        *args,
        max_batch_size = None,
        **kwargs
    ):
        total_loss = 0.

        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            with self.accelerator.autocast():
                loss = self.diffusion_prior(*chunked_args, **chunked_kwargs)
                loss = loss * chunk_size_frac

            total_loss += loss.item()

            if self.training:
                self.accelerator.backward(loss)

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
        dataloaders = None,
        use_ema = True,
        lr = 1e-4,
        wd = 1e-2,
        eps = 1e-8,
        warmup_steps = None,
        cosine_decay_max_steps = None,
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

        lr, wd, eps, warmup_steps, cosine_decay_max_steps = map(partial(cast_tuple, length = self.num_unets), (lr, wd, eps, warmup_steps, cosine_decay_max_steps))

        assert all([unet_lr <= 1e-2 for unet_lr in lr]), 'your learning rate is too high, recommend sticking with 1e-4, at most 5e-4'

        optimizers = []
        schedulers = []
        warmup_schedulers = []

        for unet, unet_lr, unet_wd, unet_eps, unet_warmup_steps, unet_cosine_decay_max_steps in zip(decoder.unets, lr, wd, eps, warmup_steps, cosine_decay_max_steps):
            if isinstance(unet, nn.Identity):
                optimizers.append(None)
                schedulers.append(None)
                warmup_schedulers.append(None)
            else:
                optimizer = get_optimizer(
                    unet.parameters(),
                    lr = unet_lr,
                    wd = unet_wd,
                    eps = unet_eps,
                    group_wd_params = group_wd_params,
                    **kwargs
                )

                optimizers.append(optimizer)

                if exists(unet_cosine_decay_max_steps):
                    scheduler = CosineAnnealingLR(optimizer, T_max = unet_cosine_decay_max_steps)
                else:
                    scheduler = LambdaLR(optimizer, lr_lambda = lambda step: 1.0)

                warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period = unet_warmup_steps) if exists(unet_warmup_steps) else None
                warmup_schedulers.append(warmup_scheduler)

                schedulers.append(scheduler)

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

        self.register_buffer('steps', torch.tensor([0] * self.num_unets))

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED and decoder.clip is not None:
            # Then we need to make sure clip is using the correct precision or else deepspeed will error
            cast_type_map = {
                "fp16": torch.half,
                "bf16": torch.bfloat16,
                "no": torch.float
            }
            precision_type = cast_type_map[accelerator.mixed_precision]
            assert precision_type == torch.float, "DeepSpeed currently only supports float32 precision when using on the fly embedding generation from clip"
            clip = decoder.clip
            clip.to(precision_type)

        decoder, *optimizers = list(self.accelerator.prepare(decoder, *optimizers))

        self.decoder = decoder

        # prepare dataloaders

        train_loader = val_loader = None
        if exists(dataloaders):
            train_loader, val_loader = self.accelerator.prepare(dataloaders["train"], dataloaders["val"])

        self.train_loader = train_loader
        self.val_loader = val_loader

        # store optimizers

        for opt_ind, optimizer in zip(range(len(optimizers)), optimizers):
            setattr(self, f'optim{opt_ind}', optimizer)

        # store schedulers

        for sched_ind, scheduler in zip(range(len(schedulers)), schedulers):
            setattr(self, f'sched{sched_ind}', scheduler)

        # store warmup schedulers

        self.warmup_schedulers = warmup_schedulers

    def validate_and_return_unet_number(self, unet_number = None):
        if self.num_unets == 1:
            unet_number = default(unet_number, 1)

        assert exists(unet_number) and 1 <= unet_number <= self.num_unets
        return unet_number

    def num_steps_taken(self, unet_number = None):
        unet_number = self.validate_and_return_unet_number(unet_number)
        return self.steps[unet_number - 1].item()

    def save(self, path, overwrite = True, **kwargs):
        path = Path(path)
        assert not (path.exists() and not overwrite)
        path.parent.mkdir(parents = True, exist_ok = True)

        save_obj = dict(
            model = self.accelerator.unwrap_model(self.decoder).state_dict(),
            version = __version__,
            steps = self.steps.cpu(),
            **kwargs
        )

        for ind in range(0, self.num_unets):
            optimizer_key = f'optim{ind}'
            scheduler_key = f'sched{ind}'

            optimizer = getattr(self, optimizer_key)
            scheduler = getattr(self, scheduler_key)

            optimizer_state_dict = optimizer.state_dict() if exists(optimizer) else None
            scheduler_state_dict = scheduler.state_dict() if exists(scheduler) else None

            save_obj = {**save_obj, optimizer_key: optimizer_state_dict, scheduler_key: scheduler_state_dict}

        if self.use_ema:
            save_obj = {**save_obj, 'ema': self.ema_unets.state_dict()}

        self.accelerator.save(save_obj, str(path))

    def load_state_dict(self, loaded_obj, only_model = False, strict = True):
        if version.parse(__version__) != version.parse(loaded_obj['version']):
            self.accelerator.print(f'loading saved decoder at version {loaded_obj["version"]}, but current package version is {__version__}')

        self.accelerator.unwrap_model(self.decoder).load_state_dict(loaded_obj['model'], strict = strict)
        self.steps.copy_(loaded_obj['steps'])

        if only_model:
            return loaded_obj

        for ind, last_step in zip(range(0, self.num_unets), self.steps.tolist()):

            optimizer_key = f'optim{ind}'
            optimizer = getattr(self, optimizer_key)

            scheduler_key = f'sched{ind}'
            scheduler = getattr(self, scheduler_key)

            warmup_scheduler = self.warmup_schedulers[ind]

            if exists(optimizer):
                optimizer.load_state_dict(loaded_obj[optimizer_key])

            if exists(scheduler):
                scheduler.load_state_dict(loaded_obj[scheduler_key])

            if exists(warmup_scheduler):
                warmup_scheduler.last_step = last_step

        if self.use_ema:
            assert 'ema' in loaded_obj
            self.ema_unets.load_state_dict(loaded_obj['ema'], strict = strict)

    def load(self, path, only_model = False, strict = True):
        path = Path(path)
        assert path.exists()

        loaded_obj = torch.load(str(path), map_location = 'cpu')

        self.load_state_dict(loaded_obj, only_model = only_model, strict = strict)

        return loaded_obj

    @property
    def unets(self):
        return nn.ModuleList([ema.ema_model for ema in self.ema_unets])

    def increment_step(self, unet_number):
        assert 1 <= unet_number <= self.num_unets

        unet_index_tensor = torch.tensor(unet_number - 1, device = self.steps.device)
        self.steps += F.one_hot(unet_index_tensor, num_classes = len(self.steps))

    def update(self, unet_number = None):
        unet_number = self.validate_and_return_unet_number(unet_number)
        index = unet_number - 1

        optimizer = getattr(self, f'optim{index}')
        scheduler = getattr(self, f'sched{index}')

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.decoder.parameters(), self.max_grad_norm)  # Automatically unscales gradients

        optimizer.step()
        optimizer.zero_grad()

        warmup_scheduler = self.warmup_schedulers[index]
        scheduler_context = warmup_scheduler.dampening if exists(warmup_scheduler) else nullcontext

        with scheduler_context():
            scheduler.step()

        if self.use_ema:
            ema_unet = self.ema_unets[index]
            ema_unet.update()

        self.increment_step(unet_number)

    @torch.no_grad()
    @cast_torch_tensor
    @decoder_sample_in_chunks
    def sample(self, *args, **kwargs):
        distributed = self.accelerator.num_processes > 1
        base_decoder = self.accelerator.unwrap_model(self.decoder)

        was_training = base_decoder.training
        base_decoder.eval()

        if kwargs.pop('use_non_ema', False) or not self.use_ema:
            out = base_decoder.sample(*args, **kwargs, distributed = distributed)
            base_decoder.train(was_training)
            return out

        trainable_unets = self.accelerator.unwrap_model(self.decoder).unets
        base_decoder.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = base_decoder.sample(*args, **kwargs, distributed = distributed)

        base_decoder.unets = trainable_unets             # restore original training unets

        # cast the ema_model unets back to original device
        for ema in self.ema_unets:
            ema.restore_ema_model_device()

        base_decoder.train(was_training)
        return output

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_text(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.decoder).clip.embed_text(*args, **kwargs)

    @torch.no_grad()
    @cast_torch_tensor
    @prior_sample_in_chunks
    def embed_image(self, *args, **kwargs):
        return self.accelerator.unwrap_model(self.decoder).clip.embed_image(*args, **kwargs)

    @cast_torch_tensor
    def forward(
        self,
        *args,
        unet_number = None,
        max_batch_size = None,
        return_lowres_cond_image=False,
        **kwargs
    ):
        unet_number = self.validate_and_return_unet_number(unet_number)

        total_loss = 0.
        cond_images = []
        for chunk_size_frac, (chunked_args, chunked_kwargs) in split_args_and_kwargs(*args, split_size = max_batch_size, **kwargs):
            with self.accelerator.autocast():
                loss_obj = self.decoder(*chunked_args, unet_number = unet_number, return_lowres_cond_image=return_lowres_cond_image, **chunked_kwargs)
                # loss_obj may be a tuple with loss and cond_image
                if return_lowres_cond_image:
                    loss, cond_image = loss_obj
                else:
                    loss = loss_obj
                    cond_image = None
                loss = loss * chunk_size_frac
                if cond_image is not None:
                    cond_images.append(cond_image)

            total_loss += loss.item()

            if self.training:
                self.accelerator.backward(loss)

        if return_lowres_cond_image:
            return total_loss, torch.stack(cond_images)
        else:
            return total_loss
