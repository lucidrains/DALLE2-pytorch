import copy
from functools import partial

import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from dalle2_pytorch.dalle2_pytorch import Decoder
from dalle2_pytorch.optimizer import get_optimizer

# helper functions

def exists(val):
    return val is not None

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

# exponential moving average wrapper

class EMA(nn.Module):
    def __init__(
        self,
        model,
        beta = 0.99,
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

# trainers

class DecoderTrainer(nn.Module):
    def __init__(
        self,
        decoder,
        use_ema = True,
        lr = 3e-4,
        wd = 1e-2,
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

        lr, wd = map(partial(cast_tuple, length = self.num_unets), (lr, wd))

        for ind, (unet, unet_lr, unet_wd) in enumerate(zip(self.decoder.unets, lr, wd)):
            optimizer = get_optimizer(
                unet.parameters(),
                lr = unet_lr,
                wd = unet_wd,
                **kwargs
            )

            setattr(self, f'optim{ind}', optimizer) # cannot use pytorch ModuleList for some reason with optimizers

            if self.use_ema:
                self.ema_unets.append(EMA(unet, **ema_kwargs))

            scaler = GradScaler(enabled = amp)
            setattr(self, f'scaler{ind}', scaler)

        # gradient clipping if needed

        self.max_grad_norm = max_grad_norm

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

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        if self.use_ema:
            trainable_unets = self.decoder.unets
            self.decoder.unets = self.unets                  # swap in exponential moving averaged unets for sampling

        output = self.decoder.sample(*args, **kwargs)

        if self.use_ema:
            self.decoder.unets = trainable_unets             # restore original training unets
        return output

    def forward(
        self,
        x,
        *,
        unet_number,
        divisor = 1,
        **kwargs
    ):
        with autocast(enabled = self.amp):
            loss = self.decoder(x, unet_number = unet_number, **kwargs)
        return self.scale(loss / divisor, unet_number = unet_number)
