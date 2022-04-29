import copy
import torch
from torch import nn

# exponential moving average wrapper

class EMA(nn.Module):
    def __init__(
        self,
        model,
        beta = 0.99,
        ema_update_after_step = 1000,
        ema_update_every = 10,
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model
        self.ema_model = copy.deepcopy(model)

        self.ema_update_after_step = ema_update_after_step # only start EMA after this step number, starting at 0
        self.ema_update_every = ema_update_every

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0.]))

    def update(self):
        self.step += 1

        if self.step <= self.ema_update_after_step or (self.step % self.ema_update_every) != 0:
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
