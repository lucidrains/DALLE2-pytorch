import os
import torch
from torch import nn

# helper functions

def exists(val):
    return val is not None

# base class

class BaseTracker(nn.Module):
    def __init__(self):
        super().__init__()

    def init(self, config, **kwargs):
        raise NotImplementedError

    def log(self, log, **kwargs):
        raise NotImplementedError

# basic stdout class

class ConsoleTracker(BaseTracker):
    def init(self, **config):
        print(config)

    def log(self, log, **kwargs):
        print(log)

# basic wandb class

class WandbTracker(BaseTracker):
    def __init__(self):
        super().__init__()
        try:
            import wandb
        except ImportError as e:
            print('`pip install wandb` to use the wandb experiment tracker')
            raise e

        os.environ["WANDB_SILENT"] = "true"
        self.wandb = wandb

    def init(self, **config):
        self.wandb.init(**config)

    def log(self, log, **kwargs):
        self.wandb.log(log, **kwargs)
