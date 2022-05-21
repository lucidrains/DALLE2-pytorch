import os
from pathlib import Path
import importlib
from itertools import zip_longest

import torch
from torch import nn

# constants

DEFAULT_DATA_PATH = './.tracker-data'

# helper functions

def exists(val):
    return val is not None

def import_or_print_error(pkg_name, err_str = None):
    try:
        return importlib.import_module(pkg_name)
    except ModuleNotFoundError as e:
        if exists(err_str):
            print(err_str)
        exit()

# load state dict functions

def load_wandb_state_dict(run_path, file_path, **kwargs):
    wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb recall function')
    file_reference = wandb.restore(file_path, run_path=run_path)
    return torch.load(file_reference.name)

def load_local_state_dict(file_path, **kwargs):
    return torch.load(file_path)

# base class

class BaseTracker(nn.Module):
    def __init__(self, data_path = DEFAULT_DATA_PATH):
        super().__init__()
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents = True, exist_ok = True)

    def init(self, config, **kwargs):
        raise NotImplementedError

    def log(self, log, **kwargs):
        raise NotImplementedError

    def log_images(self, images, **kwargs):
        raise NotImplementedError

    def save_state_dict(self, state_dict, relative_path, **kwargs):
        raise NotImplementedError

    def recall_state_dict(self, recall_source, *args, **kwargs):
        """
        Loads a state dict from any source.
        Since a user may wish to load a model from a different source than their own tracker (i.e. tracking using wandb but recalling from disk),
            this should not be linked to any individual tracker.
        """
        # TODO: Pull this into a dict or something similar so that we can add more sources without having a massive switch statement
        if recall_source == 'wandb':
            return load_wandb_state_dict(*args, **kwargs)
        elif recall_source == 'local':
            return load_local_state_dict(*args, **kwargs)
        else:
            raise ValueError('`recall_source` must be one of `wandb` or `local`')


# basic stdout class

class ConsoleTracker(BaseTracker):
    def init(self, **config):
        print(config)

    def log(self, log, **kwargs):
        print(log)

    def log_images(self, images, **kwargs): # noop for logging images
        pass
    
    def save_state_dict(self, state_dict, relative_path, **kwargs):
        torch.save(state_dict, str(self.data_path / relative_path))

# basic wandb class

class WandbTracker(BaseTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb experiment tracker')
        os.environ["WANDB_SILENT"] = "true"

    def init(self, **config):
        self.wandb.init(**config)

    def log(self, log, verbose=False, **kwargs):
        if verbose:
            print(log)
        self.wandb.log(log, **kwargs)

    def log_images(self, images, captions=[], image_section="images", **kwargs):
        """
        Takes a tensor of images and a list of captions and logs them to wandb.
        """
        wandb_images = [self.wandb.Image(image, caption=caption) for image, caption in zip_longest(images, captions)]
        self.log({ image_section: wandb_images }, **kwargs)
    
    def save_state_dict(self, state_dict, relative_path, **kwargs):
        """
        Saves a state_dict to disk and uploads it 
        """
        full_path = str(self.data_path / relative_path)
        torch.save(state_dict, full_path)
        self.wandb.save(full_path, base_path = str(self.data_path))  # Upload and keep relative to data_path
