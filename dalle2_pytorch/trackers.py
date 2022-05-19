import os
from itertools import zip_longest
from enum import Enum
import torch
from torch import nn

# helper functions

def exists(val):
    return val is not None

def load_wandb_state_dict(run_path, file_path, **kwargs):
    try:
        import wandb
    except ImportError as e:
        print('`pip install wandb` to use the wandb recall function')
        raise e
    file_reference = wandb.restore(file_path, run_path=run_path)
    return torch.load(file_reference.name)

def load_local_state_dict(file_path, **kwargs):
    return torch.load(file_path)

# base class

class BaseTracker(nn.Module):
    def __init__(self, data_path):
        super().__init__()
        assert data_path is not None, "Tracker must have a data_path to save local content"
        self.data_path = os.path.abspath(data_path)
        os.makedirs(self.data_path, exist_ok=True)

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

    def log_images(self, images, **kwargs):
        """
        Currently, do nothing with console logged images 
        """
        pass
    
    def save_state_dict(self, state_dict, relative_path, **kwargs):
        torch.save(state_dict, os.path.join(self.data_path, relative_path))

# basic wandb class

class WandbTracker(BaseTracker):
    def __init__(self, data_path):
        super().__init__(data_path)
        try:
            import wandb
        except ImportError as e:
            print('`pip install wandb` to use the wandb experiment tracker')
            raise e

        os.environ["WANDB_SILENT"] = "true"
        self.wandb = wandb

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
        full_path = os.path.join(self.data_path, relative_path)
        torch.save(state_dict, full_path)
        self.wandb.save(full_path, base_path=self.data_path)  # Upload and keep relative to data_path