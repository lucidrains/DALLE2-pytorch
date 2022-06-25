import urllib.request
import os
from pathlib import Path
import shutil
from itertools import zip_longest
from typing import Optional, List, Union
from pydantic import BaseModel

import torch

from dalle2_pytorch.utils import import_or_print_error
from dalle2_pytorch.trainer import DecoderTrainer, DiffusionPriorTrainer

# constants

DEFAULT_DATA_PATH = './.tracker-data'

# helper functions

def exists(val):
    return val is not None

# load file functions

def load_wandb_file(run_path, file_path, **kwargs):
    wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb recall function')
    file_reference = wandb.restore(file_path, run_path=run_path)
    return file_reference.name

def load_local_file(file_path, **kwargs):
    return file_path

class BaseLogger:
    """
    An abstract class representing an object that can log data.
    Parameters:
        data_path (str): A file path for storing temporary data.
        verbose (bool): Whether of not to always print logs to the console.
    """
    def __init__(self, data_path: str, verbose: bool = False, **kwargs):
        self.data_path = Path(data_path)
        self.verbose = verbose

    def init(self, full_config: BaseModel, extra_config: dict, **kwargs) -> None:
        """
        Initializes the logger.
        Errors if the logger is invalid.
        """
        raise NotImplementedError

    def log(self, log, **kwargs) -> None:
        raise NotImplementedError

    def log_images(self, images, captions=[], image_section="images", **kwargs) -> None:
        raise NotImplementedError

    def log_file(self, file_path, **kwargs) -> None:
        raise NotImplementedError

class ConsoleLogger(BaseLogger):
    def init(self, full_config: BaseModel, extra_config: dict, **kwargs) -> None:
        pass

    def log(self, log, **kwargs) -> None:
        print(log)

    def log_images(self, images, captions=[], image_section="images", **kwargs) -> None:
        pass

    def log_file(self, file_path, **kwargs) -> None:
        pass

class WandbLogger(BaseLogger):
    """
    Logs to a wandb run.
    Parameters:
        data_path (str): A file path for storing temporary data.
        wandb_entity (str): The wandb entity to log to.
        wandb_project (str): The wandb project to log to.
        wandb_run_id (str): The wandb run id to resume.
        wandb_run_name (str): The wandb run name to use.
        wandb_resume (bool): Whether to resume a wandb run.
    """
    def __init__(self,
        data_path: str,
        wandb_entity: str,
        wandb_project: str,
        wandb_run_id: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_resume: bool = False,
        **kwargs
    ):
        super().__init__(data_path, **kwargs)
        self.entity = wandb_entity
        self.project = wandb_project
        self.run_id = wandb_run_id
        self.run_name = wandb_run_name
        self.resume = wandb_resume

    def init(self, full_config: BaseModel, extra_config: dict, **kwargs) -> None:
        assert self.entity is not None, "wandb_entity must be specified for wandb logger"
        assert self.project is not None, "wandb_project must be specified for wandb logger"
        self.wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb logger')
        os.environ["WANDB_SILENT"] = "true"
        # Initializes the wandb run
        init_object = {
            "entity": self.entity,
            "project": self.project,
            "config": {**full_config.dict(), **extra_config}
        }
        if self.run_name is not None:
            init_object['name'] = self.run_name
        if self.resume:
            assert self.run_id is not None, '`wandb_run_id` must be provided if `wandb_resume` is True'
            if self.run_name is not None:
                print("You are renaming a run. I hope that is what you intended.")
            init_object['resume'] = 'must'
            init_object['id'] = self.run_id

        self.wandb.init(**init_object)

    def log(self, log, **kwargs) -> None:
        if self.verbose:
            print(log)
        self.wandb.log(log, **kwargs)

    def log_images(self, images, captions=[], image_section="images", **kwargs) -> None:
        """
        Takes a tensor of images and a list of captions and logs them to wandb.
        """
        wandb_images = [self.wandb.Image(image, caption=caption) for image, caption in zip_longest(images, captions)]
        self.wandb.log({ image_section: wandb_images }, **kwargs)

    def log_file(self, file_path, base_path: Optional[str] = None, **kwargs) -> None:
        if base_path is None:
            # Then we take the basepath as the parent of the file_path
            base_path = Path(file_path).parent
        self.wandb.save(str(file_path), base_path = str(base_path))

logger_type_map = {
    'console': ConsoleLogger,
    'wandb': WandbLogger,
}
def create_logger(logger_type: str, data_path: str, **kwargs) -> BaseLogger:
    if logger_type == 'custom':
        raise NotImplementedError('Custom loggers are not supported yet. Please use a different logger type.')
    try:
        logger_class = logger_type_map[logger_type]
    except KeyError:
        raise ValueError(f'Unknown logger type: {logger_type}. Must be one of {list(logger_type_map.keys())}')
    return logger_class(data_path, **kwargs)

class BaseLoader:
    """
    An abstract class representing an object that can load a model checkpoint.
    Parameters:
        data_path (str): A file path for storing temporary data.
    """
    def __init__(self, data_path: str, **kwargs):
        self.data_path = Path(data_path)

    def init(self, logger: BaseLogger, **kwargs) -> None:
        raise NotImplementedError

    def recall() -> dict:
        raise NotImplementedError

class UrlLoader(BaseLoader):
    """
    A loader that downloads the file from a url and loads it
    Parameters:
        data_path (str): A file path for storing temporary data.
        url (str): The url to download the file from.
    """
    def __init__(self, data_path: str, url: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.url = url

    def init(self, logger: BaseLogger, **kwargs) -> None:
        # Makes sure the file exists to be downloaded
        pass  # TODO: Actually implement that

    def recall(self) -> dict:
        # Download the file
        save_path = self.data_path / 'loaded_checkpoint.pth'
        urllib.request.urlretrieve(self.url, str(save_path))
        # Load the file
        return torch.load(str(save_path), map_location='cpu')
        

class LocalLoader(BaseLoader):
    """
    A loader that loads a file from a local path
    Parameters:
        data_path (str): A file path for storing temporary data.
        file_path (str): The path to the file to load.
    """
    def __init__(self, data_path: str, file_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.file_path = Path(file_path)

    def init(self, logger: BaseLogger, **kwargs) -> None:
        # Makes sure the file exists to be loaded
        if not self.file_path.exists():
            raise FileNotFoundError(f'Model not found at {self.file_path}')

    def recall(self) -> dict:
        # Load the file
        return torch.load(str(self.file_path), map_location='cpu')

class WandbLoader(BaseLoader):
    """
    A loader that loads a model from an existing wandb run
    """
    def __init__(self, data_path: str, wandb_file_path: str, wandb_run_path: Optional[str] = None, **kwargs):
        super().__init__(data_path, **kwargs)
        self.run_path = wandb_run_path
        self.file_path = wandb_file_path

    def init(self, logger: BaseLogger, **kwargs) -> None:
        self.wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb recall function')
        # Make sure the file can be downloaded
        if self.wandb.run is not None and self.run_path is None:
            self.run_path = self.wandb.run.path
        assert self.run_path is not None, '`wandb_run_path` must be provided for the wandb loader'
        assert self.file_path is not None, '`wandb_file_path` must be provided for the wandb loader'
        
        os.environ["WANDB_SILENT"] = "true"
        pass  # TODO: Actually implement that

    def recall(self) -> dict:
        file_reference = self.wandb.restore(self.file_path, run_path=self.run_path)
        return torch.load(file_reference.name, map_location='cpu')

loader_type_map = {
    'url': UrlLoader,
    'local': LocalLoader,
    'wandb': WandbLoader,
}
def create_loader(loader_type: str, data_path: str, **kwargs) -> BaseLoader:
    if loader_type == 'custom':
        raise NotImplementedError('Custom loaders are not supported yet. Please use a different loader type.')
    try:
        loader_class = loader_type_map[loader_type]
    except KeyError:
        raise ValueError(f'Unknown loader type: {loader_type}. Must be one of {list(loader_type_map.keys())}')
    return loader_class(data_path, **kwargs)

class BaseSaver:
    def __init__(self,
        data_path: str,
        save_all: bool = False,
        save_latest: bool = True,
        save_best: bool = True,
        save_type: str = 'checkpoint',
        **kwargs
    ):
        self.data_path = Path(data_path)
        assert save_type in ['checkpoint', 'model'], '`save_type` must be one of `checkpoint` or `model`'
        assert save_all or save_latest or save_best, 'At least one of `save_all`, `save_latest`, or `save_best` must be True'
        self.save_all = save_all
        self.save_latest = save_latest
        self.save_best = save_best
        self.save_type = save_type
    
    def init(self, logger: BaseLogger, **kwargs):
        raise NotImplementedError

    def _save_state_dict(self, trainer: Union[DiffusionPriorTrainer, DecoderTrainer], file_path: str, **kwargs) -> Path:
        """
        Gets the state dict to be saved and writes it to file_path.
        If save_type is 'checkpoint', we save the entire trainer state dict.
        If save_type is 'model', we save only the model state dict.
        """
        if self.save_type == 'checkpoint':
            trainer.save(file_path, overwrite=True, **kwargs)
        elif self.save_type == 'model':
            if isinstance(trainer, DiffusionPriorTrainer):
                prior = trainer.ema_diffusion_prior.ema_model if trainer.use_ema else trainer.diffusion_prior
                state_dict = trainer.unwrap_model(prior).state_dict()
                torch.save(state_dict, file_path)
            elif isinstance(trainer, DecoderTrainer):
                decoder = trainer.accelerator.unwrap_model(trainer.decoder)
                if trainer.use_ema:
                    trainable_unets = decoder.unets
                    decoder.unets = trainer.unets  # Swap EMA unets in
                    state_dict = decoder.state_dict()
                    decoder.unets = trainable_unets  # Swap back
                else:
                    state_dict = decoder.state_dict()
                torch.save(state_dict, file_path)
            else:
                raise NotImplementedError('Saving this type of model with EMA mode enabled is not yet implemented. Actually, how did you get here?')
        return Path(file_path)

    def save_config(self, config_path, config_name = 'config.json'):
        """
        Uploads the config under the given name
        """
        raise NotImplementedError

    def save(self, trainer, is_best: bool, is_latest: bool, **kwargs):
        """
        Saves the checkpoint or model
        """
        raise NotImplementedError

class LocalSaver(BaseSaver):
    def __init__(self, data_path: str, file_path: str, **kwargs):
        super().__init__(data_path, **kwargs)
        self.file_path = Path(file_path)

    def init(self, logger: BaseLogger, **kwargs):
        # Makes sure the directory exists to be saved to
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def save_config(self, config_path, config_name = 'config.json'):
        # Copy the config to file_path / config_name
        config_path = Path(config_path)
        new_path = self.file_path / config_name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(config_path), str(new_path))

    def save(self, trainer, is_best: bool, is_latest: bool, epoch: int, sample: int, **kwargs):
        # Create a place to house the checkpoint temporarily
        checkpoint_path = self.data_path / f'checkpoint_{epoch}_{sample}.pt'
        # Save the checkpoint
        self._save_state_dict(trainer, checkpoint_path, epoch=epoch, sample=sample, **kwargs)
        # Save the file
        if self.save_latest and is_latest:
            # Copy the checkpoint to file_path / latest.pth
            latest_path = self.file_path / 'latest.pth'
            shutil.copyfile(checkpoint_path, latest_path)
        if self.save_best and is_best:
            # Copy the checkpoint to file_path / best.pth
            best_path = self.file_path / 'best.pth'
            shutil.copyfile(checkpoint_path, best_path)
        if self.save_all:
            # Copy the checkpoint to file_path / checkpoints / epoch_step.pth
            all_path = self.file_path / 'checkpoints' / f'{epoch}_{sample}.pth'
            shutil.copyfile(checkpoint_path, all_path)
        # Remove the temporary checkpoint
        checkpoint_path.unlink()

class WandbSaver(BaseSaver):
    def __init__(self, data_path: str, wandb_run_path: Optional[str] = None, **kwargs):
        super().__init__(data_path, **kwargs)
        self.run_path = wandb_run_path

    def init(self, logger: BaseLogger, **kwargs):
        self.wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb logger')
        os.environ["WANDB_SILENT"] = "true"
        # Makes sure that the user can upload tot his run
        if self.run_path is not None:
            entity, project, run_id = run_path.split("/")
            self.run = self.wandb.init(entity=entity, project=project, id=run_id)
        else:
            assert self.wandb.run is not None, 'You must be using the wandb logger if you are saving to wandb and have not set `wandb_run_path`'
            self.run = self.wandb.run
        # TODO: Now actually check if upload is possible

    def save_config(self, config_path, config_name = 'config.json'):
        # Upload the config to wandb
        config_path = Path(config_path)
        new_path = self.data_path / config_name
        shutil.copy(config_path, new_path)
        self.run.save(str(new_path), base_path = str(self.data_path), policy='now')

    def save(self, trainer, is_best: bool, is_latest: bool, epoch: int, sample: int, **kwargs):
        # Create a place to house the checkpoint temporarily
        checkpoint_path = self.data_path / f'checkpoint_{epoch}_{sample}.pt'
        # Save the checkpoint
        self._save_state_dict(trainer, checkpoint_path, epoch=epoch, sample=sample, **kwargs)
        # Save the file
        if self.save_latest and is_latest:
            # Copy the checkpoint to file_path / latest.pth
            latest_path = self.data_path / 'latest.pth'
            shutil.copyfile(checkpoint_path, latest_path)
            self.run.save(str(latest_path), base_path = str(self.data_path), policy='now')
        if self.save_best and is_best:
            # Copy the checkpoint to file_path / best.pth
            best_path = self.data_path / 'best.pth'
            shutil.copyfile(checkpoint_path, best_path)
            self.run.save(str(best_path), base_path = str(self.data_path), policy='now')
        if self.save_all:
            # Copy the checkpoint to file_path / checkpoints / epoch_step.pth
            all_path = self.data_path / 'checkpoints' / f'{epoch}_{sample}.pth'
            shutil.copyfile(checkpoint_path, all_path)
            self.run.save(str(all_path), base_path = str(self.data_path), policy='now')
        # Remove the temporary checkpoint
        checkpoint_path.unlink()

class HuggingfaceSaver(BaseSaver):
    def __init__(self, data_path: str, huggingface_repo: str, huggingface_base_path: Optional[str] = "./", token_path: Optional[str] = None, **kwargs):
        super().__init__(data_path, **kwargs)
        self.huggingface_repo = huggingface_repo
        self.hf_base_path = huggingface_base_path
        self.token_path = token_path

    def init(self, logger: BaseLogger, **kwargs):
        # Makes sure this user can upload to the repo
        self.hub = import_or_print_error('huggingface_hub', '`pip install huggingface_hub` to use the huggingface saver')
        try:
            identity = self.hub.whoami()  # Errors if not logged in
            # Then we are logged in
        except:
            # We are not logged in. Use the token_path to set the token.
            if not os.path.exists(self.token_path):
                raise Exception("Not logged in to huggingface and no token_path specified. Please login with `huggingface-cli login` or if that does not work set the token_path.")
            with open(self.token_path, "r") as f:
                token = f.read().strip()
            self.hub.HfApi.set_access_token(token)
            identity = self.hub.whoami()

    def save_config(self, config_path, config_name = 'config.json'):
        self.hub.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo=str(Path(self.hf_base_path) / config_name),
            repo_id=self.huggingface_repo
        )

    def save(self, trainer, is_best: bool, is_latest: bool, epoch: int, sample: int, **kwargs):
        # Create a place to house the checkpoint temporarily
        checkpoint_path = self.data_path / f'checkpoint_{epoch}_{sample}.pt'
        # Save the checkpoint
        self._save_state_dict(trainer, checkpoint_path, epoch=epoch, sample=sample, **kwargs)
        # Save the file
        if self.save_latest and is_latest:
            self.hub.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=str(Path(self.hf_base_path) / 'latest.pth'),
                repo_id=self.huggingface_repo
            )
        if self.save_best and is_best:
            self.hub.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=str(Path(self.hf_base_path) / 'best.pth'),
                repo_id=self.huggingface_repo
            )
        if self.save_all:
            self.hub.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo=str(Path(self.hf_base_path) / 'checkpoints' / f'{epoch}_{sample}.pth'),
                repo_id=self.huggingface_repo
            )
        # Remove the temporary checkpoint
        checkpoint_path.unlink()

saver_type_map = {
    'local': LocalSaver,
    'wandb': WandbSaver,
    'huggingface': HuggingfaceSaver
}
def create_saver(saver_type: str, data_path: str, **kwargs) -> BaseSaver:
    if saver_type == 'custom':
        raise NotImplementedError('Custom savers are not supported yet. Please use a different saver type.')
    try:
        saver_class = saver_type_map[saver_type]
    except KeyError:
        raise ValueError(f'Unknown saver type: {saver_type}. Must be one of {list(saver_type_map.keys())}')
    return saver_class(data_path, **kwargs)



# base class
class Tracker:
    def __init__(self, data_path: Optional[str] = DEFAULT_DATA_PATH, overwrite_data_path: bool = False, dummy_mode: bool = False):
        self.data_path = Path(data_path)
        if not dummy_mode:
            if overwrite_data_path:
                if self.data_path.exists():
                    shutil.rmtree(self.data_path)
                self.data_path.mkdir(parents=True)
            else:
                assert not self.data_path.exists(), f'Data path {self.data_path} already exists. Set overwrite_data_path to True to overwrite.'
                if not self.data_path.exists():
                    self.data_path.mkdir(parents=True)
        self.logger: BaseLogger = None
        self.loader: Optional[BaseLoader] = None
        self.savers: List[BaseSaver]= []
        self.dummy_mode = dummy_mode

    def init(self, full_config: BaseModel, extra_config: dict):
        assert self.logger is not None, '`logger` must be set before `init` is called'
        if self.dummy_mode:
            # The only thing we need is a loader
            if self.loader is not None:
                self.loader.init(self.logger)
            return
        assert len(self.savers) > 0, '`savers` must be set before `init` is called'
        self.logger.init(full_config, extra_config)
        if self.loader is not None:
            self.loader.init(self.logger)
        for saver in self.savers:
            saver.init(self.logger)

    def add_logger(self, logger: BaseLogger):
        if self.dummy_mode:
            return
        self.logger = logger

    def add_loader(self, loader: BaseLoader):
        if self.dummy_mode:
            return
        self.loader = loader

    def add_saver(self, saver: BaseSaver):
        if self.dummy_mode:
            return
        self.savers.append(saver)

    def log(self, *args, **kwargs):
        if self.dummy_mode:
            return
        self.logger.log(*args, **kwargs)
    
    def log_images(self, *args, **kwargs):
        if self.dummy_mode:
            return
        self.logger.log_images(*args, **kwargs)

    def log_file(self, *args, **kwargs):
        if self.dummy_mode:
            return
        self.logger.log_file(*args, **kwargs)

    def save_config(self, config_path, config_name = 'config.json'):
        if self.dummy_mode:
            return
        for saver in self.savers:
            saver.save_config(config_path, config_name)
    
    def save(self, trainer, is_best: bool, is_latest: bool, *args, **kwargs):
        if self.dummy_mode:
            return
        for saver in self.savers:
            saver.save(trainer, is_best, is_latest, *args, **kwargs)
    
    def recall(self):
        if self.loader is not None:
            return self.loader.recall()
        else:
            raise ValueError('No loader specified')


    