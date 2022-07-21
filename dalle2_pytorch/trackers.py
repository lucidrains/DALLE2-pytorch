import urllib.request
import os
import json
from pathlib import Path
import shutil
from itertools import zip_longest
from typing import Any, Optional, List, Union
from pydantic import BaseModel

import torch
from dalle2_pytorch.dalle2_pytorch import Decoder, DiffusionPrior
from dalle2_pytorch.utils import import_or_print_error
from dalle2_pytorch.trainer import DecoderTrainer, DiffusionPriorTrainer
from dalle2_pytorch.version import __version__
from packaging import version

# constants

DEFAULT_DATA_PATH = './.tracker-data'

# helper functions

def exists(val):
    return val is not None

class BaseLogger:
    """
    An abstract class representing an object that can log data.
    Parameters:
        data_path (str): A file path for storing temporary data.
        verbose (bool): Whether of not to always print logs to the console.
    """
    def __init__(self, data_path: str, resume: bool = False, auto_resume: bool = False, verbose: bool = False, **kwargs):
        self.data_path = Path(data_path)
        self.resume = resume
        self.auto_resume = auto_resume
        self.verbose = verbose

    def init(self, full_config: BaseModel, extra_config: dict, **kwargs) -> None:
        """
        Initializes the logger.
        Errors if the logger is invalid.
        full_config is the config file dict while extra_config is anything else from the script that is not defined the config file.
        """
        raise NotImplementedError

    def log(self, log, **kwargs) -> None:
        raise NotImplementedError

    def log_images(self, images, captions=[], image_section="images", **kwargs) -> None:
        raise NotImplementedError

    def log_file(self, file_path, **kwargs) -> None:
        raise NotImplementedError

    def log_error(self, error_string, **kwargs) -> None:
        raise NotImplementedError

    def get_resume_data(self, **kwargs) -> dict:
        """
        Sets tracker attributes that along with { "resume": True } will be used to resume training.
        It is assumed that after init is called this data will be complete.
        If the logger does not have any resume functionality, it should return an empty dict.
        """
        raise NotImplementedError

class ConsoleLogger(BaseLogger):
    def init(self, full_config: BaseModel, extra_config: dict, **kwargs) -> None:
        print("Logging to console")

    def log(self, log, **kwargs) -> None:
        print(log)

    def log_images(self, images, captions=[], image_section="images", **kwargs) -> None:
        pass

    def log_file(self, file_path, **kwargs) -> None:
        pass

    def log_error(self, error_string, **kwargs) -> None:
        print(error_string)

    def get_resume_data(self, **kwargs) -> dict:
        return {}

class WandbLogger(BaseLogger):
    """
    Logs to a wandb run.
    Parameters:
        data_path (str): A file path for storing temporary data.
        wandb_entity (str): The wandb entity to log to.
        wandb_project (str): The wandb project to log to.
        wandb_run_id (str): The wandb run id to resume.
        wandb_run_name (str): The wandb run name to use.
    """
    def __init__(self,
        data_path: str,
        wandb_entity: str,
        wandb_project: str,
        wandb_run_id: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(data_path, **kwargs)
        self.entity = wandb_entity
        self.project = wandb_project
        self.run_id = wandb_run_id
        self.run_name = wandb_run_name

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
        print(f"Logging to wandb run {self.wandb.run.path}-{self.wandb.run.name}")

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

    def log_error(self, error_string, step=None, **kwargs) -> None:
        if self.verbose:
            print(error_string)
        self.wandb.log({"error": error_string, **kwargs}, step=step)

    def get_resume_data(self, **kwargs) -> dict:
        # In order to resume, we need wandb_entity, wandb_project, and wandb_run_id
        return {
            "entity": self.entity,
            "project": self.project,
            "run_id": self.wandb.run.id
        }

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
    def __init__(self, data_path: str, only_auto_resume: bool = False, **kwargs):
        self.data_path = Path(data_path)
        self.only_auto_resume = only_auto_resume

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
        if not self.file_path.exists() and not self.only_auto_resume:
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
            assert self.run_path is not None, 'wandb run was not found to load from. If not using the wandb logger must specify the `wandb_run_path`.'
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
        save_latest_to: Optional[Union[str, bool]] = None,
        save_best_to: Optional[Union[str, bool]] = None,
        save_meta_to: Optional[str] = None,
        save_type: str = 'checkpoint',
        **kwargs
    ):
        self.data_path = Path(data_path)
        self.save_latest_to = save_latest_to
        self.saving_latest = save_latest_to is not None and save_latest_to is not False
        self.save_best_to = save_best_to
        self.saving_best = save_best_to is not None and save_best_to is not False
        self.save_meta_to = save_meta_to
        self.saving_meta = save_meta_to is not None
        self.save_type = save_type
        assert save_type in ['checkpoint', 'model'], '`save_type` must be one of `checkpoint` or `model`'
        assert self.saving_latest or self.saving_best or self.saving_meta, 'At least one saving option must be specified'

    def init(self, logger: BaseLogger, **kwargs) -> None:
        raise NotImplementedError

    def save_file(self, local_path: Path, save_path: str, is_best=False, is_latest=False, **kwargs) -> None:
        """
        Save a general file under save_meta_to
        """
        raise NotImplementedError

class LocalSaver(BaseSaver):
    def __init__(self,
        data_path: str,
        **kwargs
    ):
        super().__init__(data_path, **kwargs)

    def init(self, logger: BaseLogger, **kwargs) -> None:
        # Makes sure the directory exists to be saved to
        print(f"Saving {self.save_type} locally")
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True)

    def save_file(self, local_path: str, save_path: str, **kwargs) -> None:
        # Copy the file to save_path
        save_path_file_name = Path(save_path).name
        # Make sure parent directory exists
        save_path_parent = Path(save_path).parent
        if not save_path_parent.exists():
            save_path_parent.mkdir(parents=True)
        print(f"Saving {save_path_file_name} {self.save_type} to local path {save_path}")
        shutil.copy(local_path, save_path)

class WandbSaver(BaseSaver):
    def __init__(self, data_path: str, wandb_run_path: Optional[str] = None, **kwargs):
        super().__init__(data_path, **kwargs)
        self.run_path = wandb_run_path

    def init(self, logger: BaseLogger, **kwargs) -> None:
        self.wandb = import_or_print_error('wandb', '`pip install wandb` to use the wandb logger')
        os.environ["WANDB_SILENT"] = "true"
        # Makes sure that the user can upload tot his run
        if self.run_path is not None:
            entity, project, run_id = self.run_path.split("/")
            self.run = self.wandb.init(entity=entity, project=project, id=run_id)
        else:
            assert self.wandb.run is not None, 'You must be using the wandb logger if you are saving to wandb and have not set `wandb_run_path`'
            self.run = self.wandb.run
        # TODO: Now actually check if upload is possible
        print(f"Saving to wandb run {self.run.path}-{self.run.name}")

    def save_file(self, local_path: Path, save_path: str, **kwargs) -> None:
        # In order to log something in the correct place in wandb, we need to have the same file structure here
        save_path_file_name = Path(save_path).name
        print(f"Saving {save_path_file_name} {self.save_type} to wandb run {self.run.path}-{self.run.name}")
        save_path = Path(self.data_path) / save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, save_path)
        self.run.save(str(save_path), base_path = str(self.data_path), policy='now')

class HuggingfaceSaver(BaseSaver):
    def __init__(self, data_path: str, huggingface_repo: str, token_path: Optional[str] = None, **kwargs):
        super().__init__(data_path, **kwargs)
        self.huggingface_repo = huggingface_repo
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
        print(f"Saving to huggingface repo {self.huggingface_repo}")

    def save_file(self, local_path: Path, save_path: str, **kwargs) -> None:
        # Saving to huggingface is easy, we just need to upload the file with the correct name
        save_path_file_name = Path(save_path).name
        print(f"Saving {save_path_file_name} {self.save_type} to huggingface repo {self.huggingface_repo}")
        self.hub.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=str(save_path),
            repo_id=self.huggingface_repo
        )
        
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


class Tracker:
    def __init__(self, data_path: Optional[str] = DEFAULT_DATA_PATH, overwrite_data_path: bool = False, dummy_mode: bool = False):
        self.data_path = Path(data_path)
        if not dummy_mode:
            if not overwrite_data_path:
                assert not self.data_path.exists(), f'Data path {self.data_path} already exists. Set overwrite_data_path to True to overwrite.'
                if not self.data_path.exists():
                    self.data_path.mkdir(parents=True)
        self.logger: BaseLogger = None
        self.loader: Optional[BaseLoader] = None
        self.savers: List[BaseSaver]= []
        self.dummy_mode = dummy_mode

    def _load_auto_resume(self) -> bool:
        # If the file does not exist, we return False. If autoresume is enabled we print a warning so that the user can know that this is the first run.
        if not self.auto_resume_path.exists():
            if self.logger.auto_resume:
                print("Auto_resume is enabled but no auto_resume.json file exists. Assuming this is the first run.")
            return False

        # Now we know that the autoresume file exists, but if we are not auto resuming we should remove it so that we don't accidentally load it next time
        if not self.logger.auto_resume:
            print(f'Removing auto_resume.json because auto_resume is not enabled in the config')
            self.auto_resume_path.unlink()
            return False

        # Otherwise we read the json into a dictionary will will override parts of logger.__dict__
        with open(self.auto_resume_path, 'r') as f:
            auto_resume_dict = json.load(f)
        # Check if the logger is of the same type as the autoresume save
        if auto_resume_dict["logger_type"] != self.logger.__class__.__name__:
            raise Exception(f'The logger type in the auto_resume file is {auto_resume_dict["logger_type"]} but the current logger is {self.logger.__class__.__name__}. Either use the original logger type, set `auto_resume` to `False`, or delete your existing tracker-data folder.')
        # Then we are ready to override the logger with the autoresume save
        self.logger.__dict__["resume"] = True
        print(f"Updating {self.logger.__dict__} with {auto_resume_dict}")
        self.logger.__dict__.update(auto_resume_dict)
        return True

    def _save_auto_resume(self):
        # Gets the autoresume dict from the logger and adds "logger_type" to it then saves it to the auto_resume file
        auto_resume_dict = self.logger.get_resume_data()
        auto_resume_dict['logger_type'] = self.logger.__class__.__name__
        with open(self.auto_resume_path, 'w') as f:
            json.dump(auto_resume_dict, f)

    def init(self, full_config: BaseModel, extra_config: dict):
        self.auto_resume_path = self.data_path / 'auto_resume.json'
        # Check for resuming the run
        self.did_auto_resume = self._load_auto_resume()
        if self.did_auto_resume:
            print(f'\n\nWARNING: RUN HAS BEEN AUTO-RESUMED WITH THE LOGGER TYPE {self.logger.__class__.__name__}.\nIf this was not your intention, stop this run and set `auto_resume` to `False` in the config.\n\n')
            print(f"New logger config: {self.logger.__dict__}")
        
        self.save_metadata = dict(
            version = version.parse(__version__)
        )  # Data that will be saved alongside the checkpoint or model
        self.blacklisted_checkpoint_metadata_keys = ['scaler', 'optimizer', 'model', 'version', 'step', 'steps']  # These keys would cause us to error if we try to save them as metadata

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

        if self.logger.auto_resume:
            # Then we need to save the autoresume file. It is assumed after logger.init is called that the logger is ready to be saved.
            self._save_auto_resume()

    def add_logger(self, logger: BaseLogger):
        self.logger = logger

    def add_loader(self, loader: BaseLoader):
        self.loader = loader

    def add_saver(self, saver: BaseSaver):
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

    def save_config(self, current_config_path: str, config_name = 'config.json'):
        if self.dummy_mode:
            return
        # Save the config under config_name in the root folder of data_path
        shutil.copy(current_config_path, self.data_path / config_name)
        for saver in self.savers:
            if saver.saving_meta:
                remote_path = Path(saver.save_meta_to) / config_name
                saver.save_file(current_config_path, str(remote_path))

    def add_save_metadata(self, state_dict_key: str, metadata: Any):
        """
        Adds a new piece of metadata that will be saved along with the model or decoder.
        """
        self.save_metadata[state_dict_key] = metadata

    def _save_state_dict(self, trainer: Union[DiffusionPriorTrainer, DecoderTrainer], save_type: str, file_path: str, **kwargs) -> Path:
        """
        Gets the state dict to be saved and writes it to file_path.
        If save_type is 'checkpoint', we save the entire trainer state dict.
        If save_type is 'model', we save only the model state dict.
        """
        assert save_type in ['checkpoint', 'model']
        if save_type == 'checkpoint':
            # Create a metadata dict without the blacklisted keys so we do not error when we create the state dict
            metadata = {k: v for k, v in self.save_metadata.items() if k not in self.blacklisted_checkpoint_metadata_keys}
            trainer.save(file_path, overwrite=True, **kwargs, **metadata)
        elif save_type == 'model':
            if isinstance(trainer, DiffusionPriorTrainer):
                prior = trainer.ema_diffusion_prior.ema_model if trainer.use_ema else trainer.diffusion_prior
                state_dict = trainer.accelerator.unwrap_model(prior).state_dict()
                torch.save(state_dict, file_path)
            elif isinstance(trainer, DecoderTrainer):
                decoder: Decoder = trainer.accelerator.unwrap_model(trainer.decoder)
                # Remove CLIP if it is part of the model
                original_clip = decoder.clip
                decoder.clip = None
                if trainer.use_ema:
                    trainable_unets = decoder.unets
                    decoder.unets = trainer.unets  # Swap EMA unets in
                    model_state_dict = decoder.state_dict()
                    decoder.unets = trainable_unets  # Swap back
                else:
                    model_state_dict = decoder.state_dict()
                decoder.clip = original_clip
            else:
                raise NotImplementedError('Saving this type of model with EMA mode enabled is not yet implemented. Actually, how did you get here?')
            state_dict = {
                **self.save_metadata,
                'model': model_state_dict
            }
            torch.save(state_dict, file_path)
        return Path(file_path)

    def save(self, trainer, is_best: bool, is_latest: bool, **kwargs):
        if self.dummy_mode:
            return
        if not is_best and not is_latest:
            # Nothing to do
            return
        # Save the checkpoint and model to data_path
        checkpoint_path = self.data_path / 'checkpoint.pth'
        self._save_state_dict(trainer, 'checkpoint', checkpoint_path, **kwargs)
        model_path = self.data_path / 'model.pth'
        self._save_state_dict(trainer, 'model', model_path, **kwargs)
        print("Saved cached models")
        # Call the save methods on the savers
        for saver in self.savers:
            local_path = checkpoint_path if saver.save_type == 'checkpoint' else model_path
            if saver.saving_latest and is_latest:
                latest_checkpoint_path = saver.save_latest_to.format(**kwargs)
                try:
                    saver.save_file(local_path, latest_checkpoint_path, is_latest=True, **kwargs)
                except Exception as e:
                    self.logger.log_error(f'Error saving checkpoint: {e}', **kwargs)
                    print(f'Error saving checkpoint: {e}')
            if saver.saving_best and is_best:
                best_checkpoint_path = saver.save_best_to.format(**kwargs)
                try:
                    saver.save_file(local_path, best_checkpoint_path, is_best=True, **kwargs)
                except Exception as e:
                    self.logger.log_error(f'Error saving checkpoint: {e}', **kwargs)
                    print(f'Error saving checkpoint: {e}')
    
    @property
    def can_recall(self):
        # Defines whether a recall can be performed.
        return self.loader is not None and (not self.loader.only_auto_resume or self.did_auto_resume)
    
    def recall(self):
        if self.can_recall:
            return self.loader.recall()
        else:
            raise ValueError('Tried to recall, but no loader was set or auto-resume was not performed.')


    