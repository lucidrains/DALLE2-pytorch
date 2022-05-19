"""
Defines the default values for the decoder config
"""

from enum import Enum
class ConfigField(Enum):
    REQUIRED = 0  # This had more options. It's a bit unnecessary now, but I can't think of a better way to do it.

default_config = {
    "unets": ConfigField.REQUIRED,
    "decoder": {
        "image_sizes": ConfigField.REQUIRED,  # The side lengths of the upsampled image at the end of each unet
        "image_size": ConfigField.REQUIRED,  # Usually the same as image_sizes[-1] I think
        "channels": 3,
        "timesteps": 1000,
        "loss_type": "l2",
        "beta_schedule": "cosine",
        "learned_variance": True
    },
    "data": {
        "webdataset_base_url": ConfigField.REQUIRED,  # Path to a webdataset with jpg images
        "embeddings_url": ConfigField.REQUIRED,  # Path to .npy files with embeddings
        "num_workers": 4,
        "batch_size": 64,
        "start_shard": 0,
        "end_shard": 9999999,
        "shard_width": 6,
        "index_width": 4,
        "splits": {
            "train": 0.75,
            "val": 0.15,
            "test": 0.1
        },
        "shuffle_train": True,
        "resample_train": False,
        "preprocessing": {
            "ToTensor": True
        }
    },
    "train": {
        "epochs": 20,
        "lr": 1e-4,
        "wd": 0.01,
        "max_grad_norm": 0.5,
        "save_every_n_samples": 100000,
        "n_sample_images": 6,  # The number of example images to produce when sampling the train and test dataset
        "device": "cuda:0",
        "epoch_samples": None,  # Limits the number of samples per epoch. None means no limit. Required if resample_train is true as otherwise the number of samples per epoch is infinite.
        "validation_samples": None,  # Same as above but for validation.
        "use_ema": True,
        "ema_beta": 0.99,
        "amp": False,
        "save_all": False,  # Whether to preserve all checkpoints
        "save_latest": True,  # Whether to always save the latest checkpoint
        "save_best": True,  # Whether to save the best checkpoint
        "unet_training_mask": None  # If None, use all unets
    },
    "evaluate": {
        "n_evalation_samples": 1000,
        "FID": None,
        "IS": None,
        "KID": None,
        "LPIPS": None
    },
    "tracker": {
        "tracker_type": "console",  # Decoder currently supports console and wandb
        "data_path": "./models",  # The path where files will be saved locally

        "wandb_entity": "",  # Only needs to be set if tracker_type is wandb
        "wandb_project": "",

        "verbose": False  # Whether to print console logging for non-console trackers
    },
    "load": {
        "source": None,  # Supports file and wandb

        "run_path": "",  # Used only if source is wandb
        "file_path": "",  # The local filepath if source is file. If source is wandb, the relative path to the model file in wandb.

        "resume": False  # If using wandb, whether to resume the run
    }
}
