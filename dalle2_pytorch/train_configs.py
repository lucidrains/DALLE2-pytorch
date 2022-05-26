import json
from torchvision import transforms as T
from pydantic import BaseModel, validator, root_validator
from typing import List, Iterable, Optional, Union, Tuple, Dict, Any

from x_clip import CLIP as XCLIP
from coca_pytorch import CoCa

from dalle2_pytorch.dalle2_pytorch import (
    CoCaAdapter,
    OpenAIClipAdapter,
    Unet,
    Decoder,
    DiffusionPrior,
    DiffusionPriorNetwork,
    XClipAdapter,
)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]

def SingularOrIterable(inner_type):
    return Union[inner_type, ListOrTuple(inner_type)]

# general pydantic classes

class TrainSplitConfig(BaseModel):
    train: float = 0.75
    val: float = 0.15
    test: float = 0.1

    @root_validator
    def validate_all(cls, fields):
        actual_sum = sum([*fields.values()])
        if actual_sum != 1.:
            raise ValueError(f'{fields.keys()} must sum to 1.0. Found: {actual_sum}')
        return fields

class TrackerConfig(BaseModel):
    tracker_type: str = 'console'           # Decoder currently supports console and wandb
    data_path: str = './models'             # The path where files will be saved locally
    init_config: Dict[str, Any] = None
    wandb_entity: str = ''                  # Only needs to be set if tracker_type is wandb
    wandb_project: str = ''
    verbose: bool = False                   # Whether to print console logging for non-console trackers

# diffusion prior pydantic classes

class AdapterConfig(BaseModel):
    make: str = "openai"
    model: str = "ViT-L/14"
    base_model_kwargs: Dict[str, Any] = None

    def create(self):
        if self.make == "openai":
            return OpenAIClipAdapter(self.model)
        elif self.make == "x-clip":
            return XClipAdapter(XCLIP(**self.base_model_kwargs))
        elif self.make == "coca":
            return CoCaAdapter(CoCa(**self.base_model_kwargs))
        else:
            raise AttributeError("No adapter with that name is available.")

class DiffusionPriorNetworkConfig(BaseModel):
    dim: int
    depth: int
    num_timesteps: int = None
    num_time_embeds: int = 1
    num_image_embeds: int = 1
    num_text_embeds: int = 1
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    norm_out: bool = True
    attn_dropout: float = 0.
    ff_dropout: float = 0.
    final_proj: bool = True
    normformer: bool = False
    rotary_emb: bool = True

    def create(self):
        kwargs = self.dict()
        return DiffusionPriorNetwork(**kwargs)

class DiffusionPriorConfig(BaseModel):
    clip: AdapterConfig = None
    net: DiffusionPriorNetworkConfig
    image_embed_dim: int
    image_size: int
    image_channels: int = 3
    timesteps: int = 1000
    cond_drop_prob: float = 0.
    loss_type: str = 'l2'
    predict_x_start: bool = True
    beta_schedule: str = 'cosine'
    condition_on_text_encodings: bool = True

    class Config:
        extra = "allow"

    def create(self):
        kwargs = self.dict()

        has_clip = exists(kwargs.pop('clip'))
        kwargs.pop('net')

        clip = None
        if has_clip:
            clip = self.clip.create()

        diffusion_prior_network = self.net.create()
        return DiffusionPrior(net = diffusion_prior_network, clip = clip, **kwargs)

class DiffusionPriorTrainConfig(BaseModel):
    epochs: int = 1
    lr: float = 1.1e-4
    wd: float = 6.02e-2
    max_grad_norm: float = 0.5
    use_ema: bool = True
    ema_beta: float = 0.99
    amp: bool = False
    save_every: int = 10000 # what steps to save on

class DiffusionPriorDataConfig(BaseModel):
    image_url: str     # path to embeddings folder
    meta_url: str      # path to metadata (captions) for images
    splits: TrainSplitConfig
    batch_size: int = 64

class DiffusionPriorLoadConfig(BaseModel):
    source: str = None
    resume: bool = False

class TrainDiffusionPriorConfig(BaseModel):
    prior: DiffusionPriorConfig
    data: DiffusionPriorDataConfig
    train: DiffusionPriorTrainConfig
    load: DiffusionPriorLoadConfig
    tracker: TrackerConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)

# decoder pydantic classes

class UnetConfig(BaseModel):
    dim: int
    dim_mults: ListOrTuple(int)
    image_embed_dim: int = None
    cond_dim: int = None
    channels: int = 3
    attn_dim_head: int = 32
    attn_heads: int = 16

    class Config:
        extra = "allow"

class DecoderConfig(BaseModel):
    unets: ListOrTuple(UnetConfig)
    image_size: int = None
    image_sizes: ListOrTuple(int) = None
    channels: int = 3
    timesteps: int = 1000
    loss_type: str = 'l2'
    beta_schedule: str = 'cosine'
    learned_variance: bool = True
    image_cond_drop_prob: float = 0.1
    text_cond_drop_prob: float = 0.5

    def create(self):
        decoder_kwargs = self.dict()
        unet_configs = decoder_kwargs.pop('unets')
        unets = [Unet(**config) for config in unet_configs]
        return Decoder(unets, **decoder_kwargs)

    @validator('image_sizes')
    def check_image_sizes(cls, image_sizes, values):
        if exists(values.get('image_size')) ^ exists(image_sizes):
            return image_sizes
        raise ValueError('either image_size or image_sizes is required, but not both')

    class Config:
        extra = "allow"

class DecoderDataConfig(BaseModel):
    webdataset_base_url: str     # path to a webdataset with jpg images
    embeddings_url: str          # path to .npy files with embeddings
    num_workers: int = 4
    batch_size: int = 64
    start_shard: int = 0
    end_shard: int = 9999999
    shard_width: int = 6
    index_width: int = 4
    splits: TrainSplitConfig
    shuffle_train: bool = True
    resample_train: bool = False
    preprocessing: Dict[str, Any] = {'ToTensor': True}

    @property
    def img_preproc(self):
        def _get_transformation(transformation_name, **kwargs):
            if transformation_name == "RandomResizedCrop":
                return T.RandomResizedCrop(**kwargs)
            elif transformation_name == "RandomHorizontalFlip":
                return T.RandomHorizontalFlip()
            elif transformation_name == "ToTensor":
                return T.ToTensor()

        transforms = []
        for transform_name, transform_kwargs_or_bool in self.preprocessing.items():
            transform_kwargs = {} if not isinstance(transform_kwargs_or_bool, dict) else transform_kwargs_or_bool
            transforms.append(_get_transformation(transform_name, **transform_kwargs))
        return T.Compose(transforms)

class DecoderTrainConfig(BaseModel):
    epochs: int = 20
    lr: SingularOrIterable(float) = 1e-4
    wd: SingularOrIterable(float) = 0.01
    max_grad_norm: SingularOrIterable(float) = 0.5
    save_every_n_samples: int = 100000
    n_sample_images: int = 6                       # The number of example images to produce when sampling the train and test dataset
    device: str = 'cuda:0'
    epoch_samples: int = None                      # Limits the number of samples per epoch. None means no limit. Required if resample_train is true as otherwise the number of samples per epoch is infinite.
    validation_samples: int = None                 # Same as above but for validation.
    use_ema: bool = True
    ema_beta: float = 0.999
    amp: bool = False
    save_all: bool = False                         # Whether to preserve all checkpoints
    save_latest: bool = True                       # Whether to always save the latest checkpoint
    save_best: bool = True                         # Whether to save the best checkpoint
    unet_training_mask: ListOrTuple(bool) = None   # If None, use all unets

class DecoderEvaluateConfig(BaseModel):
    n_evaluation_samples: int = 1000
    FID: Dict[str, Any] = None
    IS: Dict[str, Any] = None
    KID: Dict[str, Any] = None
    LPIPS: Dict[str, Any] = None

class DecoderLoadConfig(BaseModel):
    source: str = None                      # Supports file and wandb
    run_path: str = ''                      # Used only if source is wandb
    file_path: str = ''                     # The local filepath if source is file. If source is wandb, the relative path to the model file in wandb.
    resume: bool = False                    # If using wandb, whether to resume the run

class TrainDecoderConfig(BaseModel):
    decoder: DecoderConfig
    data: DecoderDataConfig
    train: DecoderTrainConfig
    evaluate: DecoderEvaluateConfig
    tracker: TrackerConfig
    load: DecoderLoadConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)
