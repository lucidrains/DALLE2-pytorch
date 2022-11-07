import json
from torchvision import transforms as T
from pydantic import BaseModel, validator, root_validator
from typing import List, Optional, Union, Tuple, Dict, Any, TypeVar

from x_clip import CLIP as XCLIP
from open_clip import list_pretrained
from coca_pytorch import CoCa

from dalle2_pytorch.dalle2_pytorch import (
    CoCaAdapter,
    OpenAIClipAdapter,
    OpenClipAdapter,
    Unet,
    Decoder,
    DiffusionPrior,
    DiffusionPriorNetwork,
    XClipAdapter
)
from dalle2_pytorch.trackers import Tracker, create_loader, create_logger, create_saver

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

InnerType = TypeVar('InnerType')
ListOrTuple = Union[List[InnerType], Tuple[InnerType]]
SingularOrIterable = Union[InnerType, ListOrTuple[InnerType]]

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

class TrackerLogConfig(BaseModel):
    log_type: str = 'console'
    resume: bool = False  # For logs that are saved to unique locations, resume a previous run
    auto_resume: bool = False  # If the process crashes and restarts, resume from the run that crashed
    verbose: bool = False

    class Config:
        # Each individual log type has it's own arguments that will be passed through the config
        extra = "allow"

    def create(self, data_path: str):
        kwargs = self.dict()
        return create_logger(self.log_type, data_path, **kwargs)

class TrackerLoadConfig(BaseModel):
    load_from: Optional[str] = None
    only_auto_resume: bool = False  # Only attempt to load if the logger is auto-resuming

    class Config:
        extra = "allow"

    def create(self, data_path: str):
        kwargs = self.dict()
        if self.load_from is None:
            return None
        return create_loader(self.load_from, data_path, **kwargs)

class TrackerSaveConfig(BaseModel):
    save_to: str = 'local'
    save_all: bool = False
    save_latest: bool = True
    save_best: bool = True

    class Config:
        extra = "allow"

    def create(self, data_path: str):
        kwargs = self.dict()
        return create_saver(self.save_to, data_path, **kwargs)

class TrackerConfig(BaseModel):
    data_path: str = '.tracker_data'
    overwrite_data_path: bool = False
    log: TrackerLogConfig
    load: Optional[TrackerLoadConfig]
    save: Union[List[TrackerSaveConfig], TrackerSaveConfig]

    def create(self, full_config: BaseModel, extra_config: dict, dummy_mode: bool = False) -> Tracker:
        tracker = Tracker(self.data_path, dummy_mode=dummy_mode, overwrite_data_path=self.overwrite_data_path)
        # Add the logger
        tracker.add_logger(self.log.create(self.data_path))
        # Add the loader
        if self.load is not None:
            tracker.add_loader(self.load.create(self.data_path))
        # Add the saver or savers
        if isinstance(self.save, list):
            for save_config in self.save:
                tracker.add_saver(save_config.create(self.data_path))
        else:
            tracker.add_saver(self.save.create(self.data_path))
        # Initialize all the components and verify that all data is valid
        tracker.init(full_config, extra_config)
        return tracker

# diffusion prior pydantic classes

class AdapterConfig(BaseModel):
    make: str = "openai"
    model: str = "ViT-L/14"
    base_model_kwargs: Dict[str, Any] = None

    def create(self):
        if self.make == "openai":
            return OpenAIClipAdapter(self.model)
        elif self.make == "open_clip":
            pretrained = dict(list_pretrained())
            checkpoint = pretrained[self.model]
            return OpenClipAdapter(name=self.model, pretrained=checkpoint)
        elif self.make == "x-clip":
            return XClipAdapter(XCLIP(**self.base_model_kwargs))
        elif self.make == "coca":
            return CoCaAdapter(CoCa(**self.base_model_kwargs))
        else:
            raise AttributeError("No adapter with that name is available.")

class DiffusionPriorNetworkConfig(BaseModel):
    dim: int
    depth: int
    max_text_len: int = None
    num_timesteps: int = None
    num_time_embeds: int = 1
    num_image_embeds: int = 1
    num_text_embeds: int = 1
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4
    norm_in: bool = False
    norm_out: bool = True
    attn_dropout: float = 0.
    ff_dropout: float = 0.
    final_proj: bool = True
    normformer: bool = False
    rotary_emb: bool = True

    class Config:
        extra = "allow"

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
    sample_timesteps: Optional[int] = None
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
    warmup_steps: int = None             # number of warmup steps
    save_every_seconds: int = 3600       # how often to save
    eval_timesteps: List[int] = [64]     # which sampling timesteps to evaluate with
    best_validation_loss: float = 1e9    # the current best valudation loss observed
    current_epoch: int = 0               # the current epoch
    num_samples_seen: int = 0            # the current number of samples seen
    random_seed: int = 0                 # manual seed for torch

class DiffusionPriorDataConfig(BaseModel):
    image_url: str                   # path to embeddings folder
    meta_url: str                    # path to metadata (captions) for images
    splits: TrainSplitConfig         # define train, validation, test splits for your dataset
    batch_size: int                  # per-gpu batch size used to train the model
    num_data_points: int = 25e7      # total number of datapoints to train on
    eval_every_seconds: int = 3600   # validation statistics will be performed this often

class TrainDiffusionPriorConfig(BaseModel):
    prior: DiffusionPriorConfig
    data: DiffusionPriorDataConfig
    train: DiffusionPriorTrainConfig
    tracker: TrackerConfig

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)

# decoder pydantic classes

class UnetConfig(BaseModel):
    dim: int
    dim_mults: ListOrTuple[int]
    image_embed_dim: int = None
    text_embed_dim: int = None
    cond_on_text_encodings: bool = None
    cond_dim: int = None
    channels: int = 3
    self_attn: ListOrTuple[int]
    attn_dim_head: int = 32
    attn_heads: int = 16
    init_cross_embed: bool = True

    class Config:
        extra = "allow"

class DecoderConfig(BaseModel):
    unets: ListOrTuple[UnetConfig]
    image_size: int = None
    image_sizes: ListOrTuple[int] = None
    clip: Optional[AdapterConfig]   # The clip model to use if embeddings are not provided
    channels: int = 3
    timesteps: int = 1000
    sample_timesteps: Optional[SingularOrIterable[Optional[int]]] = None
    loss_type: str = 'l2'
    beta_schedule: ListOrTuple[str] = None  # None means all cosine
    learned_variance: SingularOrIterable[bool] = True
    image_cond_drop_prob: float = 0.1
    text_cond_drop_prob: float = 0.5

    def create(self):
        decoder_kwargs = self.dict()

        unet_configs = decoder_kwargs.pop('unets')
        unets = [Unet(**config) for config in unet_configs]

        has_clip = exists(decoder_kwargs.pop('clip'))
        clip = None
        if has_clip:
            clip = self.clip.create()

        return Decoder(unets, clip=clip, **decoder_kwargs)

    @validator('image_sizes')
    def check_image_sizes(cls, image_sizes, values):
        if exists(values.get('image_size')) ^ exists(image_sizes):
            return image_sizes
        raise ValueError('either image_size or image_sizes is required, but not both')

    class Config:
        extra = "allow"

class DecoderDataConfig(BaseModel):
    webdataset_base_url: str               # path to a webdataset with jpg images
    img_embeddings_url: Optional[str]      # path to .npy files with embeddings
    text_embeddings_url: Optional[str]     # path to .npy files with embeddings
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
    lr: SingularOrIterable[float] = 1e-4
    wd: SingularOrIterable[float] = 0.01
    warmup_steps: Optional[SingularOrIterable[int]] = None
    find_unused_parameters: bool = True
    static_graph: bool = True
    max_grad_norm: SingularOrIterable[float] = 0.5
    save_every_n_samples: int = 100000
    n_sample_images: int = 6                       # The number of example images to produce when sampling the train and test dataset
    cond_scale: Union[float, List[float]] = 1.0
    device: str = 'cuda:0'
    epoch_samples: int = None                      # Limits the number of samples per epoch. None means no limit. Required if resample_train is true as otherwise the number of samples per epoch is infinite.
    validation_samples: int = None                 # Same as above but for validation.
    save_immediately: bool = False
    use_ema: bool = True
    ema_beta: float = 0.999
    amp: bool = False
    unet_training_mask: ListOrTuple[bool] = None   # If None, use all unets

class DecoderEvaluateConfig(BaseModel):
    n_evaluation_samples: int = 1000
    FID: Dict[str, Any] = None
    IS: Dict[str, Any] = None
    KID: Dict[str, Any] = None
    LPIPS: Dict[str, Any] = None

class TrainDecoderConfig(BaseModel):
    decoder: DecoderConfig
    data: DecoderDataConfig
    train: DecoderTrainConfig
    evaluate: DecoderEvaluateConfig
    tracker: TrackerConfig
    seed: int = 0

    @classmethod
    def from_json_path(cls, json_path):
        with open(json_path) as f:
            config = json.load(f)
        return cls(**config)
    
    @root_validator
    def check_has_embeddings(cls, values):
        # Makes sure that enough information is provided to get the embeddings specified for training
        data_config, decoder_config = values.get('data'), values.get('decoder')

        if not exists(data_config) or not exists(decoder_config):
            # Then something else errored and we should just pass through
            return values

        using_text_embeddings = any([unet.cond_on_text_encodings for unet in decoder_config.unets])
        using_clip = exists(decoder_config.clip)
        img_emb_url = data_config.img_embeddings_url
        text_emb_url = data_config.text_embeddings_url

        if using_text_embeddings:
            # Then we need some way to get the embeddings
            assert using_clip or exists(text_emb_url), 'If text conditioning, either clip or text_embeddings_url must be provided'

        if using_clip:
            if using_text_embeddings:
                assert not exists(text_emb_url) or not exists(img_emb_url), 'Loaded clip, but also provided text_embeddings_url and img_embeddings_url. This is redundant. Remove the clip model or the text embeddings'
            else:
                assert not exists(img_emb_url), 'Loaded clip, but also provided img_embeddings_url. This is redundant. Remove the clip model or the embeddings'

        if text_emb_url:
            assert using_text_embeddings, "Text embeddings are being loaded, but text embeddings are not being conditioned on. This will slow down the dataloader for no reason."

        return values
