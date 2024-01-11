## DALLE2 Training Configurations

For more complex configuration, we provide the option of using a configuration file instead of command line arguments.

### Decoder Trainer

The decoder trainer has 7 main configuration options. A full example of their use can be found in the [example decoder configuration](train_decoder_config.example.json).

**<ins>Unet</ins>:**

This is a single unet config, which belongs as an array nested under the decoder config as a list of `unets`

| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `dim`  | Yes      | N/A     | The starting channels of the unet. |
| `image_embed_dim` | Yes | N/A | The dimension of the image embeddings. |
| `dim_mults` | No | `(1, 2, 4, 8)` | The growth factors of the channels. |

Any parameter from the `Unet` constructor can also be given here.

**<ins>Decoder</ins>:**

Defines the configuration options for the decoder model. The unets defined above will automatically be inserted.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `unets` | Yes | N/A | A list of unets, using the configuration above |
| `image_sizes` | Yes | N/A | The resolution of the image after each upsampling step. The length of this array should be the number of unets defined. |
| `image_size` | Yes | N/A | Not used. Can be any number. |
| `timesteps` | No | `1000` | The number of diffusion timesteps used for generation. |
| `loss_type` | No | `l2` | The loss function. Options are `l1`, `huber`, or `l2`. |
| `beta_schedule` | No | `cosine` | The noising schedule. Options are `cosine`, `linear`, `quadratic`, `jsd`, or `sigmoid`. |
| `learned_variance` | No | `True` | Whether to learn the variance. |
| `clip` | No | `None` | The clip model to use if embeddings are being generated on the fly. Takes keys `make` and `model` with defaults `openai` and `ViT-L/14`. |

Any parameter from the `Decoder` constructor can also be given here.

**<ins>Data</ins>:**

Settings for creation of the dataloaders.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `webdataset_base_url` | Yes | N/A | The url of a shard in the webdataset with the shard replaced with `{}`[^1]. |
| `img_embeddings_url` | No | `None` | The url of the folder containing image embeddings shards. Not required if embeddings are in webdataset or clip is being used. |
| `text_embeddings_url` | No | `None` | The url of the folder containing text embeddings shards. Not required if embeddings are in webdataset or clip is being used. |
| `num_workers` | No | `4` | The number of workers used in the dataloader. |
| `batch_size` | No | `64` | The batch size. |
| `start_shard` | No | `0` | Defines the start of the shard range the dataset will recall. |
| `end_shard` | No | `9999999` | Defines the end of the shard range the dataset will recall. |
| `shard_width` | No | `6` | Defines the width of one webdataset shard number[^2]. |
| `index_width` | No | `4` | Defines the width of the index of a file inside a shard[^3]. |
| `splits` | No | `{ "train": 0.75, "val": 0.15, "test": 0.1 }` | Defines the proportion of shards that will be allocated to the training, validation, and testing datasets. |
| `shuffle_train` | No | `True` | Whether to shuffle the shards of the training dataset. |
| `resample_train` | No | `False` | If true, shards will be randomly sampled with replacement from the datasets making the epoch length infinite if a limit is not set. Cannot be enabled if `shuffle_train` is enabled. |
| `preprocessing` | No | `{ "ToTensor": True }` | Defines preprocessing applied to images from the datasets. |

[^1]: If your shard files have the paths `protocol://path/to/shard/00104.tar`, then the base url would be `protocol://path/to/shard/{}.tar`. If you are using a protocol like `s3`, you need to pipe the tars. For example `pipe:s3cmd get s3://bucket/path/{}.tar -`.

[^2]: This refers to the string length of the shard number for your webdataset shards. For instance, if your webdataset shard has the filename `00104.tar`, your shard length is 5.

[^3]: Inside the webdataset `tar`, you have files named something like `001045945.jpg`. 5 of these characters refer to the shard, and 4 refer to the index of the file in the webdataset (shard is `001041` and index is `5945`). The `index_width` in this case is 4.

**<ins>Train</ins>:**

Settings for controlling the training hyperparameters.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `epochs` | No | `20` | The number of epochs in the training run. |
| `lr` | No | `1e-4` | The learning rate. |
| `wd` | No | `0.01` | The weight decay. |
| `max_grad_norm`| No | `0.5` | The grad norm clipping. |
| `save_every_n_samples` | No | `100000` | Samples will be generated and a checkpoint will be saved every `save_every_n_samples` samples. |
| `cond_scale` | No | `1.0` | Conditioning scale to use for sampling. Can also be an array of values, one for each unet. |
| `device` | No | `cuda:0` | The device to train on. |
| `epoch_samples` | No | `None` | Limits the number of samples iterated through in each epoch. This must be set if resampling. None means no limit. |
| `validation_samples` | No | `None` | The number of samples to use for validation. None mean the entire validation set. |
| `use_ema` | No | `True` | Whether to use exponential moving average models for sampling. |
| `ema_beta` | No | `0.99` | The ema coefficient. |
| `unet_training_mask` | No | `None` | A boolean array of the same length as the number of unets. If false, the unet is frozen. A value of `None` trains all unets. |

**<ins>Evaluate</ins>:**

Defines which evaluation metrics will be used to test the model.
Each metric can be enabled by setting its configuration. The configuration keys for each metric are defined by the torchmetrics constructors which will be linked.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `n_evaluation_samples` | No | `1000` | The number of samples to generate to test the model. |
| `FID` | No | `None` | Setting to an object enables the [Frechet Inception Distance](https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html) metric. 
| `IS` | No | `None` | Setting to an object enables the [Inception Score](https://torchmetrics.readthedocs.io/en/stable/image/inception_score.html) metric.
| `KID` | No | `None` | Setting to an object enables the [Kernel Inception Distance](https://torchmetrics.readthedocs.io/en/stable/image/kernel_inception_distance.html) metric. |
| `LPIPS` | No | `None` | Setting to an object enables the [Learned Perceptual Image Patch Similarity](https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html) metric. |

**<ins>Tracker</ins>:**

Selects how the experiment will be tracked.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `data_path` | No | `./.tracker-data` | The path to the folder where temporary tracker data will be saved. |
| `overwrite_data_path` | No | `False` | If true, the data path will be overwritten. Otherwise, you need to delete it yourself. |
| `log` | Yes | N/A | Logging configuration. |
| `load` | No | `None` | Checkpoint loading configuration. |
| `save` | Yes | N/A | Checkpoint/Model saving configuration. |
Tracking is split up into three sections:
* Log: Where to save run metadata and image output. Options are `console` or `wandb`.
* Load: Where to load a checkpoint from. Options are `local`, `url`, or `wandb`.
* Save: Where to save a checkpoint to. Options are `local`, `huggingface`, or `wandb`.

**Logging:**

All loggers have the following keys:
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `log_type` | Yes | N/A | The type of logger class to use. |
| `resume` | No | `False` | For loggers that have the option to resume an old run, resume it using maually input parameters. |
| `auto_resume` | No | `False` | If true, the logger will attempt to resume an old run using parameters from that previous run. |

If using `console` there is no further configuration than setting `log_type` to `console`.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `log_type` | Yes | N/A | Must be `console`. |

If using `wandb`
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `log_type` | Yes | N/A | Must be `wandb`. |
| `wandb_entity` | Yes | N/A | The wandb entity to log to. |
| `wandb_project` | Yes | N/A | The wandb project save the run to. |
| `wandb_run_name` | No | `None` | The wandb run name. |
| `wandb_run_id` | No | `None` | The wandb run id. Used if resuming an old run. |

**Loading:**

All loaders have the following keys:
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `load_from` | Yes | N/A | The type of loader class to use. |
| `only_auto_resume` | No | `False` | If true, the loader will only load the model if the run is being auto resumed. |

If using `local`
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `load_from` | Yes | N/A | Must be `local`. |
| `file_path` | Yes | N/A | The path to the checkpoint file. |

If using `url`
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `load_from` | Yes | N/A | Must be `url`. |
| `url` | Yes | N/A | The url of the checkpoint file. |

If using `wandb`
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `load_from` | Yes | N/A | Must be `wandb`. |
| `wandb_run_path` | No | `None` | The wandb run path. If `None`, uses the run that is being resumed. |
| `wandb_file_path` | Yes | N/A | The path to the checkpoint file in the W&B file system. |

**Saving:**
Unlike `log` and `load`, `save` may be an array of options so that you can save to different locations in a run.

All save locations have these configuration options
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `save_to` | Yes | N/A | Must be `local`, `huggingface`, or `wandb`. |
| `save_latest_to` | No | `None` | Sets the relative path to save the latest model to. |
| `save_best_to` | No | `None` | Sets the relative path to save the best model to every time the model has a lower validation loss than all previous models. |
| `save_meta_to` | No | `None` | The path to save metadata files in. This includes the config files used to start the training. |
| `save_type` | No | `checkpoint` | The type of save. `checkpoint` saves a checkpoint, `model` saves a model without any fluff (Saves with ema if ema is enabled). |

If using `local`
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `save_to` | Yes | N/A | Must be `local`. |

If using `huggingface`
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `save_to` | Yes | N/A | Must be `huggingface`. |
| `huggingface_repo` | Yes | N/A | The huggingface repository to save to. |
| `token_path` | No | `None` | If logging in with the huggingface cli is not possible, point to a token file instead. |

If using `wandb`
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `save_to` | Yes | N/A | Must be `wandb`. |
| `wandb_run_path` | No | `None` | The wandb run path. If `None`, uses the current run. You will almost always want this to be `None`. |
