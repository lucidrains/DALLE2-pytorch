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

Any parameter from the `Decoder` constructor can also be given here.

**<ins>Data</ins>:**

Settings for creation of the dataloaders.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `webdataset_base_url` | Yes | N/A | The url of a shard in the webdataset with the shard replaced with `{}`[^1]. |
| `embeddings_url` | No | N/A | The url of the folder containing embeddings shards. Not required if embeddings are in webdataset. |
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
| `device` | No | `cuda:0` | The device to train on. |
| `epoch_samples` | No | `None` | Limits the number of samples iterated through in each epoch. This must be set if resampling. None means no limit. |
| `validation_samples` | No | `None` | The number of samples to use for validation. None mean the entire validation set. |
| `use_ema` | No | `True` | Whether to use exponential moving average models for sampling. |
| `ema_beta` | No | `0.99` | The ema coefficient. |
| `save_all` | No | `False` | If True, preserves a checkpoint for every epoch. |
| `save_latest` | No | `True` | If True, overwrites the `latest.pth` every time the model is saved. |
| `save_best` | No | `True` | If True, overwrites the `best.pth` every time the model has a lower validation loss than all previous models. |
| `unet_training_mask` | No | `None` | A boolean array of the same length as the number of unets. If false, the unet is frozen. A value of `None` trains all unets. |

**<ins>Evaluate</ins>:**

Defines which evaluation metrics will be used to test the model.
Each metric can be enabled by setting its configuration. The configuration keys for each metric are defined by the torchmetrics constructors which will be linked.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `n_evalation_samples` | No | `1000` | The number of samples to generate to test the model. |
| `FID` | No | `None` | Setting to an object enables the [Frechet Inception Distance](https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html) metric. 
| `IS` | No | `None` | Setting to an object enables the [Inception Score](https://torchmetrics.readthedocs.io/en/stable/image/inception_score.html) metric.
| `KID` | No | `None` | Setting to an object enables the [Kernel Inception Distance](https://torchmetrics.readthedocs.io/en/stable/image/kernel_inception_distance.html) metric. |
| `LPIPS` | No | `None` | Setting to an object enables the [Learned Perceptual Image Patch Similarity](https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html) metric. |

**<ins>Tracker</ins>:**

Selects which tracker to use and configures it.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `tracker_type` | No | `console` | Which tracker to use. Currently accepts `console` or `wandb`. |
| `data_path` | No | `./models` | Where the tracker will store local data. |
| `verbose` | No | `False` | Enables console logging for non-console trackers. |

Other configuration options are required for the specific trackers. To see which are required, reference the initializer parameters of each [tracker](../dalle2_pytorch/trackers.py).

**<ins>Load</ins>:**

Selects where to load a pretrained model from.
| Option | Required | Default | Description |
| ------ | -------- | ------- | ----------- |
| `source` | No | `None` | Supports `file` or `wandb`. |
| `resume` | No | `False` | If the tracker support resuming the run, resume it. |

Other configuration options are required for loading from a specific source. To see which are required, reference the load methods at the top of the [tracker file](../dalle2_pytorch/trackers.py).
