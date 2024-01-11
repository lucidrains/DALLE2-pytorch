## Dataloaders
In order to make loading data simple and efficient, we include some general dataloaders that can be used to train portions of the network.

### Decoder: Image Embedding Dataset
When training the decoder (and up samplers if training together) in isolation, you will need to load images and corresponding image embeddings. This dataset can read two similar types of datasets. First, it can read a [webdataset](https://github.com/webdataset/webdataset) that contains `.jpg` and `.npy` files in the `.tar`s that contain the images and associated image embeddings respectively. Alternatively, you can also specify a source for the embeddings outside of the webdataset. In this case, the path to the embeddings should contain `.npy` files with the same shard numbers as the webdataset and there should be a correspondence between the filename of the `.jpg` and the index of the embedding in the `.npy`. So, for example, `0001.tar` from the webdataset with image `00010509.jpg` (the first 4 digits are the shard number and the last 4 are the index) in it should be paralleled by a `img_emb_0001.npy` which contains a NumPy array with the embedding at index 509.

Generating a dataset of this type:
1. Use [img2dataset](https://github.com/rom1504/img2dataset) to generate a webdataset.
2. Use [clip-retrieval](https://github.com/rom1504/clip-retrieval) to convert the images to embeddings.
3. Use [embedding-dataset-reordering](https://github.com/Veldrovive/embedding-dataset-reordering) to reorder the embeddings into the expected format.

Usage:
```python
from dalle2_pytorch.dataloaders import ImageEmbeddingDataset, create_image_embedding_dataloader

# Create a dataloader directly.
dataloader = create_image_embedding_dataloader(
    tar_url="/path/or/url/to/webdataset/{0000..9999}.tar", # Uses bracket expanding notation. This specifies to read all tars from 0000.tar to 9999.tar
    embeddings_url="path/or/url/to/embeddings/folder",     # Included if .npy files are not in webdataset. Left out or set to None otherwise
    num_workers=4,
    batch_size=32,
    shard_width=4,                                         # If a file in the webdataset shard 3 is named 0003039.jpg, we know the shard width is 4 and the last three digits are the index
    shuffle_num=200,                                       # Does a shuffle of the data with a buffer size of 200
    shuffle_shards=True,                                   # Shuffle the order the shards are read in
    resample_shards=False,                                 # Sample shards with replacement. If true, an epoch will be infinite unless stopped manually
)
for img, emb in dataloader:
    print(img.shape)  # torch.Size([32, 3, 256, 256])
    print(emb.shape)  # torch.Size([32, 512])
    # Train decoder only as shown above

# Or create a dataset without a loader so you can configure it manually
dataset = ImageEmbeddingDataset(
    urls="/path/or/url/to/webdataset/{0000..9999}.tar",
    embedding_folder_url="path/or/url/to/embeddings/folder",
    shard_width=4,
    shuffle_shards=True,
    resample=False
)
```

### Diffusion Prior: Prior Embedding Dataset
When training the prior it is much more efficient to work with pre-computed embeddings. The `PriorEmbeddingDataset` class enables you to leverage the same script (with minimal modification) for both embedding-only and text-conditioned prior training. This saves you from having to worry about a lot of the boilerplate code.

To utilize the `PriorEmbeddingDataset`, all you need to do is make a single call to `get_reader()` which will create `EmbeddingReader` object(s) for you. Afterwards, you can utilize `make_splits()` to cleanly create DataLoader objects from for your training run.

If you are training in a distributed manner, `make_splits()` accepts `rank` and `world_size` arguments to properly distribute to each process. The defaults for these values are `rank=0` and `world_size=1`, so single-process training can safely ignore these parameters.

Usage:
```python
from dalle2_pytorch.dataloaders import get_reader, make_splits

# grab embeddings from some specified location
IMG_URL = "data/img_emb/"
META_URL = "data/meta/"

reader = get_reader(text_conditioned=True, img_url=IMG_URL, meta_url=META_URL)

# some config for training
TRAIN_ARGS = {
    "world_size": 3,
    "text_conditioned": True,
    "start": 0,
    "num_data_points": 10000,
    "batch_size": 2,
    "train_split": 0.5,
    "eval_split": 0.25,
    "image_reader": reader,
}

# specifying a rank will handle allocation internally
rank0_train, rank0_eval, rank0_test = make_splits(rank=0, **TRAIN_ARGS)
rank1_train, rank1_eval, rank1_test = make_splits(rank=1, **TRAIN_ARGS)
rank2_train, rank2_eval, rank2_test = make_splits(rank=2, **TRAIN_ARGS)
```
