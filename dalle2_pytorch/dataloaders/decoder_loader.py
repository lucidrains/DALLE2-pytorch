import os
import webdataset as wds
import torch
import numpy as np
import fsspec

def get_shard(filename):
    """
    Filenames with shards in them have a consistent structure that we can take advantage of
    Standard structure: path/to/file/prefix_string_00001.ext
    """
    try:
        return filename.split("_")[-1].split(".")[0]
    except ValueError:
        raise RuntimeError(f"Could not find shard for filename {filename}")

def get_example_file(fs, path, file_format):
    """
    Given a file system and a file extension, return the example file
    """
    return fs.glob(os.path.join(path, f"*.{file_format}"))[0]

def embedding_inserter(samples, embeddings_url, shard_width, handler=wds.handlers.reraise_exception):
    """Given a datum of {"__key__": str, "__url__": str, ...} adds the cooresponding embedding and yields"""
    previous_tar_url = None
    current_embeddings = None
    # Get a reference to an abstract file system where the embeddings are stored
    embeddings_fs, embeddings_path = fsspec.core.url_to_fs(embeddings_url)
    example_embedding_file = get_example_file(embeddings_fs, embeddings_path, "npy")
    example_embedding_shard = get_shard(example_embedding_file)
    emb_shard_width = len(example_embedding_shard)
    # Easier to get the basename without the shard once than search through for the correct file every time
    embedding_file_basename = '_'.join(example_embedding_file.split("_")[:-1]) + "_"

    def load_corresponding_embeds(tar_url):
      """Finds and reads the npy files that contains embeddings for the given webdataset tar"""
      shard = int(tar_url.split("/")[-1].split(".")[0])
      embedding_url = embedding_file_basename + str(shard).zfill(emb_shard_width) + '.npy'
      with embeddings_fs.open(embedding_url) as f:
        data = np.load(f)
      return torch.from_numpy(data)

    for sample in samples:
        try:
            tar_url = sample["__url__"]
            key = sample["__key__"]
            if tar_url != previous_tar_url:
                # If the tar changed, we need to download new embeddings
                # This means if we shuffle before inserting it will load many more files than we expect and be very inefficient.
                previous_tar_url = tar_url
                current_embeddings = load_corresponding_embeds(tar_url)
                
            embedding_index = int(key[shard_width:])
            sample["npy"] = current_embeddings[embedding_index]
            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break
insert_embedding = wds.filters.pipelinefilter(embedding_inserter)

def verify_keys(samples, handler=wds.handlers.reraise_exception):
    """
    Requires that both the image and embedding are present in the sample
    This is important to do as a user may forget they do not have embeddings in their webdataset and neglect to add them using the embedding_folder_url parameter.
    """
    for sample in samples:
        try:
            assert "jpg" in sample, f"Sample {sample['__key__']} missing image"
            assert "npy" in sample, f"Sample {sample['__key__']} missing embedding. Did you set embedding_folder_url?"
            yield sample
        except Exception as exn:  # From wds implementation
            if handler(exn):
                continue
            else:
                break

class ImageEmbeddingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            embedding_folder_url=None,
            shard_width=None,
            handler=wds.handlers.reraise_exception,
            resample=False,
            shuffle_shards=True
    ):
        """
        Modeled directly off of the WebDataset constructor

        :param urls: A url pointing to the tar files of the webdataset formatted as /path/to/webdataset/{0000..9999}.tar
        :param embedding_folder_url: Required if webdataset does not contain embeddings. A url pointing to the npy files of the embeddings. Should have the same number of shards as the webdataset.
            Webdataset image keys should align with the index of the embedding. This means missing image indices must have a corresponding embedding of all zeros.
        :param shard_width: The number of digits in the shard number. This is used to align the embedding index with the image index.
            For example, if a file in the webdataset shard 3 is named 0003039.jpg, we know the shard with this 4 and the last three digits are the index.
        :param handler: A webdataset handler.
        :param resample: If true, resample webdataset shards with replacement. You need to set your own epoch size if this is true since it will resample infinitely.
        :param shuffle_shards: If true, shuffle the shards before resampling. This cannot be true if resample is true.
        """
        super().__init__()
        # Add the shardList and randomize or resample if requested
        if resample:
            assert not shuffle_shards, "Cannot both resample and shuffle"
            self.append(wds.ResampledShards(urls))
        else:
            self.append(wds.SimpleShardList(urls))
            if shuffle_shards:
                self.append(wds.filters.shuffle(1000))

        self.append(wds.split_by_node)
        self.append(wds.split_by_worker)

        self.append(wds.tarfile_to_samples(handler=handler))
        self.append(wds.decode("torchrgb"))
        if embedding_folder_url is not None:
            assert shard_width is not None, "Reading embeddings separately requires shard length to be given"
            self.append(insert_embedding(embeddings_url=embedding_folder_url, shard_width=shard_width, handler=handler))
        self.append(verify_keys)
        self.append(wds.to_tuple("jpg", "npy"))

def create_image_embedding_dataloader(
    tar_url,
    num_workers,
    batch_size,
    embeddings_url=None,
    shard_width=None,
    shuffle_num = None,
    shuffle_shards = True,
    resample_shards = False, 
    handler=wds.handlers.warn_and_continue
):
    """
    Convenience function to create an image embedding dataseta and dataloader in one line

    :param tar_url: A url pointing to the tar files of the webdataset formatted as /path/to/webdataset/{0000..9999}.tar
    :param num_workers: The number of workers to use for the dataloader
    :param batch_size: The batch size to use for the dataloader
    :param embeddings_url: Required if webdataset does not contain embeddings. A url pointing to the npy files of the embeddings. Should have the same number of shards as the webdataset.
        Webdataset image keys should align with the index of the embedding. This means missing image indices must have a corresponding embedding of all zeros.
    :param shard_width: The number of digits in the shard number. This is used to align the embedding index with the image index.
        For example, if a file in the webdataset shard 3 is named 0003039.jpg, we know the shard width is 4 and the last three digits are the index.
    :param shuffle_num: If not None, shuffle the dataset with this size buffer after sampling.
    :param shuffle_shards: If true, shuffle the shards before sampling. This cannot be true if resample is true.
    :param resample_shards: If true, resample webdataset shards with replacement. You need to set your own epoch size if this is true since it will resample infinitely.
    :param handler: A webdataset handler.
    """
    ds = ImageEmbeddingDataset(
        tar_url,
        embeddings_url,
        shard_width=shard_width,
        shuffle_shards=shuffle_shards,
        resample=resample_shards,
        handler=handler
    )
    if shuffle_num is not None and shuffle_num > 0:
        ds.shuffle(1000)
    return wds.WebLoader(
        ds,
        num_workers=num_workers,
        batch_size=batch_size,
        prefetch_factor=2,  # This might be good to have high so the next npy file is prefetched
        pin_memory=True,
        shuffle=False
    )