import os
import io
from urllib.parse import urlparse
import webdataset as wds
import requests
import torch
import numpy as np


def embedding_inserter(samples, embeddings_url, shard_width, imb_shard_width, handler=wds.handlers.reraise_exception):
    """Given a datum of {"__key__": str, "__url__": str, ...} adds the cooresponding embedding and yields"""
    previous_tar_url = None
    current_embeddings = None

    def load_corresponding_embeds(tar_url):  # TODO: Start downloading the next one in parallel? How do we get the next tar_url?
        """Finds and reads the npy files that contains embeddings for the given webdataset tar"""
        shard = int(tar_url.split("/")[-1].split(".")[0])
        embedding_url = os.path.join(embeddings_url, f'img_emb_{str(shard).zfill(imb_shard_width)}.npy')
        scheme = urlparse(embedding_url).scheme
        if len(scheme) > 0:
            response = requests.get(embedding_url)
            response.raise_for_status()
            data = np.load(io.BytesIO(response.content))
        else:
            data = np.load(embedding_url)
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

class ImageEmbedingDataset(wds.DataPipeline, wds.compat.FluidInterface):
    """
    A fluid interface wrapper for DataPipline that returns image embedding pairs
    Reads embeddings as npy files from the webdataset if they exist. If embedding_folder_url is set, they will be inserted in from the alternate source.
    """

    def __init__(
            self,
            urls,
            embedding_folder_url=None,
            shard_width=None,
            imb_shard_width=None,
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
        :param imb_shard_width: The number of digits in the shard number for the embedding. If an embedding file is named img_emb_0000.npy, there is an imb_shard_width of 4.
            This is often, but not always, the same as shard_width.
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
            assert imb_shard_width is not None, "Reading embedding separately requires embedding shard length to be given"
            self.append(insert_embedding(embeddings_url=embedding_folder_url, shard_width=shard_width, imb_shard_width=imb_shard_width, handler=handler))
        self.append(verify_keys)
        self.append(wds.to_tuple("jpg", "npy"))

def create_dataloader(
    tar_url,
    num_workers,
    batch_size,
    embeddings_url=None,
    shard_width=None,
    imb_shard_width=None,
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
        For example, if a file in the webdataset shard 3 is named 0003039.jpg, we know the shard with this 4 and the last three digits are the index.
    :param imb_shard_width: The number of digits in the shard number for the embedding. If an embedding file is named img_emb_0000.npy, there is an imb_shard_width of 4.
        This is often, but not always, the same as shard_width.
    :param shuffle_num: If not None, shuffle the dataset with this size buffer after sampling.
    :param shuffle_shards: If true, shuffle the shards before sampling. This cannot be true if resample is true.
    :param resample_shards: If true, resample webdataset shards with replacement. You need to set your own epoch size if this is true since it will resample infinitely.
    :param handler: A webdataset handler.
    """
    ds = ImageEmbedingDataset(
        tar_url,
        embeddings_url,
        shard_width=shard_width,
        imb_shard_width=imb_shard_width,
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