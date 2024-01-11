from math import ceil
from clip import tokenize
from embedding_reader import EmbeddingReader
from torch import from_numpy
from torch.utils.data import IterableDataset, DataLoader


class PriorEmbeddingDataset(IterableDataset):
    """
    PriorEmbeddingDataset is a wrapper of EmbeddingReader.

    It enables one to simplify the logic necessary to yield samples from
    the different EmbeddingReader configurations available.
    """

    def __init__(
        self,
        text_conditioned: bool,
        batch_size: int,
        start: int,
        stop: int,
        image_reader,
        text_reader: EmbeddingReader = None,
    ) -> None:
        super(PriorEmbeddingDataset).__init__()

        self.text_conditioned = text_conditioned

        if not self.text_conditioned:
            self.text_reader = text_reader

        self.image_reader = image_reader
        self.start = start
        self.stop = stop
        self.batch_size = batch_size

    def __len__(self):
        return self.stop - self.start

    def __iter__(self):
        # D.R.Y loader args
        loader_args = dict(
            batch_size=self.batch_size,
            start=self.start,
            end=self.stop,
            show_progress=False,
        )

        # if the data requested is text conditioned, only load images
        if self.text_conditioned:
            self.loader = self.image_reader(**loader_args)
        # otherwise, include text embeddings and bypass metadata
        else:
            self.loader = zip(
                self.image_reader(**loader_args), self.text_reader(**loader_args)
            )

        # return the data loader in its formatted state
        return self

    def __next__(self):
        try:
            return self.get_sample()
        except StopIteration:
            raise StopIteration

    def __str__(self):
        return f"<PriorEmbeddingDataset: start: {self.start}, stop: {self.stop}, len: {self.__len__()}>"

    def set_start(self, start):
        """
        Adjust the starting point within the reader, useful for resuming an epoch
        """
        self.start = start

    def get_start(self):
        return self.start

    def get_sample(self):
        """
        pre-proocess data from either reader into a common format
        """
        if self.text_conditioned:
            image_embedding, caption = next(self.loader)

            image_embedding = from_numpy(image_embedding)
            tokenized_caption = tokenize(caption["caption"].to_list(), truncate=True)

            return image_embedding, tokenized_caption

        else:
            (image_embedding, _), (text_embedding, _) = next(self.loader)

            image_embedding = from_numpy(image_embedding)
            text_embedding = from_numpy(text_embedding)

            return image_embedding, text_embedding


# helper functions


def distribute_to_rank(start, stop, rank, world_size):
    """
    Distribute data to each rank given the world size.

    Return:
        - New start and stop points for this rank.
    """
    num_samples = int(stop - start)

    per_rank = int(ceil((num_samples) / float(world_size)))

    assert (
        per_rank > 0
    ), f"Number of samples per rank must be larger than 0, (found: {per_rank})"

    rank_start = start + rank * per_rank

    rank_stop = min(rank_start + per_rank, stop)

    new_length = rank_stop - rank_start

    assert (
        new_length > 0
    ), "Calculated start and stop points result in a length of zero for this rank."

    return rank_start, rank_stop


def get_reader(
    text_conditioned: bool, img_url: str, meta_url: str = None, txt_url: str = None
):
    """
    Create an EmbeddingReader object from the specified URLs

    get_reader() will always expect a url to image embeddings.

    If text-conditioned, it will also expect a meta_url for the captions.
    Otherwise, it will need txt_url for the matching text embeddings.

    Returns an image_reader object if text-conditioned.
    Otherwise it returns both an image_reader and a text_reader
    """

    assert img_url is not None, "Must supply a image url"

    if text_conditioned:
        assert meta_url is not None, "Must supply meta url if text-conditioned"

        image_reader = EmbeddingReader(
            embeddings_folder=img_url,
            file_format="parquet_npy",
            # will assume the caption column exists and is the only one requested
            meta_columns=["caption"],
            metadata_folder=meta_url,
        )

        return image_reader

    # otherwise we will require text embeddings as well and return two readers
    assert (
        txt_url is not None
    ), "Must supply text embedding url if not text-conditioning"

    image_reader = EmbeddingReader(img_url, file_format="npy")
    text_reader = EmbeddingReader(txt_url, file_format="npy")

    return image_reader, text_reader


def make_splits(
    text_conditioned: bool,
    batch_size: int,
    num_data_points: int,
    train_split: float,
    eval_split: float,
    image_reader: EmbeddingReader,
    text_reader: EmbeddingReader = None,
    start=0,
    rank=0,
    world_size=1,
):
    """
    Split an embedding reader object as needed.

    NOTE: make_splits() will infer the test set size from your train and eval.

    Input:
        - text_conditioned: whether to prepare text-conditioned training data
        - batch_size: the batch size for a single gpu
        - num_data_points: the total number of data points you wish to train on
        - train_split: the percentage of data you wish to train on
        - eval_split: the percentage of data you wish to validate on
        - image_reader: the image_reader you wish to split
        - text_reader: the text_reader you want to split (if !text_conditioned)
        - start: the starting point within your dataset
        - rank: the rank of your worker
        - world_size: the total world size of your distributed training run

    Returns:
        - PyTorch Dataloaders that yield tuples of (img, txt) data.
    """

    assert start < image_reader.count, "start position cannot exceed reader count."

    # verify that the num_data_points does not exceed the max points
    if num_data_points > (image_reader.count - start):
        print(
            "Specified count is larger than what's available...defaulting to reader's count."
        )
        num_data_points = image_reader.count

    # compute split points
    train_set_size = int(train_split * num_data_points)
    eval_set_size = int(eval_split * num_data_points)
    eval_start = train_set_size
    eval_stop = int(eval_start + eval_set_size)

    assert (
        train_split + eval_split
    ) < 1.0, "Specified train and eval split is too large to infer a test split."

    # distribute to rank
    rank_train_start, rank_train_stop = distribute_to_rank(
        start, train_set_size, rank, world_size
    )
    rank_eval_start, rank_eval_stop = distribute_to_rank(
        train_set_size, eval_stop, rank, world_size
    )
    rank_test_start, rank_test_stop = distribute_to_rank(
        eval_stop, num_data_points, rank, world_size
    )

    # wrap up splits into a dict
    train_split_args = dict(
        start=rank_train_start, stop=rank_train_stop, batch_size=batch_size
    )
    eval_split_args = dict(
        start=rank_eval_start, stop=rank_eval_stop, batch_size=batch_size
    )
    test_split_args = dict(
        start=rank_test_start, stop=rank_test_stop, batch_size=batch_size
    )

    if text_conditioned:
        # add the text-conditioned args to a unified dict
        reader_args = dict(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
        )

        train_split_args = dict(**reader_args, **train_split_args)
        eval_split_args = dict(**reader_args, **eval_split_args)
        test_split_args = dict(**reader_args, **test_split_args)

        train = PriorEmbeddingDataset(**train_split_args)
        val = PriorEmbeddingDataset(**eval_split_args)
        test = PriorEmbeddingDataset(**test_split_args)

    else:
        # add the non-conditioned args to a unified dict
        reader_args = dict(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
            text_reader=text_reader,
        )

        train_split_args = dict(**reader_args, **train_split_args)
        eval_split_args = dict(**reader_args, **eval_split_args)
        test_split_args = dict(**reader_args, **test_split_args)

        train = PriorEmbeddingDataset(**train_split_args)
        val = PriorEmbeddingDataset(**eval_split_args)
        test = PriorEmbeddingDataset(**test_split_args)

    # true batch size is specifed in the PriorEmbeddingDataset
    train_loader = DataLoader(train, batch_size=None)
    eval_loader = DataLoader(val, batch_size=None)
    test_loader = DataLoader(test, batch_size=None)

    return train_loader, eval_loader, test_loader
