from torch.utils.data import IterableDataset
from torch import from_numpy
from clip import tokenize
from embedding_reader import EmbeddingReader


class PriorEmbeddingLoader(IterableDataset):
    def __init__(
        self,
        text_conditioned: bool,
        batch_size: int,
        start: int,
        stop: int,
        image_reader,
        text_reader: EmbeddingReader = None,
        device: str = "cpu",
    ) -> None:
        super(PriorEmbeddingLoader).__init__()

        self.text_conditioned = text_conditioned

        if not self.text_conditioned:
            self.text_reader = text_reader

        self.image_reader = image_reader
        self.batch_size = batch_size
        self.start = start
        self.stop = stop
        self.device = device

    def __iter__(self):
        self.n = 0
        loader_args = dict(
            batch_size=self.batch_size,
            start=self.start,
            end=self.stop,
            show_progress=False,
        )
        if self.text_conditioned:
            self.loader = self.image_reader(**loader_args)
        else:
            self.loader = zip(
                self.image_reader(**loader_args), self.text_reader(**loader_args)
            )
        return self

    def __next__(self):
        try:
            return self.get_sample()
        except StopIteration:
            raise StopIteration

    def get_sample(self):
        """
        pre-proocess data from either reader into a common format
        """
        self.n += 1

        if self.text_conditioned:
            image_embedding, caption = next(self.loader)

            image_embedding = from_numpy(image_embedding).to(self.device)
            tokenized_caption = tokenize(
                caption["caption"].to_list(), truncate=True
            ).to(self.device)

            return image_embedding, tokenized_caption

        else:
            (image_embedding, _), (text_embedding, _) = next(self.loader)

            image_embedding = from_numpy(image_embedding).to(self.device)
            text_embedding = from_numpy(text_embedding).to(self.device)

            return image_embedding, text_embedding


def make_splits(
    text_conditioned: bool,
    batch_size: int,
    num_data_points: int,
    train_split: float,
    eval_split: float,
    device: str,
    img_url: str,
    meta_url: str = None,
    txt_url: str = None,
):

    assert img_url is not None, "Must supply some image embeddings"

    if text_conditioned:
        assert meta_url is not None, "Must supply metadata url if text-conditioning"
        image_reader = EmbeddingReader(
            embeddings_folder=img_url,
            file_format="parquet_npy",
            meta_columns=["caption"],
            metadata_folder=meta_url,
        )

        # compute split points
        if num_data_points > image_reader.count:
            print("Specified point count is larger than the number of points available...defaulting to max length of reader.")
            num_data_points = image_reader.count

        train_set_size = int(train_split * num_data_points)
        eval_set_size = int(eval_split * num_data_points)
        eval_stop = int(train_set_size + eval_set_size)

        train_loader = PriorEmbeddingLoader(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
            batch_size=batch_size,
            start=0,
            stop=train_set_size,
            device=device,
        )
        eval_loader = PriorEmbeddingLoader(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
            batch_size=batch_size,
            start=train_set_size,
            stop=eval_stop,
            device=device,
        )
        test_loader = PriorEmbeddingLoader(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
            batch_size=batch_size,
            start=eval_stop,
            stop=int(num_data_points),
            device=device,
        )

    else:
        assert (
            txt_url is not None
        ), "Must supply text embedding url if not text-conditioning"

        image_reader = EmbeddingReader(img_url, file_format="npy")
        text_reader = EmbeddingReader(txt_url, file_format="npy")

        # compute split points
        if num_data_points > image_reader.count:
            print("Specified point count is larger than the number of points available...defaulting to max length of reader.")
            num_data_points = image_reader.count

        train_set_size = int(train_split * num_data_points)
        eval_set_size = int(eval_split * num_data_points)
        eval_stop = int(train_set_size + eval_set_size)

        train_loader = PriorEmbeddingLoader(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
            text_reader=text_reader,
            batch_size=batch_size,
            start=0,
            stop=train_set_size,
            device=device,
        )
        eval_loader = PriorEmbeddingLoader(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
            text_reader=text_reader,
            batch_size=batch_size,
            start=train_set_size,
            stop=eval_stop,
            device=device,
        )
        test_loader = PriorEmbeddingLoader(
            text_conditioned=text_conditioned,
            image_reader=image_reader,
            text_reader=text_reader,
            batch_size=batch_size,
            start=eval_stop,
            stop=int(num_data_points),
            device=device,
        )

    return train_loader, eval_loader, test_loader
