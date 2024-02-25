import copy
import random
import typing

import tqdm

from .batchify import batchify

__all__ = ["iter"]


def iter(
    queries, documents, batch_size=512, epochs: int = 1, shuffle=True, tqdm_bar=True
) -> typing.Generator:
    """Iterate over the dataset.

    Parameters
    ----------
    queries
        List of queries paired with documents.
    documents
        List of documents paired with queries.
    batch_size
        Size of the batch.
    epochs
        Number of epochs.
    """
    step = 0
    queries = copy.deepcopy(x=queries)
    documents = copy.deepcopy(x=documents)

    bar = tqdm.tqdm(iterable=range(epochs), position=0) if tqdm_bar else range(epochs)

    for _ in bar:
        if shuffle:
            queries_documents = list(zip(queries, documents))
            random.shuffle(x=queries_documents)
            queries, documents = zip(*queries_documents)

        for batch_queries, batch_documents in zip(
            batchify(X=queries, batch_size=batch_size, tqdm_bar=False),
            batchify(X=documents, batch_size=batch_size, tqdm_bar=False),
        ):
            yield step, batch_queries, batch_documents
            step += 1
