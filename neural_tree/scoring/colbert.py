import numpy as np
import torch
from neural_cherche import models

from ..retrievers import ColBERT as colbert_retriever
from ..utils import batchify, set_env
from .base import BaseScore

__all__ = ["ColBERT"]


class ColBERT(BaseScore):
    """TfIdf scoring function.

    Examples
    --------
    >>> from neural_tree import trees, scoring
    >>> from neural_cherche import models
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> from pprint import pprint
    >>> import torch

    >>> _ = torch.manual_seed(42)

    >>> documents = [
    ...     {"id": 0, "text": "Paris is the capital of France."},
    ...     {"id": 1, "text": "Berlin is the capital of Germany."},
    ...     {"id": 2, "text": "Paris and Berlin are European cities."},
    ...     {"id": 3, "text": "Paris and Berlin are beautiful cities."},
    ... ]

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-mpnet-base-v2",
    ...     embedding_size=128,
    ...     max_length_document=96,
    ...     max_length_query=32,
    ... )

    >>> tree = trees.ColBERTTree(
    ...    key="id",
    ...    on="text",
    ...    model=model,
    ...    documents=documents,
    ...    leaf_balance_factor=1,
    ...    branch_balance_factor=2,
    ...    n_jobs=1,
    ... )

    >>> print(tree)
    node 1
        node 10
            leaf 100
            leaf 101
        node 11
            leaf 110
            leaf 111

    >>> tree.leafs_to_documents
    {'100': [0], '101': [1], '110': [2], '111': [3]}

    >>> candidates = tree(
    ...    queries=["Paris is the capital of France.", "Paris and Berlin are European cities."],
    ...    k_leafs=2,
    ...    k=2,
    ... )

    >>> candidates["scores"]
    array([[28.12037659, 18.32332611],
           [29.28324509, 21.38923264]])

    >>> candidates["leafs"]
    array([['100', '101'],
           ['110', '111']], dtype='<U3')

    >>> pprint(candidates["tree_scores"])
    [{'10': tensor(28.1204),
      '100': tensor(28.1204),
      '101': tensor(18.3233),
      '11': tensor(20.9327)},
     {'10': tensor(21.6886),
      '11': tensor(29.2832),
      '110': tensor(29.2832),
      '111': tensor(21.3892)}]

    >>> pprint(candidates["documents"])
    [[{'id': 0, 'leaf': '100', 'similarity': 28.120376586914062},
      {'id': 1, 'leaf': '101', 'similarity': 18.323326110839844}],
     [{'id': 2, 'leaf': '110', 'similarity': 29.283245086669922},
      {'id': 3, 'leaf': '111', 'similarity': 21.389232635498047}]]

    """

    def __init__(
        self,
        key: str,
        on: list | str,
        documents: list,
        model: models.ColBERT = None,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        """Initialize the scoring function."""
        set_env()
        self.key = key
        self.on = [on] if isinstance(on, str) else on
        self.model = model
        self.device = device

    @property
    def distinct_documents_encoder(self) -> bool:
        """Return True if the encoder is distinct for documents and nodes."""
        return False

    def transform_queries(
        self, queries: list[str], batch_size: int, tqdm_bar: bool, *kwargs
    ) -> torch.Tensor:
        """Transform queries to embeddings."""
        queries_embeddings = []

        for batch in batchify(X=queries, batch_size=batch_size, tqdm_bar=tqdm_bar):
            queries_embeddings.append(
                self.model.encode(texts=batch, query_mode=True)["embeddings"]
            )

        return (
            queries_embeddings[0].to(self.device)
            if len(queries_embeddings) == 1
            else torch.cat(tensors=queries_embeddings, dim=0).to(device=self.device)
        )

    def transform_documents(
        self, documents: list[dict], batch_size: int, tqdm_bar: bool, **kwargs
    ) -> torch.Tensor:
        """Transform documents to embeddings."""
        documents_embeddings = []

        for batch in batchify(
            X=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
            batch_size=batch_size,
            tqdm_bar=tqdm_bar,
        ):
            documents_embeddings.append(
                self.model.encode(texts=batch, query_mode=False)["embeddings"]
            )

        return (
            documents_embeddings[0].to(self.device)
            if len(documents_embeddings) == 1
            else torch.cat(tensors=documents_embeddings, dim=0).to(device=self.device)
        )

    def get_retriever(self) -> None:
        """Create a retriever"""
        return colbert_retriever(key=self.key, on=self.on, device=self.device)

    def encode_queries_for_retrieval(self, queries: list[str]) -> None:
        """Encode queries for retrieval."""
        pass

    @staticmethod
    def convert_to_tensor(
        embeddings: np.ndarray | torch.Tensor, device: str
    ) -> torch.Tensor:
        """Transform sparse matrix to tensor."""
        if isinstance(embeddings, np.ndarray):
            return torch.tensor(data=embeddings, device=device, dtype=torch.float32)
        return embeddings.to(device=device)

    @staticmethod
    def nodes_scores(
        queries_embeddings: torch.Tensor, nodes_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Score between queries and nodes embeddings."""
        return torch.stack(
            tensors=[
                torch.einsum(
                    "sh,bth->bst",
                    query_embedding,
                    nodes_embeddings,
                )
                .max(dim=2)
                .values.sum(axis=1)
                .max(dim=0)
                .values
                for query_embedding in queries_embeddings
            ],
            dim=0,
        )

    @staticmethod
    def leaf_scores(
        queries_embeddings: torch.Tensor, leaf_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Return the scores of the embeddings."""
        return torch.stack(
            tensors=[
                torch.einsum(
                    "sh,th->st",
                    query_embedding,
                    leaf_embedding,
                )
                .max(dim=1)
                .values.sum()
                for query_embedding in queries_embeddings
            ],
            dim=0,
        )

    def stack(self, embeddings: list[torch.Tensor | np.ndarray]) -> torch.Tensor:
        """Stack list of embeddings."""
        if isinstance(embeddings, np.ndarray):
            return self.convert_to_tensor(
                embeddings=embeddings, device=self.model.device
            )
        return torch.stack(tensors=embeddings, dim=0)

    @staticmethod
    def average(embeddings: torch.Tensor) -> torch.Tensor:
        """Average embeddings."""
        return embeddings.mean(axis=0)
