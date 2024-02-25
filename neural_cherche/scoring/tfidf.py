import numpy as np
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from ..retrievers import TfIdf as tfidf_retriever
from ..utils import set_env
from .base import BaseScore

__all__ = ["TfIdf"]


class TfIdf(BaseScore):
    """TfIdf scoring function.

    Examples
    --------
    >>> from neural_tree import trees, scoring
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> from pprint import pprint

    >>> documents = [
    ...     {"id": 0, "text": "Paris is the capital of France."},
    ...     {"id": 1, "text": "Berlin is the capital of Germany."},
    ...     {"id": 2, "text": "Paris and Berlin are European cities."},
    ...     {"id": 3, "text": "Paris and Berlin are beautiful cities."},
    ... ]

    >>> tree = trees.Tree(
    ...    key="id",
    ...    documents=documents,
    ...    scoring=scoring.TfIdf(key="id", on=["text"], documents=documents),
    ...    leaf_balance_factor=1,
    ...    branch_balance_factor=2,
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
    array([[0.99999994, 0.63854915],
           [0.99999994, 0.72823119]])

    >>> candidates["leafs"]
    array([['100', '101'],
           ['110', '111']], dtype='<U3')

    >>> pprint(candidates["tree_scores"])
    [{'10': tensor(1.0000),
      '100': tensor(1.0000),
      '101': tensor(0.6385),
      '11': tensor(0.1076)},
     {'10': tensor(0.1076),
      '11': tensor(1.0000),
      '110': tensor(1.0000),
      '111': tensor(0.7282)}]

    >>> pprint(candidates["documents"])
    [[{'id': 0, 'leaf': '100', 'similarity': 0.9999999999999978},
      {'id': 1, 'leaf': '101', 'similarity': 0.39941742405759667}],
     [{'id': 2, 'leaf': '110', 'similarity': 0.9999999999999978},
      {'id': 3, 'leaf': '111', 'similarity': 0.5385719658738707}]]

    """

    def __init__(
        self,
        key: str,
        on: list | str,
        documents: list,
        tfidf_nodes: TfidfVectorizer | None = None,
        tfidf_documents: TfidfVectorizer | None = None,
        **kwargs,
    ) -> None:
        """Initialize the scoring function."""
        set_env()

        self.key = key
        self.on = [on] if isinstance(on, str) else on

        if tfidf_nodes is None:
            tfidf_nodes = TfidfVectorizer()

        if tfidf_documents is None:
            tfidf_documents = TfidfVectorizer(
                lowercase=True, ngram_range=(3, 7), analyzer="char_wb"
            )

        self.tfidf_nodes = tfidf_nodes.fit(
            raw_documents=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
        )

        self.model = tfidf_documents.fit(
            raw_documents=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
        )

    @property
    def distinct_documents_encoder(self) -> bool:
        """Return True if the encoder is distinct for documents and nodes."""
        return True

    def transform_queries(self, queries: list[str], **kwargs) -> sparse.csr_matrix:
        """Transform queries to embeddings."""
        return self.tfidf_nodes.transform(raw_documents=queries)

    def transform_documents(self, documents: list[dict], **kwargs) -> sparse.csr_matrix:
        """Transform documents to embeddings."""
        return self.tfidf_nodes.transform(
            raw_documents=[
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
        )

    def get_retriever(self) -> tfidf_retriever:
        """Create a retriever"""
        return tfidf_retriever(
            key=self.key,
            on=self.on,
        )

    def encode_queries_for_retrieval(self, queries: list[str]) -> sparse.csr_matrix:
        """Encode queries for retrieval."""
        return self.model.transform(raw_documents=queries)

    @staticmethod
    def convert_to_tensor(embeddings: sparse.csr_matrix, device: str) -> torch.Tensor:
        """Transform sparse matrix to tensor."""
        embeddings = embeddings.tocoo()
        return torch.sparse_coo_tensor(
            indices=torch.tensor(
                data=np.vstack(tup=(embeddings.row, embeddings.col)),
                dtype=torch.long,
                device=device,
            ),
            values=torch.tensor(
                data=embeddings.data, dtype=torch.float32, device=device
            ),
            size=torch.Size(embeddings.shape),
        ).to(device=device)

    @staticmethod
    def nodes_scores(
        queries_embeddings: torch.Tensor, nodes_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Score between queries and nodes embeddings."""
        return torch.max(
            input=torch.mm(
                input=queries_embeddings,
                mat2=nodes_embeddings.T,
            ).to_dense(),
            dim=1,
        ).values

    @staticmethod
    def leaf_scores(
        queries_embeddings: torch.Tensor, leaf_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Return the scores of the embeddings."""
        return (
            torch.mm(input=queries_embeddings, mat2=leaf_embedding.unsqueeze(dim=0).T)
            .to_dense()
            .flatten()
        )

    @staticmethod
    def stack(embeddings: list[sparse.csr_matrix]) -> sparse.csr_matrix:
        """Stack list of embeddings."""
        return (
            sparse.vstack(blocks=embeddings) if len(embeddings) > 1 else embeddings[0]
        )

    @staticmethod
    def average(embeddings: sparse.csr_matrix) -> sparse.csr_matrix:
        """Average embeddings."""
        return embeddings.mean(axis=0)
