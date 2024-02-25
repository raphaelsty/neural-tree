import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..retrievers import SentenceTransformer as SentenceTransformerRetriever
from ..utils import set_env

__all__ = ["SentenceTransformer"]


class SentenceTransformer:
    """Sentence Transformer scoring function.

    Examples
    --------
    >>> from neural_tree import trees, scoring
    >>> from sentence_transformers import SentenceTransformer
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
    ...    scoring=scoring.SentenceTransformer(key="id", on=["text"], model=SentenceTransformer("all-mpnet-base-v2")),
    ...    leaf_balance_factor=1,
    ...    branch_balance_factor=2,
    ...    n_jobs=1,
    ... )

    >>> print(tree)
    node 1
        node 11
            node 110
                leaf 1100
                leaf 1101
            leaf 111
        leaf 10

    >>> candidates = tree(
    ...    queries=["paris", "berlin"],
    ...    k_leafs=2,
    ... )

    >>> candidates["scores"]
    array([[0.72453916, 0.60635257],
           [0.58386189, 0.57546711]])

    >>> candidates["leafs"]
    array([['111', '10'],
           ['1101', '1100']], dtype='<U4')

    >>> pprint(candidates["tree_scores"])
    [{'10': tensor(0.6064),
      '11': tensor(0.7245),
      '110': tensor(0.5542),
      '1100': tensor(0.5403),
      '1101': tensor(0.5542),
      '111': tensor(0.7245)},
     {'10': tensor(0.5206),
      '11': tensor(0.5797),
      '110': tensor(0.5839),
      '1100': tensor(0.5755),
      '1101': tensor(0.5839),
      '111': tensor(0.4026)}]

    >>> pprint(candidates["documents"])
    [[{'id': 0, 'leaf': '111', 'similarity': 0.6447779347587058},
      {'id': 1, 'leaf': '10', 'similarity': 0.43175890864117644}],
     [{'id': 3, 'leaf': '1101', 'similarity': 0.545769273959571},
      {'id': 2, 'leaf': '1100', 'similarity': 0.54081365990618}]]

    """

    def __init__(
        self,
        key: str,
        on: str | list,
        model: SentenceTransformer,
        device: str = "cpu",
        faiss_device: str = "cpu",
    ) -> None:
        set_env()
        self.key = key
        self.on = [on] if isinstance(on, str) else on
        self.model = model
        self.device = device
        self.faiss_device = faiss_device

    @property
    def distinct_documents_encoder(self) -> bool:
        """Return True if the encoder is distinct for documents and nodes."""
        return False

    def transform_queries(self, queries: list[str], batch_size: int) -> np.ndarray:
        """Transform queries to embeddings."""
        return self.model.encode(queries, batch_size=batch_size)

    def transform_documents(self, documents: list[dict], batch_size: int) -> np.ndarray:
        """Transform documents to embeddings."""
        return self.model.encode(
            [
                " ".join([document[field] for field in self.on])
                for document in documents
            ],
            batch_size=batch_size,
        )

    def get_retriever(self) -> None:
        """Create a retriever"""
        return SentenceTransformerRetriever(key=self.key, device=self.faiss_device)

    @staticmethod
    def encode_queries_for_retrieval(queries: list[str]) -> None:
        """Encode queries for retrieval."""
        pass

    @staticmethod
    def convert_to_tensor(embeddings: np.ndarray, device: str) -> torch.Tensor:
        """Convert numpy array to torch tensor."""
        return torch.tensor(data=embeddings, device=device, dtype=torch.float32)

    @staticmethod
    def nodes_scores(
        queries_embeddings: torch.Tensor, nodes_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Score between queries and nodes embeddings."""
        return torch.max(
            input=torch.mm(input=queries_embeddings, mat2=nodes_embeddings.T),
            dim=1,
        ).values

    @staticmethod
    def leaf_scores(
        queries_embeddings: torch.Tensor, leaf_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Computes scores between query and leaf embedding."""
        return queries_embeddings @ leaf_embedding.T

    @staticmethod
    def stack(embeddings: list[np.ndarray]) -> np.ndarray:
        """Stack embeddings."""
        return (
            np.vstack(tup=embeddings)
            if len(embeddings) > 1
            else embeddings[0].reshape(1, -1)
        )

    @staticmethod
    def average(embeddings: np.ndarray) -> np.ndarray:
        """Average embeddings."""
        return np.mean(a=embeddings, axis=0)
