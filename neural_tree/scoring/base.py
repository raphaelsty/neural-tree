"""Base class for scoring functions."""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from scipy import sparse

__all__ = ["BaseScore"]


class BaseScore(ABC):
    """Base class for scoring functions."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def distinct_documents_encoder(self) -> bool:
        """Return True if the encoder is distinct for documents and nodes."""

    @abstractmethod
    def transform_queries(
        self, queries: list[str]
    ) -> sparse.csr_matrix | np.ndarray | dict:
        """Transform queries to embeddings."""

    @abstractmethod
    def transform_documents(
        self, documents: list[dict]
    ) -> sparse.csr_matrix | np.ndarray | dict:
        """Transform documents to embeddings."""

    @abstractmethod
    def get_retriever(self) -> Any:
        """Create a retriever"""

    @abstractmethod
    def encode_queries_for_retrieval(
        self, queries: list[str]
    ) -> sparse.csr_matrix | np.ndarray | dict:
        """Encode queries for retrieval."""

    @abstractmethod
    def convert_to_tensor(
        embeddings: sparse.csr_matrix | np.ndarray, device: str
    ) -> torch.Tensor:
        """Transform sparse matrix to tensor."""

    @abstractmethod
    def nodes_scores(
        queries_embeddings: torch.Tensor, nodes_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Score between queries and nodes embeddings."""

    @abstractmethod
    def leaf_scores(
        queries_embeddings: torch.Tensor, leaf_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Return the scores of the embeddings."""

    @abstractmethod
    def stack(
        embeddings: list[sparse.csr_matrix | np.ndarray | dict],
    ) -> sparse.csr_matrix | np.ndarray | dict:
        """Stack list of embeddings."""
