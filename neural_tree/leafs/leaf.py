__all__ = ["Leaf"]

import collections

import torch

from ..scoring import SentenceTransformer, TfIdf


class Leaf(torch.nn.Module):
    """Leaf class."""

    def __init__(
        self,
        key: str,
        level: int,
        documents: list,
        documents_embeddings: list,
        node_name: int,
        scoring: TfIdf | SentenceTransformer,
        parent: int = 0,
        create_retrievers: bool = True,
        **kwargs,
    ) -> None:
        super(Leaf, self).__init__()
        self.key = key
        self.level = level
        self.node_name = node_name
        self.parent = parent
        self.create_retrievers = create_retrievers

        self.documents = {}

        if self.create_retrievers:
            self.retriever = scoring.get_retriever()

        if scoring.distinct_documents_encoder:
            documents_embeddings = None
        elif self.create_retrievers:
            documents_embeddings = {
                document[self.key]: embedding
                for document, embedding in zip(documents, documents_embeddings)
            }

        self.add(
            scoring=scoring,
            documents=documents,
            documents_embeddings=documents_embeddings,
        )

    def __str__(self) -> str:
        """String representation of a leaf."""
        sep = "\t"
        return f"{self.level * sep} leaf {self.node_name}"

    def add(
        self,
        scoring: SentenceTransformer | TfIdf,
        documents: list,
        documents_embeddings: dict | None = None,
    ) -> "Leaf":
        """Add document to the leaf."""
        if not self.create_retrievers:
            # If we don't want to create retrievers for the leaves
            for document in documents:
                self.documents[document[self.key]] = True
            return self

        if documents_embeddings is None:
            documents_embeddings = self.retriever.encode_documents(
                documents=documents,
                model=scoring.model,
            )

        documents_embeddings = {
            document: embedding
            for document, embedding in documents_embeddings.items()
            if document not in self.documents
        }

        if not documents_embeddings:
            return self

        self.retriever.add(
            documents_embeddings=documents_embeddings,
        )

        for document in documents:
            self.documents[document[self.key]] = True

        return self

    def nodes_scores(
        self,
        scoring: SentenceTransformer | TfIdf,
        queries_embeddings: torch.Tensor,
        node_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the scores between the queries and the leaf."""
        return scoring.leaf_scores(
            queries_embeddings=queries_embeddings, leaf_embedding=node_embedding
        )

    def __call__(
        self,
        queries_embeddings,
        k: int,
    ) -> torch.Tensor:
        """Return scores between query and leaf documents."""
        if not self.documents:
            return [[] for _ in range(len(queries_embeddings))]

        candidates = self.retriever(
            queries_embeddings=queries_embeddings,
            tqdm_bar=False,
            k=k,
        )

        return [
            [{**document, "leaf": self.node_name} for document in query_documents]
            for query_documents in candidates
        ]

    def search(
        self,
        tree_scores: collections.defaultdict,
        **kwargs,
    ) -> tuple[torch.Tensor, list]:
        """Return the documents in the leaf."""
        return tree_scores

    def to_json(self) -> dict:
        """Return the leaf as a json."""
        return [{self.key: document} for document in self.documents]
