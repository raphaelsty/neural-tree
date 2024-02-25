import torch
from neural_cherche import rank

from .. import utils

__all__ = ["ColBERT"]


class ColBERT(rank.ColBERT):
    """ColBERT retriever."""

    def __init__(
        self,
        key: str,
        on: str | list[str],
        device: str,
    ) -> None:
        self.key = key
        self.on = on if isinstance(on, list) else [on]
        self.embeddings = {}
        self.documents = []
        self.device = device

    def add(self, documents_embeddings: dict) -> "ColBERT":
        """Add documents embeddings."""
        documents_embeddings = {
            document_id: embedding
            for document_id, embedding in documents_embeddings.items()
            if document_id not in self.embeddings
        }

        self.embeddings.update(documents_embeddings)
        self.documents.extend(
            [{self.key: document_id} for document_id in documents_embeddings.keys()]
        )

        return self

    def __call__(
        self,
        queries_embeddings: dict[str, torch.Tensor],
        batch_size: int = 32,
        k: int = None,
        tqdm_bar: bool = False,
    ) -> list[list[str]]:
        """Rank documents  givent queries.

        Parameters
        ----------
        queries
            Queries.
        documents
            Documents.
        queries_embeddings
            Queries embeddings.
        batch_size
            Batch size.
        tqdm_bar
            Show tqdm bar.
        k
            Number of documents to retrieve.
        """
        scores = []

        for query, embedding_query in queries_embeddings.items():
            query_scores = []

            embedding_query = embedding_query.to(device=self.device)

            for batch_documents in utils.batchify(
                X=self.documents,
                batch_size=batch_size,
                tqdm_bar=tqdm_bar,
            ):
                embeddings_batch_documents = torch.stack(
                    tensors=[
                        self.embeddings[document[self.key]]
                        for document in batch_documents
                    ],
                    dim=0,
                )

                query_documents_scores = torch.einsum(
                    "sh,bth->bst",
                    embedding_query,
                    embeddings_batch_documents,
                )

                query_scores.append(
                    query_documents_scores.max(dim=2).values.sum(axis=1)
                )

            scores.append(torch.cat(tensors=query_scores, dim=0))

        return self._rank(scores=scores, documents=self.documents, k=k)

    def _rank(
        self, scores: torch.Tensor, documents: list[dict], k: int
    ) -> list[list[dict]]:
        """Rank documents by scores.

        Parameters
        ----------
        scores
            Scores.
        documents
            Documents.
        k
            Number of documents to retrieve.
        """
        ranked = []

        for query_scores in scores:
            top_k = torch.topk(
                input=query_scores,
                k=min(k, len(self.documents)) if k is not None else len(self.documents),
                dim=-1,
            )

            ranked.append(
                [
                    {**self.documents[indice], "similarity": similarity}
                    for indice, similarity in zip(top_k.indices, top_k.values.tolist())
                ]
            )

        return ranked
