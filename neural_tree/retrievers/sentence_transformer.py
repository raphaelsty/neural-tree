from typing import Any

import numpy as np

__all__ = ["SentenceTransformer"]


class SentenceTransformer:
    """Sentence Transformer retriever.

    Examples
    --------
    >>> from neural_tree import retrievers
    >>> from sentence_transformers import SentenceTransformer
    >>> from pprint import pprint

    >>> model = SentenceTransformer("all-mpnet-base-v2")

    >>> retriever = retrievers.SentenceTransformer(key="id")

    >>> retriever = retriever.add(
    ...     documents_embeddings={
    ...         0: model.encode("Paris is the capital of France."),
    ...         1: model.encode("Berlin is the capital of Germany."),
    ...         2: model.encode("Paris and Berlin are European cities."),
    ...         3: model.encode("Paris and Berlin are beautiful cities."),
    ...     }
    ... )

    >>> queries_embeddings = {
    ...     0: model.encode("Paris"),
    ...     1: model.encode("Berlin"),
    ... }

    >>> candidates = retriever(queries_embeddings=queries_embeddings, k=2)
    >>> pprint(candidates)
    [[{'id': 0, 'similarity': 0.644777984318611},
      {'id': 3, 'similarity': 0.52865785276988}],
     [{'id': 1, 'similarity': 0.6901492368348436},
      {'id': 3, 'similarity': 0.5457692206973245}]]

    """

    def __init__(self, key: str, device: str = "cpu") -> None:
        self.key = key
        self.device = device
        self.index = None
        self.documents = []

    def _build(self, embeddings: np.ndarray) -> Any:
        """Build faiss index.

        Parameters
        ----------
        index
            faiss index.
        embeddings
            Embeddings of the documents.

        """
        if self.index is None:
            try:
                import faiss
            except:
                raise ImportError(
                    'Run pip install "neural-tree[cpu]" or pip install "neural-tree[gpu]" to run faiss on cpu / gpu.'
                )
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            if self.device == "cuda":
                try:
                    self.index = faiss.index_cpu_to_gpu(
                        faiss.StandardGpuResources(), 0, self.index
                    )
                except:
                    raise ImportError(
                        'Run pip install "neural-tree[gpu]" to run faiss on gpu.'
                    )

        if not self.index.is_trained and embeddings:
            self.index.train(embeddings)

        self.index.add(embeddings)
        return self.index

    def add(self, documents_embeddings: dict[int, np.ndarray]) -> "SentenceTransformer":
        """Add documents to the faiss index."""
        self.documents.extend(list(documents_embeddings.keys()))
        self.index = self._build(
            embeddings=np.array(object=list(documents_embeddings.values()))
        )
        return self

    def __call__(
        self,
        queries_embeddings: dict[int, np.ndarray],
        k: int | None = 100,
        **kwargs,
    ) -> list:
        """Retrieve documents."""
        if k is None:
            k = len(self.documents)

        k = min(k, len(self.documents))
        queries_embeddings = np.array(object=list(queries_embeddings.values()))
        distances, indexes = self.index.search(queries_embeddings, k)
        matchs = np.take(a=self.documents, indices=np.where(indexes < 0, 0, indexes))
        rank = []
        for distance, index, match in zip(distances, indexes, matchs):
            rank.append(
                [
                    {
                        self.key: m,
                        "similarity": 1 / (1 + d),
                    }
                    for d, idx, m in zip(distance, index, match)
                    if idx > -1
                ]
            )

        return rank
