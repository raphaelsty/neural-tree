from neural_cherche import retrieve
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ["TfIdf"]


class TfIdf(retrieve.TfIdf):
    """TfIdf retriever"""

    def __init__(
        self,
        key: str,
        on: list[str],
    ) -> None:
        super().__init__(key=key, on=on, fit=False)
        self.tfidf = None

    def encode_documents(
        self, documents: list[dict], model: TfidfVectorizer
    ) -> dict[str, csr_matrix]:
        """Encode queries into sparse matrix."""
        content = [
            " ".join([doc.get(field, "") for field in self.on]) for doc in documents
        ]

        # matrix is a csr matrix of shape (n_documents, n_features)
        matrix = model.transform(raw_documents=content)
        return {document[self.key]: row for document, row in zip(documents, matrix)}

    def encode_queries(
        self, queries: list[str], model: TfidfVectorizer
    ) -> dict[str, csr_matrix]:
        """Encode queries into sparse matrix."""
        matrix = model.transform(raw_documents=queries)
        return {query: row for query, row in zip(queries, matrix)}
