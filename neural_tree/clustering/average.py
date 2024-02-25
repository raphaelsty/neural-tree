import numpy as np
import torch
from scipy import sparse

__all__ = ["average", "get_mapping_nodes_documents"]


def average(
    key: str,
    documents: list,
    documents_embeddings: np.ndarray | sparse.csr_matrix,
    graph,
    scoring,
    device: str,
) -> tuple[torch.Tensor, list, list]:
    """Replace KMeans clustering with average clustering when an existing graph is provided.

    Examples
    --------
    >>> from neural_tree import clustering, scoring
    >>> import numpy as np

    >>> documents = [
    ...     {"id": 0, "text": "Paris is the capital of France."},
    ...     {"id": 1, "text": "Berlin is the capital of Germany."},
    ...     {"id": 2, "text": "Paris and Berlin are European cities."},
    ...     {"id": 3, "text": "Paris and Berlin are beautiful cities."},
    ... ]

    >>> documents_embeddings = np.array([
    ...     [1, 1],
    ...     [1, 2],
    ...     [10, 10],
    ...     [1, 3],
    ... ])

    >>> graph = {1: {11: {111: [{'id': 0}, {'id': 3}], 112: [{'id': 1}]}, 12: {121: [{'id': 2}], 122: [{'id': 3}]}}}

    >>> clustering.average(
    ...     key="id",
    ...     documents_embeddings=documents_embeddings,
    ...     documents=documents,
    ...     graph=graph[1],
    ...     scoring=scoring.SentenceTransformer(key="id", on=["text"], model=None),
    ... )

    """
    mapping_documents_embeddings = {
        document[key]: embedding
        for document, embedding in zip(documents, documents_embeddings)
    }

    mapping_nodes_documents = {
        node: get_mapping_nodes_documents(graph=graph[node]) for node in graph.keys()
    }

    mappings_nodes_embeddings = {
        node: scoring.average(
            scoring.stack(
                [
                    mapping_documents_embeddings[document[key]]
                    for document in node_documents
                ]
            )
        )
        for node, node_documents in mapping_nodes_documents.items()
    }

    mapping_documents_ids = {document[key]: document for document in documents}

    mappings_nodes_embeddings = list(mappings_nodes_embeddings.values())

    if isinstance(mappings_nodes_embeddings[0], np.ndarray):
        node_embeddings = torch.tensor(
            data=np.stack(arrays=mappings_nodes_embeddings),
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        )
    else:
        node_embeddings = torch.stack(tensors=mappings_nodes_embeddings, dim=0)
        node_embeddings = node_embeddings.to(device=device)
        node_embeddings.requires_grad = True

    extended_documents, extended_documents_embeddings, labels = [], [], []
    for node, node_documents in mapping_nodes_documents.items():
        extended_documents.extend(
            [mapping_documents_ids[document[key]] for document in node_documents]
        )
        extended_documents_embeddings.extend(
            [mapping_documents_embeddings[document[key]] for document in node_documents]
        )
        labels.extend([node] * len(node_documents))

    return (
        node_embeddings,
        labels,
        extended_documents,
        scoring.stack(extended_documents_embeddings),
    )


def get_mapping_nodes_documents(graph: dict | list, documents: list | None = None):
    """Get documents from specific node."""
    if documents is None:
        documents = []

    if isinstance(graph, list):
        documents.extend(graph)
        return documents

    for node, child in graph.items():
        if isinstance(child, dict):
            documents = get_mapping_nodes_documents(graph=child, documents=documents)
        else:
            documents.extend(child)
    return documents
