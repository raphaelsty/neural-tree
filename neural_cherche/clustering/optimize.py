import collections

import numpy as np
from cherche import retrieve
from scipy import sparse
from scipy.sparse import csr_matrix, dok_matrix

__all__ = ["optimize_leafs"]


def create_sparse_matrix_retriever(
    candidates: list[list], mapping_documents: dict, key: str
) -> csr_matrix:
    """Build a sparse matrix (queries, documents) with 1 when document is relevant to a
    query."""
    query_documents_mapping = collections.defaultdict(list)
    for query, query_candidates in enumerate(iterable=candidates):
        for document in query_candidates:
            query_documents_mapping[query].append(mapping_documents[document[key]])

    query_documents_matrix = dok_matrix(
        arg1=(len(candidates), len(mapping_documents)), dtype=np.int8
    )

    for query, query_documents in query_documents_mapping.items():
        query_documents_matrix[query, query_documents] = 1

    return query_documents_matrix.tocsr()


def create_sparse_matrix_tree(
    candidates_tree: np.ndarray,
) -> tuple[sparse.csr_matrix, dict[int, list[int]]]:
    """Build a sparse matrix (queries, leafs) with 1 when leaf is relevant to a query."""
    leafs_to_queries = collections.defaultdict(list)

    for query, query_leafs in enumerate(iterable=candidates_tree):
        for leaf in query_leafs.tolist():
            leafs_to_queries[leaf].append(query)

    query_leafs_matrix = dok_matrix(
        arg1=(len(candidates_tree), len(leafs_to_queries)), dtype=np.int8
    )

    mapping_leafs = {
        leaf: index for index, leaf in enumerate(iterable=leafs_to_queries)
    }

    for leaf, leaf_queries in leafs_to_queries.items():
        for query in leaf_queries:
            query_leafs_matrix[query, mapping_leafs[leaf]] = 1

    return query_leafs_matrix.tocsr(), {
        index: leaf for leaf, index in mapping_leafs.items()
    }


def top_k(similarities, k: int):
    """Return the top k documents for each query."""
    similarities *= -1
    matchs = []
    for row in similarities:
        _k = min(row.data.shape[0] - 1, k)
        ind = np.argpartition(a=row.data, kth=_k, axis=0)[:k]
        similarity = np.take_along_axis(arr=row.data, indices=ind, axis=0)
        indices = np.take_along_axis(arr=row.indices, indices=ind, axis=0)
        ind = np.argsort(a=similarity, axis=0)
        matchs.append(np.take_along_axis(arr=indices, indices=ind, axis=0))
    return matchs


def optimize_leafs(
    tree,
    documents: list[dict],
    queries: list[str],
    k_tree: int = 2,
    k_retriever: int = 10,
    k_leafs: int = 2,
    **kwargs,
) -> dict:
    """Optimize the clusters."""
    mapping_documents = {
        document[tree.key]: index for index, document in enumerate(iterable=documents)
    }

    retriever = retrieve.TfIdf(key=tree.key, on=tree.scoring.on, documents=documents)
    query_documents_matrix = create_sparse_matrix_retriever(
        candidates=retriever(q=queries, k=k_retriever, batch_size=512, tqdm_bar=False),
        mapping_documents=mapping_documents,
        key=tree.key,
    )

    inverse_mapping_document = {
        index: document for document, index in mapping_documents.items()
    }

    query_leafs_matrix, inverse_mapping_leafs = create_sparse_matrix_tree(
        candidates_tree=tree(
            queries=queries,
            k=k_tree,
            score_documents=False,
            **kwargs,
        )["leafs"]
    )

    documents_to_leafs = collections.defaultdict(list)
    for document, leafs in enumerate(
        iterable=top_k(
            similarities=query_documents_matrix.T @ query_leafs_matrix,
            k=k_leafs,
        )
    ):
        for leaf in leafs.tolist():
            documents_to_leafs[inverse_mapping_document[document]].append(
                inverse_mapping_leafs[leaf]
            )

    return documents_to_leafs
